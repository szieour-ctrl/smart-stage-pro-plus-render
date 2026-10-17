// reelRender.js — Reels v1 production render (Oct 17, 2026).
//
// Payload (from Netlify reel-render.js):
//   { jobId, event, lengthSec, data: {address, city, neighborhood, price, beds, baths, sqft, headline, cta, ...intake,
//     highlights[], supporting[]},  scenes: [{ imageUrl, motion, badgeType, roomLabel }],   // one per scene, in order
//     (a scene may carry reveal: {beforeUrl, preset} = Before -> After wipe, then `motion` as the end motion)
//     brand: {licensedName, dre, brokerage, primary, secondary, border, logoUrl, headshotUrl} | null,
//     qrUrl, musicStyle }
// Output: BRANDED (only when brand is present) and UNBRANDED (MLS) mp4s in the private
// smart-stage-reel-drafts/ S3 prefix. Text-free base (native 9:16 Ken Burns from the full photo, Vision
// focal point) + sharp PNG overlays composited by ffmpeg. Beta: no charging here.
const fs = require("fs");
const os = require("os");
const path = require("path");
const axios = require("axios");
const { execFile } = require("child_process");
const QRCode = require("qrcode");
const { downloadFrames } = require("./downloadFrames");
const { mapWithConcurrencyLimit, REVEAL_PRESETS, REVEAL_OPENER_DURATION, REVEAL_WIPE_DURATION, buildRevealClip } = require("./assemble");
const { computeFocals } = require("./verticalPipeline");
const { applyMotionPreset, VALID_PRESETS } = require("./motionPresets");
const { toVerticalKenBurns } = require("./verticalMotionMap");
const { generateMusic } = require("./musicGen");
const sb = require("./reelStoryboard");
const { renderBadgeLayer, renderQrLayer, probeDuration } = require("./reelOverlay");
const { uploadReelDraft } = require("./s3Upload");

const run = (cmd, args) => new Promise((resolve, reject) => execFile(cmd, args, { maxBuffer: 1 << 26 }, (e, so, se) => (e ? reject(new Error(`${cmd} failed: ${String(se).slice(-400)}`)) : resolve(so))));

function verticalPreset(name) {
  if (!name || name === "auto") return "auto";
  const mapped = toVerticalKenBurns(name);
  return mapped && VALID_PRESETS.has(mapped) ? mapped : "push_in";
}
const num = (v) => (v !== undefined && v !== null && v !== "" && !isNaN(Number(v)) ? Number(v).toLocaleString("en-US") : v);

async function fetchImg(label, u, warnings) {
  if (!u) return { buf: null, type: "" };
  try {
    const r = await axios.get(u, { responseType: "arraybuffer", timeout: 15000 });
    return { buf: Buffer.from(r.data), type: String(r.headers["content-type"] || "") };
  } catch (e) { warnings.push(`${label} skipped: could not fetch (${e.message})`); return { buf: null, type: "" }; }
}

async function renderReel(job, onStep = () => {}) {
  const { jobId, event, lengthSec } = job;
  const scenes = job.scenes || [];
  const n = scenes.length;
  if (!sb.EVENTS.includes(event)) throw new Error(`Unknown Reel event "${event}"`);
  const range = sb.REEL_PHOTO_RANGES[lengthSec];
  if (!range) throw new Error(`Unsupported Reel length ${lengthSec}s`);
  if (n < range[0] || n > range[1]) throw new Error(`A ${lengthSec}s Reel uses ${range[0]}-${range[1]} photos (got ${n})`);

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), `reel-${jobId}-`));
  const warnings = [];
  const timings = {};
  try {
    // 1. photos + subject focal points
    onStep("Preparing photos");
    let t = Date.now();
    const reveals = scenes.map((s) => !!(s.reveal && s.reveal.beforeUrl));
    const durations = sb.allocateDurations(lengthSec, reveals);
    if (!durations) throw new Error(`Too many Before -> After reveals for a ${lengthSec}s Reel (each reveal scene needs at least ${sb.REEL_REVEAL_MIN}s). Switch some photos to Standard Motion or choose a longer Reel.`);
    const frames = await downloadFrames(scenes.map((s, i) => ({ imageUrl: s.imageUrl, ...(reveals[i] ? { beforeUrl: s.reveal.beforeUrl, isBeforeAfter: true, useRevealEffect: true } : {}) })), workDir);
    const focals = await computeFocals(frames.map((f, i) => ({ ...f, roomLabel: scenes[i].roomLabel || null })), workDir);
    timings.photosMs = Date.now() - t;

    // 2. one Ken Burns clip per photo, equal share of the Reel length, hard cuts
    onStep("Rendering motion");
    t = Date.now();
    // Clips are independent (hard cuts, no zoom carry-over), so render a few at a time.
    const clips = await mapWithConcurrencyLimit(scenes, 3, async (sc, i) => {
      const base = { ...frames[i], roomLabel: sc.roomLabel || null, vertical: true, focal: focals[i] || { fx: 0.5, fy: 0.5 } };
      if (reveals[i] && !frames[i].beforeLocalPath) warnings.push(`Scene ${i + 1}: the Original photo could not be downloaded, so the Before wipe was skipped (staged photo only).`);
      if (reveals[i] && frames[i].beforeLocalPath) {
        // Before -> After: the original photo ("Original" label, no badge), a wipe to the staged photo, then the end motion.
        const rp = REVEAL_PRESETS[sc.reveal.preset] ? sc.reveal.preset : "classic_reveal";
        const contDur = Math.max(1.0, durations[i] - REVEAL_OPENER_DURATION + REVEAL_WIPE_DURATION);
        const opener = await applyMotionPreset({ ...base, localPath: frames[i].beforeLocalPath, motionPreset: REVEAL_PRESETS[rp].openerMotion,
          durationSeconds: REVEAL_OPENER_DURATION, labelText: "Original" }, workDir, 1.0);
        const cont = await applyMotionPreset({ ...base, motionPreset: verticalPreset(sc.motion === "auto" ? "push_in" : sc.motion), durationSeconds: contDur }, workDir, 1.0);
        return buildRevealClip(opener.path, cont.path, rp, workDir, `reveal_${i}.mp4`, contDur, { w: 1080, h: 1920 });
      }
      const c = await applyMotionPreset({ ...base, motionPreset: verticalPreset(sc.motion), durationSeconds: durations[i] }, workDir, 1.0);
      return c.path;
    });
    const windows = [];
    let cursor = 0;
    for (const c of clips) { const d = await probeDuration(c); windows.push({ start: cursor, end: cursor + d }); cursor += d; }
    const listFile = path.join(workDir, "clips.txt");
    fs.writeFileSync(listFile, clips.map((c) => `file '${c}'`).join("\n"));
    const silentBase = path.join(workDir, "base_silent.mp4");
    await run("ffmpeg", ["-y", "-f", "concat", "-safe", "0", "-i", listFile, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", silentBase]);
    const total = await probeDuration(silentBase);
    timings.motionMs = Date.now() - t;

    // 3. music bed (no narration in v1): ~0.3s fade-in, ~1s fade-out, exactly the Reel length
    onStep("Adding music");
    t = Date.now();
    const music = await generateMusic({ durationSeconds: Math.ceil(total) + 1, musicStyle: job.musicStyle || "default", workDir });
    const basePath = path.join(workDir, "base.mp4");
    await run("ffmpeg", ["-y", "-i", silentBase, "-i", music, "-filter_complex",
      `[1:a]atrim=0:${total.toFixed(2)},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.3,afade=t=out:st=${Math.max(0, total - 1).toFixed(2)}:d=1,volume=0.9[a]`,
      "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", basePath]);
    timings.musicMs = Date.now() - t;

    // 4. overlays (storyboard text, details, closing panel, AB 723 badge, QR)
    onStep("Adding text and compliance overlays");
    t = Date.now();
    const d = { ...(job.data || {}), sqft: num((job.data || {}).sqft) };
    const specs = sb.resolveScenes(event, d, n);
    const brand = job.brand || null;
    const [logo, headshot] = brand ? await Promise.all([fetchImg("logo", brand.logoUrl, warnings), fetchImg("headshot", brand.headshotUrl, warnings)]) : [{}, {}];
    const common = { address: d.address || "", warnings };
    const variants = [];
    if (brand) variants.push({ key: "branded", params: { ...common, brand, logoBuf: logo.buf, logoType: logo.type, headshotBuf: headshot.buf, headshotType: headshot.type } });
    variants.push({ key: "unbranded", params: { ...common, unbranded: true, brand: {}, logoBuf: null, headshotBuf: null } });

    // AB 723 badges: one full-length layer when every scene needs the same badge; else per scene.
    const badgeTypes = scenes.map((s) => (s.badgeType && s.badgeType !== "none" ? s.badgeType : null));
    const typesUsed = [...new Set(badgeTypes.filter(Boolean))];
    const badgePngs = {};
    for (const bt of typesUsed) badgePngs[bt] = await renderBadgeLayer({ badgeType: bt }, path.join(workDir, `badge_${bt}.png`));
    const allSame = typesUsed.length === 1 && badgeTypes.every((b) => b === typesUsed[0]) && !reveals.some(Boolean);
    const badgeOpts = allSame ? { badgePng: badgePngs[typesUsed[0]] }
      : { badgeLayers: badgeTypes.map((bt, i) => (bt ? { png: badgePngs[bt], startSec: windows[i].start + (reveals[i] ? REVEAL_OPENER_DURATION : 0), endSec: windows[i].end } : null)).filter(Boolean) };
    const qrBuf = await QRCode.toBuffer(job.qrUrl || "https://smartstagepro.com", { margin: 1, width: 300 });
    const qrPng = await renderQrLayer({ qrPng: qrBuf }, path.join(workDir, "qr.png"));
    timings.overlayMs = Date.now() - t;

    // 5. composite + upload each variant
    onStep("Finishing the Reel");
    t = Date.now();
    const out = {};
    // One variant at a time: two parallel composites doubled peak memory and got the job OOM-killed.
    for (const v of variants) {
      const pngs = [];
      for (let i = 0; i < specs.length; i++) pngs.push(await sb.renderScenePng(specs[i], v.params, path.join(workDir, `scene_${v.key}_${i}.png`), workDir, `${v.key}${i}`));
      const mp4 = await sb.compositeStoryboard(basePath, pngs, path.join(workDir, `reel_${v.key}.mp4`), { ...badgeOpts, qrPng, windows, textOffsets: reveals.map((r) => (r ? REVEAL_OPENER_DURATION - REVEAL_WIPE_DURATION : 0.35)) });
      out[v.key] = await uploadReelDraft(mp4, `smart-stage-reel-drafts/${jobId}/${v.key}_${Date.now()}.mp4`);
    }
    timings.compositeMs = Date.now() - t;
    return { keys: out, durationSec: Math.round(total * 10) / 10, scenes: n, timingsMs: timings, warnings };
  } finally {
    fs.rm(workDir, { recursive: true, force: true }, () => {});
  }
}

module.exports = { renderReel };
