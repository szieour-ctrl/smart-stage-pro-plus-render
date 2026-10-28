// agentMarketingRender.js — renders an Agent Marketing Reel from a frozen manifest (Oct 26, 2026).
//
// Reads ONLY the manifest and the prepared assets (backgrounds, narration, headshot); it makes no calls to GPT Image 2, ElevenLabs or any
// other outside service, so a failed render can be run again without paying for anything twice. One output per selected format.
// Everything with words or numbers is a transparent PNG drawn by agentMarketingCards.js; this file only moves them through time:
//   background plate -> slow Ken Burns clip per scene (exact frame counts, from the narration windows)
//   narration + music -> one audio bed
//   cards + caption lines -> faded onto the video by reelOverlay.compositeLayers (the same function Reel v1 uses, unchanged)
// Afterwards the finished file is checked: frames are pulled at the opening and closing and the MetroList notice is compared
// with the one the code drew.
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const sharp = require("sharp");
const { execFile } = require("child_process");
const cards = require("./agentMarketingCards");
const zones = require("./agentMarketingZones");
const styleMod = require("./agentMarketingStyle");
const { compositeLayers, probeDuration } = require("./reelOverlay");

const FPS = 25;
const FORMAT_CANVAS = { "9x16": "portrait", "16x9": "landscape" };
const run = (cmd, args) => new Promise((resolve, reject) => execFile(cmd, args, { maxBuffer: 1 << 26 }, (e, so, se) => (e ? reject(new Error(`${cmd} failed: ${String(se).slice(-500)}`)) : resolve(so))));

// ── background motion ─────────────────────────────────────────────────────────────────────────────────────────
const CENTER_X = "iw/2-(iw/zoom/2)", CENTER_Y = "ih/2-(ih/zoom/2)";
function motionExpr(preset, frames) {
  const N = Math.max(1, frames - 1);
  switch (preset) {
    case "pull_back": return { z: `1.08-0.08*on/${N}`, x: CENTER_X, y: CENTER_Y };
    case "pan_left": return { z: "1.08", x: `(iw-iw/zoom)*(1-on/${N})`, y: CENTER_Y };
    case "pan_right": return { z: "1.08", x: `(iw-iw/zoom)*on/${N}`, y: CENTER_Y };
    case "tilt_up": return { z: "1.08", x: CENTER_X, y: `(ih-ih/zoom)*(1-on/${N})` };
    case "tilt_down": return { z: "1.08", x: CENTER_X, y: `(ih-ih/zoom)*on/${N}` };
    case "static_hold": return { z: "1", x: CENTER_X, y: CENTER_Y };
    default: return { z: `1+0.08*on/${N}`, x: CENTER_X, y: CENTER_Y }; // push_in
  }
}
/** One scene's background: the plate, slowly moving, exactly `frames` frames at 25 fps. */
async function backgroundClip(imagePath, frames, preset, size, outPath) {
  const { z, x, y } = motionExpr(preset, frames);
  const vf = `scale=${size.w * 2}:${size.h * 2}:force_original_aspect_ratio=increase,crop=${size.w * 2}:${size.h * 2},` +
    `zoompan=z='${z}':x='${x}':y='${y}':d=${frames}:s=${size.w}x${size.h}:fps=${FPS},format=yuv420p`;
  await run("ffmpeg", ["-y", "-threads", "2", "-i", imagePath, "-vf", vf, "-frames:v", String(frames), "-r", String(FPS), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", outPath]);
  return outPath;
}

// ── captions ──────────────────────────────────────────────────────────────────────────────────────────────────
/** A caption line as a transparent full-frame PNG, drawn inside caption_zone: white text on a soft dark pill. */
async function captionLayer(text, canvas, brandFont, outPath) {
  const cv = cards.CANVAS[canvas], zone = zones.zonesFor("stat", canvas).caption_zone, portrait = canvas === "portrait";
  const size = portrait ? 54 : 44, family = brandFont || "Montserrat", maxW = portrait ? 900 : 1500;
  const words = String(text).split(/\s+/).filter(Boolean);
  const approx = (s) => s.length * size * 0.55;
  let lines = [words.join(" ")];
  if (approx(lines[0]) > maxW) { // two balanced lines
    let best = null;
    for (let k = 1; k < words.length; k++) {
      const a = words.slice(0, k).join(" "), b = words.slice(k).join(" ");
      const score = Math.max(a.length, b.length);
      if (!best || score < best.score) best = { a, b, score };
    }
    lines = best ? [best.a, best.b] : lines;
  }
  const lh = Math.round(size * 1.22), blockH = lines.length * lh;
  const w = Math.min(cv.w - 100, Math.round(Math.max(...lines.map(approx)) + 80));
  const x0 = Math.round((cv.w - w) / 2), y0 = Math.round(zone.y + (zone.h - blockH) / 2 - 14), h = blockH + 28;
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}">
  <rect x="${x0}" y="${y0}" width="${w}" height="${h}" rx="18" fill="#000000" fill-opacity="0.58"/>
  ${lines.map((l, i) => `<text x="${cv.w / 2}" y="${Math.round(y0 + 14 + size * 0.88 + i * lh)}" text-anchor="middle" font-family="${family}" font-weight="700" font-size="${size}" fill="#FFFFFF">${esc(l)}</text>`).join("\n  ")}
</svg>`;
  await sharp(Buffer.from(svg)).png().toFile(outPath);
  return outPath;
}

// ── narration voice shaping ──────────────────────────────────────────────────────────────
// manifest.audio.voice_shape = { semitones: -1.5, bass_db: 3 }: lowers the narration's pitch (a little deeper, bigger voice) and adds
// low-end warmth. Timing is untouched: the pitch drop is a sample-rate change that atempo cancels, so every word stays where the
// timestamps say it is. Absent or zero = the voice as ElevenLabs made it.
function voiceShapeFilter(shape) {
  const st = Math.max(-4, Math.min(0, Number(shape && shape.semitones) || 0));
  const bass = Math.max(0, Math.min(8, Number(shape && shape.bass_db) || 0));
  const parts = [];
  if (st < 0) { const f = Math.pow(2, st / 12); parts.push(`aresample=44100,asetrate=${(44100 * f).toFixed(2)},aresample=44100,atempo=${(1 / f).toFixed(6)}`); }
  if (bass > 0) parts.push(`bass=g=${bass}:f=150:w=0.6`);
  return parts.length ? parts.join(",") + "," : "";
}

// ── cards from the manifest ───────────────────────────────────────────────────────────────────────────────────
// text_style "image" (default): ink, accent and soft fades follow the picture behind each scene; the brand contributes only the
// headshot, name and brokerage. "brand" keeps the legacy look (Brand DNA colours + font).
const imageMode = (m) => !(m.brand && m.brand.text_style === "brand");
function cardParams(m, scene, headshotBuf, style) {
  const b = m.brand || {};
  const brand = style ? { border: b.border } : { primary: b.primary, secondary: b.secondary, border: b.border, font: b.font };
  // A web-research Reel carries a code-built source line instead of the MetroList notice (notice_required:false); the card's footer
  // names the publishers, not MetroList.
  const sl = m.notice_required === false ? m.source_line : null;
  if (m.notice_required === false && !(sl && sl.text)) throw new Error("This Reel has no source line and no MetroList notice; one of them is required.");
  // A MetroList Market Summary has no date range, only its printed publication line; the Netlify-built notice text carries that line word for word.
  const printed = !sl && m.notice && m.notice.disclosure && m.notice.text ? m.notice.text : null;
  const common = { period: sl || printed ? undefined : m.notice && m.notice.period, ...(printed ? { noticeText: printed } : {}), ...(sl ? { noticeText: sl.text, sourceLabel: sl.label || "Sources" } : {}), brand, agent: b.agent, headshotBuf, style: style || undefined };
  return { ...scene.data, ...common };
}
/** Zones where words sit directly on the picture (opaque panels and the notice bar are not part of it). */
function textZones(scene, canvas, opts) {
  const z = zones.zonesFor(scene.composition, canvas, opts);
  const out = [z.headline_zone || z.editorial_content_zone, z.agent_name_zone, z.agent_identity_zone].filter(Boolean);
  return out;
}
async function renderSceneCard(m, scene, canvas, headshotBuf, outPng, plateBuf) {
  const map = zones.COMPOSITION_LAYOUT[scene.composition];
  if (!map) throw new Error(`Scene ${scene.n}: no renderer for composition "${scene.composition}"`);
  const [kind, layout] = map;
  const cv = cards.CANVAS[canvas];
  const zopts = { side: (scene.layout_options || {}).side, headshot: !!headshotBuf };
  const lay = layout === "headline_focus" ? {} : { layout };
  const fn = { opening: cards.renderOpeningCard, closing: cards.renderClosingCard, stat: cards.renderStatCard, compare: cards.renderCompareCard, expert: cards.renderExpertCard }[kind];
  const tz = textZones(scene, canvas, zopts);
  let style = null, legibility = null;
  if (imageMode(m) && plateBuf) style = await styleMod.styleFor(plateBuf, cv, tz);
  let r;
  for (let attempt = 0; ; attempt++) {
    r = await fn({ ...cardParams(m, scene, headshotBuf, style), ...lay, ...(scene.layout_options || {}) }, outPng, canvas);
    if (!r.fits) throw new Error(`Scene ${scene.n} (${scene.composition}) does not fit its zone on ${canvas}`);
    if (!style) break;
    // Measure what was actually drawn against the picture; if too much of the ink is under the floor, add scrim and draw again.
    legibility = { share: 0, ...(await styleMod.checkLegibility(outPng, plateBuf, cv, style.ink, tz[0])) };
    if (legibility.ratioOfInkPixels <= 0.04 || attempt >= 2 || style.scrim >= 1.6) break;
    style = { ...style, scrim: Math.min(1.6, (style.scrim || 0) + 0.5) };
  }
  const verdict = await zones.checkCard(scene.composition, canvas, outPng, zopts);
  if (!verdict.ok) throw new Error(`Scene ${scene.n} (${scene.composition}) layout problem on ${canvas}: ${verdict.problems.join("; ")}`);
  const warnings = [...(r.warnings || [])];
  if (legibility && legibility.ratioOfInkPixels > 0.04) warnings.push(`text over the picture is hard to read in ${Math.round(legibility.ratioOfInkPixels * 100)}% of the headline area`);
  return { png: outPng, warnings, style: style && { ink: style.ink, accent: style.accent, scrim: style.scrim }, legibility };
}

// ── after-the-fact checks ─────────────────────────────────────────────────────────────────────────────────────
/** Pull one frame from the finished video. */
async function frameAt(video, sec, outPng) {
  await run("ffmpeg", ["-y", "-ss", sec.toFixed(2), "-i", video, "-frames:v", "1", outPng]);
  return outPng;
}
/**
 * Compare the MetroList notice bar in a finished frame with the one the code drew. The bar is opaque ink with white text, so the same
 * pixels must show up in the video (within compression noise). Returns { ok, meanDiff, frame }.
 */
async function verifyNoticeInFrame(video, sec, cardPng, canvas, outDir, label) {
  const zone = zones.zonesFor("opening", canvas).disclosure_zone;
  const framePng = await frameAt(video, sec, path.join(outDir, `${label}.png`));
  const read = async (p) => sharp(p).removeAlpha().extract({ left: zone.x, top: zone.y, width: zone.w, height: zone.h }).raw().toBuffer();
  const [a, b] = await Promise.all([read(framePng), read(cardPng)]);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  const meanDiff = sum / a.length;
  return { ok: meanDiff < 12, meanDiff: Math.round(meanDiff * 100) / 100, frame: framePng, sec: Math.round(sec * 100) / 100, region: zone };
}

// ── the render ────────────────────────────────────────────────────────────────────────────────────────────────
/**
 * renderAgentMarketing(job, onStep, deps)
 *   job   { jobId, manifest, assets: { [key]: url }, outDir? }   manifest is the validated am-manifest-2 document
 *   deps  { fetchAsset(key) -> Buffer, getMusic(workDir, seconds, style) -> path, upload(localPath, key) -> key }  (all injectable for tests)
 * Returns { outputs: { "9x16": { key|path, ... } }, checks, warnings }.
 */
async function renderAgentMarketing(job, onStep = () => {}, deps = {}) {
  const m = job.manifest;
  if (!m || m.version !== "am-manifest-2") throw new Error("Not an Agent Marketing manifest (am-manifest-2)");
  const formats = (m.formats || []).filter((f) => FORMAT_CANVAS[f]);
  if (!formats.length) throw new Error("The manifest names no format to render");
  const fetchAsset = deps.fetchAsset || (async (key) => {
    const axios = require("axios");
    const url = job.assets && job.assets[key];
    if (!url) throw new Error(`No download link for asset "${key}"`);
    const r = await axios.get(url, { responseType: "arraybuffer", timeout: 60000 });
    return Buffer.from(r.data);
  });
  const getMusic = deps.getMusic || (async (workDir, seconds, style) => require("./musicGen").generateMusic({ durationSeconds: Math.ceil(seconds) + 1, musicStyle: style || "default", workDir }));
  const upload = deps.upload || (async (p, key) => require("./s3Upload").uploadReelDraft(p, key));

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), `am-${job.jobId}-`));
  const warnings = [], outputs = {}, checks = {};
  try {
    onStep("Fetching the prepared assets");
    const headshotBuf = m.brand && m.brand.headshot_key ? await fetchAsset(m.brand.headshot_key) : null;
    const narrationPath = path.join(workDir, "narration.mp3");
    fs.writeFileSync(narrationPath, await fetchAsset(m.audio.narration_key));
    const total = m.scenes[m.scenes.length - 1].window.end;

    for (const format of formats) {
      const canvas = FORMAT_CANVAS[format], cv = cards.CANVAS[canvas], size = { w: cv.w, h: cv.h };
      const dir = path.join(workDir, format);
      fs.mkdirSync(dir, { recursive: true });

      onStep(`Rendering the ${format} backgrounds`);
      const clips = [];
      for (const sc of m.scenes) {
        const img = path.join(dir, `plate_${sc.n}`);
        fs.writeFileSync(img, await fetchAsset(sc.visual.image_keys[format]));
        const frames = Math.max(1, Math.round(sc.window.end * FPS) - Math.round(sc.window.start * FPS));
        clips.push(await backgroundClip(img, frames, sc.visual.motion || "push_in", size, path.join(dir, `clip_${sc.n}.mp4`)));
      }
      const list = path.join(dir, "clips.txt");
      fs.writeFileSync(list, clips.map((c) => `file '${c}'`).join("\n"));
      const silent = path.join(dir, "silent.mp4");
      await run("ffmpeg", ["-y", "-f", "concat", "-safe", "0", "-i", list, "-c", "copy", "-an", silent]);

      onStep(`Mixing the ${format} audio`);
      const music = await getMusic(workDir, total, job.musicStyle);
      const base = path.join(dir, "base.mp4");
      await run("ffmpeg", ["-y", "-i", silent, "-i", narrationPath, "-i", music, "-filter_complex",
        `[1:a]${voiceShapeFilter(m.audio && m.audio.voice_shape)}apad,atrim=0:${total.toFixed(3)},asetpts=PTS-STARTPTS[v];` +
        `[2:a]atrim=0:${total.toFixed(3)},asetpts=PTS-STARTPTS,volume=0.13,afade=t=in:st=0:d=0.5,afade=t=out:st=${Math.max(0, total - 1.2).toFixed(2)}:d=1.2[m];` +
        `[v][m]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]`,
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", base]);

      onStep(`Drawing the ${format} cards and captions`);
      const layers = [], noticeCards = [];
      for (let i = 0; i < m.scenes.length; i++) {
        const sc = m.scenes[i], last = i === m.scenes.length - 1;
        const r = await renderSceneCard(m, sc, canvas, headshotBuf, path.join(dir, `card_${sc.n}.png`), fs.readFileSync(path.join(dir, `plate_${sc.n}`)));
        (checks.style = checks.style || {})[`${format}_${sc.n}`] = { style: r.style, legibility: r.legibility && r.legibility.ratioOfInkPixels };
        r.warnings.forEach((w) => warnings.push(`${format} scene ${sc.n}: ${w}`));
        // a figure card can be told to wait for the moment its number is spoken (card_start_sec); the picture and captions play meanwhile
        const start = Number.isFinite(sc.card_start_sec) && sc.card_start_sec > sc.window.start && sc.card_start_sec < sc.window.end - 1 ? sc.card_start_sec : sc.window.start + (i === 0 ? 0.2 : 0.15);
        const l = { png: r.png, startSec: start, fadeSec: 0.45 };
        if (!last) { l.endSec = sc.window.end - 0.02; l.fadeOutSec = 0.3; }
        layers.push(l);
        if (/^(opening|closing)_/.test(sc.composition)) noticeCards.push({ png: r.png, mid: last ? Math.max(start + 0.55, Math.min(total - 0.6, start + 1.6)) : Math.min(start + 1.6, (start + 0.5 + sc.window.end - 0.32) / 2), label: `notice_scene${sc.n}` });
      }
      for (let k = 0; k < m.captions.length; k++) {
        const c = m.captions[k];
        const png = await captionLayer(c.text, canvas, imageMode(m) ? undefined : (m.brand && m.brand.font), path.join(dir, `cap_${k}.png`));
        layers.push({ png, startSec: c.start, endSec: Math.min(total, c.end), fadeSec: 0.12, fadeOutSec: 0.12 });
      }

      onStep(`Finishing the ${format} video`);
      const outMp4 = path.join(dir, `agent_marketing_${format}.mp4`);
      await compositeLayers(base, layers, outMp4);
      const dur = await probeDuration(outMp4);

      onStep(`Checking the ${format} video`);
      const checkDir = job.outDir ? path.join(job.outDir, "checks", format) : path.join(dir, "checks");
      fs.mkdirSync(checkDir, { recursive: true });
      const notices = [];
      for (const nc of noticeCards) notices.push(await verifyNoticeInFrame(outMp4, nc.mid, nc.png, canvas, checkDir, nc.label));
      checks[format] = { durationSec: Math.round(dur * 100) / 100, notice: notices, noticeOk: notices.length > 0 && notices.every((n) => n.ok) };
      if (!checks[format].noticeOk) throw new Error(`The MetroList notice could not be confirmed in the finished ${format} video`);

      if (job.outDir) {
        const dest = path.join(job.outDir, `agent_marketing_${format}.mp4`);
        fs.mkdirSync(job.outDir, { recursive: true });
        fs.copyFileSync(outMp4, dest);
        outputs[format] = { path: dest };
      } else {
        outputs[format] = { key: await upload(outMp4, `smart-stage-reel-drafts/agent-marketing/${job.jobId}/${format}_${Date.now()}.mp4`) };
      }
    }
    const keys = {}; for (const f of Object.keys(outputs)) if (outputs[f].key) keys[f] = outputs[f].key;
    return { outputs, keys, checks, warnings, durationSec: total };
  } finally {
    fs.rm(workDir, { recursive: true, force: true }, () => {});
  }
}

module.exports = { renderAgentMarketing, backgroundClip, captionLayer, verifyNoticeInFrame, motionExpr, FORMAT_CANVAS, voiceShapeFilter };
