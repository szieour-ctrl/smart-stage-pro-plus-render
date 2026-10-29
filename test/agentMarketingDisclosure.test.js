"use strict";
// Oct 29, 2026: AI-generated pictures carry an automatic disclosure, drawn by the renderer from the manifest's image provenance.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const sharp = require("sharp");
const { execFileSync } = require("child_process");
const R = require("../src/lib/agentMarketingRender");
const cards = require("../src/lib/agentMarketingCards");

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "amd-"));
const period = { start: "2026-09-01", end: "2026-09-30" };
const EVENT = "AI-generated event illustration. Not actual event photography.";
const solid = (w, h, c) => sharp({ create: { width: w, height: h, channels: 3, background: c } }).png().toBuffer();
const voice = path.join(dir, "voice.mp3"), music = path.join(dir, "music.mp3");
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=180:duration=7", "-c:a", "libmp3lame", voice]);
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=9", "-c:a", "libmp3lame", music]);
const probe = (f, what) => execFileSync("ffprobe", ["-v", "error", "-show_entries", what, "-of", "csv=p=0", f]).toString().trim();

function manifest(formats, prov, text = EVENT) {
  const scene = (n, composition, start, end, data, motion, p) => ({ n, composition, narrative_role: "x", branding: "full", window: { start, end }, claims: [], data, visual: { research_visual_context: "stand-in", motion, image_keys: { "9x16": "p9", "16x9": "p16" }, ...(p ? { image_provenance: p } : {}) } });
  return {
    version: "am-manifest-2", project_id: "t", formats, notice: { period }, ...(text ? { disclosure: { kind: "event", text } } : {}),
    brand: { primary: "#B08D3F", secondary: "#1D2023", border: "#B08D3F", headshot_key: "hs", agent: { name: "Sam Zieour", brokerage: "SZ Real Estate Group" } },
    audio: { narration_key: "voice", duration_sec: 6 },
    captions: [{ text: "A craft fair this November", start: 0.2, end: 2.5 }, { text: "hope to see you there.", start: 2.6, end: 5.5 }],
    scenes: [
      scene(1, "opening_headline_focus", 0, 2, { market: "Roseville", kind: "Local Events", headline: "There's A Craft Fair This November", eyebrow: "Saturday, Nov 14 · 10 a.m. to 2 p.m." }, "pan_left", prov[0]),
      scene(2, "expert_takeaway", 2, 4, { headline: ["Vendors, Bake Sale", "And Antiques"], statement: "Free and open to the public", support: "" }, "static_hold", prov[1]),
      scene(3, "closing_headline_focus", 4, 6, { headline: ["I hope you enjoy", "every minute of it"], support: "Hope to see you there.", tagline: "LOCAL LIFE" }, "tilt_up", prov[2]),
    ],
  };
}
const AI = { "9x16": "ai", "16x9": "ai" }, PHOTO = { "9x16": "photo", "16x9": "photo" };
async function assets() { return { p9: await solid(1080, 1920, "#4a6a5a"), p16: await solid(1920, 1080, "#4a5f7a"), hs: await solid(500, 500, "#7d8794"), voice: fs.readFileSync(voice) }; }

test("disclosureFor: only scenes whose picture is AI in that format, only with the manifest's wording", () => {
  const m = manifest(["9x16", "16x9"], [AI, PHOTO, { "9x16": "ai", "16x9": "photo" }]);
  assert.strictEqual(R.disclosureFor(m, m.scenes[0], "portrait"), EVENT);
  assert.strictEqual(R.disclosureFor(m, m.scenes[1], "portrait"), "");
  assert.strictEqual(R.disclosureFor(m, m.scenes[2], "portrait"), EVENT);
  assert.strictEqual(R.disclosureFor(m, m.scenes[2], "landscape"), "");
  const none = manifest(["9x16"], [AI, AI, AI], "");
  assert.strictEqual(R.disclosureFor(none, none.scenes[0], "portrait"), "", "an older manifest without the wording draws nothing");
});

test("renders both formats with the disclosure on every AI scene and verifies it in the finished frames", async () => {
  const a = await assets();
  const out = path.join(dir, "out");
  const r = await R.renderAgentMarketing({ jobId: "d1", manifest: manifest(["9x16", "16x9"], [AI, AI, AI]), outDir: out }, () => {}, { fetchAsset: async (k) => a[k], getMusic: async () => music });
  for (const fmt of ["9x16", "16x9"]) {
    const d = r.checks[fmt].disclosure;
    assert.deepStrictEqual(d.expectedScenes, [1, 2, 3], fmt);
    assert.strictEqual(d.checked.length, 3);
    assert.ok(d.ok, JSON.stringify(d));
    assert.ok(r.checks[fmt].noticeOk, "the source notice is still verified beside it");
    assert.ok(fs.existsSync(r.outputs[fmt].path));
  }
});

test("a photo scene draws no disclosure, and a photo-only Reel is unchanged", async () => {
  const a = await assets();
  const r = await R.renderAgentMarketing({ jobId: "d2", manifest: manifest(["9x16"], [AI, PHOTO, PHOTO]), outDir: path.join(dir, "out2") }, () => {}, { fetchAsset: async (k) => a[k], getMusic: async () => music });
  assert.deepStrictEqual(r.checks["9x16"].disclosure.expectedScenes, [1]);
  const photoOnly = await R.renderAgentMarketing({ jobId: "d3", manifest: manifest(["9x16"], [PHOTO, PHOTO, PHOTO]), outDir: path.join(dir, "out3") }, () => {}, { fetchAsset: async (k) => a[k], getMusic: async () => music });
  assert.deepStrictEqual(photoOnly.checks["9x16"].disclosure, { expectedScenes: [], checked: [], ok: true });
});

test("the opening and closing source bar keeps its notice and gains the disclosure line without changing size", async () => {
  for (const canvas of ["portrait", "landscape"]) {
    const cv = cards.CANVAS[canvas];
    const plain = path.join(dir, `open_plain_${canvas}.png`), withDis = path.join(dir, `open_dis_${canvas}.png`);
    const p = { period, market: "Natomas", headline: "A Headline", agent: { name: "Sam" } };
    await cards.renderOpeningCard(p, plain, canvas);
    await cards.renderOpeningCard({ ...p, disclosure: EVENT }, withDis, canvas);
    const bar = (f) => sharp(f).extract({ left: 0, top: cv.h - cv.notice.h, width: cv.w, height: cv.notice.h }).raw().toBuffer();
    assert.notDeepStrictEqual(await bar(plain), await bar(withDis), `${canvas}: the bar changed`);
    const above = (f) => sharp(f).extract({ left: 0, top: 0, width: cv.w, height: cv.h - cv.notice.h }).raw().toBuffer();
    assert.deepStrictEqual(await above(plain), await above(withDis), `${canvas}: nothing above the bar moved`);
  }
  // a notice with no disclosure is byte-for-byte what it was
  const a = path.join(dir, "n1.png"), b = path.join(dir, "n2.png");
  await cards.renderClosingCard({ period, headline: ["Hope to see", "you there"], support: "", agent: { name: "Sam" } }, a, "portrait");
  await cards.renderClosingCard({ period, headline: ["Hope to see", "you there"], support: "", agent: { name: "Sam" }, disclosure: "" }, b, "portrait");
  assert.deepStrictEqual(await sharp(a).raw().toBuffer(), await sharp(b).raw().toBuffer());
});

test("the frame check fails when the disclosure is not in the video", async () => {
  const plain = path.join(dir, "plain.mp4");
  execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x4a6a5a:s=1080x1920:d=1:r=25", "-pix_fmt", "yuv420p", plain]);
  const card = path.join(dir, "negcard.png");
  await cards.renderOpeningCard({ period, market: "Natomas", headline: "x", agent: { name: "Sam" }, disclosure: EVENT }, card, "portrait");
  const v = await R.verifyDisclosureInFrame(plain, card, 0.5, "portrait", dir, "neg");
  assert.ok(v.linePixels >= 120, "the card really has the line");
  assert.strictEqual(v.ok, false);
});
