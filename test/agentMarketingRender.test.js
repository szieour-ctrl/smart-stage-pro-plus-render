"use strict";
// End-to-end check of the Agent Marketing renderer with stand-in assets (needs ffmpeg, ffprobe and the bundled fonts).
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const sharp = require("sharp");
const { execFileSync } = require("child_process");
const R = require("../src/lib/agentMarketingRender");
const cards = require("../src/lib/agentMarketingCards");

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "amr-"));
const period = { start: "2026-09-01", end: "2026-09-30" };
const solid = (w, h, c) => sharp({ create: { width: w, height: h, channels: 3, background: c } }).png().toBuffer();
const voice = path.join(dir, "voice.mp3"), music = path.join(dir, "music.mp3");
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=180:duration=7", "-c:a", "libmp3lame", voice]);
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=9", "-c:a", "libmp3lame", music]);

function manifest(formats) {
  const scene = (n, composition, start, end, data, motion) => ({ n, composition, narrative_role: "x", branding: "full", window: { start, end }, claims: [], data, visual: { research_visual_context: "stand-in", motion, image_keys: { "9x16": "p9", "16x9": "p16" } } });
  return {
    version: "am-manifest-2", project_id: "t", formats, notice: { period },
    brand: { primary: "#B08D3F", secondary: "#1D2023", border: "#B08D3F", headshot_key: "hs", agent: { name: "Sam Zieour", brokerage: "SZ Real Estate Group" } },
    audio: { narration_key: "voice", duration_sec: 6 },
    captions: [{ text: "Natomas sellers are competing", start: 0.2, end: 2.5 }, { text: "with brand new homes.", start: 2.6, end: 5.5 }],
    scenes: [
      scene(1, "opening_news_brief", 0, 2, { market: "Natomas", headline: "New Construction Is Part of Your Competition" }, "pan_left"),
      scene(2, "stat_big_number", 2, 4, { lead: "Active Listings Are Sitting a Median of", value: "41", unit: "days" }, "static_hold"),
      scene(3, "closing_editorial_portrait", 4, 6, { headline: ["Want to talk through where", "your home fits in this picture?"], support: "I'm happy to walk you through it." }, "tilt_up"),
    ],
  };
}
async function assets() {
  return { p9: await solid(1080, 1920, "#9fb7a0"), p16: await solid(1920, 1080, "#a7b7c9"), hs: await solid(500, 500, "#7d8794"), voice: fs.readFileSync(voice) };
}
const probe = (f, what) => execFileSync("ffprobe", ["-v", "error", "-show_entries", what, "-of", "csv=p=0", f]).toString().trim();

test("renders both formats with narration, captions, cards and a verified notice", async () => {
  const a = await assets();
  const out = path.join(dir, "out");
  const steps = [];
  const r = await R.renderAgentMarketing({ jobId: "t1", manifest: manifest(["9x16", "16x9"]), outDir: out }, (s) => steps.push(s), { fetchAsset: async (k) => a[k], getMusic: async () => music });
  for (const [fmt, wh] of [["9x16", "1080,1920"], ["16x9", "1920,1080"]]) {
    const f = r.outputs[fmt].path;
    assert.ok(fs.existsSync(f));
    assert.strictEqual(probe(f, "stream=width,height").split("\n")[0], wh);
    assert.ok(Math.abs(Number(probe(f, "format=duration")) - 6) < 0.15, `${fmt} duration`);
    assert.match(probe(f, "stream=codec_name"), /h264/);
    assert.match(probe(f, "stream=codec_name"), /aac/);
    assert.ok(r.checks[fmt].noticeOk);
    assert.strictEqual(r.checks[fmt].notice.length, 2, "opening and closing are both checked");
  }
  assert.ok(steps.some((s) => /Checking the 16x9/.test(s)));
  assert.ok(fs.existsSync(path.join(out, "checks", "9x16", "notice_scene1.png")), "the checked frames are saved for review");
});

test("a 9:16-only manifest renders only 9:16 and never asks for a 16:9 asset", async () => {
  const a = await assets();
  const asked = [];
  const r = await R.renderAgentMarketing({ jobId: "t2", manifest: manifest(["9x16"]), outDir: path.join(dir, "out2") }, () => {}, { fetchAsset: async (k) => { asked.push(k); return a[k]; }, getMusic: async () => music });
  assert.deepStrictEqual(Object.keys(r.outputs), ["9x16"]);
  assert.ok(!asked.includes("p16"));
});

test("the notice check fails when the notice is not in the video", async () => {
  const card = path.join(dir, "card.png");
  await cards.renderOpeningCard({ period, market: "Natomas", headline: "x", agent: { name: "Sam" } }, card, "portrait");
  const plain = path.join(dir, "plain.mp4");
  execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x9fb7a0:s=1080x1920:d=1:r=25", "-pix_fmt", "yuv420p", plain]);
  const v = await R.verifyNoticeInFrame(plain, 0.5, card, "portrait", dir, "neg");
  assert.strictEqual(v.ok, false);
});

test("bad manifests stop before any rendering", async () => {
  await assert.rejects(() => R.renderAgentMarketing({ jobId: "t3", manifest: { version: "nope" } }), /am-manifest-2/);
  await assert.rejects(() => R.renderAgentMarketing({ jobId: "t4", manifest: { ...manifest(["9x16"]), formats: ["4x3"] } }), /no format/);
  const bad = manifest(["9x16"]); bad.notice = { period: { start: "x", end: "y" } };
  const a = await assets();
  await assert.rejects(() => R.renderAgentMarketing({ jobId: "t5", manifest: bad, outDir: path.join(dir, "o5") }, () => {}, { fetchAsset: async (k) => a[k], getMusic: async () => music }), /MetroList notice required/);
});

test("a web-research manifest draws its source line in the bar, not the MetroList notice, and the render still verifies it", async () => {
  const a = await assets();
  const m = manifest(["9x16"]);
  m.notice_required = false; m.notice = {};
  m.source_line = { text: "Source: Redfin 95834 Housing Market (redfin.com, Oct 20, 2026). Figures cover ZIP 95834, not all of Natomas.", label: "redfin.com", sources: [], scope_note: "" };
  const out = path.join(dir, "out_web");
  const r = await R.renderAgentMarketing({ jobId: "t-web", manifest: m, outDir: out }, () => {}, { fetchAsset: async (k) => a[k], getMusic: async () => music });
  assert.ok(r.checks["9x16"].noticeOk);
  assert.strictEqual(r.checks["9x16"].notice.length, 2);
  // the saved checked frame is the opening: bar region must not be the MetroList sentence (the card differs from a MetroList card)
  const web = path.join(dir, "web_open.png"), ml = path.join(dir, "ml_open.png");
  await cards.renderOpeningCard({ noticeText: m.source_line.text, market: "Natomas", headline: "x", agent: { name: "Sam" } }, web, "portrait");
  await cards.renderOpeningCard({ period, market: "Natomas", headline: "x", agent: { name: "Sam" } }, ml, "portrait");
  const crop = (f) => sharp(f).extract({ left: 0, top: 1920 - 150, width: 1080, height: 150 }).raw().toBuffer();
  assert.notDeepStrictEqual(await crop(web), await crop(ml));
});

test("a web-research manifest with no source line, and a source line too long for its bar, are refused", async () => {
  const a = await assets();
  const none = manifest(["9x16"]); none.notice_required = false; none.notice = {};
  await assert.rejects(R.renderAgentMarketing({ jobId: "t-web2", manifest: none, outDir: path.join(dir, "o3") }, () => {}, { fetchAsset: async (k) => a[k], getMusic: async () => music }), /no source line|required/i);
  const card = path.join(dir, "toolong.png");
  await assert.rejects(cards.renderOpeningCard({ noticeText: "Source: " + "very long ".repeat(60), market: "N", headline: "x", agent: { name: "Sam" } }, card, "portrait"), /too long for its bar/);
});
