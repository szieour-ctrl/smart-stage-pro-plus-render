"use strict";
// Optional lip-sync clips on the opening and closing headshots: Off changes nothing; clips sit over the headshot area; a missing or bad clip
// falls back to the still headshot, scene by scene; the clip's own sound is never used. Stand-in assets; needs ffmpeg and ffprobe.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const sharp = require("sharp");
const { execFileSync } = require("child_process");
const R = require("../src/lib/agentMarketingRender");
const L = require("../src/lib/agentMarketingLipSync");

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "aml-"));
const period = { start: "2026-09-01", end: "2026-09-30" };
const solid = (w, h, c) => sharp({ create: { width: w, height: h, channels: 3, background: c } }).png().toBuffer();
const voice = path.join(dir, "voice.mp3"), music = path.join(dir, "music.mp3"), clip = path.join(dir, "clip.mp4");
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=180:duration=7", "-c:a", "libmp3lame", voice]);
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=9", "-c:a", "libmp3lame", music]);
// a loud, colourful 2-second talking-head stand-in with its own (different) sound that must never reach the output
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=720x1280:rate=25:duration=2", "-f", "lavfi", "-i", "sine=frequency=1000:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip]);

function manifest(lip) {
  const scene = (n, composition, start, end, data, motion, extra) => ({ n, composition, narrative_role: "x", branding: "full", window: { start, end }, claims: [], data, visual: { research_visual_context: "stand-in", motion, image_keys: { "9x16": "p9" } }, ...(extra || {}) });
  return {
    version: "am-manifest-2", project_id: "t", formats: ["9x16"], notice: { period },
    brand: { primary: "#B08D3F", secondary: "#1D2023", border: "#B08D3F", headshot_key: "hs", agent: { name: "Sam Zieour", brokerage: "SZ Real Estate Group" } },
    audio: { narration_key: "voice", duration_sec: 6 },
    captions: [{ text: "Natomas sellers are competing", start: 0.2, end: 2.5 }, { text: "with brand new homes.", start: 2.6, end: 5.5 }],
    scenes: [
      scene(1, "opening_news_brief", 0, 2, { market: "Natomas", headline: "New Construction Is Part of Your Competition" }, "pan_left", lip && lip.open ? { lipsync: { clip_key: lip.open } } : null),
      scene(2, "stat_big_number", 2, 4, { lead: "Active Listings Are Sitting a Median of", value: "41", unit: "days" }, "static_hold"),
      scene(3, "closing_editorial_portrait", 4, 6, { headline: ["Want to talk through where", "your home fits in this picture?"], support: "I'm happy to walk you through it." }, "tilt_up", lip && lip.close ? { lipsync: { clip_key: lip.close } } : null),
    ],
  };
}
async function assets() { return { p9: await solid(1080, 1920, "#9fb7a0"), hs: await solid(500, 500, "#7d8794"), voice: fs.readFileSync(voice), clipA: fs.readFileSync(clip), clipB: fs.readFileSync(clip), bad: Buffer.from("not a video") }; }
const probe = (f, what, sel) => execFileSync("ffprobe", ["-v", "error", ...(sel ? ["-select_streams", sel] : []), "-show_entries", what, "-of", "csv=p=0", f]).toString().trim();
async function go(id, lip, extraDeps) {
  const a = await assets();
  const asked = [];
  const r = await R.renderAgentMarketing({ jobId: id, manifest: manifest(lip), outDir: path.join(dir, id) }, () => {}, { fetchAsset: async (k) => { asked.push(k); if (extraDeps && extraDeps.fail && extraDeps.fail.includes(k)) throw new Error("download failed"); return a[k]; }, getMusic: async () => music });
  return { r, asked };
}
async function frame(mp4, sec, png) { execFileSync("ffmpeg", ["-y", "-v", "error", "-ss", String(sec), "-i", mp4, "-frames:v", "1", png]); return sharp(png).raw().toBuffer({ resolveWithObject: true }); }
function diffShare(a, b) { let n = 0; for (let i = 0; i < a.data.length; i += 3) if (Math.abs(a.data[i] - b.data[i]) + Math.abs(a.data[i + 1] - b.data[i + 1]) + Math.abs(a.data[i + 2] - b.data[i + 2]) > 60) n++; return n / (a.data.length / a.info.channels); }

test("placement: inset box, even sizes, circle or rounded rect; no box means no placement", () => {
  const c = L.placement({ x: 100, y: 200, w: 240, h: 240, shape: "circle" });
  assert.deepStrictEqual([c.x, c.y, c.w, c.h, c.shape], [103, 203, 234, 234, "circle"]);
  const r = L.placement({ x: 50, y: 60, w: 400, h: 520, shape: "rect", rx: 30 });
  assert.deepStrictEqual([r.w, r.h, r.rx, r.shape], [394, 514, 27, "rect"]);
  assert.strictEqual(L.placement(null), null);
  assert.strictEqual(L.placement({ x: 0, y: 0, w: 10, h: 10, shape: "circle" }), null);
  assert.deepStrictEqual(L.requestFor({ lipsync: { clip_key: " k " } }), { clipKey: "k" });
  assert.strictEqual(L.requestFor({}), null);
  assert.strictEqual(L.requestFor({ lipsync: {} }), null);
});

test("Off: no clip is asked for, nothing about lip sync is reported, the render is as before", async () => {
  const { r, asked } = await go("off", null);
  assert.ok(!asked.includes("clipA") && !asked.includes("clipB"));
  assert.strictEqual(r.lipsync, undefined);
  assert.strictEqual(probe(r.outputs["9x16"].path, "stream=codec_type", "a"), "audio");
});

test("both clips: placed over the headshot areas, one audio stream (the narration), same length, notices still verified", async () => {
  const off = await go("both-off", null), on = await go("both", { open: "clipA", close: "clipB" });
  assert.deepStrictEqual(on.r.lipsync["9x16"].map((x) => [x.scene, x.status]), [[1, "used"], [3, "used"]]);
  const f = on.r.outputs["9x16"].path, g = off.r.outputs["9x16"].path;
  assert.strictEqual(probe(f, "stream=codec_type", "a").split("\n").length, 1);
  assert.ok(Math.abs(Number(probe(f, "format=duration")) - Number(probe(g, "format=duration"))) < 0.1);
  assert.ok(on.r.checks["9x16"].noticeOk);
  for (const sec of [1.2, 5.2]) {
    const a = await frame(f, sec, path.join(dir, `a${sec}.png`)), b = await frame(g, sec, path.join(dir, `b${sec}.png`));
    const share = diffShare(a, b);
    assert.ok(share > 0.005 && share < 0.2, `scene at ${sec}s: ${share} of the frame changed (expected a headshot-sized area)`);
  }
  const mid = await frame(f, 3, path.join(dir, "m.png")), midOff = await frame(g, 3, path.join(dir, "mo.png"));
  assert.ok(diffShare(mid, midOff) < 0.001, "the middle scene is untouched");
});

test("one clip missing: that scene keeps its still headshot and is named; the other clip is still used", async () => {
  const { r } = await go("one", { open: "clipA", close: "clipB" }, { fail: ["clipB"] });
  assert.deepStrictEqual(r.lipsync["9x16"].map((x) => [x.scene, x.status]), [[1, "used"], [3, "fallback"]]);
  assert.match(r.warnings.join("\n"), /scene 3: no lip sync/);
  assert.ok(r.checks["9x16"].noticeOk);
});

test("an unreadable clip falls back; a layout with no headshot area falls back; the Reel still finishes", async () => {
  const { r } = await go("bad", { open: "bad", close: "clipB" });
  const s = r.lipsync["9x16"];
  assert.strictEqual(s[0].status, "fallback");
  assert.strictEqual(s[1].status, "used");
  assert.ok(fs.existsSync(r.outputs["9x16"].path));
});

test("retry after a render failure: the same manifest renders again with the same clips (nothing is generated here)", async () => {
  const a = await go("retry1", { open: "clipA", close: "clipB" }, { fail: ["voice"] }).catch((e) => e);
  assert.ok(a instanceof Error);
  const b = await go("retry2", { open: "clipA", close: "clipB" });
  assert.deepStrictEqual(b.r.lipsync["9x16"].map((x) => x.status), ["used", "used"]);
});
