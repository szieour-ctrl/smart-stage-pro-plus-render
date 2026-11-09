"use strict";
// The subscriber's logo on Agent Marketing opening and closing cards (Nov 9, 2026).
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const sharp = require("sharp");
const { logoPlate } = require("../src/lib/agentMarketingLogo");
const cards = require("../src/lib/agentMarketingCards");
const Z = require("../src/lib/agentMarketingZones");

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "aml-"));
const period = { start: "2026-09-01", end: "2026-09-30" };
const agent = { name: "Sam Zieour", brokerage: "SZ Real Estate Group" };
const solid = (w, h, c) => sharp({ create: { width: w, height: h, channels: 3, background: c } }).png().toBuffer();
const pixels = async (f) => (await sharp(f).raw().toBuffer({ resolveWithObject: true }));

test("no logo, no change: an unreadable or absent logo draws nothing", async () => {
  assert.deepStrictEqual(await logoPlate(null, { x0: 0, x1: 500, cy: 100, maxH: 100 }), { svg: "", placed: false });
  const bad = await logoPlate(Buffer.from("not an image"), { x0: 0, x1: 500, cy: 100, maxH: 100 });
  assert.strictEqual(bad.placed, false);
  assert.match(bad.warning, /could not be read/);
});

test("a logo is left off, with a warning, when the free space is too small", async () => {
  const logo = await solid(600, 200, "#b8975a");
  const narrow = await logoPlate(logo, { x0: 400, x1: 480, cy: 500, maxH: 120 });
  assert.strictEqual(narrow.placed, false);
  assert.match(narrow.warning, /no room/);
  const short = await logoPlate(logo, { x0: 0, x1: 500, cy: 500, maxH: 100, minTop: 480 });
  assert.strictEqual(short.placed, false);
});

test("the plate stays inside the free span and the height limit, right- or left-aligned", async () => {
  const logo = await solid(900, 300, "#b8975a");
  const r = await logoPlate(logo, { x0: 300, x1: 700, cy: 500, maxH: 120 });
  assert.ok(r.placed);
  assert.ok(r.box.x >= 300 && r.box.x + r.box.w === 700, "right-aligned inside the span");
  assert.ok(r.box.h <= 120 && Math.abs(r.box.y + r.box.h / 2 - 500) <= 1, "centered on cy within maxH");
  const l = await logoPlate(logo, { x0: 300, x1: 700, cy: 500, maxH: 120, align: "left" });
  assert.strictEqual(l.box.x, 300);
});

test("every opening and closing layout on both canvases: without a logo the card is unchanged, with one it is drawn and the caption zone stays clear", async () => {
  const hs = await solid(600, 600, "#7a8aa0");
  const logo = await solid(760, 260, "#b8975a");
  for (const canvas of ["portrait", "landscape"]) for (const layout of ["headline_focus", "editorial_portrait", "news_brief"]) for (const kind of ["opening", "closing"]) {
    const base = { period, agent, market: "Lincoln", headshotBuf: hs, headline: kind === "opening" ? "New Construction Is Part of Your Competition" : ["Want to talk through where", "your home fits in this picture?"], support: "I'm happy to walk you through it.", ...(layout === "headline_focus" ? {} : { layout }) };
    const fn = kind === "opening" ? cards.renderOpeningCard : cards.renderClosingCard;
    const name = `${kind}_${layout}_${canvas}`;
    const a = await fn(base, path.join(dir, `${name}_a.png`), canvas);
    const b = await fn({ ...base, logoBuf: null }, path.join(dir, `${name}_b.png`), canvas);
    const w = await fn({ ...base, logoBuf: logo }, path.join(dir, `${name}_w.png`), canvas);
    assert.ok(w.fits, `${name} still fits`);
    assert.ok(!(w.warnings || []).some((x) => /logo/.test(x)), `${name}: ${(w.warnings || []).join("; ")}`);
    const pa = await pixels(a.png), pb = await pixels(b.png), pw = await pixels(w.png);
    assert.ok(pa.data.equals(pb.data), `${name}: no logo gives the same card as before`);
    assert.ok(!pa.data.equals(pw.data), `${name}: the logo is drawn`);
    const verdict = await Z.checkCard(`${kind}_${layout}`, canvas, w.png, { headshot: true });
    assert.ok(verdict.ok, `${name}: ${verdict.problems.join("; ")}`);
  }
});

test("a long brokerage name leaves no room for the logo: it is left off with a warning, never overlapping the name", async () => {
  const hs = await solid(600, 600, "#7a8aa0");
  const logo = await solid(760, 260, "#b8975a");
  const long = { name: "Alexandria Montgomery-Whitfield", brokerage: "Coldwell Banker Realty Associates of Greater Sacramento" };
  const r = await cards.renderOpeningCard({ period, agent: long, market: "Lincoln", headshotBuf: hs, logoBuf: logo, headline: "New Construction Is Part of Your Competition" }, path.join(dir, "long.png"), "portrait");
  assert.ok(r.fits);
  assert.ok(r.warnings.some((x) => /^logo: .*no room/.test(x)));
});

test("a middle card (stat) ignores the logo", async () => {
  const logo = await solid(760, 260, "#b8975a");
  const p = { period, lead: "Active Listings Are Sitting a Median of", value: "41", unit: "days", trail: "Before Going Under Contract", icon: "calendar" };
  const a = await cards.renderStatCard(p, path.join(dir, "s_a.png"), "portrait");
  const b = await cards.renderStatCard({ ...p, logoBuf: logo }, path.join(dir, "s_b.png"), "portrait");
  assert.ok((await pixels(a.png)).data.equals((await pixels(b.png)).data));
});

// ── end to end through renderAgentMarketing (needs ffmpeg, ffprobe and the bundled fonts, like agentMarketingRender.test.js) ──
const { execFileSync } = require("child_process");
const R = require("../src/lib/agentMarketingRender");
const voice = path.join(dir, "voice.mp3"), music = path.join(dir, "music.mp3");
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=180:duration=7", "-c:a", "libmp3lame", voice]);
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=9", "-c:a", "libmp3lame", music]);
function manifest(brandExtra) {
  const scene = (n, composition, start, end, data) => ({ n, composition, narrative_role: "x", branding: "full", window: { start, end }, claims: [], data, visual: { research_visual_context: "st", image_keys: { "9x16": "p9" }, motion: "static_hold" } });
  return {
    version: "am-manifest-2", project_id: "t", formats: ["9x16"], notice: { period },
    brand: { primary: "#B08D3F", secondary: "#1D2023", border: "#B08D3F", headshot_key: "hs", agent, ...brandExtra },
    audio: { narration_key: "voice", duration_sec: 4 },
    captions: [{ text: "Natomas sellers are competing", start: 0.2, end: 2.5 }],
    scenes: [
      scene(1, "opening_news_brief", 0, 2, { market: "Natomas", headline: "New Construction Is Part of Your Competition" }),
      scene(2, "closing_headline_focus", 2, 4, { headline: ["Want to talk through where", "your home fits in this picture?"], support: "I'm happy to walk you through it." }),
    ],
  };
}
test("a manifest with brand.logo_key fetches the logo and renders; a logo that cannot be fetched only adds a warning", async () => {
  const assets = { p9: await solid(1080, 1920, "#9fb7a0"), hs: await solid(500, 500, "#7d8794"), voice: fs.readFileSync(voice), logo: await solid(760, 260, "#b8975a") };
  const asked = [];
  const ok = await R.renderAgentMarketing({ jobId: "l1", manifest: manifest({ logo_key: "logo" }), outDir: path.join(dir, "o1") }, () => {}, { fetchAsset: async (k) => { asked.push(k); return assets[k]; }, getMusic: async () => music });
  assert.ok(asked.includes("logo"));
  assert.ok(fs.existsSync(ok.outputs["9x16"].path));
  assert.ok(ok.checks["9x16"].noticeOk);
  assert.ok(!ok.warnings.some((w) => /logo/i.test(w)));

  const noLogo = await R.renderAgentMarketing({ jobId: "l2", manifest: manifest({ logo_key: "gone" }), outDir: path.join(dir, "o2") }, () => {}, { fetchAsset: async (k) => { if (k === "gone") throw new Error("404"); return assets[k]; }, getMusic: async () => music });
  assert.ok(fs.existsSync(noLogo.outputs["9x16"].path), "the Reel is still made");
  assert.ok(noLogo.warnings.some((w) => /logo could not be downloaded/.test(w)));

  const none = await R.renderAgentMarketing({ jobId: "l3", manifest: manifest({}), outDir: path.join(dir, "o3") }, () => {}, { fetchAsset: async (k) => { asked.push("L3:" + k); return assets[k]; }, getMusic: async () => music });
  assert.ok(!asked.includes("L3:logo"), "no logo_key: no logo request");
  assert.ok(fs.existsSync(none.outputs["9x16"].path));
});
