"use strict";
// Nov 12, 2026: a Reel with the MetroList notice credits the approved article on the scene(s) that draw on it. The credit comes from the manifest, is drawn on
// the card (never narrated, never a caption), sits clear of the captions, the agent branding and the AI-image disclosure, and is verified in the finished MP4.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const sharp = require("sharp");
const { execFileSync } = require("child_process");
const R = require("../src/lib/agentMarketingRender");
const cards = require("../src/lib/agentMarketingCards");
const zones = require("../src/lib/agentMarketingZones");

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "amc-"));
const period = { start: "2026-10-01", end: "2026-10-31" };
const CREDIT = "Source: Roseville Market Update September 2026: Prices Hold (thegoolsbygroup.com, Oct 12, 2026).";
const AI_TEXT = "AI-generated image for illustration only.";
const solid = (w, h, c) => sharp({ create: { width: w, height: h, channels: 3, background: c } }).png().toBuffer();
const voice = path.join(dir, "voice.mp3"), music = path.join(dir, "music.mp3");
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=180:duration=7", "-c:a", "libmp3lame", voice]);
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=9", "-c:a", "libmp3lame", music]);
const AI = { "9x16": "ai", "16x9": "ai" }, PHOTO = { "9x16": "photo", "16x9": "photo" };

function manifest(formats, { credit = true, prov = [AI, AI, AI], disclosure = true, web = false } = {}) {
  const scene = (n, composition, start, end, data, motion, p, role) => ({ n, composition, narrative_role: "x", branding: "full", source_role: role || "data", window: { start, end }, claims: [], data, visual: { research_visual_context: "Roseville neighborhood concept image for the market update", image_keys: { "9x16": "p9", "16x9": "p16" }, motion, source: "GPT_IMAGE_2_CONCEPT", image_provenance: p } });
  return {
    version: "am-manifest-2", project_id: "t", formats,
    ...(web ? { notice_required: false, notice: {}, source_line: { sources: [], scope_note: "", text: CREDIT, label: "thegoolsbygroup.com" } } : { notice: { period } }),
    ...(disclosure ? { disclosure: { kind: "market", text: AI_TEXT } } : {}),
    ...(credit ? { source_credit: { sources: [], text: CREDIT, scenes: [2] } } : {}),
    brand: { primary: "#B08D3F", secondary: "#1D2023", border: "#B08D3F", headshot_key: "hs", agent: { name: "Sam Zieour", brokerage: "SZ Real Estate Group" } },
    audio: { narration_key: "voice", duration_sec: 6 },
    captions: [{ text: "A local market update put September", start: 0.2, end: 2.5 }, { text: "at about two point eight months.", start: 2.6, end: 5.5 }],
    scenes: [
      scene(1, "opening_headline_focus", 0, 2, { market: "Roseville", kind: "Market Update", headline: "Roseville Is Shifting Toward Balanced", eyebrow: "" }, "pan_left", prov[0]),
      scene(2, "expert_takeaway", 2, 4.4, { headline: ["What the article", "reported for September"], statement: "About 2.8 months of supply, based on pending sales", support: "" }, "static_hold", prov[1], "article"),
      scene(3, "closing_headline_focus", 4.4, 6, { headline: ["Price for", "today's buyer"], support: "I track this monthly.", tagline: "LOCAL MARKET" }, "tilt_up", prov[2]),
    ],
  };
}
async function assets() { return { p9: await solid(1080, 1920, "#4a6a5a"), p16: await solid(1920, 1080, "#4a5f7a"), hs: await solid(500, 500, "#7d8794"), voice: fs.readFileSync(voice) }; }
const render = async (id, m, a) => R.renderAgentMarketing({ jobId: id, manifest: m, outDir: path.join(dir, id) }, () => {}, { fetchAsset: async (k) => a[k], getMusic: async () => music });

test("creditFor: only an article-sourced middle scene of a MetroList Reel, with the manifest's text", () => {
  const m = manifest(["9x16"]);
  assert.strictEqual(R.creditFor(m, m.scenes[1]), CREDIT);
  assert.strictEqual(R.creditFor(m, m.scenes[0]), "", "the opening scene never carries it");
  assert.strictEqual(R.creditFor(m, m.scenes[2]), "", "nor the closing scene");
  const dataScene = { ...m.scenes[1], source_role: "data" };
  assert.strictEqual(R.creditFor(m, dataScene), "", "a MetroList-data scene is not credited to the article");
  const none = manifest(["9x16"], { credit: false });
  assert.strictEqual(R.creditFor(none, none.scenes[1]), "", "a MetroList-only manifest draws nothing new");
  const web = manifest(["9x16"], { web: true });
  assert.strictEqual(R.creditFor(web, web.scenes[1]), "", "a web Reel keeps its source line only");
});

test("the credit fits its lines, stays below the caption band, above the disclosure, and the MetroList label is not drawn over it (9:16 and 16:9)", async () => {
  for (const canvas of ["portrait", "landscape"]) {
    const cv = cards.CANVAS[canvas];
    const g = cards.creditGeometry(canvas);
    assert.ok(g.band.top >= cv.captionBand.y + cv.captionBand.h + 20, `${canvas}: clear of the caption band`);
    const params = { period, brand: { border: "#B08D3F" }, agent: { name: "Sam" } };
    const png = path.join(dir, `exp_${canvas}.png`);
    await cards.renderExpertCard({ headline: ["What the article", "reported for September"], statement: "About 2.8 months of supply", support: "", ...params, hideFooter: true }, png, canvas);
    await cards.applyDisclosure(png, canvas, AI_TEXT, params);
    const c = await cards.applySourceCredit(png, canvas, CREDIT, params, { withFade: false });
    assert.ok(c.lines.length <= g.maxLines && c.size >= 15, `${canvas}: ${c.lines.length} line(s) at ${c.size}px`);
    // credit ink is inside its band and the disclosure ink is below it: the two never share a row
    const rowsWithInk = async (band) => {
      const { data, info } = await sharp(png).ensureAlpha().extract(band).raw().toBuffer({ resolveWithObject: true });
      const rows = new Set();
      for (let i = 0; i < band.width * band.height; i++) { const o = i * info.channels; if (data[o + 3] >= 200 && Math.min(data[o], data[o + 1], data[o + 2]) >= 170) rows.add(Math.floor(i / band.width)); }
      return rows;
    };
    const credRows = await rowsWithInk(g.band);
    assert.ok(credRows.size > 10, `${canvas}: the credit is drawn in its band`);
    const belowBand = { left: g.band.left, top: g.band.top + g.band.height, width: g.band.width, height: cv.h - (g.band.top + g.band.height) };
    const disRows = await rowsWithInk(belowBand);
    assert.ok(disRows.size > 5, `${canvas}: the disclosure line is still there below the credit`);
    const aboveBand = { left: g.band.left, top: cv.captionBand.y + cv.captionBand.h, width: g.band.width, height: g.band.top - (cv.captionBand.y + cv.captionBand.h) };
    assert.strictEqual((await rowsWithInk(aboveBand)).size, 0, `${canvas}: nothing between the caption band and the credit`);
  }
});

test("the MetroList footer label is dropped only on a credited scene; an uncredited card is pixel-identical to before", async () => {
  const a = await assets();
  const m = manifest(["9x16"]);
  const plate = await solid(1080, 1920, "#4a6a5a");
  const credited = await R.renderSceneCard(m, m.scenes[1], "portrait", a.hs, path.join(dir, "c_credited.png"), plate);
  assert.ok(credited.credit && credited.credit.text === CREDIT);
  const noCredit = manifest(["9x16"], { credit: false });
  const plain = await R.renderSceneCard(noCredit, noCredit.scenes[1], "portrait", a.hs, path.join(dir, "c_plain.png"), plate);
  assert.strictEqual(plain.credit, null);
  const again = await R.renderSceneCard(noCredit, noCredit.scenes[1], "portrait", a.hs, path.join(dir, "c_plain2.png"), plate);
  assert.deepStrictEqual(await sharp(path.join(dir, "c_plain.png")).raw().toBuffer(), await sharp(path.join(dir, "c_plain2.png")).raw().toBuffer(), "deterministic without a credit");
  assert.notDeepStrictEqual(await sharp(path.join(dir, "c_credited.png")).raw().toBuffer(), await sharp(path.join(dir, "c_plain.png")).raw().toBuffer());
  // the opening and closing bars (the MetroList notice) never change because of a credit
  const o1 = await R.renderSceneCard(m, m.scenes[0], "portrait", a.hs, path.join(dir, "o1.png"), plate);
  const o2 = await R.renderSceneCard(noCredit, noCredit.scenes[0], "portrait", a.hs, path.join(dir, "o2.png"), plate);
  assert.strictEqual(o1.credit, null);
  assert.deepStrictEqual(await sharp(path.join(dir, "o1.png")).raw().toBuffer(), await sharp(path.join(dir, "o2.png")).raw().toBuffer());
  void again;
});

test("renders both formats: the MetroList notice, the AI disclosure and the article credit are all confirmed in the finished frames", async () => {
  const a = await assets();
  const r = await render("cr1", manifest(["9x16", "16x9"]), a);
  for (const fmt of ["9x16", "16x9"]) {
    const c = r.checks[fmt].credit;
    assert.deepStrictEqual(c.expectedScenes, [2], fmt);
    assert.strictEqual(c.checked.length, 1);
    assert.ok(c.ok, JSON.stringify(c));
    assert.ok(r.checks[fmt].noticeOk, "the MetroList notice is still verified");
    assert.ok(r.checks[fmt].disclosure.ok);
    assert.ok(fs.existsSync(r.outputs[fmt].path));
  }
});

test("a MetroList-only Reel and a web Reel render with no credit and no change to their checks", async () => {
  const a = await assets();
  const plain = await render("cr2", manifest(["9x16"], { credit: false }), a);
  assert.deepStrictEqual(plain.checks["9x16"].credit, { expectedScenes: [], checked: [], ok: true });
  assert.ok(plain.checks["9x16"].noticeOk);
  const web = await render("cr3", manifest(["9x16"], { web: true, credit: false }), a);
  assert.deepStrictEqual(web.checks["9x16"].credit, { expectedScenes: [], checked: [], ok: true });
});

test("the credit is on the card only: the captions are the manifest's captions and carry no source text", async () => {
  const m = manifest(["9x16"]);
  assert.ok(!m.captions.some((c) => /source|goolsby/i.test(c.text)));
  const cap = await R.captionLayer(m.captions[0].text, "portrait", undefined, path.join(dir, "cap.png"));
  assert.ok(fs.existsSync(cap));
  const credited = zones.zonesFor("expert_takeaway", "portrait");
  assert.ok(credited.caption_zone && credited.disclosure_zone);
});

test("a credit too long for its lines stops the render rather than clipping", async () => {
  const png = path.join(dir, "long.png");
  await cards.renderExpertCard({ headline: ["A", "B"], statement: "C", support: "", period, brand: { border: "#B08D3F" } }, png, "portrait");
  await assert.rejects(() => cards.applySourceCredit(png, "portrait", "Source: " + "very long title ".repeat(30) + "(example.com, Oct 12, 2026).", {}), /too long/);
});
