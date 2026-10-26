"use strict";
// Fonts: the cards follow the subscriber's brand font and fall back to the default look. Needs the bundled fonts installed
// (the Dockerfile does this; locally copy src/assets/fonts/*.ttf into ~/.fonts and run fc-cache).
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const crypto = require("crypto");
const sharp = require("sharp");
const R = require("../src/lib/agentMarketingCards");

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "amc-fonts-"));
const params = (font) => ({ period: { start: "2026-09-01", end: "2026-09-30" }, lead: "Active Listings Are Sitting a Median of", value: "41", unit: "days", trail: "Before Going Under Contract", icon: "calendar", brand: font === undefined ? {} : { font } });
async function render(font, tag = "x") {
  const f = path.join(dir, `${tag}_${font || "default"}.png`);
  await R.renderStatCard(params(font), f, "portrait");
  const { data } = await sharp(f).raw().toBuffer({ resolveWithObject: true });
  return crypto.createHash("md5").update(data).digest("hex");
}

test("every allowed brand font renders and differs from the default look", async () => {
  const base = await render(undefined);
  const seen = new Set([base]);
  for (const f of R.BRAND_FONTS) { const h = await render(f); assert.ok(!seen.has(h), `${f} should look different`); seen.add(h); }
});

test("an unknown font name falls back to the default look", async () => {
  assert.strictEqual(await render("Comic Sans MS"), await render(undefined));
});

test("two renders at the same time keep their own fonts", async () => {
  const [a, b, a2, b2] = await Promise.all([render("Lato", "p1"), render("Bodoni Moda", "p2"), render("Lato", "p3"), render("Bodoni Moda", "p4")]);
  assert.strictEqual(a, a2); assert.strictEqual(b, b2); assert.notStrictEqual(a, b);
  assert.strictEqual(a, await render("Lato", "p5"));
});
