"use strict";
const test = require("node:test");
const assert = require("node:assert");
const sharp = require("sharp");
const style = require("../src/lib/agentMarketingStyle");
const cards = require("../src/lib/agentMarketingCards");

const cv = cards.CANVAS.portrait, Z = [{ x: 60, y: 96, w: 960, h: 700 }];
const flat = (c) => sharp({ create: { width: 540, height: 960, channels: 3, background: c } }).png().toBuffer();
const stripes = () => {
  const r = Array.from({ length: 30 }, (_, i) => `<rect x="${i * 18}" y="0" width="9" height="960" fill="${i % 2 ? "#000" : "#fff"}"/>`).join("");
  return sharp(Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="540" height="960">${r}</svg>`)).png().toBuffer();
};

test("a light picture gets dark ink and no scrim; a dark one gets white ink", async () => {
  const l = await style.styleFor(await flat("#e8e2d4"), cv, Z), d = await style.styleFor(await flat("#202428"), cv, Z);
  assert.strictEqual(l.ink, style.DARK); assert.strictEqual(l.scrim, 0);
  assert.strictEqual(d.ink, style.WHITE); assert.strictEqual(d.scrim, 0);
});
test("the accent stays readable on the picture it sits on", async () => {
  for (const c of ["#e8e2d4", "#202428", "#8a7a3a"]) {
    const s = await style.styleFor(await flat(c), cv, Z);
    const n = parseInt(c.slice(1), 16), L = style.relLum((n >> 16) & 255, (n >> 8) & 255, n & 255), A = style.hexLum(s.accent);
    assert.ok((Math.max(L, A) + 0.05) / (Math.min(L, A) + 0.05) >= 2.9, `${c} → ${s.accent}`);
  }
});
test("a busy picture asks for a scrim", async () => {
  const s = await style.styleFor(await stripes(), cv, Z);
  assert.ok(s.scrim > 0);
});
test("legibility check finds dark text on a dark picture", async () => {
  const card = await sharp({ create: { width: 1080, height: 1920, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } })
    .composite([{ input: { create: { width: 400, height: 100, channels: 4, background: style.DARK } }, left: 100, top: 200 }]).png().toBuffer();
  const bad = await style.checkLegibility(card, await flat("#202428"), cv, style.DARK, Z[0]);
  const ok = await style.checkLegibility(card, await flat("#e8e2d4"), cv, style.DARK, Z[0]);
  assert.ok(bad.ratioOfInkPixels > 0.9); assert.strictEqual(ok.ratioOfInkPixels, 0);
});

test("a takeaway card draws with no icon rows, in both formats", async () => {
  const os = require("os"), path = require("path");
  for (const canvas of ["portrait", "landscape"]) {
    const out = path.join(os.tmpdir(), `expert_${canvas}.png`);
    const r = await cards.renderExpertCard({ headline: ["This Means Condition and", "Pricing Precision Matter"], statement: "New construction sets a baseline so your home has to answer for itself.", period: { start: "2026-09-01", end: "2026-09-30" }, brand: {} }, out, canvas);
    assert.ok(r.fits, canvas);
  }
});
