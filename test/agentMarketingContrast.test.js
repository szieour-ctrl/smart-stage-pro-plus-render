"use strict";
const test = require("node:test");
const assert = require("node:assert");
const sharp = require("sharp");
const cards = require("../src/lib/agentMarketingCards");

test("a brand colour that matches the surface is moved until it reads", () => {
  assert.ok(cards.contrast("#0B2E7A", "#0B2E7A") < 1.1);
  const onDark = cards.readableOn("#0B2E7A", "#0B2E7A", 7);
  assert.ok(cards.contrast(onDark, "#0B2E7A") >= 7);
  const onLight = cards.readableOn("#E8D9A8", "#FBF8F1");
  assert.ok(cards.contrast(onLight, "#FBF8F1") >= 4.5);
});
test("a colour that already reads is left alone", () => {
  assert.strictEqual(cards.readableOn("#B08D3F", "#1D2023"), "#B08D3F");
  assert.strictEqual(cards.readableOn("#0B2E7A", "#FBF8F1"), "#0B2E7A");
});
test("navy brand: the '6 of 14' figure is visible on the navy card", async () => {
  const dir = require("fs").mkdtempSync(require("path").join(require("os").tmpdir(), "amc-"));
  const p = { market: "Natomas", period: { start: "2026-02-19", end: "2026-10-19" }, brand: { primary: "#0B2E7A", secondary: "#0B2E7A", border: "#0B2E7A", font: "Roboto" }, agent: { name: "A" },
    left: { lead: "Homes That Closed Sold At", percent: 99.4, caption: "of list price on average" }, right: { lead: "But Here in the Active Market,", highlighted: 6, total: 14, caption: "listings have already taken a price reduction." } };
  const out = dir + "/c.png";
  const r = await cards.renderCompareCard(p, out, "landscape");
  assert.ok(r.fits);
  // the figure sits at about y 560-670 in the right card (x 1230-1560): there must be light pixels on the navy
  const raw = await sharp(out).extract({ left: 1200, top: 560, width: 380, height: 110 }).removeAlpha().raw().toBuffer();
  let light = 0; for (let i = 0; i < raw.length; i += 3) if (raw[i] > 120 && raw[i + 1] > 120) light++;
  assert.ok(light > 800, `figure pixels visible: ${light}`);
});

test("a market of more than 42 listings (28 of 56) draws a share bar instead of failing; small markets keep the dot grid", async () => {
  const dir = require("fs").mkdtempSync(require("path").join(require("os").tmpdir(), "amc-"));
  const mk = (hi, total) => ({ market: "Antelope", period: { start: "2026-08-27", end: "2026-10-27" }, brand: { primary: "#012169", secondary: "#012169", border: "#FFFFFF", font: "Roboto" }, agent: { name: "A" },
    left: { lead: "Homes That Closed Sold At", percent: 99.8, caption: "of list price on average" }, right: { lead: "But Here in the Active Market,", highlighted: hi, total, caption: "listings have already taken a price reduction." } });
  for (const canvas of ["portrait", "landscape"]) {
    for (const [hi, total] of [[28, 56], [15, 30], [16, 31], [7, 14], [8, 15]]) {
      const r = await cards.renderCompareCard(mk(hi, total), `${dir}/c_${canvas}_${total}.png`, canvas);
      assert.ok(r.fits, `${hi} of ${total} fits on the ${canvas} card`);
    }
  }
  await assert.rejects(() => cards.renderCompareCard(mk(30, 28), `${dir}/bad.png`, "portrait"), /highlighted <= total/);
});
