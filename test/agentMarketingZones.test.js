"use strict";
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const Z = require("../src/lib/agentMarketingZones");
const R = require("../src/lib/agentMarketingCards");

const period = { start: "2026-09-01", end: "2026-09-30" };
const agent = { name: "Sam Zieour", brokerage: "SZ Real Estate Group" };
const dir = fs.mkdtempSync(path.join(os.tmpdir(), "amz-"));

test("every composition has clean zones on both canvases", () => {
  for (const kind of Z.KINDS) for (const canvas of ["portrait", "landscape"]) {
    assert.deepStrictEqual(Z.checkZones(Z.zonesFor(kind, canvas), canvas), [], `${kind} ${canvas}`);
  }
});

test("checkZones names overlaps, frame violations and safe-margin violations", () => {
  const z = Z.zonesFor("opening", "portrait");
  assert.ok(Z.checkZones({ ...z, caption_zone: { ...z.caption_zone, y: z.disclosure_zone.y - 10 } }, "portrait").some((p) => /overlaps/.test(p)));
  assert.ok(Z.checkZones({ ...z, headline_zone: { ...z.headline_zone, x: -5 } }, "portrait").some((p) => /leaves the frame/.test(p)));
  assert.ok(Z.checkZones({ ...z, headline_zone: { ...z.headline_zone, y: 10 } }, "portrait").some((p) => /safe margin/.test(p)));
  assert.throws(() => Z.zonesFor("nope"), /Unknown composition/);
});

test("the five cards keep the caption zone empty on both canvases", async () => {
  for (const canvas of ["portrait", "landscape"]) {
    const jobs = {
      opening: () => R.renderOpeningCard({ period, agent, market: "Natomas", headline: "New Construction Is Part of Your Competition", eyebrow: "Seller Insight" }, path.join(dir, `o_${canvas}.png`), canvas),
      closing: () => R.renderClosingCard({ period, agent, headline: ["Want to talk through where", "your home fits in this picture?"], support: "I'm happy to walk you through it." }, path.join(dir, `c_${canvas}.png`), canvas),
      stat: () => R.renderStatCard({ period, lead: "Active Listings Are Sitting a Median of", value: "41", unit: "days", trail: "Before Going Under Contract", icon: "calendar" }, path.join(dir, `s_${canvas}.png`), canvas),
      compare: () => R.renderCompareCard({ period, market: "Natomas", left: { lead: "Homes That Closed Sold At", percent: 99.4, caption: "of list price on average", icon: "sold" }, right: { lead: "But Here in the Active Market,", highlighted: 6, total: 14, caption: "listings have already taken a price reduction.", icon: "reduced" } }, path.join(dir, `m_${canvas}.png`), canvas),
      expert: () => R.renderExpertCard({ period, headline: ["Condition +", "Pricing Precision Matter"], statement: "That gap tells a real story.", points: [{ title: "Condition", caption: "Well-maintained homes stand out.", icon: "condition" }] }, path.join(dir, `e_${canvas}.png`), canvas),
    };
    for (const [kind, run] of Object.entries(jobs)) {
      const r = await run();
      const verdict = await Z.checkCard(kind, canvas, r.png);
      assert.ok(verdict.ok, `${kind} ${canvas}: ${verdict.problems.join("; ")}`);
    }
  }
});

test("ink inside a zone is detected", async () => {
  const sharp = require("sharp");
  const f = path.join(dir, "ink.png");
  await sharp({ create: { width: 1080, height: 1920, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } })
    .composite([{ input: { create: { width: 50, height: 50, channels: 4, background: "#fff" } }, left: 100, top: 1600 }]).png().toFile(f);
  const v = await Z.checkCard("stat", "portrait", f);
  assert.ok(!v.ok && v.problems.some((p) => /inside caption_zone/.test(p)));
});
