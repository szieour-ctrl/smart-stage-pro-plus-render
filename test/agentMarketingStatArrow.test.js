const test = require("node:test");
const assert = require("node:assert");
const cards = require("../src/lib/agentMarketingCards.js");

const base = { period: "2026-09", lead: "For-sale listings, down from a year earlier", value: "5.9", unit: "%", trail: "Fewer homes", icon: "none" };

test("stat card draws a gold arrow only when a direction is given", async () => {
  const fs = require("fs"), os = require("os"), path = require("path");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "arrow-"));
  const sizes = {};
  for (const d of [undefined, "down", "up"]) {
    const f = path.join(dir, `${d || "none"}.png`);
    const r = await cards.renderStatCard({ ...base, direction: d }, f, "portrait");
    assert.ok(r.fits);
    sizes[d || "none"] = fs.statSync(f).size;
  }
  assert.notEqual(sizes.down, sizes.none);
  assert.notEqual(sizes.up, sizes.none);
  assert.notEqual(sizes.up, sizes.down);
});
