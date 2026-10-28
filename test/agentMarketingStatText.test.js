const test = require("node:test");
const assert = require("node:assert");
const { statText } = require("../src/lib/agentMarketingCards");

test("big-number card text: thousands commas, $ prefix, % attached, other units spaced", () => {
  assert.strictEqual(statText({ value: "482000", prefix: "$" }), "$482,000");
  assert.strictEqual(statText({ value: "99", unit: "%" }), "99%");
  assert.strictEqual(statText({ value: "4.5", unit: "%" }), "4.5%");
  assert.strictEqual(statText({ value: "45", unit: "days" }), "45 DAYS");
  assert.strictEqual(statText({ value: "1250000", prefix: "$" }), "$1,250,000");
});
