// Renders sample opening/closing cards (both canvases) over a PLACEHOLDER plate so layout can be checked
// without calling GPT. Usage: node scripts/agent-marketing-card-samples.js <outDir> [headshot.png]
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const { renderOpeningCard, renderClosingCard, renderStatCard, renderCompareCard, renderExpertCard, CANVAS } = require("../src/lib/agentMarketingCards");

async function plate(cv) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}"><defs>
    <linearGradient id="s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#cfe3f5"/><stop offset="0.55" stop-color="#f6efe0"/><stop offset="1" stop-color="#8aa05a"/></linearGradient></defs>
    <rect width="100%" height="100%" fill="url(#s)"/>
    <rect y="${cv.h * 0.62}" width="100%" height="${cv.h * 0.38}" fill="#6f8a45" opacity="0.7"/>
    <rect x="${cv.w * 0.45}" y="${cv.h * 0.42}" width="${cv.w * 0.5}" height="${cv.h * 0.22}" fill="#c9b9a0" opacity="0.8"/></svg>`;
  return sharp(Buffer.from(svg)).png().toBuffer();
}

(async () => {
  const out = process.argv[2] || ".";
  fs.mkdirSync(out, { recursive: true });
  const headshotBuf = process.argv[3] ? fs.readFileSync(process.argv[3]) : null;
  const common = {
    period: { start: "2026-09-01", end: "2026-09-30" },
    agent: { name: "Sam Zieour", brokerage: "SZ Real Estate Group" },
    headshotBuf,
  };
  for (const canvas of ["portrait", "landscape"]) {
    const cv = CANVAS[canvas];
    const base = await plate(cv);
    const o = await renderOpeningCard({ ...common, market: "Natomas", kind: "Market Update", headline: "New Construction Is Part of Your Competition", eyebrow: "Seller Insight" }, path.join(out, `opening_${canvas}.png`), canvas);
    const c = await renderClosingCard({ ...common, headline: ["Want to talk through where", "your home fits in this picture?"], support: "I'm happy to walk you through it.", tagline: "Pricing · Condition · Competition" }, path.join(out, `closing_${canvas}.png`), canvas);
    const c2 = await renderClosingCard({ ...common, variant: "portrait", headline: ["Want to talk through where", "your home fits in this picture?"], support: "I'm happy to walk you through it.", tagline: "Pricing · Condition · Competition" }, path.join(out, `closing_portraitvariant_${canvas}.png`), canvas);
    const st = await renderStatCard({ period: common.period, lead: "Active Listings Are Sitting a Median of", value: "41", unit: "days", trail: "Before Going Under Contract", icon: "calendar" }, path.join(out, `stat_${canvas}.png`), canvas);
    const cp = await renderCompareCard({ period: common.period, market: "Natomas", kind: "Market Update",
      left: { lead: "Homes That Closed Sold At", percent: 99.4, caption: "of list price on average", icon: "sold" },
      right: { lead: "But Here in the Active Market,", highlighted: 6, total: 14, caption: "listings have already taken a price reduction.", icon: "reduced" } }, path.join(out, `compare_${canvas}.png`), canvas);
    const ex = await renderExpertCard({ period: common.period, headline: ["Condition +", "Pricing Precision Matter"],
      statement: "That gap between homes that sold and homes that didn't tells a real story.",
      support: "New construction sets a baseline\u2014your home has to answer for itself.",
      points: [{ title: "Condition", caption: "Well-maintained homes stand out.", icon: "condition" }, { title: "Pricing", caption: "Accurate pricing drives results.", icon: "pricing" }] }, path.join(out, `expert_${canvas}.png`), canvas);
    for (const [name, r] of [["opening", o], ["closing", c], ["closing_portraitvariant", c2], ["stat", st], ["compare", cp], ["expert", ex]]) {
      const f = path.join(out, `${name}_${canvas}.png`);
      await sharp(base).composite([{ input: f }]).png().toFile(path.join(out, `preview_${name}_${canvas}.png`));
      console.log(canvas, name, "fits:", r.fits, "textBottom", Math.round(r.textBottom), "limit", r.limit, r.warnings.join(";"));
    }
  }
})().catch((e) => { console.error(e); process.exit(1); });
