// Pipeline check for agentMarketingRender.js with STAND-IN assets: gradient plates, a tone for the voice, a tone for music, and a test
// script. It proves timing, captions, cards, fades, notice and the file checks. It is NOT the Natomas acceptance Reel (that uses
// GPT Image 2 backgrounds, ElevenLabs narration and the approved script from the project).
//   NODE_PATH=<sharp> node scripts/agent-marketing-render-sample.js <outDir> [headshot.png] [9x16,16x9]
const fs = require("fs");
const path = require("path");
const os = require("os");
const sharp = require("sharp");
const { execFileSync } = require("child_process");
const { renderAgentMarketing } = require("../src/lib/agentMarketingRender");

const out = process.argv[2] || ".";
const headshotPath = process.argv[3];
const formats = (process.argv[4] || "9x16,16x9").split(",");
fs.mkdirSync(out, { recursive: true });
const period = { start: "2026-09-01", end: "2026-09-30" };
const SIZE = { "9x16": [1080, 1920], "16x9": [1920, 1080] };

async function plate(format, n) {
  const [w, h] = SIZE[format];
  const hues = ["#cfe3f5", "#f1e4cf", "#dfe9d2", "#e8dccb", "#d5e0ea"];
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><defs><linearGradient id="s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${hues[n % 5]}"/><stop offset="0.6" stop-color="#f6efe0"/><stop offset="1" stop-color="#8aa05a"/></linearGradient></defs>
  <rect width="100%" height="100%" fill="url(#s)"/><rect y="${h * 0.6}" width="100%" height="${h * 0.4}" fill="#6f8a45" opacity="0.75"/>
  <rect x="${w * (0.12 + 0.1 * n)}" y="${h * 0.38}" width="${w * 0.4}" height="${h * 0.22}" fill="#c9b9a0"/><rect x="${w * (0.4 + 0.08 * n)}" y="${h * 0.44}" width="${w * 0.3}" height="${h * 0.16}" fill="#b6a58d"/></svg>`;
  let img = sharp(Buffer.from(svg));
  // PLATE=dark|light|busy changes the stand-in picture so the image-aware text style can be checked on each kind.
  if (process.env.PLATE === "dark") img = img.modulate({ brightness: 0.28 });
  else if (process.env.PLATE === "busy") {
    const stripes = Array.from({ length: 30 }, (_, i) => `<rect x="${i * w / 30}" y="0" width="${w / 60}" height="${h}" fill="${i % 2 ? '#111' : '#fff'}" opacity="0.7"/>`).join("");
    img = img.composite([{ input: Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">${stripes}</svg>`) }]);
  }
  return img.png().toBuffer();
}

const WORDS = ("Natomas sellers are competing with brand new homes. Active listings are sitting a median of 41 days before going under contract. " +
  "Homes that closed sold at 99.4 percent of list price, but 6 of 14 active listings have already taken a price reduction. " +
  "Condition and pricing precision matter more than ever. Want to talk through where your home fits? I'm happy to walk you through it.").split(" ");
const SCENES = [
  { composition: "opening_headline_focus", role: "hook", from: 0, to: 9, end: 6.5, data: { market: "Natomas", kind: "Market Update", headline: "New Construction Is Part of Your Competition", eyebrow: "Seller Insight" }, motion: "push_in" },
  { composition: "stat_big_number", role: "insight", from: 9, to: 20, end: 13.5, data: { lead: "Active Listings Are Sitting a Median of", value: "41", unit: "days", trail: "Before Going Under Contract", icon: "calendar" }, claims: [{ field: "value", figure: "r1.f1" }], motion: "pan_right" },
  { composition: "compare_two_card", role: "contrast", from: 20, to: 33, end: 21, data: { market: "Natomas", kind: "Market Update", left: { lead: "Homes That Closed Sold At", percent: 99.4, caption: "of list price on average", icon: "sold" }, right: { lead: "But Here in the Active Market,", highlighted: 6, total: 14, caption: "listings have already taken a price reduction.", icon: "reduced" } }, motion: "pull_back" },
  { composition: "expert_takeaway", role: "consequence", from: 33, to: 42, end: 0, data: { headline: ["Condition +", "Pricing Precision Matter"], statement: "That gap between homes that sold and homes that didn't tells a real story.", support: "New construction sets a baseline—your home has to answer for itself.", points: [{ title: "Condition", caption: "Well-maintained homes stand out.", icon: "condition" }, { title: "Pricing", caption: "Accurate pricing drives results.", icon: "pricing" }] }, motion: "pan_left" },
  { composition: "closing_headline_focus", role: "next_step", from: 42, to: WORDS.length, end: 0, data: { headline: ["Want to talk through where", "your home fits in this picture?"], support: "I'm happy to walk you through it.", tagline: "Pricing · Condition · Competition" }, motion: "push_in" },
];

(async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "amsample-"));
  const perWord = 0.42, total = Math.round((WORDS.length * perWord + 1.5) * 10) / 10;
  const wordT = WORDS.map((w, i) => ({ text: w, start: 0.3 + i * perWord, end: 0.3 + (i + 0.85) * perWord }));
  // scene windows from the word clock
  let t = 0;
  const scenes = SCENES.map((s, i) => {
    const endT = i === SCENES.length - 1 ? total : wordT[s.to].start;
    const sc = { n: i + 1, composition: s.composition, narrative_role: s.role, branding: /^(opening|closing)/.test(s.composition) ? "full" : "subtle", window: { start: t, end: endT },
      visual: { research_visual_context: "stand-in plate for the pipeline test", image_keys: { "9x16": `plate${i}_9x16.png`, "16x9": `plate${i}_16x9.png` }, motion: s.motion }, claims: s.claims || [], data: s.data };
    t = endT;
    return sc;
  });
  const captions = [];
  for (let i = 0; i < WORDS.length; i += 5) { const g = wordT.slice(i, i + 5); captions.push({ text: g.map((w) => w.text).join(" "), start: g[0].start, end: g[g.length - 1].end + 0.1 }); }
  const manifest = { version: "am-manifest-2", project_id: "sample", formats, notice_required: true, notice: { period },
    brand: { primary: "#B08D3F", secondary: "#1D2023", border: "#B08D3F", font: process.env.BRAND_FONT || undefined, headshot_key: "hs.png", agent: { name: "Sam Zieour", brokerage: "SZ Real Estate Group" } },
    audio: { narration_key: "voice.mp3", duration_sec: total }, captions, scenes };
  const assets = {};
  for (let i = 0; i < 5; i++) for (const f of formats) assets[`plate${i}_${f}.png`] = await plate(f, i);
  assets["hs.png"] = headshotPath ? fs.readFileSync(headshotPath) : await sharp({ create: { width: 600, height: 600, channels: 3, background: "#8d99a8" } }).png().toBuffer();
  const voice = path.join(dir, "voice.mp3");
  execFileSync("ffmpeg", ["-y", "-f", "lavfi", "-i", `aevalsrc='0.35*sin(2*PI*180*t)*(0.5+0.5*sin(2*PI*3.1*t))*gt(sin(2*PI*0.9*t)\,-0.3)':d=${total}:s=44100`, "-c:a", "libmp3lame", voice]);
  assets["voice.mp3"] = fs.readFileSync(voice);
  const music = path.join(dir, "music.mp3");
  execFileSync("ffmpeg", ["-y", "-f", "lavfi", "-i", `sine=frequency=220:duration=${total + 2}`, "-c:a", "libmp3lame", music]);
  const res = await renderAgentMarketing({ jobId: "sample", manifest, outDir: out }, (s) => console.log("step:", s), {
    fetchAsset: async (k) => { if (!assets[k]) throw new Error("missing " + k); return assets[k]; },
    getMusic: async () => music,
  });
  console.log(JSON.stringify({ outputs: res.outputs, durationSec: res.durationSec, checks: Object.fromEntries(Object.entries(res.checks).filter(([k]) => k !== "style").map(([k, v]) => [k, { durationSec: v.durationSec, noticeOk: v.noticeOk, notice: v.notice.map((n) => ({ sec: n.sec, meanDiff: n.meanDiff, ok: n.ok })) }])), warnings: res.warnings }, null, 2));
})().catch((e) => { console.error(e); process.exit(1); });
