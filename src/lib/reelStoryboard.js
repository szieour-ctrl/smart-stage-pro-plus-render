// reelStoryboard.js — Reels v1 storyboard engine (Oct 2026).
//
// The 12 event templates from "Smart Stage PRO Plus Reel v1 (SSPP v1) Build" as DATA: six scene
// roles each, with the overlay text each scene carries. resolveScenes() fills a template from
// the Reel's data (listing details + event intake + photo-attached highlights) and returns scene
// specs; renderScenePng() draws each scene as one transparent 1080x1920 layer; compositeStoryboard()
// times the layers onto the finished base video (no text baked in), adds the AB 723 badge and the
// end QR. Rendering stays sharp + ffmpeg (Remotion skipped for v1, per the Oct 16 decision).
//
// Scene spec: {
//   role, image,                       // role + suggested photo (for the storyboard UI / Vision)
//   top?:     { headline, subline, cityLine, pill: [line1, line2], pillY },
//   caption?: "highlight text",        // lower-third caption over the photo
//   closing?: { lines: [l1, l2], addressRow: bool, cta }   // panel with CTA + identity
// }
// Optional fields never leave an empty box: a missing value simply drops its block.
// UNBRANDED (MLS): headline/address/details/highlights stay; CTA, name, DRE, brokerage, headshot
// and logo are dropped (the closing panel keeps only details/date/address).
const path = require("path");
const sharp = require("sharp");
const { renderBrandedOverlay, compositeLayers, probeDuration, W, H } = require("./reelOverlay");

const has = (v) => v !== undefined && v !== null && String(v).trim() !== "";
const T = (v) => (has(v) ? String(v).trim() : "");

// ── detail helpers ───────────────────────────────────────────────────────────
function specItems(d) {
  const out = [];
  if (has(d.beds)) out.push(`${T(d.beds)} Bd`);
  if (has(d.baths)) out.push(`${T(d.baths)} Ba`);
  if (has(d.sqft)) out.push(`${T(d.sqft)} Sq Ft`);
  return out;
}
// Price (optional) over the specs; or specs alone.
function specLines(d, withPrice = true) {
  const specs = specItems(d);
  const price = withPrice ? T(d.price) : "";
  if (price && specs.length) return [price, specs.join("  •  ")];
  if (price) return [price];
  return specs.length ? [specs.join("  •  ")] : [];
}
const place = (d) => T(d.neighborhood) || [T(d.city)].filter(Boolean).join(", ");
const hl = (d, i) => T((d.highlights || [])[i]);
const money = (v) => T(v);

// ── templates ────────────────────────────────────────────────────────────────
// Each scene is (d) => spec. closingCta = default CTA when the user gave none.
const IMG = {
  strongest: "Strongest image", exterior: "Front exterior", living: "Main living area",
  kitchen: "Kitchen", suite: "Primary suite or specialty space", outdoor: "Backyard or outdoor living",
};

const closing = (d, cta, extra = {}) => ({ lines: specLines(d, true), addressRow: true, cta: T(d.cta) || cta, ...extra });

const TEMPLATES = {
  teaser: {
    label: "Teaser", mergePair: [2, 3],
    scenes: [
      { role: "Create curiosity", image: IMG.strongest, build: (d) => ({ top: { headline: T(d.headline) || "SOMETHING SPECIAL IS COMING" } }) },
      { role: "Establish location", image: "Front exterior or neighborhood-facing view", build: (d) => (place(d) ? { top: { subline: `Coming to ${place(d)}` } } : {}) },
      { role: "Introduce the home", image: IMG.living, build: (d) => ({ top: { pill: specLines(d, false) } }) },
      { role: "Show a reason to care", image: "Kitchen or standout room", build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Build anticipation", image: "Backyard or second standout space", build: (d) => ({ caption: hl(d, 1), top: has(d.availability) ? { subline: `Available ${T(d.availability).replace(/^available\s*/i, "")}` } : undefined }) },
      { role: "Invite a response", image: IMG.strongest, build: (d) => ({ closing: { lines: specLines(d, false), addressRow: false, cta: T(d.cta) || "Want the first look?" } }) },
    ],
  },
  coming_soon: {
    label: "Coming Soon", mergePair: [1, 2],
    scenes: [
      { role: "Announce", image: IMG.exterior, build: (d) => ({ top: { headline: "COMING SOON", subline: T(d.address) || place(d) } }) },
      { role: "Orient", image: IMG.living, build: (d) => ({ top: { pill: specLines(d, true) } }) },
      { role: "Highlight", image: IMG.kitchen, build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Add appeal", image: "Primary suite or flexible space", build: (d) => ({ caption: hl(d, 1) }) },
      { role: "Establish timing", image: "Backyard or strongest remaining image", build: (d) => ({ top: has(d.availability) ? { subline: /^available/i.test(T(d.availability)) ? T(d.availability) : `Available ${T(d.availability)}` } : undefined }) },
      { role: "Capture interest", image: IMG.strongest, build: (d) => ({ closing: closing(d, "Request details") }) },
    ],
  },
  just_listed: {
    label: "Just Listed Highlight", mergePair: [1, 2],
    scenes: [
      { role: "Launch", image: "Front exterior or strongest image", build: (d) => ({ top: { headline: "JUST LISTED", subline: T(d.address), cityLine: T(d.city) } }) },
      { role: "Establish value", image: IMG.living, build: (d) => ({ top: { pill: specLines(d, true) } }) },
      { role: "Feature", image: IMG.kitchen, build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Feature", image: IMG.suite, build: (d) => ({ caption: hl(d, 1) }) },
      { role: "Feature", image: IMG.outdoor, build: (d) => ({ caption: hl(d, 2) }) },
      { role: "Drive action", image: IMG.strongest, build: (d) => ({ closing: closing(d, "Schedule your private tour") }) },
    ],
  },
  for_sale: {
    label: "For Sale", mergePair: [1, 2],
    scenes: [
      { role: "Position the home", image: IMG.strongest, build: (d) => ({ top: { headline: "FOR SALE", subline: T(d.positioning || d.headline) } }) },
      { role: "Establish the basics", image: "Front exterior or living area", build: (d) => ({ top: { subline: T(d.address), pill: specLines(d, false), pillY: 600 } }) },
      { role: "Show daily living", image: "Living area or kitchen", build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Show a differentiator", image: "Office, suite, or specialty space", build: (d) => ({ caption: hl(d, 1) }) },
      { role: "Reinforce the offer", image: "Backyard or strongest remaining image", build: (d) => (has(d.price) ? { top: { pill: [T(d.price)] } } : { caption: T(d.sellingPoint) }) },
      { role: "Drive action", image: IMG.strongest, build: (d) => ({ closing: closing(d, "See it in person") }) },
    ],
  },
  for_lease: {
    label: "For Lease", mergePair: [1, 2],
    scenes: [
      { role: "Announce", image: IMG.exterior, build: (d) => ({ top: { headline: "FOR LEASE", subline: T(d.address) } }) },
      { role: "Establish the offer", image: IMG.living, build: (d) => ({ top: { pill: has(d.rent) ? [`${money(d.rent)}/month`, ...(specItems(d).length ? [specItems(d).join("  •  ")] : [])] : specLines(d, false) } }) },
      { role: "Show a feature", image: IMG.kitchen, build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Show flexibility", image: "Bedroom, office, or bonus room", build: (d) => ({ caption: hl(d, 1) }) },
      { role: "Clarify availability", image: "Outdoor space or remaining image", build: (d) => ({ top: has(d.availability) ? { subline: `Available ${T(d.availability).replace(/^available\s*/i, "")}`, pill: has(d.leaseTerms) ? [T(d.leaseTerms)] : undefined, pillY: 600 } : undefined }) },
      { role: "Generate inquiries", image: IMG.strongest, build: (d) => ({ closing: { lines: has(d.rent) ? [`${money(d.rent)}/month`] : specLines(d, false), addressRow: true, cta: T(d.cta) || "Request a showing" } }) },
    ],
  },
  feature_spotlight: {
    label: "Feature Spotlight", mergePair: [1, 2],
    scenes: [
      { role: "Introduce the feature", image: "Best image of the featured space", build: (d) => ({ top: { headline: T(d.feature || d.headline).toUpperCase() } }) },
      { role: "Establish context", image: "Wider view", build: (d) => ({ top: { subline: T(d.address) || place(d), pill: specLines(d, false), pillY: 600 } }) },
      { role: "Explain one detail", image: "Supporting view", build: (d) => ({ caption: T((d.supporting || [])[0]) }) },
      { role: "Explain another", image: "Related room or angle", build: (d) => ({ caption: T((d.supporting || [])[1]) }) },
      { role: "Connect to the experience", image: "Related indoor/outdoor space", build: (d) => ({ caption: T((d.supporting || [])[2]) }) },
      { role: "Invite exploration", image: "Strongest feature image", build: (d) => ({ closing: { lines: T(d.closingMessage) ? [T(d.closingMessage)] : specLines(d, false), addressRow: true, cta: T(d.cta) || "Come see it" } }) },
    ],
  },
  under_contract: {
    label: "Under Contract", mergePair: [1, 2],
    scenes: [
      { role: "Announce the milestone", image: IMG.exterior, build: (d) => ({ top: { headline: "UNDER CONTRACT", subline: T(d.address) } }) },
      { role: "Reintroduce the property", image: IMG.living, build: (d) => ({ top: { pill: specLines(d, false) } }) },
      { role: "Showcase", image: IMG.kitchen, build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Showcase", image: "Suite or specialty space", build: (d) => ({ caption: hl(d, 1) }) },
      { role: "Add the message", image: "Backyard or strongest remaining image", build: (d) => ({ caption: T(d.statusMessage) }) },
      { role: "Turn the milestone into an inquiry", image: IMG.strongest, build: (d) => ({ closing: closing(d, "Planning your next move?") }) },
    ],
  },
  back_on_market: {
    label: "Back on Market", mergePair: [1, 2],
    scenes: [
      { role: "Reannounce", image: IMG.strongest, build: (d) => ({ top: { headline: "BACK ON MARKET", subline: T(d.address) } }) },
      { role: "Reestablish the offer", image: IMG.living, build: (d) => ({ top: { pill: specLines(d, true) } }) },
      { role: "Renew interest", image: IMG.kitchen, build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Show another reason to visit", image: "Suite or specialty space", build: (d) => ({ caption: hl(d, 1) }) },
      { role: "Establish availability", image: "Backyard or remaining image", build: (d) => ({ caption: T(d.availability) || "Now scheduling showings" }) },
      { role: "Drive action", image: IMG.strongest, build: (d) => ({ closing: closing(d, "Schedule your tour") }) },
    ],
  },
  price_drop: {
    label: "Price Drop", mergePair: [2, 3],
    scenes: [
      { role: "Announce the change", image: IMG.strongest, build: (d) => ({ top: { headline: T(d.headline) || "PRICE DROP", subline: T(d.address) } }) },
      { role: "Reveal the price", image: IMG.living, build: (d) => ({ top: { pill: has(d.newPrice) ? [`Now ${money(d.newPrice)}`, ...(has(d.oldPrice) ? [`Previously ${money(d.oldPrice)}`] : [])] : undefined } }) },
      { role: "Ground the offer", image: "Kitchen or wide interior", build: (d) => ({ top: { pill: specLines(d, false) } }) },
      { role: "Reinforce appeal", image: "Standout room", build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Reinforce appeal", image: IMG.outdoor, build: (d) => ({ caption: hl(d, 1) }) },
      { role: "Drive action", image: IMG.strongest, build: (d) => ({ closing: { lines: has(d.newPrice) ? [`Now ${money(d.newPrice)}`, ...specItems(d).slice(0, 3).length ? [specItems(d).join("  •  ")] : []] : specLines(d, false), addressRow: true, cta: T(d.cta) || "Schedule your tour" } }) },
    ],
  },
  price_improvement: {
    label: "Price Improvement Alert", mergePair: [2, 3],
    scenes: [
      { role: "Announce", image: IMG.strongest, build: (d) => ({ top: { headline: T(d.headline) || "PRICE IMPROVEMENT", subline: T(d.address) } }) },
      { role: "Present the updated offer", image: IMG.living, build: (d) => ({ top: { pill: has(d.newPrice) ? [`Now offered at ${money(d.newPrice)}`] : undefined } }) },
      { role: "Establish the home", image: "Kitchen or wide interior", build: (d) => ({ top: { pill: [...specLines(d, false), ...(has(d.oldPrice) ? [`Previously ${money(d.oldPrice)}`] : [])].slice(0, 2) } }) },
      { role: "Support the positioning", image: "Standout room", build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Add another benefit", image: IMG.outdoor, build: (d) => ({ caption: hl(d, 1) }) },
      { role: "Invite a visit", image: IMG.strongest, build: (d) => ({ closing: { lines: has(d.newPrice) ? [`Now offered at ${money(d.newPrice)}`] : specLines(d, false), addressRow: true, cta: T(d.cta) || "Discover the possibilities" } }) },
    ],
  },
  open_house: {
    label: "Open House Invite", mergePair: [1, 2],
    scenes: [
      // The date and time are the primary message: opening AND closing.
      { role: "Make the invitation", image: IMG.exterior, build: (d) => ({ top: { headline: "OPEN HOUSE", pill: [T(d.date), [T(d.startTime), T(d.endTime)].filter(Boolean).join(" – ")].filter(Boolean), pillY: 470 } }) },
      { role: "Identify the property", image: IMG.living, build: (d) => ({ top: { subline: T(d.address), pill: specLines(d, false), pillY: 600 } }) },
      { role: "Give a reason to visit", image: IMG.kitchen, build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Add another reason", image: "Suite or specialty space", build: (d) => ({ caption: hl(d, 1) }) },
      { role: "Reinforce the invitation", image: IMG.outdoor, build: (d) => ({ caption: "Come explore in person", top: has(d.price) ? { pill: [T(d.price)] } : undefined }) },
      { role: "Make attendance easy", image: "Front exterior or strongest image", build: (d) => ({ closing: { lines: [T(d.date), [T(d.startTime), T(d.endTime)].filter(Boolean).join(" – ")].filter(Boolean), addressRow: true, cta: T(d.cta) || "See you there" } }) },
    ],
  },
  just_sold: {
    label: "Just Sold Recap", mergePair: [1, 2],
    scenes: [
      { role: "Announce the result", image: IMG.exterior, build: (d) => ({ top: { headline: "JUST SOLD", subline: T(d.address) } }) },
      { role: "Recap the home", image: IMG.living, build: (d) => ({ top: { pill: specLines(d, false).concat(has(d.salePrice) ? [`Sold for ${money(d.salePrice)}`] : []).slice(0, 2) } }) },
      { role: "Remember a feature", image: IMG.kitchen, build: (d) => ({ caption: hl(d, 0) }) },
      { role: "Remember another", image: "Suite or specialty space", build: (d) => ({ caption: hl(d, 1) }) },
      // Claims like "sold over asking" appear only when the user entered and confirmed them.
      { role: "Add the human message", image: "Backyard or strongest remaining image", build: (d) => ({ caption: T(d.thankYou) || (d.resultConfirmed ? T(d.verifiedResult) : "") }) },
      { role: "Start the next conversation", image: IMG.strongest, build: (d) => ({ closing: closing(d, "Thinking about selling?") }) },
    ],
  },
};

const EVENTS = Object.keys(TEMPLATES);

// Merge two scene specs (5-photo version): union of blocks, first wins on conflicts.
function mergeSpecs(a, b) {
  const out = { ...b, ...a };
  const top = { ...(b.top || {}), ...(a.top || {}) };
  if (Object.keys(top).length) out.top = top;
  out.role = `${a.role} + ${b.role}`;
  out.image = a.image;
  return out;
}

// Returns the ordered scene specs (6, or 5 with the details + first-feature scenes merged).
function resolveScenes(eventKey, data, sceneCount = 6) {
  const tpl = TEMPLATES[eventKey];
  if (!tpl) throw new Error(`Unknown Reel event "${eventKey}". Valid: ${EVENTS.join(", ")}`);
  const d = { ...data, highlights: (data.highlights || []).filter(has), supporting: (data.supporting || []).filter(has) };
  let scenes = tpl.scenes.map((s) => ({ role: s.role, image: s.image, ...s.build(d) }));
  if (sceneCount === 5) {
    const [i, j] = tpl.mergePair;
    const merged = mergeSpecs(scenes[i], scenes[j]);
    scenes = [...scenes.slice(0, i), merged, ...scenes.slice(j + 1)];
    // mergePair indices are adjacent by construction
  }
  return scenes;
}

// ── rendering ────────────────────────────────────────────────────────────────
const blank = () => sharp({ create: { width: W, height: H, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } });

// One scene -> one transparent PNG. params = { brand, address, logoBuf, headshotBuf, warnings, unbranded }
async function renderScenePng(spec, params, outPng, workDir, tag) {
  const layers = [];
  const base = { brand: params.brand, logoBuf: params.logoBuf, logoType: params.logoType, headshotBuf: params.headshotBuf, headshotType: params.headshotType, warnings: params.warnings, unbranded: !!params.unbranded };
  const t = spec.top;
  if (t && (T(t.headline) || T(t.subline) || T(t.cityLine) || (t.pill && t.pill.filter(has).length))) {
    const p = renderBrandedOverlay({ ...base, headline: T(t.headline), address: T(t.subline), cityLine: T(t.cityLine), detailLines: (t.pill || []).filter(has), pillY: t.pillY }, path.join(workDir, `sb_${tag}_top.png`), "opening");
    layers.push({ input: await p });
  }
  if (has(spec.caption)) {
    const p = renderBrandedOverlay({ ...base, caption: spec.caption }, path.join(workDir, `sb_${tag}_cap.png`), "caption");
    layers.push({ input: await p });
  }
  const cl = spec.closing;
  const closingHasContent = cl && ((cl.lines || []).filter(has).length || cl.addressRow || !params.unbranded);
  if (closingHasContent) {
    const c = spec.closing;
    const p = renderBrandedOverlay({
      ...base, cta: c.cta, closingAddress: c.addressRow ? T(params.address) : "",
      closingDetailLines: (c.lines || []).filter(has),
    }, path.join(workDir, `sb_${tag}_close.png`), "closing");
    layers.push({ input: await p });
  }
  if (!layers.length) return null;
  await blank().composite(layers).png().toFile(outPng);
  return outPng;
}

// Scene windows on the final timeline: equal scenes with xfade overlap.
function sceneWindows(duration, n = 6, overlap = 0.5) {
  const s = (duration + (n - 1) * overlap) / n;
  return Array.from({ length: n }, (_, i) => ({ start: i * (s - overlap), end: i * (s - overlap) + s }));
}

// Render all scene layers and composite them (+ badge + end QR) onto baseVideo.
// windows (optional): [{start,end}] from the real timeline; default = equal split.
async function compositeStoryboard(baseVideo, sceneLayers, outVideo, { badgePng, qrPng, windows, qrLeadSec = 1.0, textInSec = 0.35, textOutLeadSec = 0.2 } = {}) {
  const dur = await probeDuration(baseVideo);
  const n = sceneLayers.length;
  const wins = windows || sceneWindows(dur, n);
  const layers = [];
  sceneLayers.forEach((png, i) => {
    if (!png) return;
    const w = wins[i], last = i === n - 1;
    const start = w.start + textInSec;
    const l = { png, startSec: start, fadeSec: 0.4 };
    if (!last) { l.endSec = Math.max(start + 1, w.end - textOutLeadSec); l.fadeOutSec = 0.4; }
    layers.push(l);
  });
  if (badgePng) layers.push({ png: badgePng, startSec: 0, fadeSec: 0.2 });
  if (qrPng) layers.push({ png: qrPng, startSec: Math.max(0, dur - qrLeadSec), fadeSec: 0.3 });
  return compositeLayers(baseVideo, layers, outVideo);
}

module.exports = { TEMPLATES, EVENTS, resolveScenes, renderScenePng, compositeStoryboard, sceneWindows, specLines };
