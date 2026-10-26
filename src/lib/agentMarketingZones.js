// agentMarketingZones.js — named layout regions for Agent Marketing scenes (Oct 26, 2026).
//
// Instead of one fixed caption position for every scene, each composition declares its regions for each aspect ratio:
//   headline_zone, editorial_content_zone, agent_identity_zone, caption_zone, disclosure_zone.
// The checks here run on every card: regions may not overlap, must sit inside the frame and the safe margins, and the
// ink a card actually drew must stay out of the caption zone (captions are drawn separately, under the card).
// Pure geometry plus one pixel check; nothing here renders a card.
"use strict";

const sharp = require("sharp");
const { CANVAS, GEOM } = require("./agentMarketingCards");

// Compositions that exist today. Opening and closing carry the agent identity and the full MetroList notice bar;
// the stat, comparison and expert cards carry content plus a one-line source label in a thin footer strip.
const KINDS = ["opening", "closing", "stat", "compare", "expert"];

// Composition ids (the names the treatment library uses) -> the card kind and layout that draws them.
const COMPOSITION_LAYOUT = {
  opening_headline_focus: ["opening", "headline_focus"], opening_editorial_portrait: ["opening", "editorial_portrait"], opening_news_brief: ["opening", "news_brief"],
  closing_headline_focus: ["closing", "headline_focus"], closing_editorial_portrait: ["closing", "editorial_portrait"], closing_news_brief: ["closing", "news_brief"],
  stat_big_number: ["stat", "headline_focus"], compare_two_card: ["compare", "headline_focus"], expert_takeaway: ["expert", "headline_focus"],
};

// Platform-safe margins (px): where the host app's own buttons or crop can cover a frame.
const SAFE = {
  portrait: { top: 120, right: 60, bottom: 40, left: 60 },
  landscape: { top: 60, right: 90, bottom: 40, left: 90 },
};

const rect = (x, y, w, h) => ({ x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) });
const right = (r) => r.x + r.w;
const bottom = (r) => r.y + r.h;
const overlap = (a, b) => a.x < right(b) && b.x < right(a) && a.y < bottom(b) && b.y < bottom(a);

/** The named regions for a composition on a canvas ("portrait" 1080x1920 or "landscape" 1920x1080). */
function zonesFor(kindOrComposition, canvas = "portrait", { side = "left", headshot = true } = {}) {
  let kind = kindOrComposition, layout = "headline_focus";
  if (COMPOSITION_LAYOUT[kindOrComposition]) [kind, layout] = COMPOSITION_LAYOUT[kindOrComposition];
  if (!KINDS.includes(kind)) throw new Error(`Unknown composition "${kindOrComposition}" (known: ${KINDS.join(", ")}, ${Object.keys(COMPOSITION_LAYOUT).join(", ")})`);
  const cv = CANVAS[canvas];
  if (!cv) throw new Error(`Unknown canvas "${canvas}"`);
  const caption = rect(0, cv.captionBand.y, cv.w, cv.captionBand.h);
  if ((kind === "opening" || kind === "closing") && layout === "editorial_portrait") {
    const g = GEOM[canvas].ep, disclosure = rect(0, cv.h - cv.notice.h, cv.w, cv.notice.h);
    if (canvas === "portrait") {
      return { headline_zone: rect(cv.marginX, cv.top, cv.maxW, g.stackLimit - cv.top), agent_identity_zone: rect(cv.marginX, g.panelTop, cv.maxW, g.panelH), caption_zone: caption, disclosure_zone: disclosure };
    }
    const panelX = side === "left" ? cv.marginX : cv.w - cv.marginX - g.panelW;
    const colX = headshot && side === "left" ? cv.marginX + g.panelW + g.colGap : cv.marginX;
    const colW = headshot ? cv.w - 2 * cv.marginX - g.panelW - g.colGap : cv.maxW;
    const z = { headline_zone: rect(colX, cv.top, colW, g.stackLimit - cv.top), agent_name_zone: rect(colX, cv.lockupCy - 70, Math.min(colW, 760), 140), caption_zone: caption, disclosure_zone: disclosure };
    if (headshot) z.agent_identity_zone = rect(panelX, g.panelTop, g.panelW, g.panelH);
    return z;
  }
  if ((kind === "opening" || kind === "closing") && layout === "news_brief") {
    const g = GEOM[canvas].nb;
    return { headline_zone: rect(cv.marginX, g.chipTop, cv.maxW, g.bigLimit - g.chipTop), editorial_content_zone: rect(0, g.bandTop, cv.w, g.bandH), caption_zone: caption, disclosure_zone: rect(0, cv.h - cv.notice.h, cv.w, cv.notice.h) };
  }
  if (kind === "opening" || kind === "closing") {
    const noticeTop = cv.h - cv.notice.h;
    const hs = cv.hs;
    return {
      headline_zone: rect(cv.marginX, cv.top, cv.maxW, cv.bottomLimit - cv.top),
      agent_identity_zone: rect(cv.marginX, cv.lockupCy - hs / 2, cv.maxW, hs),
      caption_zone: caption,
      disclosure_zone: rect(0, noticeTop, cv.w, cv.notice.h),
    };
  }
  const footTop = cv.h - (canvas === "portrait" ? 110 : 80);
  return {
    editorial_content_zone: rect(cv.marginX, cv.top, cv.maxW + (canvas === "landscape" ? 560 : 0), cv.bottomLimit - cv.top),
    caption_zone: caption,
    disclosure_zone: rect(0, footTop, cv.w, cv.h - footTop),
  };
}

/** Problems with a set of zones: overlaps, frame and safe-margin violations. [] when clean. */
function checkZones(zones, canvas = "portrait") {
  const cv = CANVAS[canvas], safe = SAFE[canvas], problems = [];
  const names = Object.keys(zones);
  for (const n of names) {
    const z = zones[n];
    if (z.x < 0 || z.y < 0 || right(z) > cv.w || bottom(z) > cv.h) problems.push(`${n} leaves the frame`);
  }
  for (let i = 0; i < names.length; i++) for (let j = i + 1; j < names.length; j++) {
    if (overlap(zones[names[i]], zones[names[j]])) problems.push(`${names[i]} overlaps ${names[j]}`);
  }
  // Text zones (not the full-width bars) must respect the safe margins.
  for (const n of names) {
    const z = zones[n];
    if (n === "caption_zone" || n === "disclosure_zone" || (z.x === 0 && z.w === cv.w)) continue; // full-width bars
    if (z.x < safe.left || right(z) > cv.w - safe.right || z.y < safe.top) problems.push(`${n} is inside the platform-safe margin`);
  }
  const cap = zones.caption_zone;
  if (cap && bottom(cap) > cv.h - safe.bottom && zones.disclosure_zone && overlap(cap, zones.disclosure_zone)) problems.push("caption_zone runs into disclosure_zone");
  return problems;
}

/**
 * How many pixels of a transparent card PNG inside a rectangle are text or graphics. A soft scrim behind the captions is allowed:
 * a gradient changes slowly, so a pixel counts only when its alpha jumps away from the average of its neighbours on both sides
 * (3 px left and right, 3 px above and below) by more than 24. Text and shapes have edges; a scrim has none.
 */
async function inkIn(pngPath, r) {
  const { data, info } = await sharp(pngPath).ensureAlpha().extract({ left: r.x, top: r.y, width: r.w, height: r.h }).raw().toBuffer({ resolveWithObject: true });
  const { width: W, height: H, channels: ch } = info;
  const al = (x, y) => data[(y * W + x) * ch + 3];
  const K = 3;
  let n = 0;
  for (let y = K; y < H - K; y++) for (let x = K; x < W - K; x++) {
    const a = al(x, y);
    const h = (al(x - K, y) + al(x + K, y)) / 2, v = (al(x, y - K) + al(x, y + K)) / 2;
    if (Math.abs(a - h) > 24 || Math.abs(a - v) > 24) n++;
  }
  return n;
}

/** Full check of a rendered card: its zones are clean and it left the caption zone empty. Returns { ok, problems[] }. */
async function checkCard(kind, canvas, pngPath, opts) {
  const zones = zonesFor(kind, canvas, opts);
  const problems = checkZones(zones, canvas);
  const ink = await inkIn(pngPath, zones.caption_zone);
  if (ink > 0) problems.push(`the card drew ${ink} pixels inside caption_zone`);
  return { ok: problems.length === 0, problems, zones };
}

module.exports = { KINDS, COMPOSITION_LAYOUT, SAFE, zonesFor, checkZones, inkIn, checkCard, overlap };
