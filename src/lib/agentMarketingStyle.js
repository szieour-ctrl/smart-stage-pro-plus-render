// agentMarketingStyle.js — image-aware text style for Agent Marketing cards (Oct 26, 2026).
//
// Sam's direction after the first Natomas render: branding is the headshot and the name; the text graphics should work with each
// picture, not follow the Brand DNA colours and font. So for every scene we look at the picture BEHIND the text and choose:
//   ink     dark or white, whichever reads better over the pixels where the words will sit
//   scrim   how much soft fade the card should add behind its text (0 = none; a clean picture stays clean)
//   accent  the gold rule / sub-heading colour, moved only as far as needed to stay visible on that picture
// After the card is drawn, checkLegibility() measures the finished pixels against the picture and reports the share of text that
// falls under the contrast floor, so a bad pairing is caught by code and not by eye.
const sharp = require("sharp");

const DARK = "#1D2023", WHITE = "#FFFFFF", GOLD = "#B08D3F";
const lin = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
const relLum = (r, g, b) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
const hexLum = (h) => { const n = parseInt(h.slice(1), 16); return relLum((n >> 16) & 255, (n >> 8) & 255, n & 255); };
const ratio = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
const LUM_DARK = hexLum(DARK);
const FLOOR = 3.5; // large display text; the headline sizes here are 80px and up

/** Downscaled luminance map of the plate as the video shows it (cover-fit to the canvas). */
async function lumaMap(plateBuf, cv, scale = 8) {
  const w = Math.round(cv.w / scale), h = Math.round(cv.h / scale);
  const raw = await sharp(plateBuf).rotate().resize(w, h, { fit: "cover", position: "centre" }).removeAlpha().raw().toBuffer();
  const L = new Float32Array(w * h), C = new Uint8Array(raw);
  for (let i = 0; i < w * h; i++) L[i] = relLum(C[i * 3], C[i * 3 + 1], C[i * 3 + 2]);
  return { w, h, L, scale };
}
const zoneIdx = (map, z) => {
  const x0 = Math.max(0, Math.floor(z.x / map.scale)), y0 = Math.max(0, Math.floor(z.y / map.scale));
  const x1 = Math.min(map.w, Math.ceil((z.x + z.w) / map.scale)), y1 = Math.min(map.h, Math.ceil((z.y + z.h) / map.scale));
  const out = [];
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) out.push(y * map.w + x);
  return out;
};
const share = (vals, inkLum, a, scrimLum) => {
  let ok = 0;
  for (const l of vals) { const bg = l * (1 - a) + scrimLum * a; if (ratio(inkLum, bg) >= FLOOR) ok++; }
  return vals.length ? ok / vals.length : 1;
};

/**
 * pickStyle(map, zones) → { ink, scrim, accent, score, mean }
 * zones: rectangles (in canvas pixels) where text sits directly on the picture.
 */
function pickStyle(map, zones) {
  const idx = zones.flatMap((z) => zoneIdx(map, z));
  const vals = idx.map((i) => map.L[i]);
  if (!vals.length) return { ink: DARK, scrim: 0, accent: GOLD, score: 1, mean: 0.5 };
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  // each ink with no scrim; scrim colour is the opposite (white behind dark ink, black behind white ink)
  const dark = share(vals, LUM_DARK, 0, 1), white = share(vals, 1, 0, 0);
  const useDark = dark >= white - 0.02;
  const inkLum = useDark ? LUM_DARK : 1, scrimLum = useDark ? 1 : 0;
  let k = 0, score = useDark ? dark : white;
  // cards draw gradients of up to 0.5*k opacity; over the text zone the average effect is about 0.65 of that peak
  while (score < 0.93 && k < 1.6) { k = Math.round((k + 0.2) * 100) / 100; score = share(vals, inkLum, Math.min(0.9, 0.5 * k * 0.65), scrimLum); }
  const ink = useDark ? DARK : WHITE;
  const bgAfter = mean * (1 - 0.5 * k * 0.65) + scrimLum * 0.5 * k * 0.65;
  return { ink, scrim: k, accent: accentFor(bgAfter), score: Math.round(score * 1000) / 1000, mean: Math.round(mean * 1000) / 1000 };
}

// The gold, moved toward white or black only as far as needed to read at 3:1 on the average background behind the text.
function accentFor(bgLum) {
  const n = parseInt(GOLD.slice(1), 16), base = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  if (ratio(hexLum(GOLD), bgLum) >= 3) return GOLD;
  const target = bgLum < 0.3 ? 255 : 0;
  for (let t = 0.08; t <= 1.0001; t += 0.06) {
    const c = base.map((v) => Math.round(v + (target - v) * t));
    if (ratio(relLum(...c), bgLum) >= 3) return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
  }
  return target ? WHITE : DARK;
}

/** styleFor(plateBuf, canvasGeometry, textZones) — the one call the renderer makes per scene and format. */
async function styleFor(plateBuf, cv, zones) { return pickStyle(await lumaMap(plateBuf, cv), zones); }

/**
 * checkLegibility(cardPng, plateBuf, cv, ink) → { underFloor, ratioOfInkPixels, samples }
 * Ink pixels are the card's opaque pixels close to the ink colour; each is compared with the picture under it (the picture is slightly
 * blurred, because the video drifts a little).
 */
async function checkLegibility(cardPng, plateBuf, cv, ink, region) {
  const W = cv.w, H = cv.h;
  const card = await sharp(cardPng).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const plate = await sharp(plateBuf).rotate().resize(W, H, { fit: "cover", position: "centre" }).blur(6).removeAlpha().raw().toBuffer();
  const n = parseInt(ink.slice(1), 16), ir = (n >> 16) & 255, ig = (n >> 8) & 255, ib = n & 255, inkL = hexLum(ink);
  const isInk = (i) => card.data[i + 3] >= 235 && Math.abs(card.data[i] - ir) + Math.abs(card.data[i + 1] - ig) + Math.abs(card.data[i + 2] - ib) <= 60;
  // The picture under a letter also has the card's soft fade over it. Ink pixels hide that fade, so take it from the nearest
  // non-ink pixel on the same row (the fades are smooth, so a few pixels away is the same value).
  const x0 = region ? region.x : 0, y0 = region ? region.y : 0, x1 = region ? Math.min(W, region.x + region.w) : W, y1 = region ? Math.min(H, region.y + region.h) : H;
  let total = 0, bad = 0;
  for (let y = y0; y < y1; y += 2) {
    for (let x = x0; x < x1; x += 2) {
      const i = (y * W + x) * 4;
      if (!isInk(i)) continue;
      let k = -1;
      for (let d = 3; d <= 40 && k < 0; d += 3) for (const xx of [x - d, x + d]) { if (xx < 0 || xx >= W) continue; const q = (y * W + xx) * 4; if (!isInk(q)) { k = q; break; } }
      const j = (y * W + x) * 3;
      let r = plate[j], g = plate[j + 1], b = plate[j + 2];
      if (k >= 0) { const a = card.data[k + 3] / 255; r = r * (1 - a) + card.data[k] * a; g = g * (1 - a) + card.data[k + 1] * a; b = b * (1 - a) + card.data[k + 2] * a; }
      total++;
      if (ratio(inkL, relLum(r, g, b)) < 3) bad++;
    }
  }
  return { samples: total, underFloor: bad, ratioOfInkPixels: total ? Math.round((bad / total) * 1000) / 1000 : 0 };
}

module.exports = { DARK, WHITE, GOLD, FLOOR, lumaMap, pickStyle, styleFor, checkLegibility, accentFor, relLum, hexLum };
