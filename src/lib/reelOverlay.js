// reelOverlay.js — Reels v1 overlay engine SPIKE ("Just Listed" template).
//
// Overlays are transparent 1080x1920 PNG layers drawn with sharp (SVG text via
// librsvg/fontconfig) and composited onto an already-rendered 9:16 base video
// by ffmpeg. Text is never baked into the Ken Burns render, so editing a price
// or headline re-runs only this pass.
//
// Layers (each its own PNG with its own start time):
//   text   — BRANDED only. Headline, address, details panel, CTA, and an
//            identity row: optional headshot, licensed name, DRE #, brokerage,
//            optional logo. Falls back to a text-only row when the profile has
//            no headshot/logo. Kept inside the y=250..1500 safe zone.
//   badge  — AB 723 "Virtually Staged" / "Digitally Enhanced" badge, bottom-left,
//            on screen the whole video (both variants).
//   qr     — compliance-page QR. Appears only for the LAST second and stays
//            until the video ends, so it never covers the room during the tour.
//
// UNBRANDED (MLS, MetroList 11.6) = badge + end QR only: no text, logo, name,
// DRE or contact. BRANDED = text layer + badge + end QR.

const { spawn } = require("child_process");
const sharp = require("sharp");
const { badgePathFor } = require("./badgeAssets");

const W = 1080, H = 1920;
const QR_CARD = 348;      // white card incl. quiet-zone padding
const QR_IMG = 300;
const HS = 124;           // headshot circle diameter
const LOGO_W = 300, LOGO_H = 108; // logo fits inside this box
const ID_H = 124;         // identity row height
// Reels badge geometry (badgeAssets' geometry is untouched, so existing videos don't move).
// The badge is composited as a fixed overlay on the clean (unbadged) image, so it is not at
// risk of being cropped and can sit in the lowest part of the frame: bottom edge ~y=1805.
// NOTE: Instagram/TikTok caption/profile UI covers roughly y>1632 on those platforms; raise
// bottomFrac (e.g. 0.165 puts the bottom edge at ~y=1603) if a Reel will be posted there.
const REEL_BADGE = { wFrac: 0.45, leftFrac: 0.05, bottomFrac: 0.06 };

const esc = (s) =>
  String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const hex = (v, fallback) => (/^#[0-9a-fA-F]{6}$/.test(String(v || "")) ? v : fallback);

// Shrink the font until the string fits maxW (rough Montserrat avg glyph width).
function fit(text, maxW, start, min, k = 0.62) {
  let size = start;
  while (size > min && String(text).length * size * k > maxW) size -= 2;
  return size;
}

// Readable text color on a given panel color.
function textOn(bgHex) {
  const n = parseInt(bgHex.slice(1), 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return lum > 0.62 ? "#111111" : "#ffffff";
}

// Marketing Profile images come from user uploads / URLs, so never let a bad one (HEIC,
// an HTML error page, a truncated file) fail the whole render. Returns the Buffer if sharp
// can read it, else null plus a human-readable warning explaining what it actually was.
async function usableImage(buf, label, warnings, hint = "") {
  if (!buf) return null;
  try {
    await sharp(buf).metadata();
    return buf;
  } catch (e) {
    const head = buf.subarray(0, 24).toString("latin1").replace(/[^\x20-\x7e]/g, ".");
    if (warnings) warnings.push(`${label} skipped: ${String(e.message).replace(/\s+/g, " ").slice(0, 160)}${hint ? ` (content-type ${hint})` : ""}; first bytes "${head}"`);
    return null;
  }
}

const dataUri = (buf) => `data:image/png;base64,${buf.toString("base64")}`;

// Headshot -> square PNG (circle clip is applied in the SVG).
async function prepHeadshot(buf) {
  return sharp(buf).rotate().resize(HS, HS, { fit: "cover", position: "attention" }).png().toBuffer();
}
// Logo -> fits inside LOGO_W x LOGO_H, transparency preserved.
async function prepLogo(buf) {
  const out = await sharp(buf).rotate().resize(LOGO_W, LOGO_H, { fit: "inside", withoutEnlargement: true }).png().toBuffer();
  const m = await sharp(out).metadata();
  return { buf: out, w: m.width, h: m.height };
}

// part: "opening" (headline/address/details pill), "closing" (details + CTA + identity panel),
// or "all". The two parts are separate layers so each appears only in its own clip.
async function justListedSvg(p, part = "all") {
  const showTop = part !== "closing", showBottom = part !== "opening";
  const brand = p.brand || {};
  const prim = hex(brand.primary, "#B8860B");
  const sec = hex(brand.secondary, "#0B2545");
  const bord = hex(brand.border, "#D4AF37");
  const ink = textOn(sec);
  const headline = p.headline || "JUST LISTED";
  // Details render on two lines so they can be big: price on top, "Bd • Ba • Sq Ft" beneath.
  const dItems = (p.details || []).filter(Boolean);
  const dLine1 = dItems.length > 1 ? dItems[0] : "";
  const dLine2 = dItems.length > 1 ? dItems.slice(1).join("  •  ") : dItems.join("  •  ");
  const detail = dItems.join(" ");
  const cta = p.cta || "Schedule a Private Tour";
  const hs = fit(headline, 920, 120, 70);
  const as = fit(p.address, 920, 56, 36, 0.6);
  const d1 = dLine1 ? fit(dLine1, 860, 68, 44, 0.6) : 0;
  const d2 = dLine2 ? fit(dLine2, 860, 50, 34, 0.58) : 0;
  const detailH = (d1 ? 74 : 0) + (d2 ? 62 : 0);
  const detailW = Math.max(dLine1.length * d1 * 0.6, dLine2.length * d2 * 0.58);
  const detailSvg = (cx, top) => {
    let yy = top, out = "";
    if (d1) { yy += d1 * 0.95; out += `<text x="${cx}" y="${Math.round(yy)}" text-anchor="middle" font-family="Montserrat" font-weight="700" font-size="${d1}" fill="${ink}">${esc(dLine1)}</text>`; yy += 18; }
    if (d2) { yy += d2 * 0.95; out += `<text x="${cx}" y="${Math.round(yy)}" text-anchor="middle" font-family="Montserrat" font-weight="600" font-size="${d2}" fill="${ink}">${esc(dLine2)}</text>`; }
    return out;
  };
  const cs = fit(cta, 560, 38, 26, 0.64);
  const pillW = Math.min(980, Math.round(detailW) + 110);
  const pillH = detailH + 40;
  const ctaW = Math.min(820, Math.max(520, cta.length * cs * 0.66 + 110));

  const warnings = p.warnings;
  let headshot = null, logo = null;
  try {
    const hb = await usableImage(p.headshotBuf, "headshot", warnings, p.headshotType);
    headshot = hb ? await prepHeadshot(hb) : null;
  } catch (e) { headshot = null; if (warnings) warnings.push(`headshot skipped: ${e.message}`); }
  try {
    const lb = await usableImage(p.logoBuf, "logo", warnings, p.logoType);
    logo = lb ? await prepLogo(lb) : null;
  } catch (e) { logo = null; if (warnings) warnings.push(`logo skipped: ${e.message}`); }
  const hasDetail = detail.length > 0;
  const hasIdentity = !p.unbranded && !!(brand.licensedName || brand.dre || brand.brokerage || headshot || logo);

  // Panel stacks from a fixed bottom edge (y=1462) so it never reaches the badge zone below.
  const PAD = 26, ROW_D = hasDetail ? detailH + 14 : 0, ROW_C = 82, ROW_I = hasIdentity ? 22 + ID_H : 0;
  const panelH = PAD + ROW_D + ROW_C + ROW_I + PAD;
  const panelBottom = 1462;
  const panelY = panelBottom - panelH;
  let y = panelY + PAD;
  const detailTop = y; y += ROW_D;
  const ctaY = y; y += ROW_C;
  const idY = y + 22; // identity row top

  let identity = "";
  if (hasIdentity) {
    const left = 92, right = 988;
    const hsX = left, hsY = idY + Math.round((ID_H - HS) / 2);
    const logoX = logo ? right - logo.w : right;
    const logoY = idY + Math.round((ID_H - (logo ? logo.h : 0)) / 2);
    const textX = headshot ? hsX + HS + 24 : left;
    const maxTextW = (logo ? logoX - 24 : right) - textX;
    const centered = !headshot && !logo;
    const anchor = centered ? "middle" : "start";
    const tx = centered ? W / 2 : textX;
    const line1 = brand.licensedName || "";
    const line2 = [brand.dre ? `DRE #${brand.dre}` : "", brand.brokerage || ""].filter(Boolean);
    const f1 = fit(line1, maxTextW, 34, 24, 0.62);
    const f2 = fit(line2.join("  ·  "), maxTextW, 26, 20, 0.58);
    // If DRE + brokerage don't fit on one line at a readable size, split them.
    const split = f2 < 23 && line2.length === 2;
    const rows = [];
    if (line1) rows.push({ t: line1, s: f1, w: 700 });
    if (split) { rows.push({ t: line2[0], s: 24, w: 500 }); rows.push({ t: fitText(line2[1], maxTextW, 24), s: 24, w: 500 }); }
    else if (line2.length) rows.push({ t: line2.join("  ·  "), s: f2, w: 500 });
    const lh = rows.length >= 3 ? 34 : 40;
    const blockH = rows.length * lh;
    let ty = idY + Math.round((ID_H - blockH) / 2) + lh - 8;
    const textSvg = rows.map((r) => {
      const el = `<text x="${tx}" y="${ty}" text-anchor="${anchor}" font-family="Montserrat" font-weight="${r.w}" font-size="${r.s}" fill="${ink}" fill-opacity="${r.w === 700 ? 1 : 0.9}">${esc(r.t)}</text>`;
      ty += lh; return el;
    }).join("\n  ");
    identity = `
  ${headshot ? `<clipPath id="hs"><circle cx="${hsX + HS / 2}" cy="${hsY + HS / 2}" r="${HS / 2}"/></clipPath>
  <image href="${dataUri(headshot)}" x="${hsX}" y="${hsY}" width="${HS}" height="${HS}" clip-path="url(#hs)" preserveAspectRatio="xMidYMid slice"/>
  <circle cx="${hsX + HS / 2}" cy="${hsY + HS / 2}" r="${HS / 2}" fill="none" stroke="${bord}" stroke-width="3"/>` : ""}
  ${textSvg}
  ${logo ? `<image href="${dataUri(logo.buf)}" x="${logoX}" y="${logoY}" width="${logo.w}" height="${logo.h}"/>` : ""}`;
  }

  // Scrims fade to fully transparent on the photo side and run to the frame edge, so
  // there is no visible hard edge anywhere.
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
  <defs>
    <linearGradient id="t" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0.6"/><stop offset="1" stop-color="#000" stop-opacity="0"/></linearGradient>
    <linearGradient id="b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.55"/></linearGradient>
  </defs>
${showTop ? `  <rect x="0" y="0" width="${W}" height="760" fill="url(#t)"/>
  <text x="${W / 2}" y="400" text-anchor="middle" font-family="Montserrat" font-weight="700" font-size="${hs}" fill="#fff" letter-spacing="4">${esc(headline)}</text>
  <rect x="${W / 2 - 90}" y="436" width="180" height="6" rx="3" fill="${prim}"/>
  <text x="${W / 2}" y="512" text-anchor="middle" font-family="Montserrat" font-weight="600" font-size="${as}" fill="#fff">${esc(p.address)}</text>
  ${p.cityLine ? `<text x="${W / 2}" y="568" text-anchor="middle" font-family="Montserrat" font-weight="500" font-size="34" fill="#fff" fill-opacity="0.9">${esc(p.cityLine)}</text>` : ""}
  ${hasDetail ? `<rect x="${(W - pillW) / 2}" y="604" width="${pillW}" height="${pillH}" rx="40" fill="${sec}" fill-opacity="0.94" stroke="${bord}" stroke-width="3"/>
  ${detailSvg(W / 2, 604 + 20)}` : ""}` : ""}
${showBottom ? `  <rect x="0" y="${panelY - 120}" width="${W}" height="${H - panelY + 120}" fill="url(#b)"/>
  <rect x="60" y="${panelY}" width="960" height="${panelH}" rx="28" fill="${sec}" fill-opacity="0.94" stroke="${bord}" stroke-width="3"/>
  ${hasDetail ? detailSvg(W / 2, detailTop) : ""}
  <rect x="${W / 2 - ctaW / 2}" y="${ctaY}" width="${ctaW}" height="76" rx="38" fill="${prim}"/>
  <text x="${W / 2}" y="${ctaY + 50}" text-anchor="middle" font-family="Montserrat" font-weight="700" font-size="${cs}" fill="${textOn(prim)}" letter-spacing="1">${esc(cta)}</text>${identity}
` : ""}
</svg>`;
}

// Truncate with an ellipsis if even the minimum size can't fit.
function fitText(text, maxW, size, k = 0.58) {
  const maxChars = Math.floor(maxW / (size * k));
  return String(text).length > maxChars ? String(text).slice(0, Math.max(1, maxChars - 1)) + "…" : text;
}

// params.logoBuf / params.headshotBuf: optional image Buffers from the Marketing Profile.
// params.warnings: optional array; images that can't be decoded are skipped and reported there.
async function renderBrandedOverlay(params, outPng, part = "all") {
  const svg = await justListedSvg(params, part);
  await sharp(Buffer.from(svg)).png().toFile(outPng);
  return outPng;
}

// AB 723 badge layer (bottom-left, above the platform caption zone).
async function renderBadgeLayer({ badgeType = "virtually_staged" } = {}, outPng) {
  const geo = REEL_BADGE;
  const badgePath = badgePathFor(badgeType);
  const layers = [];
  if (badgePath) {
    const badge = await sharp(badgePath).resize({ width: Math.round(W * geo.wFrac) }).png().toBuffer();
    const bh = (await sharp(badge).metadata()).height;
    layers.push({ input: badge, left: Math.round(W * geo.leftFrac), top: H - Math.round(H * geo.bottomFrac) - bh });
  }
  await sharp({ create: { width: W, height: H, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } })
    .composite(layers).png().toFile(outPng);
  return outPng;
}

// Compliance QR as a centered white card. Only composited for the last second.
// qrPng is a PNG Buffer (generate-qr.js output on the Netlify side).
async function renderQrLayer({ qrPng }, outPng) {
  const qr = await sharp(qrPng).resize(QR_IMG, QR_IMG, { kernel: "nearest" }).png().toBuffer();
  const card = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${QR_CARD}" height="${QR_CARD}"><rect width="${QR_CARD}" height="${QR_CARD}" rx="28" fill="#fff"/></svg>`
  );
  const pad = Math.round((QR_CARD - QR_IMG) / 2);
  const cardPng = await sharp(card).composite([{ input: qr, left: pad, top: pad }]).png().toBuffer();
  await sharp({ create: { width: W, height: H, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } })
    .composite([{ input: cardPng, left: Math.round((W - QR_CARD) / 2), top: 640 }])
    .png().toFile(outPng);
  return outPng;
}

function runFfmpeg(args, label) {
  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", args, { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    p.stderr.on("data", (d) => { err = (err + d).slice(-2000); });
    p.on("error", reject);
    p.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg ${label} failed (${code}): ${err}`))));
  });
}

function probeDuration(file) {
  return new Promise((resolve, reject) => {
    const p = spawn("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", file], { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    p.stdout.on("data", (d) => (out += d));
    p.on("error", reject);
    p.on("close", () => { const n = parseFloat(out); Number.isFinite(n) ? resolve(n) : reject(new Error("ffprobe: no duration")); });
  });
}

// Composite any number of PNG layers onto baseVideo. layers: [{ png, startSec, fadeSec }].
// Each layer fades in at startSec and stays on to the end. overlay=shortest=1 ends each
// (looped) PNG input with the base video. Audio is copied untouched.
async function compositeLayers(baseVideo, layers, outVideo) {
  const args = ["-y", "-i", baseVideo];
  layers.forEach((l) => args.push("-loop", "1", "-i", l.png));
  const parts = [];
  let prev = "0:v";
  layers.forEach((l, i) => {
    const start = Math.max(0, l.startSec || 0), fade = l.fadeSec ?? 0.4;
    let f = `[${i + 1}:v]format=rgba,fade=t=in:st=${start.toFixed(3)}:d=${fade}:alpha=1`;
    if (l.endSec != null) {
      const fo = l.fadeOutSec ?? 0.4;
      f += `,fade=t=out:st=${Math.max(start, l.endSec - fo).toFixed(3)}:d=${fo}:alpha=1`;
    }
    parts.push(`${f}[ov${i}]`);
    parts.push(`[${prev}][ov${i}]overlay=0:0:format=auto:shortest=1[v${i}]`);
    prev = `v${i}`;
  });
  parts.push(`[${prev}]format=yuv420p[out]`);
  args.push("-filter_complex", parts.join(";"), "-map", "[out]", "-map", "0:a?", "-shortest",
    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "copy", outVideo);
  await runFfmpeg(args, "layers");
  return outVideo;
}

// Build the Reel timeline and composite it.
//   openingPng — headline/address/details pill. Shown early, then fades out so it never sits on
//                every room.
//   closingPng — details + CTA + identity panel. Fades in as an overlay on the LAST photo and
//                holds to the end (no separate end-card clip).
//   badgePng   — AB 723 badge, always on.
//   qrPng      — compliance QR, last qrLeadSec seconds, holds to the end.
// Scene hooks (optional, seconds on the final timeline): firstSceneEndSec and lastSceneStartSec
// let the real storyboard pin the opening and closing windows to actual clips. Without them the
// windows are time-based (openSec from the start, closeSec before the end).
async function compositeReel(baseVideo, { openingPng, closingPng, textPng, badgePng, qrPng }, outVideo, opts = {}) {
  const { openStartSec = 0.4, openSec = 3.5, closeSec = 3.5, qrLeadSec = 1.0,
    firstSceneEndSec = null, lastSceneStartSec = null } = opts;
  const dur = await probeDuration(baseVideo);
  const openEnd = firstSceneEndSec != null ? Math.max(openStartSec + 1, firstSceneEndSec - 0.2) : openStartSec + openSec;
  let closeStart = lastSceneStartSec != null ? lastSceneStartSec + 0.4 : dur - closeSec;
  closeStart = Math.min(Math.max(closeStart, openEnd), Math.max(0, dur - 1.5));
  const layers = [];
  if (textPng) layers.push({ png: textPng, startSec: 0.5, fadeSec: 0.4 }); // legacy single-layer
  if (openingPng) layers.push({ png: openingPng, startSec: openStartSec, fadeSec: 0.4, endSec: openEnd, fadeOutSec: 0.4 });
  if (closingPng) layers.push({ png: closingPng, startSec: closeStart, fadeSec: 0.5 });
  if (badgePng) layers.push({ png: badgePng, startSec: 0, fadeSec: 0.2 });
  if (qrPng) layers.push({ png: qrPng, startSec: Math.max(0, dur - qrLeadSec), fadeSec: 0.3 });
  return compositeLayers(baseVideo, layers, outVideo);
}

module.exports = {
  renderBrandedOverlay, renderBadgeLayer, renderQrLayer,
  compositeLayers, compositeReel, probeDuration, W, H,
};
