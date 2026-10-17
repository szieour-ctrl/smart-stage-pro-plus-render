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

// Canvas layouts. Portrait = Reels / native 9:16 video; landscape = 16:9 video end card.
const PORTRAIT = { w: W, h: H, cx: 540, panelX: 60, panelW: 960, panelBottom: 1462, qrTop: 640, qrCard: QR_CARD, qrImg: QR_IMG, detailMaxW: 860, d1: 68, d2: 50 };
// Oct 17, 2026: the 16:9 panel is centered. The AB 723 badge occupies x 38-614, y ~882-1037 on 1920x1080
// (badgeAssets.js, 30% wide, bottom-left), so the panel's bottom edge sits at y=860 to clear it.
const LANDSCAPE = { w: 1920, h: 1080, cx: 960, panelX: 410, panelW: 1100, panelBottom: 860, qrTop: 50, qrCard: 300, qrImg: 256, detailMaxW: 1000, d1: 60, d2: 44 };
// Native 9:16 VIDEO (not Reels): its AB 723 badge keeps badgeAssets' geometry (y~1368-1498), so the
// end panel sits above it.
const PORTRAIT_VIDEO = { ...PORTRAIT, panelBottom: 1340, qrTop: 430 };
const canvasFor = (c) => (c === "landscape" ? LANDSCAPE : c === "portraitVideo" ? PORTRAIT_VIDEO : PORTRAIT);

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
async function justListedSvg(p, part = "all", cv = PORTRAIT) {
  const showTop = part === "all" || part === "opening", showBottom = part === "all" || part === "closing", showCaption = part === "caption";
  const brand = p.brand || {};
  const prim = hex(brand.primary, "#B8860B");
  const sec = hex(brand.secondary, "#0B2545");
  const bord = hex(brand.border, "#D4AF37");
  const ink = textOn(sec);
  const headline = p.headline === undefined ? "JUST LISTED" : p.headline;
  // Details render on two lines so they can be big: price on top, "Bd • Ba • Sq Ft" beneath.
  // Explicit lines (detailLines / closingDetailLines) win over the details array, so storyboard
  // scenes can show e.g. a date over a time, or "Now $X" over "Previously $Y".
  const dItems = (p.details || []).filter(Boolean);
  const linesFrom = (arr) => {
    const a = (arr || []).filter(Boolean);
    if (!a.length) return ["", ""];
    return a.length === 1 ? [a[0], ""] : [a[0], a.slice(1).join("  •  ")];
  };
  const baseLines = p.detailLines ? linesFrom(p.detailLines)
    : (dItems.length > 1 ? [dItems[0], dItems.slice(1).join("  •  ")] : ["", dItems.join("  •  ")]);
  const mkSet = ([l1, l2]) => {
    const d1 = l1 ? fit(l1, cv.detailMaxW, cv.d1, 44, 0.6) : 0;
    const d2 = l2 ? fit(l2, cv.detailMaxW, cv.d2, 34, 0.58) : 0;
    return { l1, l2, d1, d2, h: (d1 ? 74 : 0) + (d2 ? 62 : 0), w: Math.max(l1.length * d1 * 0.6, l2.length * d2 * 0.58), any: !!(l1 || l2) };
  };
  const pillSet = mkSet(baseLines);
  const closeSet = p.closingDetailLines ? mkSet(linesFrom(p.closingDetailLines)) : pillSet;
  const detail = baseLines.join(" ");
  const cta = p.cta || "Schedule a Private Tour";
  // Long headlines wrap to two balanced lines; the last line keeps the single-line baseline (y=400)
  // so the rule, address and details below never move.
  let headLines = [headline];
  if (headline.length > 15 && headline.includes(" ")) {
    const w = headline.split(" ");
    let best = null;
    for (let i = 1; i < w.length; i++) {
      const l1 = w.slice(0, i).join(" "), l2 = w.slice(i).join(" ");
      const m = Math.max(l1.length, l2.length);
      if (!best || m < best.m) best = { m, l: [l1, l2] };
    }
    if (best && headline.length * 70 * 0.62 > 920) headLines = best.l;
  }
  const hs = fit(headLines.reduce((x, y) => (y.length > x.length ? y : x), ""), 920, headLines.length > 1 ? 96 : 120, 56);
  const as = fit(p.address || "", 920, 56, 36, 0.6);
  const detailSvg = (set, cx, top) => {
    let yy = top, out = "";
    if (set.d1) { yy += set.d1 * 0.95; out += `<text x="${cx}" y="${Math.round(yy)}" text-anchor="middle" font-family="Montserrat" font-weight="700" font-size="${set.d1}" fill="${ink}">${esc(set.l1)}</text>`; yy += 18; }
    if (set.d2) { yy += set.d2 * 0.95; out += `<text x="${cx}" y="${Math.round(yy)}" text-anchor="middle" font-family="Montserrat" font-weight="600" font-size="${set.d2}" fill="${ink}">${esc(set.l2)}</text>`; }
    return out;
  };
  const cs = fit(cta, 560, 38, 26, 0.64);
  const pillW = Math.min(980, Math.round(pillSet.w) + 110);
  const pillH = pillSet.h + 40;
  const ctaW = Math.min(820, Math.max(520, cta.length * cs * 0.66 + 110));

  const pillY = p.pillY ?? (headline ? (p.address ? (p.cityLine ? 604 : 548) : 476) : 420);
  // Highlight caption: wrap to at most two balanced lines, sized to fit.
  const capWords = String(p.caption || "").trim().split(/\s+/).filter(Boolean);
  let capLines = [];
  if (capWords.length) {
    const total = capWords.join(" ");
    if (total.length <= 22) capLines = [total];
    else {
      let best = null;
      for (let i = 1; i < capWords.length; i++) {
        const l1 = capWords.slice(0, i).join(" "), l2 = capWords.slice(i).join(" ");
        const w = Math.max(l1.length, l2.length);
        if (!best || w < best.w) best = { w, l: [l1, l2] };
      }
      capLines = best ? best.l : [total];
    }
  }
  const capFs = capLines.length ? fit(capLines.reduce((a, b) => (b.length > a.length ? b : a), ""), 820, 58, 36, 0.62) : 0;
  const capW = capLines.length ? Math.min(960, Math.round(Math.max(...capLines.map((l) => l.length)) * capFs * 0.62) + 96) : 0;
  const capH = capLines.length ? Math.round(capLines.length * capFs * 1.2 + 44) : 0;
  const capY = 1180;

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
  const hasDetail = pillSet.any;
  const hasCloseDetail = closeSet.any;
  const hasIdentity = !p.unbranded && !!(brand.licensedName || brand.dre || brand.brokerage || headshot || logo);

  // Panel stacks from a fixed bottom edge (y=1462) so it never reaches the badge zone below.
  const PAD = 26, ROW_D = hasCloseDetail ? closeSet.h + 14 : 0, ROW_C = p.unbranded ? 0 : 82, ROW_A = p.closingAddress ? 52 : 0, ROW_I = hasIdentity ? 22 + ID_H : 0;
  const panelH = PAD + ROW_A + ROW_D + ROW_C + ROW_I + PAD;
  const panelBottom = cv.panelBottom;
  const panelY = panelBottom - panelH;
  let y = panelY + PAD;
  const addrTop = y; y += ROW_A;
  const detailTop = y; y += ROW_D;
  const ctaY = y; y += ROW_C;
  const idY = y + 22; // identity row top

  let identity = "";
  if (hasIdentity) {
    const left = cv.panelX + 32, right = cv.panelX + cv.panelW - 32;
    const hsX = left, hsY = idY + Math.round((ID_H - HS) / 2);
    const logoX = logo ? right - logo.w : right;
    const logoY = idY + Math.round((ID_H - (logo ? logo.h : 0)) / 2);
    const textX = headshot ? hsX + HS + 24 : left;
    const maxTextW = (logo ? logoX - 24 : right) - textX;
    const centered = !headshot && !logo;
    const anchor = centered ? "middle" : "start";
    const tx = centered ? cv.cx : textX;
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
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.w}" height="${cv.h}">
  <defs>
    <linearGradient id="t" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0.6"/><stop offset="1" stop-color="#000" stop-opacity="0"/></linearGradient>
    <linearGradient id="cap" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="0.5" stop-color="#000" stop-opacity="0.35"/><stop offset="1" stop-color="#000" stop-opacity="0"/></linearGradient>
    <linearGradient id="b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.55"/></linearGradient>
  </defs>
${showTop ? `  <rect x="0" y="0" width="${W}" height="760" fill="url(#t)"/>
  ${headline ? `${headLines.map((l, i) => `<text x="${W / 2}" y="${400 - (headLines.length - 1 - i) * Math.round(hs * 1.1)}" text-anchor="middle" font-family="Montserrat" font-weight="700" font-size="${hs}" fill="#fff" letter-spacing="4">${esc(l)}</text>`).join("\n  ")}
  <rect x="${W / 2 - 90}" y="436" width="180" height="6" rx="3" fill="${prim}"/>` : ""}
  ${p.address ? `<text x="${W / 2}" y="512" text-anchor="middle" font-family="Montserrat" font-weight="600" font-size="${as}" fill="#fff">${esc(p.address)}</text>` : ""}
  ${p.cityLine ? `<text x="${W / 2}" y="568" text-anchor="middle" font-family="Montserrat" font-weight="500" font-size="34" fill="#fff" fill-opacity="0.9">${esc(p.cityLine)}</text>` : ""}
  ${hasDetail ? `<rect x="${(W - pillW) / 2}" y="${pillY}" width="${pillW}" height="${pillH}" rx="40" fill="${sec}" fill-opacity="0.94" stroke="${bord}" stroke-width="3"/>
  ${detailSvg(pillSet, W / 2, pillY + 20)}` : ""}` : ""}
${showCaption && capLines.length ? `  <rect x="0" y="1020" width="${W}" height="520" fill="url(#cap)"/>
  <rect x="${(W - capW) / 2}" y="${capY}" width="${capW}" height="${capH}" rx="36" fill="${sec}" fill-opacity="0.9" stroke="${bord}" stroke-width="3"/>
  ${capLines.map((l, i) => `<text x="${W / 2}" y="${capY + 22 + capFs * 0.95 + i * capFs * 1.2}" text-anchor="middle" font-family="Montserrat" font-weight="700" font-size="${capFs}" fill="${ink}">${esc(l)}</text>`).join("\n  ")}` : ""}
${showBottom ? `  <rect x="0" y="${panelY - 120}" width="${cv.w}" height="${cv.h - panelY + 120}" fill="url(#b)"/>
  <rect x="${cv.panelX}" y="${panelY}" width="${cv.panelW}" height="${panelH}" rx="28" fill="${sec}" fill-opacity="0.94" stroke="${bord}" stroke-width="3"/>
  ${p.closingAddress ? `<text x="${cv.cx}" y="${addrTop + 36}" text-anchor="middle" font-family="Montserrat" font-weight="600" font-size="${fit(p.closingAddress, cv.panelW - 80, 34, 24, 0.6)}" fill="${ink}" fill-opacity="0.92">${esc(p.closingAddress)}</text>` : ""}
  ${hasCloseDetail ? detailSvg(closeSet, cv.cx, detailTop) : ""}
${p.unbranded ? "" : `<rect x="${cv.cx - ctaW / 2}" y="${ctaY}" width="${ctaW}" height="76" rx="38" fill="${prim}"/>
  <text x="${cv.cx}" y="${ctaY + 50}" text-anchor="middle" font-family="Montserrat" font-weight="700" font-size="${cs}" fill="${textOn(prim)}" letter-spacing="1">${esc(cta)}</text>`}${identity}
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
async function renderBrandedOverlay(params, outPng, part = "all", canvas = "portrait") {
  const svg = await justListedSvg(params, part, canvasFor(canvas));
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
async function renderQrLayer({ qrPng }, outPng, canvas = "portrait") {
  const cv = canvasFor(canvas);
  const qr = await sharp(qrPng).resize(cv.qrImg, cv.qrImg, { kernel: "nearest" }).png().toBuffer();
  const card = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${cv.qrCard}" height="${cv.qrCard}"><rect width="${cv.qrCard}" height="${cv.qrCard}" rx="28" fill="#fff"/></svg>`
  );
  const pad = Math.round((cv.qrCard - cv.qrImg) / 2);
  const cardPng = await sharp(card).composite([{ input: qr, left: pad, top: pad }]).png().toBuffer();
  await sharp({ create: { width: cv.w, height: cv.h, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } })
    .composite([{ input: cardPng, left: Math.round(cv.cx - cv.qrCard / 2), top: cv.qrTop }])
    .png().toFile(outPng);
  return outPng;
}

function runFfmpeg(args, label) {
  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", args, { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    p.stderr.on("data", (d) => { err = (err + d).slice(-2000); });
    p.on("error", reject);
    p.on("close", (code, signal) => {
      if (code === 0) return resolve();
      const e = new Error(`ffmpeg ${label} failed (${signal ? "killed by " + signal + (signal === "SIGKILL" ? " - likely out of memory" : "") : code}): ${err.split("\r").pop().slice(-600)}`);
      e.killed = !!signal; reject(e);
    });
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
  // Speed (Oct 17 2026): each PNG input is trimmed to the seconds it is actually on screen and shifted to its
  // start time, so ffmpeg only filters/overlays those frames (it used to process every layer for the whole
  // video, which made a 45s Reel take minutes). eof_action=pass lets the base show through after a layer ends.
  const total = await probeDuration(baseVideo);
  const args = ["-y", "-i", baseVideo];
  const meta = layers.map((l) => {
    const start = Math.max(0, l.startSec || 0), fade = l.fadeSec ?? 0.4;
    const end = l.endSec != null ? Math.min(total, l.endSec) : total;
    const fo = l.endSec != null ? (l.fadeOutSec ?? 0.4) : 0;
    return { start, fade, end, fo, dur: Math.max(0.2, end - start) };
  });
  layers.forEach((l, i) => args.push("-framerate", "25", "-loop", "1", "-t", (meta[i].dur + 0.1).toFixed(3), "-i", l.png));
  const parts = [];
  let prev = "0:v";
  layers.forEach((l, i) => {
    const m = meta[i];
    let f = `[${i + 1}:v]format=rgba,fade=t=in:st=0:d=${m.fade}:alpha=1`;
    if (m.fo > 0) f += `,fade=t=out:st=${Math.max(0, m.dur - m.fo).toFixed(3)}:d=${m.fo}:alpha=1`;
    f += `,setpts=PTS+${m.start.toFixed(3)}/TB`;
    parts.push(`${f}[ov${i}]`);
    parts.push(`[${prev}][ov${i}]overlay=0:0:format=auto:eof_action=pass[v${i}]`);
    prev = `v${i}`;
  });
  parts.push(`[${prev}]format=yuv420p[out]`);
  // Memory (Oct 17 2026): a 30s Room Reveal Reel with many layers was OOM-killed on Railway. Cap threads
  // so peak RAM stays modest; if ffmpeg still gets killed, retry once with the lightest settings.
  const tail = ["-filter_complex", parts.join(";"), "-map", "[out]", "-map", "0:a?", "-shortest"];
  const enc = (preset) => ["-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p", "-c:a", "copy", outVideo];
  try {
    await runFfmpeg(["-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1", ...args, ...tail, "-threads", "2", ...enc("veryfast")], "layers");
  } catch (e) {
    if (!e.killed) throw e;
    console.warn("[compositeLayers] ffmpeg was killed (" + e.message.slice(0, 80) + "); retrying with 1 thread");
    await runFfmpeg(["-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1", ...args, ...tail, "-threads", "1", ...enc("ultrafast")], "layers");
  }
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

// 16:9 / native 9:16 VIDEO end card: the closing panel fades in over the LAST seconds of the
// finished video (no appended clip, audio untouched) and the compliance QR joins for the final
// second. Pass only the layers you have; either may be omitted. Badge is NOT drawn here (the
// video pipeline already carries the AB 723 badge).
async function compositeEndCard(baseVideo, { panelPng, qrPng }, outVideo, opts = {}) {
  return compositeReel(baseVideo, { closingPng: panelPng, qrPng }, outVideo,
    { openStartSec: 0, openSec: 0, closeSec: 3.5, qrLeadSec: 1.0, ...opts });
}

module.exports = {
  compositeEndCard,
  renderBrandedOverlay, renderBadgeLayer, renderQrLayer,
  compositeLayers, compositeReel, probeDuration, W, H,
};
