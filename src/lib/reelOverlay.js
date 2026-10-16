// reelOverlay.js — Reels v1 overlay engine SPIKE ("Just Listed" template).
//
// Overlays are transparent 1080x1920 PNGs drawn with sharp (SVG text via
// librsvg/fontconfig) and composited onto an already-rendered 9:16 base video
// by ffmpeg. Text is never baked into the Ken Burns render, so editing a price
// or headline re-runs only this pass.
//
// Two variants:
//   branded   — headline, address, details, CTA, and the licensed identity
//               footer (name · DRE · brokerage). Safe zone y=250..1500.
//   unbranded — MLS version (MetroList 11.6): NO text, logo, name, DRE or
//               contact. Only the AB 723 badge + compliance QR.
//
// Spike scope: one template, one scene. Scene timing, storyboard and the
// remaining 11 templates come after this is confirmed on Railway.

const path = require("path");
const { spawn } = require("child_process");
const sharp = require("sharp");
const { badgePathFor, badgeGeometry } = require("./badgeAssets");

const W = 1080, H = 1920;

const esc = (s) =>
  String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const hex = (v, fallback) => (/^#[0-9a-fA-F]{6}$/.test(String(v || "")) ? v : fallback);

// Shrink the font until the string fits maxW (rough Montserrat avg glyph width).
function fit(text, maxW, start, min, k = 0.62) {
  let size = start;
  while (size > min && String(text).length * size * k > maxW) size -= 2;
  return size;
}

function justListedSvg(p) {
  const brand = p.brand || {};
  const prim = hex(brand.primary, "#B8860B");
  const sec = hex(brand.secondary, "#0B2545");
  const bord = hex(brand.border, "#D4AF37");
  const headline = p.headline || "JUST LISTED";
  const detail = (p.details || []).filter(Boolean).join("   •   ");
  const cta = p.cta || "Schedule a Private Tour";
  const hs = fit(headline, 920, 120, 70);
  const as = fit(p.address, 920, 56, 36, 0.6);
  const ds = fit(detail, 880, 40, 28, 0.58);
  const cs = fit(cta, 560, 38, 26, 0.64);
  const ctaW = Math.min(820, Math.max(520, cta.length * cs * 0.66 + 110));
  // Optional fields: no detail line => drop the panel's text row and shrink the panel.
  const hasDetail = detail.length > 0;
  const panelY = hasDetail ? 1130 : 1180, panelH = hasDetail ? 250 : 160;
  const ctaY = hasDetail ? 1255 : 1216;
  const ctaTextY = ctaY + 57;
  const foot1 = `${brand.licensedName || ""}${brand.dre ? `  ·  DRE #${brand.dre}` : ""}`;
  const foot2 = brand.brokerage || "";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
  <defs>
    <linearGradient id="t" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0.62"/><stop offset="1" stop-color="#000" stop-opacity="0"/></linearGradient>
    <linearGradient id="b" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#000" stop-opacity="0.55"/><stop offset="1" stop-color="#000" stop-opacity="0"/></linearGradient>
  </defs>
  <rect x="0" y="180" width="${W}" height="560" fill="url(#t)"/>
  <rect x="0" y="1100" width="${W}" height="500" fill="url(#b)"/>
  <text x="${W / 2}" y="400" text-anchor="middle" font-family="Montserrat" font-weight="700" font-size="${hs}" fill="#fff" letter-spacing="4">${esc(headline)}</text>
  <rect x="${W / 2 - 90}" y="436" width="180" height="6" rx="3" fill="${prim}"/>
  <text x="${W / 2}" y="512" text-anchor="middle" font-family="Montserrat" font-weight="600" font-size="${as}" fill="#fff">${esc(p.address)}</text>
  ${p.cityLine ? `<text x="${W / 2}" y="568" text-anchor="middle" font-family="Montserrat" font-weight="500" font-size="34" fill="#fff" fill-opacity="0.9">${esc(p.cityLine)}</text>` : ""}
  <rect x="60" y="${panelY}" width="960" height="${panelH}" rx="28" fill="${sec}" fill-opacity="0.94" stroke="${bord}" stroke-width="3"/>
  ${hasDetail ? `<text x="${W / 2}" y="${panelY + 85}" text-anchor="middle" font-family="Montserrat" font-weight="600" font-size="${ds}" fill="#fff">${esc(detail)}</text>` : ""}
  <rect x="${W / 2 - ctaW / 2}" y="${ctaY}" width="${ctaW}" height="86" rx="43" fill="${prim}"/>
  <text x="${W / 2}" y="${ctaTextY}" text-anchor="middle" font-family="Montserrat" font-weight="700" font-size="${cs}" fill="#fff" letter-spacing="1">${esc(cta)}</text>
  ${foot1 ? `<text x="${W / 2}" y="1440" text-anchor="middle" font-family="Montserrat" font-weight="600" font-size="30" fill="#fff" stroke="#000" stroke-opacity="0.35" stroke-width="0.8" paint-order="stroke">${esc(foot1)}</text>` : ""}
  ${foot2 ? `<text x="${W / 2}" y="1480" text-anchor="middle" font-family="Montserrat" font-weight="500" font-size="28" fill="#fff" fill-opacity="0.92" stroke="#000" stroke-opacity="0.35" stroke-width="0.8" paint-order="stroke">${esc(foot2)}</text>` : ""}
</svg>`;
}

async function renderBrandedOverlay(params, outPng) {
  await sharp(Buffer.from(justListedSvg(params))).png().toFile(outPng);
  return outPng;
}

// Unbranded (MLS) overlay: AB 723 badge + compliance QR only. qrPng is a PNG
// Buffer supplied by the caller (generate-qr.js output on the Netlify side).
async function renderUnbrandedOverlay({ badgeType = "virtually_staged", qrPng }, outPng) {
  const layers = [];
  const geo = badgeGeometry(true);
  const badgePath = badgePathFor(badgeType);
  let badgeTop = H - Math.round(H * geo.bottomFrac);
  if (badgePath) {
    const badge = await sharp(badgePath).resize({ width: Math.round(W * geo.wFrac) }).png().toBuffer();
    const bh = (await sharp(badge).metadata()).height;
    badgeTop = H - Math.round(H * geo.bottomFrac) - bh;
    layers.push({ input: badge, left: Math.round(W * geo.leftFrac), top: badgeTop });
  }
  if (qrPng) {
    const qr = await sharp(qrPng).resize({ width: 200 }).png().toBuffer();
    layers.push({ input: qr, left: W - 200 - 60, top: H - Math.round(H * geo.bottomFrac) - 200 });
  }
  await sharp({ create: { width: W, height: H, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } })
    .composite(layers).png().toFile(outPng);
  return outPng;
}

// Composite overlayPng onto baseVideo with an alpha fade-in. Audio is copied untouched.
function compositeOverlay(baseVideo, overlayPng, outVideo, { startSec = 0.5, fadeSec = 0.4 } = {}) {
  return new Promise((resolve, reject) => {
    const fc = `[1:v]format=rgba,fade=t=in:st=${startSec}:d=${fadeSec}:alpha=1[ov];[0:v][ov]overlay=0:0:format=auto:shortest=1,format=yuv420p`;
    const args = ["-y", "-i", baseVideo, "-loop", "1", "-i", overlayPng, "-filter_complex", fc,
      "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "copy", outVideo];
    const p = spawn("ffmpeg", args, { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    p.stderr.on("data", (d) => { err = (err + d).slice(-2000); });
    p.on("error", reject);
    p.on("close", (code) => (code === 0 ? resolve(outVideo) : reject(new Error(`ffmpeg overlay failed (${code}): ${err}`))));
  });
}

module.exports = { renderBrandedOverlay, renderUnbrandedOverlay, compositeOverlay, justListedSvg, W, H };
