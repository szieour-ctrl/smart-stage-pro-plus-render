// endCardOverlay.js — builds the PRO Plus VIDEO end-card overlay layers (Oct 2026).
//
// REPLACES the separate appended 2.5s end-card clip with a panel that fades in over the LAST
// seconds of the finished video, plus the AB 723 compliance QR for the final second. Built on
// the Reels overlay engine (reelOverlay.js). Nothing here touches the audio mix.
//
// Optional fields on the render job (all backward compatible; absent = old behaviour):
//   job.qrUrl     compliance-page URL. When present the QR is added (even if END_FRAME is off).
//   job.endCard = {
//     variant: "branded" | "unbranded",     // unbranded = MLS: no CTA, name, DRE, brokerage, headshot, logo
//     details: ["$689,000","4 Bd","2.5 Ba","2,079 Sq Ft"],
//     cta: "Schedule a Private Tour",       // branded only
//     brand: { licensedName, dre, brokerage, primary, secondary, border, logoUrl, headshotUrl }
//   }
// No job.endCard => legacy panel (address + END_FRAME_CTA_TEXT), shown only when END_FRAME_ENABLED.
const path = require("path");
const axios = require("axios");
const { renderBrandedOverlay, renderQrLayer } = require("./reelOverlay");
const { isEndFrameEnabled } = require("./endFrame");

async function fetchImg(url, label, warnings) {
  if (!url) return { buf: null, type: "" };
  try {
    const r = await axios.get(url, { responseType: "arraybuffer", timeout: 15000 });
    return { buf: Buffer.from(r.data), type: String(r.headers["content-type"] || "") };
  } catch (e) {
    warnings.push(`${label} skipped: could not fetch (${e.message})`);
    return { buf: null, type: "" };
  }
}

// canvas: "landscape" (16:9) or "portraitVideo" (native 9:16 video). Returns { panelPng, qrPng } (either
// may be null) or null when there is nothing to draw.
async function buildEndOverlay(job, workDir, canvas, tag = "") {
  const jid = job.jobId || "job";
  const warnings = [];
  const ec = job.endCard || null;
  const wantPanel = !!(job.address || (ec && (ec.details || []).length)) && (ec ? true : isEndFrameEnabled());
  const qrUrl = typeof job.qrUrl === "string" && /^https:\/\//i.test(job.qrUrl) ? job.qrUrl : null;
  if (job.qrUrl && !qrUrl) console.warn(`[${jid}] [QR MISSING] job.qrUrl is not an https URL — QR skipped.`);
  if (!qrUrl) console.warn(`[${jid}] [QR MISSING] no job.qrUrl on this render — video ships without the compliance QR.`);

  let panelPng = null, qrPng = null;
  if (wantPanel) {
    const unbranded = !!(ec && ec.variant === "unbranded");
    const b = (ec && ec.brand) || {};
    const [logo, hs] = unbranded ? [{}, {}] : await Promise.all([fetchImg(b.logoUrl, "logo", warnings), fetchImg(b.headshotUrl, "headshot", warnings)]);
    const params = {
      address: job.address || "",
      closingAddress: job.address || "",
      details: (ec && ec.details) || [],
      cta: unbranded ? undefined : (ec && ec.cta) || process.env.END_FRAME_CTA_TEXT || "Schedule Your Private Showing",
      unbranded,
      brand: unbranded ? {} : { licensedName: b.licensedName || "", dre: b.dre || "", brokerage: b.brokerage || "", primary: b.primary, secondary: b.secondary, border: b.border },
      logoBuf: logo.buf, logoType: logo.type, headshotBuf: hs.buf, headshotType: hs.type,
      warnings,
    };
    panelPng = await renderBrandedOverlay(params, path.join(workDir, `endcard_panel_${tag || canvas}.png`), "closing", canvas);
  }
  if (qrUrl) {
    const QRCode = require("qrcode");
    const qrBuf = await QRCode.toBuffer(qrUrl, { margin: 1, width: 300 });
    qrPng = await renderQrLayer({ qrPng: qrBuf }, path.join(workDir, `endcard_qr_${tag || canvas}.png`), canvas);
  }
  if (warnings.length) console.warn(`[${jid}] end card warnings: ${warnings.join("; ")}`);
  return panelPng || qrPng ? { panelPng, qrPng } : null;
}

module.exports = { buildEndOverlay };
