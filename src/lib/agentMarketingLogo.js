// agentMarketingLogo.js — the subscriber's logo on Agent Marketing opening and closing cards (Nov 9, 2026).
//
// The logo is optional. It sits on a light rounded plate (so a dark or transparent logo reads on any scrim or dark band), to the right
// of the agent's name block, vertically centered on it. The plate is placed only where the free space the layout leaves is wide enough;
// otherwise it is left off and a warning is recorded. A logo can never fail a Reel.
"use strict";

const sharp = require("sharp");

const MIN_PLATE_W = 120; // a narrower plate would make the logo unreadable
const MIN_PLATE_H = 56;
const PAD = 14;
const MAX_PLATE_W = 380;

/**
 * @param logoBuf  the logo image bytes (PNG/JPG/WebP) or null
 * @param area     { x0, x1, cy, maxH, minTop?, align? }  free horizontal span, vertical center, tallest allowed plate; minTop = highest y the plate may reach;
 *                 align "left" puts the plate at x0 (default: right-aligned to x1)
 * @returns        { svg, placed, warning? }
 */
async function logoPlate(logoBuf, area) {
  if (!logoBuf) return { svg: "", placed: false };
  let meta;
  try { meta = await sharp(logoBuf).metadata(); } catch { return { svg: "", placed: false, warning: "the logo image could not be read, so it was left off" }; }
  if (!meta.width || !meta.height) return { svg: "", placed: false, warning: "the logo image could not be read, so it was left off" };

  const { x0, x1, cy } = area;
  let maxH = area.maxH;
  if (Number.isFinite(area.minTop)) maxH = Math.min(maxH, 2 * (cy - area.minTop));
  const availW = Math.min(MAX_PLATE_W, x1 - x0);
  if (availW < MIN_PLATE_W || maxH < MIN_PLATE_H) return { svg: "", placed: false, warning: "there was no room for the logo on this card, so it was left off" };

  const png = await sharp(logoBuf).rotate()
    .resize(Math.round(availW - 2 * PAD), Math.round(maxH - 2 * PAD), { fit: "inside", withoutEnlargement: true })
    .png().toBuffer({ resolveWithObject: true });
  const w = png.info.width + 2 * PAD, h = png.info.height + 2 * PAD;
  const x = Math.round(area.align === "left" ? x0 : x1 - w), y = Math.round(cy - h / 2);
  const svg = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="16" fill="#FFFFFF" fill-opacity="0.95"/>
  <image href="data:image/png;base64,${png.data.toString("base64")}" x="${x + PAD}" y="${y + PAD}" width="${png.info.width}" height="${png.info.height}"/>`;
  return { svg, placed: true, box: { x, y, w, h } };
}

module.exports = { logoPlate, MIN_PLATE_W };
