// badgeAssets.js — AB 723 disclosure badge overlay assets + geometry.
//
// "Clean copy" architecture: Smart Stage PRO now saves an UNBADGED staged
// image for video use, and the badge is composited here as a fixed overlay
// on every frame of every output format (16:9 and 9:16), instead of being
// baked into the photo (where a crop could cut it off). Gallery, compliance
// pages and final downloads keep their baked badge — nothing changes there.
//
// badgeType values sent per frame by video-job.js:
//   "virtually_staged"   — Virtually Staged PNG (same asset as the site's badge)
//   "digitally_enhanced" — "Digitally Enhanced" PNG (Cormorant Garamond Italic 600,
//                          same soft engraved look as the site's canvas version)
//   "none"               — clean copy, but no disclosure needed
// Absent/undefined badgeType = legacy payload → legacy behaviour everywhere.

const path = require("path");

const BADGE_FILES = {
  virtually_staged:   path.join(__dirname, "..", "assets", "badges", "virtually_staged.png"),
  digitally_enhanced: path.join(__dirname, "..", "assets", "badges", "digitally_enhanced.png"),
};

// Fractions of the output canvas. 16:9 matches where the baked badge used
// to land inside the old 16:9 crop (bottom-left, ~30% wide); 9:16 sits above
// the platform caption/UI zone at the bottom of Reels/Shorts/TikTok.
const BADGE_GEOMETRY = {
  h: { wFrac: 0.30, leftFrac: 0.02, bottomFrac: 0.04 },
  v: { wFrac: 0.45, leftFrac: 0.05, bottomFrac: 0.22 },
};

function badgePathFor(badgeType) {
  return BADGE_FILES[badgeType] || null;
}

function badgeGeometry(vertical) {
  return vertical ? BADGE_GEOMETRY.v : BADGE_GEOMETRY.h;
}

module.exports = { badgePathFor, badgeGeometry, BADGE_FILES };
