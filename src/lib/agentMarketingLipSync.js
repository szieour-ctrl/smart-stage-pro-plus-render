// agentMarketingLipSync.js — optional lip-sync clips for the opening and closing headshots (Nov 25, 2026).
//
// A clip is a short talking video of the agent's headshot made elsewhere (HeyGen) from the exact narration excerpt for that scene.
// This file only places a prepared clip over the headshot area the card already drew, for the scene's window; it calls nobody.
//   - The card still draws the still headshot and its ring, so layout, text zones, branding, captions and notices are exactly as before.
//   - The clip is cropped to the headshot's shape (circle or rounded panel), inset a few pixels so the card's ring stays visible.
//   - The clip's own sound is never used: the finished video keeps SSP's full narration track (with its voice shaping) as the only voice.
//   - A scene whose clip is missing, unreadable or without a headshot area keeps its still headshot, and the scene is reported as a fallback.
// manifest scene field (optional; older manifests have none): scene.lipsync = { clip_key }
"use strict";

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const { execFile } = require("child_process");

const FPS = 25;
const INSET = 3;       // pixels kept clear so the card's gold ring shows around the clip
// HeyGen's frame has a thin strip of the source photo's bottom edge; a 6% zoom anchored near the top crops it away (the face stays put).
const ZOOM = 1.06;
const FADE_IN = 0.15;  // seconds; the card is fading in under it
const run = (cmd, args) => new Promise((resolve, reject) => execFile(cmd, args, { maxBuffer: 1 << 26 }, (e, so, se) => (e ? reject(new Error(`${cmd} failed: ${String(se).slice(-500)}`)) : resolve(so))));

/** The scene's lip-sync request from the manifest: { clipKey } or null. */
function requestFor(scene) {
  const k = scene && scene.lipsync && typeof scene.lipsync.clip_key === "string" ? scene.lipsync.clip_key.trim() : "";
  return k ? { clipKey: k } : null;
}

// The clip starts a little after the scene does. 80 ms (the pilot-clip measurement) still left the mouth ahead of the narration on Sam's Nov 27 reels (both clips),
// and the in-render measurement is weak on real clips, so the default is 200 ms and a measurement is trusted only at a 0.5 match or better.
// pilot clips: 75-95 ms on both). scene.lipsync.delay_sec can override it (0 to 0.5).
const DEFAULT_DELAY_SEC = 0.2;
function delayFor(scene) {
  const d = scene && scene.lipsync && Number(scene.lipsync.delay_sec);
  return Number.isFinite(d) && scene.lipsync.delay_sec !== null ? Math.max(0, Math.min(0.5, d)) : DEFAULT_DELAY_SEC;
}

/**
 * chooseDelay(scene, measured) → { delaySec, source }. An explicit scene.lipsync.delay_sec always wins. Otherwise a trusted measurement of this clip
 * (agentMarketingLipAlign: how far its mouth runs ahead of its own audio) is used, kept between 0 and 0.5 s; with none, the 200 ms default.
 */
function chooseDelay(scene, measured) {
  const d = scene && scene.lipsync && scene.lipsync.delay_sec;
  if (d !== null && d !== undefined && Number.isFinite(Number(d))) return { delaySec: delayFor(scene), source: "manifest" };
  if (measured && Number.isFinite(measured.leadSec) && measured.corr >= 0.5) return { delaySec: Math.max(0, Math.min(0.5, measured.leadSec)), source: "measured" };
  return { delaySec: DEFAULT_DELAY_SEC, source: "default" };
}

/** Where the clip goes: the card's headshot box, inset. Returns { x, y, w, h, shape, rx } in whole pixels, or null. */
function placement(headshot) {
  if (!headshot || !(headshot.w > 2 * INSET + 8) || !(headshot.h > 2 * INSET + 8)) return null;
  const even = (n) => Math.max(2, Math.round(n / 2) * 2);   // video dimensions must be even
  const w = even(headshot.w - 2 * INSET), h = even(headshot.h - 2 * INSET);
  return { x: Math.round(headshot.x + INSET), y: Math.round(headshot.y + INSET), w, h, shape: headshot.shape === "rect" ? "rect" : "circle", rx: Math.max(0, Math.round((headshot.rx || 0) - INSET)) };
}

async function maskPng(pl, outPng) {
  const shape = pl.shape === "circle"
    ? `<circle cx="${pl.w / 2}" cy="${pl.h / 2}" r="${Math.min(pl.w, pl.h) / 2}" fill="#fff"/>`
    : `<rect x="0" y="0" width="${pl.w}" height="${pl.h}" rx="${pl.rx}" fill="#fff"/>`;
  await sharp(Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${pl.w}" height="${pl.h}"><rect width="100%" height="100%" fill="#000"/>${shape}</svg>`)).grayscale().png().toFile(outPng);
  return outPng;
}

/**
 * overlayClips(baseVideo, items, outVideo, workDir)
 *   items: [{ n, clip: path, start, end, place }]  start/end are seconds on the finished timeline (the scene window); place from placement()
 * Each clip is scaled to fill its box (face kept toward the top), shaped by a mask, held to the scene's exact frame count, and overlaid for the
 * window. Audio is copied from the base untouched.
 */
async function overlayClips(baseVideo, items, outVideo, workDir) {
  const args = ["-y", "-i", baseVideo];
  const parts = [];
  let prev = "0:v";
  for (let i = 0; i < items.length; i++) {
    const it = items[i], pl = it.place;
    const frames = Math.max(1, Math.round(it.end * FPS) - Math.round(it.start * FPS)), dur = frames / FPS;
    const mask = await maskPng(pl, path.join(workDir, `lipmask_${it.n}.png`));
    args.push("-i", it.clip, "-framerate", String(FPS), "-loop", "1", "-t", (dur + 0.1).toFixed(3), "-i", mask);
    const ci = 1 + i * 2, mi = ci + 1;
    parts.push(
      `[${ci}:v]fps=${FPS},tpad=stop_mode=clone:stop_duration=2,trim=end_frame=${frames},setpts=PTS-STARTPTS,` +
      `scale=${Math.round(pl.w * ZOOM)}:${Math.round(pl.h * ZOOM)}:force_original_aspect_ratio=increase,crop=${pl.w}:${pl.h}:(iw-${pl.w})/2:(ih-${pl.h})*0.15,format=yuva420p[c${i}]`,
      `[${mi}:v]format=gray,trim=end_frame=${frames},setpts=PTS-STARTPTS[m${i}]`,
      `[c${i}][m${i}]alphamerge,fade=t=in:st=0:d=${FADE_IN}:alpha=1,setpts=PTS+${it.start.toFixed(3)}/TB[o${i}]`,
      `[${prev}][o${i}]overlay=${pl.x}:${pl.y}:format=auto:eof_action=pass[v${i}]`,
    );
    prev = `v${i}`;
  }
  parts.push(`[${prev}]format=yuv420p[out]`);
  await run("ffmpeg", [...args, "-filter_complex", parts.join(";"), "-map", "[out]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-threads", "2", "-c:a", "copy", outVideo]);
  return outVideo;
}

module.exports = { requestFor, delayFor, chooseDelay, DEFAULT_DELAY_SEC, placement, overlayClips, maskPng, INSET, FPS };
