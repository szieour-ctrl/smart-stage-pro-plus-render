// musicGen.js — Background music, sourced from a curated Suno track
// library (static, pre-hosted files), NOT a live generative API call.
//
// CHANGE (Aug 2026, S3 migration): SUNO_TRACK_LIBRARY now points at S3
// (bucket smart-stage-pro-media-938733852197-us-east-2-an, prefix
// smart-stage-music/) instead of Cloudinary. Same static-file pattern
// as before — Sam generates tracks in Suno's own app, downloads the
// mp3s, uploads each to the smart-stage-music/ S3 prefix, and adds/
// updates the entry below. No live music-generation API call happens
// at request time at all; this file just downloads a pre-made file and
// loops/trims it to fit.
//
// CHANGE (Sep 2026): added 10 more tracks (15 total). New tracks use
// their own generated titles as labels/keys rather than the original
// style/mood naming pattern (japandi_calm, luxury_cinematic, etc.) —
// both patterns coexist fine since the picker just needs a valid key.
//
// To add a new track:
//   1. Generate it in Suno, download the mp3.
//   2. Upload to S3 under smart-stage-music/ (bucket above), simple
//      URL-safe filename (letters/numbers/hyphens/underscores, no spaces).
//   3. Confirm smart-stage-music/* is covered by the bucket's public-read
//      policy (it should already be, from the initial migration).
//   4. Add a new entry below — key is what the frontend's music picker
//      sends as musicStyle; safe to rename/add/remove entries here
//      without touching any other file, same as MUSIC_STYLE_MAP worked
//      before.

const fs = require("fs");
const path = require("path");
const axios = require("axios");
const ffmpeg = require("fluent-ffmpeg");

const S3_MUSIC_BASE = "https://smart-stage-pro-media-938733852197-us-east-2-an.s3.us-east-2.amazonaws.com/smart-stage-music";

const SUNO_TRACK_LIBRARY = {
  // Original 5
  "japandi_calm":       { label: "Japandi — Calm Piano",        url: `${S3_MUSIC_BASE}/japandi-calm.mp3` },
  "luxury_cinematic":   { label: "Luxury — Cinematic Strings",  url: `${S3_MUSIC_BASE}/luxury-cinematic.mp3` },
  "modern_uplifting":   { label: "Modern — Warm & Uplifting",   url: `${S3_MUSIC_BASE}/modern-uplifting.mp3` },
  "farmhouse_acoustic": { label: "Farmhouse — Light Acoustic",  url: `${S3_MUSIC_BASE}/farmhouse-acoustic.mp3` },
  "default":            { label: "Default — Neutral Ambient",  url: `${S3_MUSIC_BASE}/default-ambient.mp3` },

  // Added Sep 2026 (10 new tracks)
  "paper_lantern_court": { label: "Paper Lantern Court", url: `${S3_MUSIC_BASE}/paper-lantern-court.mp3` },
  "breezy":              { label: "Breezy",              url: `${S3_MUSIC_BASE}/breezy.mp3` },
  "aperture_rising":     { label: "Aperture Rising",     url: `${S3_MUSIC_BASE}/aperture-rising.mp3` },
  "open_house_glow":     { label: "Open House Glow",     url: `${S3_MUSIC_BASE}/open-house-glow.mp3` },
  "open_door_plans":     { label: "Open Door Plans",     url: `${S3_MUSIC_BASE}/open-door-plans.mp3` },
  "fresh_keyframes":     { label: "Fresh Keyframes",     url: `${S3_MUSIC_BASE}/fresh-keyframes.mp3` },
  "twilight_terrace":    { label: "Twilight Terrace",    url: `${S3_MUSIC_BASE}/twilight-terrace.mp3` },
  "sunshine_tour":       { label: "Sunshine Tour",       url: `${S3_MUSIC_BASE}/sunshine-tour.mp3` },
  "sunshine":            { label: "Sunshine",            url: `${S3_MUSIC_BASE}/sunshine.mp3` },
  "sunlit_keychain":     { label: "Sunlit Keychain",     url: `${S3_MUSIC_BASE}/sunlit-keychain.mp3` },
};

function resolveTrack(musicStyle) {
  return SUNO_TRACK_LIBRARY[musicStyle] || SUNO_TRACK_LIBRARY.default;
}

// ── SILENT FALLBACK ────────────────────────────────────────────────────
// Same purpose as before: lets the full render pipeline be tested/deployed
// before Sam has replaced the placeholder URLs above with real ones, and
// is also the permanent behavior for musicStyle: "none" (user explicitly
// chose no music — the "Video Only, silent" option from Sam's original
// audio-options doc).

function generateSilentTrack(durationSeconds, workDir) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, "music_silent.mp3");
    ffmpeg()
      .input("anullsrc=channel_layout=stereo:sample_rate=44100")
      .inputFormat("lavfi")
      .duration(durationSeconds)
      .audioCodec("libmp3lame")
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", reject)
      .run();
  });
}

// ── DOWNLOAD + FIT TO DURATION ────────────────────────────────────────
// A static track is very unlikely to be exactly as long as the video.
// Loop it (concat filter) if shorter, trim it if longer — either way,
// output is always exactly durationSeconds long so mixAudio() downstream
// never has to special-case track length.

async function downloadRawTrack(url, workDir) {
  const outputPath = path.join(workDir, "music_raw.mp3");
  const response = await axios.get(url, { responseType: "arraybuffer", timeout: 20000 });
  fs.writeFileSync(outputPath, response.data);
  return outputPath;
}

function probeDuration(filePath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) return reject(new Error(`ffprobe failed for ${filePath}: ${err.message}`));
      resolve(metadata.format.duration);
    });
  });
}

function fitTrackToDuration(rawPath, rawDuration, targetDuration, workDir) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, "music_fitted.mp3");

    if (rawDuration >= targetDuration) {
      // Trim — simple, no loop needed.
      ffmpeg(rawPath)
        .setDuration(targetDuration)
        .audioCodec("libmp3lame")
        .output(outputPath)
        .on("end", () => resolve(outputPath))
        .on("error", reject)
        .run();
      return;
    }

    // Loop — stream_loop repeats the input enough times to cover the
    // target, then setDuration trims the tail to an exact match (avoids
    // an awkward hard cut mid-loop being audible as a click; -1 loops
    // indefinitely and setDuration is what actually bounds it).
    ffmpeg(rawPath)
      .inputOptions(["-stream_loop", "-1"])
      .setDuration(targetDuration)
      .audioCodec("libmp3lame")
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", reject)
      .run();
  });
}

// ── ENTRY POINT ────────────────────────────────────────────────────────

async function generateMusic({ durationSeconds, musicStyle, workDir }) {
  if (musicStyle === "none") {
    return generateSilentTrack(durationSeconds, workDir);
  }

  const track = resolveTrack(musicStyle);

  if (!track.url || track.url.startsWith("REPLACE_WITH_REAL_")) {
    console.warn(`Suno track "${musicStyle}" has no real hosted URL configured yet — using silent fallback. Replace the placeholder in SUNO_TRACK_LIBRARY (musicGen.js) with a real S3 URL.`);
    return generateSilentTrack(durationSeconds, workDir);
  }

  try {
    const rawPath = await downloadRawTrack(track.url, workDir);
    const rawDuration = await probeDuration(rawPath);
    return await fitTrackToDuration(rawPath, rawDuration, durationSeconds, workDir);
  } catch (err) {
    console.error(`Suno track download/fit failed for "${musicStyle}", falling back to silent track:`, err.message);
    return generateSilentTrack(durationSeconds, workDir);
  }
}

module.exports = { generateMusic, resolveTrack, SUNO_TRACK_LIBRARY };
