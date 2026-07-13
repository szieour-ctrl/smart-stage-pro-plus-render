// musicGen.js — Background music, sourced from a curated Suno track
// library (static, pre-hosted files), NOT a live generative API call.
//
// CHANGE (July 2026, audio build): REPLACES the old Mubert integration
// entirely. Mubert was scaffolded but never actually turned on in
// production (MUBERT_API_KEY was never set — every video shipped with a
// silent fallback track, always). Per Sam's direction: rather than a
// live prompt-to-music API (Mubert, or an unofficial Suno API wrapper),
// this is a small, fixed set of tracks Sam generates himself in Suno's
// own app, downloads, and hosts on Cloudinary — same pattern as the 8
// Photographic Presets and the ElevenLabs voice options. No live music-
// generation API call happens at request time at all; this file just
// downloads a pre-made file and loops/trims it to fit.
//
// SUNO_TRACK_LIBRARY BELOW IS PLACEHOLDER DATA — every url is a dummy.
// Sam needs to: generate tracks in Suno's own app, download the mp3s,
// upload each to Cloudinary (any folder, e.g. "smart-stage-audio/music"),
// and replace the placeholder urls below with the real Cloudinary URLs.
// Track IDs (the map keys) are what the frontend's music picker sends as
// musicStyle — safe to rename/add/remove entries here without touching
// any other file, same as MUSIC_STYLE_MAP worked before.

const fs = require("fs");
const path = require("path");
const axios = require("axios");
const ffmpeg = require("fluent-ffmpeg");

const SUNO_TRACK_LIBRARY = {
  "japandi_calm":       { label: "Japandi — Calm Piano",        url: "REPLACE_WITH_REAL_CLOUDINARY_URL_1.mp3" },
  "luxury_cinematic":   { label: "Luxury — Cinematic Strings",  url: "REPLACE_WITH_REAL_CLOUDINARY_URL_2.mp3" },
  "modern_uplifting":   { label: "Modern — Warm & Uplifting",   url: "REPLACE_WITH_REAL_CLOUDINARY_URL_3.mp3" },
  "farmhouse_acoustic": { label: "Farmhouse — Light Acoustic",  url: "REPLACE_WITH_REAL_CLOUDINARY_URL_4.mp3" },
  "default":            { label: "Default — Neutral Ambient",  url: "REPLACE_WITH_REAL_CLOUDINARY_URL_5.mp3" },
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
    console.warn(`Suno track "${musicStyle}" has no real Cloudinary URL configured yet — using silent fallback. Replace the placeholder in SUNO_TRACK_LIBRARY (musicGen.js) with a real hosted track.`);
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
