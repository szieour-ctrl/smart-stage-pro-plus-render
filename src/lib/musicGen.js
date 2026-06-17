// musicGen.js — Generates background music via Mubert API, prompted from
// the project's Design DNA staging style.
//
// If MUBERT_API_KEY is not set (e.g. during early pipeline testing before
// signup is complete), this falls back to generating silent audio of the
// correct duration so the rest of the pipeline can be tested end-to-end
// without blocking on the Mubert account.

const fs = require("fs");
const path = require("path");
const axios = require("axios");
const ffmpeg = require("fluent-ffmpeg");

const MUSIC_STYLE_MAP = {
  "Japandi":          "minimal ambient piano sparse zen calm",
  "Organic Modern":   "warm acoustic instrumental light uplifting",
  "RH Luxury":        "orchestral cinematic elegant strings",
  "Modern Farmhouse": "light country acoustic warm homey",
  "Contemporary":     "modern ambient electronic clean neutral",
  "Transitional":     "soft piano neutral warm ambient",
  "default":          "calm ambient real estate background neutral",
};

function resolveStylePrompt(musicStyle) {
  return MUSIC_STYLE_MAP[musicStyle] || MUSIC_STYLE_MAP.default;
}

// ── SILENT FALLBACK ────────────────────────────────────────────────────
// Generates a silent audio track of the exact duration needed. This lets
// the full render pipeline (download → motion → assemble → upload) be
// tested before the Mubert account exists. Once MUBERT_API_KEY is set,
// generateMusic() automatically uses the real API instead.

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

// ── MUBERT API ────────────────────────────────────────────────────────
// Real implementation. Mubert generates asynchronously — we submit a
// request, then poll until the track is ready.

async function requestMubertTrack(durationSeconds, stylePrompt) {
  const response = await axios.post(
    "https://api-b2b.mubert.com/v2/RecordTrackTTM",
    {
      method: "RecordTrackTTM",
      params: {
        pat: process.env.MUBERT_API_KEY,
        prompt: stylePrompt,
        duration: durationSeconds,
        format: "mp3",
        intensity: "low",
        mode: "track",
      },
    },
    { timeout: 20000 }
  );

  const task = response.data?.data?.tasks?.[0];
  if (!task) throw new Error("Mubert did not return a task");
  return task;
}

async function pollMubertTask(task, maxAttempts = 30, intervalMs = 4000) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (task.download_link) return task.download_link;

    await new Promise((r) => setTimeout(r, intervalMs));

    const response = await axios.post(
      "https://api-b2b.mubert.com/v2/GetTask",
      {
        method: "GetTask",
        params: { pat: process.env.MUBERT_API_KEY, tasks: [task.id] },
      },
      { timeout: 20000 }
    );

    const updated = response.data?.data?.tasks?.[0];
    if (updated?.download_link) return updated.download_link;
    task = updated || task;
  }

  throw new Error("Mubert track generation timed out");
}

async function downloadMusicFile(url, workDir) {
  const outputPath = path.join(workDir, "music.mp3");
  const response = await axios.get(url, { responseType: "arraybuffer" });
  fs.writeFileSync(outputPath, response.data);
  return outputPath;
}

// ── ENTRY POINT ────────────────────────────────────────────────────────

async function generateMusic({ durationSeconds, musicStyle, workDir }) {
  if (!process.env.MUBERT_API_KEY) {
    console.warn("MUBERT_API_KEY not set — using silent track fallback. Set the key in Railway env vars once Mubert signup is complete.");
    return generateSilentTrack(durationSeconds, workDir);
  }

  try {
    const stylePrompt = resolveStylePrompt(musicStyle);
    const task = await requestMubertTrack(durationSeconds, stylePrompt);
    const downloadUrl = await pollMubertTask(task);
    return await downloadMusicFile(downloadUrl, workDir);
  } catch (err) {
    console.error("Mubert generation failed, falling back to silent track:", err.message);
    return generateSilentTrack(durationSeconds, workDir);
  }
}

module.exports = { generateMusic, resolveStylePrompt, MUSIC_STYLE_MAP };
