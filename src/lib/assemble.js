// assemble.js — Concatenates motion clips with crossfade transitions,
// mixes in background music, and renders final output formats.

const path = require("path");
const ffmpeg = require("fluent-ffmpeg");

const CROSSFADE_DURATION = 0.6; // seconds between clips

// ── PROBE ACTUAL CLIP DURATION ───────────────────────────────────────────
// The previous version assumed every clip was exactly 4.5s when calculating
// crossfade offsets. That assumption was wrong as soon as duration varied
// even slightly, and caused xfade offsets to be miscalculated — visually
// "swallowing" earlier clips in the sequence (only the last clip appeared
// in testing with 2 frames). Probing real duration fixes this at the root.

function probeDuration(clipPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(clipPath, (err, metadata) => {
      if (err) return reject(new Error(`ffprobe failed for ${clipPath}: ${err.message}`));
      resolve(metadata.format.duration);
    });
  });
}

// ── CONCATENATE CLIPS WITH CROSSFADE ─────────────────────────────────────
// Chains xfade filters across all clips in sequence using REAL probed
// durations for offset calculation, not an assumed fixed length.

async function concatenateClips(clipPaths, workDir) {
  if (clipPaths.length === 1) {
    return clipPaths[0];
  }

  const durations = await Promise.all(clipPaths.map(probeDuration));

  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, "concatenated.mp4");
    const command = ffmpeg();

    clipPaths.forEach((clip) => command.input(clip));

    // Build chained xfade filter graph using real durations.
    // offset = where in the CUMULATIVE output timeline this transition
    // should begin — i.e. (sum of all preceding clip durations so far,
    // minus the crossfade overlaps already consumed).
    let filterParts = [];
    let lastLabel = "0:v";
    let cumulativeOffset = durations[0] - CROSSFADE_DURATION;

    for (let i = 1; i < clipPaths.length; i++) {
      const nextLabel = `${i}:v`;
      const outLabel = i === clipPaths.length - 1 ? "outv" : `v${i}`;

      filterParts.push(
        `[${lastLabel}][${nextLabel}]xfade=transition=fade:duration=${CROSSFADE_DURATION}:offset=${Math.max(0, cumulativeOffset).toFixed(2)}[${outLabel}]`
      );
      lastLabel = outLabel;
      cumulativeOffset += durations[i] - CROSSFADE_DURATION;
    }

    command
      .complexFilter(filterParts)
      .outputOptions(["-map", "[outv]", "-pix_fmt", "yuv420p"])
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`Concatenation failed: ${err.message}`)))
      .run();
  });
}

// ── BEFORE/AFTER WIPE TRANSITION ─────────────────────────────────────────
// Vacant room (pull_back) holds briefly, then wipes left-to-right into
// the staged version (push_in). This is the flagship PRO Plus feature —
// no general video tool can do this because it requires the paired
// vacant/staged Cloudinary URLs that only exist because PRO staged it.

function buildBeforeAfterClip(beforeClipPath, afterClipPath, workDir, outputName) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, outputName);

    ffmpeg()
      .input(beforeClipPath)
      .input(afterClipPath)
      .complexFilter([
        `[0:v]trim=duration=3,setpts=PTS-STARTPTS[vacant]`,
        `[1:v]trim=duration=4,setpts=PTS-STARTPTS[staged]`,
        `[vacant][staged]xfade=transition=wipeleft:duration=0.8:offset=2.5[out]`,
      ])
      .outputOptions(["-map", "[out]", "-pix_fmt", "yuv420p"])
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`Before/After wipe failed: ${err.message}`)))
      .run();
  });
}

// ── MIX MUSIC INTO VIDEO ──────────────────────────────────────────────────

function mixAudio(videoPath, musicPath, workDir) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, "with_music.mp4");

    ffmpeg()
      .input(videoPath)
      .input(musicPath)
      .complexFilter([
        // Music at -20dB under any future voiceover headroom, looped/trimmed
        // to match video length automatically via shortest flag below.
        `[1:a]volume=0.25[music]`,
      ])
      .outputOptions(["-map", "0:v", "-map", "[music]", "-c:v", "copy", "-c:a", "aac", "-shortest"])
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`Audio mix failed: ${err.message}`)))
      .run();
  });
}

// ── RENDER FORMAT (16:9 master, 9:16 reframe) ────────────────────────────

function renderFormat(inputPath, dimensions, workDir, outputName) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, outputName);
    const [w, h] = dimensions.split("x");

    // 9:16 reframe crops to center — smart subject-aware cropping is a
    // Phase 2B refinement; center crop is the correct safe default for v1.
    const filter =
      dimensions === "1080x1920"
        ? `crop=ih*9/16:ih,scale=${w}:${h}`
        : `scale=${w}:${h}:force_original_aspect_ratio=decrease,pad=${w}:${h}:(ow-iw)/2:(oh-ih)/2`;

    ffmpeg(inputPath)
      .videoFilters(filter)
      .outputOptions(["-c:a", "copy", "-movflags", "+faststart"])
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`Format render failed (${dimensions}): ${err.message}`)))
      .run();
  });
}

// ── ENTRY POINT ────────────────────────────────────────────────────────

async function assembleVideo({ clipPaths, musicPath, formats, workDir }) {
  const concatenated = await concatenateClips(clipPaths, workDir);
  const withMusic = await mixAudio(concatenated, musicPath, workDir);

  const outputs = {};

  if (formats.includes("16x9")) {
    outputs["16x9"] = await renderFormat(withMusic, "1920x1080", workDir, "output_16x9.mp4");
  }
  if (formats.includes("9x16")) {
    outputs["9x16"] = await renderFormat(withMusic, "1080x1920", workDir, "output_9x16.mp4");
  }

  return outputs;
}

module.exports = { assembleVideo, buildBeforeAfterClip, concatenateClips, mixAudio, renderFormat };
