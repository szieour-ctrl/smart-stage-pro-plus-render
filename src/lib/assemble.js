// assemble.js — Concatenates motion clips with crossfade transitions,
// mixes in background music, and renders final output formats.

const path = require("path");
const ffmpeg = require("fluent-ffmpeg");

const CROSSFADE_DURATION = 0.6; // seconds between clips

// ── NORMALIZE CLIP (fixes the real root cause of the xfade crash) ────────
// CONFIRMED ROOT CAUSE of "Error reinitializing filters! / Failed to
// inject frame into filter network: Invalid argument" — a real crash from
// a real test job with 4 clips, 2 sourced from Kling (fal.ai's own output
// resolution/fps/codec) and 2 from the local Ken Burns fallback (OpenCV/
// FFmpeg, whatever this codebase already renders at). xfade requires every
// pair of inputs it chains together to share resolution, frame rate, and
// pixel format — concatenateClips() never enforced this, so any mix of
// Kling + Ken Burns clips (or even two Kling clips at different durations/
// fal.ai output settings) was one mismatch away from crashing the whole
// render.
//
// FIX: normalize every clip to identical parameters BEFORE it ever reaches
// the xfade chain, regardless of source. This is deliberately a fixed
// target (1920x1080/30fps/yuv420p) rather than "whatever Kling outputs" —
// Ken Burns clips would still need separate normalization to MATCH
// whatever Kling's native spec is, so pinning both to one target this
// codebase controls is no more work and is robust to fal.ai ever changing
// Kling's default output spec in the future. 1920x1080 matches the final
// 16:9 output exactly, so this adds no redundant scaling at renderFormat()
// time for the common case.
// CHANGE: fps corrected from an initial 30 to 20, to match the EXISTING
// normalization convention already established in klingMotion.js's own
// concatTwoClips() (used for the Before/After continuation feature) —
// confirmed by direct inspection: OUTPUT_W=1920, OUTPUT_H=1080, fps=20,
// format=yuv420p. Using a different fps here would have meant a
// continuation-combined clip (already normalized once, at 20fps, inside
// klingMotion.js) gets silently re-normalized to a SECOND, different fps
// the moment it reaches assemble.js — wasted re-encoding at best, and a
// new source of inconsistency at worst if other clips in the same job
// were normalized to yet a third value. One target, matched across both
// files, is the actual fix — not just "pick a fixed number and move on."
const NORMALIZE_WIDTH  = 1920;
const NORMALIZE_HEIGHT = 1080;
const NORMALIZE_FPS    = 20;

function normalizeClip(clipPath, workDir, index) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, `normalized_${index}.mp4`);
    ffmpeg(clipPath)
      .videoFilters([
        // Scale to fit within target dimensions preserving aspect ratio,
        // then pad to exactly the target size — same safe-default pattern
        // already used in renderFormat() below, applied here instead at
        // the per-clip stage so xfade never sees a size mismatch.
        `scale=${NORMALIZE_WIDTH}:${NORMALIZE_HEIGHT}:force_original_aspect_ratio=decrease`,
        `pad=${NORMALIZE_WIDTH}:${NORMALIZE_HEIGHT}:(ow-iw)/2:(oh-ih)/2`,
        `fps=${NORMALIZE_FPS}`,
      ])
      .outputOptions(["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast"])
      // Audio is dropped here deliberately — concatenateClips' xfade only
      // ever operates on [x:v] video streams (see lastLabel/nextLabel
      // below), and mixAudio() adds the real soundtrack afterward across
      // the whole concatenated video. Carrying per-clip audio through this
      // step would be discarded work and a second source of format
      // mismatches (Kling clips may have audio tracks with different
      // sample rates than Ken Burns clips, which have none at all).
      .noAudio()
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`Clip normalization failed for ${clipPath}: ${err.message}`)))
      .run();
  });
}

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
  // NEW — normalize every clip BEFORE the single-clip early-return check
  // too. A single Kling clip still needs to end up at a known, consistent
  // resolution/fps/pixel format for mixAudio()/renderFormat() downstream,
  // even though there's no xfade chain to crash with only one clip.
  const normalizedPaths = await Promise.all(
    clipPaths.map((clip, i) => normalizeClip(clip, workDir, i))
  );

  if (normalizedPaths.length === 1) {
    return normalizedPaths[0];
  }

  const durations = await Promise.all(normalizedPaths.map(probeDuration));

  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, "concatenated.mp4");
    const command = ffmpeg();

    normalizedPaths.forEach((clip) => command.input(clip));

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

// ── MIX AUDIO — music + optional narration ────────────────────────────
//
// CHANGE (July 2026, audio build): REPLACES the old flat -20dB-forever
// mix. That version had no fade in/out, no loudness normalization, and
// no way to carry a narration track at all — the "Video Only, silent"
// and "Video + Cinematic Music" options from Sam's original audio-options
// doc worked, but "Video + AI Narration + Cinematic Music" had nothing to
// plug into.
//
// narrationPath is optional — when absent, behavior is music-only (fade
// in/out + loudnorm, still a real improvement over the old flat volume).
// When present:
//   - sidechaincompress automatically ducks the music WHENEVER narration
//     audio is actually present in the signal — this is real audio
//     ducking, not a manually-timed volume envelope. It works correctly
//     even though narration length rarely matches video length exactly
//     (narration plays once near the start; music fills the rest at full
//     volume once narration ends, since sidechaincompress only reduces
//     music while it detects narration signal).
//   - amix combines the ducked music and narration into one stream,
//     padded with silence to match the longer of the two (duration=longest)
//     rather than cutting off video audio at the shorter track's length.
//   - loudnorm normalizes final loudness to a standard broadcast target
//     (-16 LUFS, matches typical streaming/social-video loudness norms)
//     so tracks of very different source loudness (Suno tracks vs
//     ElevenLabs TTS output) don't need per-track manual leveling.

function downloadNarrationAudio(narrationUrl, workDir) {
  return new Promise((resolve, reject) => {
    const axios = require("axios");
    const fs = require("fs");
    axios.get(narrationUrl, { responseType: "arraybuffer", timeout: 20000 })
      .then((response) => {
        const outputPath = path.join(workDir, "narration.mp3");
        fs.writeFileSync(outputPath, response.data);
        resolve(outputPath);
      })
      .catch(reject);
  });
}

function mixAudio(videoPath, musicPath, workDir, narrationPath) {
  return new Promise(async (resolve, reject) => {
    try {
      const outputPath = path.join(workDir, "with_music.mp4");
      const videoDuration = await probeDuration(videoPath);
      const fadeOutStart = Math.max(0, videoDuration - 1.5);

      const command = ffmpeg().input(videoPath).input(musicPath);
      let filterParts;

      if (narrationPath) {
        command.input(narrationPath);
        filterParts = [
          // Music: fade in/out, no manual volume cut — sidechaincompress
          // below handles ducking dynamically instead of a flat reduction.
          `[1:a]afade=t=in:st=0:d=1,afade=t=out:st=${fadeOutStart.toFixed(2)}:d=1.5[music_faded]`,
          // Narration: short fade in only — no fade-out, since it should
          // finish cleanly on its own, not trail off mid-sentence.
          `[2:a]afade=t=in:st=0:d=0.3[narration_faded]`,
          // Ducking: music_faded is the signal being reduced, narration_faded
          // is the sidechain trigger. threshold/ratio tuned for speech-over-
          // music (aggressive enough that narration is always intelligible,
          // not so aggressive that music disappears entirely underneath it).
          `[music_faded][narration_faded]sidechaincompress=threshold=0.05:ratio=8:attack=5:release=300[music_ducked]`,
          `[music_ducked][narration_faded]amix=inputs=2:duration=longest:dropout_transition=2[premix]`,
          `[premix]loudnorm=I=-16:TP=-1.5:LRA=11[audio_out]`,
        ];
      } else {
        filterParts = [
          `[1:a]afade=t=in:st=0:d=1,afade=t=out:st=${fadeOutStart.toFixed(2)}:d=1.5,volume=0.6[music_faded]`,
          `[music_faded]loudnorm=I=-16:TP=-1.5:LRA=11[audio_out]`,
        ];
      }

      command
        .complexFilter(filterParts)
        .outputOptions(["-map", "0:v", "-map", "[audio_out]", "-c:v", "copy", "-c:a", "aac", "-shortest"])
        .output(outputPath)
        .on("end", () => resolve(outputPath))
        .on("error", (err) => reject(new Error(`Audio mix failed: ${err.message}`)))
        .run();
    } catch (err) {
      reject(err);
    }
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

async function assembleVideo({ clipPaths, musicPath, narrationUrl, formats, workDir }) {
  const concatenated = await concatenateClips(clipPaths, workDir);

  // NEW (audio build): download narration audio if this job requested it.
  // Absent for any job that didn't request narration, or where narration
  // generation failed upstream (Netlify still dispatches the job either
  // way — a failed narration should never block the video itself).
  const narrationPath = narrationUrl ? await downloadNarrationAudio(narrationUrl, workDir) : null;

  const withMusic = await mixAudio(concatenated, musicPath, workDir, narrationPath);

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
