// ─── PATCH FOR klingMotion.js ──────────────────────────────────────────────
//
// Add extractLastFrame() anywhere before applyKlingMotion().
// Then update the three places marked CHANGE inside applyContinuationMotion().
//
// WHY: Previously, continuation motion ran Ken Burns on the original staged
// image (localPath). That image is the STARTING state of Kling's clip, not
// the ending state — so the Ken Burns continuation begins at a different zoom,
// crop, and composition than where Kling left off. The viewer sees an obvious
// size/position jump at the stitch point.
//
// FIX: Extract the actual last frame of Kling's output video as a PNG,
// then feed THAT image into Ken Burns. The first frame of the Ken Burns clip
// is now pixel-identical to the last frame of the Kling clip — seamless cut.
// ───────────────────────────────────────────────────────────────────────────

const path = require("path");
const ffmpeg = require("fluent-ffmpeg");

// ── ADD THIS FUNCTION ────────────────────────────────────────────────────────

/**
 * Extracts the last frame of a video clip as a PNG.
 * Used to seed the Ken Burns continuation clip from Kling's exact end state,
 * so the first frame of Ken Burns is pixel-identical to Kling's last frame.
 *
 * @param {string} videoPath  - Path to the Kling output video on disk.
 * @param {string} workDir    - Temp directory for this job.
 * @returns {Promise<string>} - Path to the extracted PNG.
 */
async function extractLastFrame(videoPath, workDir) {
  const outputPath = path.join(workDir, `lastframe_${path.basename(videoPath, ".mp4")}.png`);
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .outputOptions([
        "-sseof", "-0.1",  // seek to 0.1s before end of file
        "-vframes", "1",   // grab exactly one frame
        "-q:v", "2",       // near-lossless quality (PNG ignores this, but safe)
      ])
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`extractLastFrame failed: ${err.message}`)));
  });
}

// ── CHANGE INSIDE applyContinuationMotion() ──────────────────────────────────
//
// Current signature:
//   async function applyContinuationMotion(klingSrc, localPath, preset, durationSeconds, workDir)
//
// Change to:
//   async function applyContinuationMotion(klingSrc, preset, durationSeconds, workDir)
//   (localPath removed — we derive the source from klingSrc directly)
//
// Current body (find this block):
//   const continuationResult = await applyMotionPreset(
//     { localPath, motionPreset: preset, roomType: null },
//     workDir,
//     1.0
//   );
//
// Replace with:
async function applyContinuationMotion(klingSrc, preset, durationSeconds, workDir) {
  // Extract the last frame of Kling's clip — this is the pixel-exact state
  // Kling ended on. Ken Burns will start from this image, making the cut
  // between Kling and Ken Burns invisible to the viewer.
  const lastFramePath = await extractLastFrame(klingSrc, workDir);

  // Run Ken Burns on the extracted last frame.
  // startZoom = 1.0 is correct here: the last frame already represents Kling's
  // full zoomed/panned composition — Ken Burns doesn't need to compensate for
  // any prior zoom state, it just continues naturally from that still image.
  const { applyMotionPreset } = require("./motionPresets");
  const continuationResult = await applyMotionPreset(
    { localPath: lastFramePath, motionPreset: preset, durationSeconds, roomType: null },
    workDir,
    1.0
  );

  return continuationResult;
}

// ── CHANGE THE CALL SITE INSIDE applyKlingMotion() ──────────────────────────
//
// Find the existing call to applyContinuationMotion and remove the localPath arg:
//
// BEFORE:
//   const continuation = await applyContinuationMotion(
//     klingSrc, params.localPath, params.continuationPreset,
//     params.continuationDurationSeconds, workDir
//   );
//
// AFTER:
//   const continuation = await applyContinuationMotion(
//     klingSrc, params.continuationPreset,
//     params.continuationDurationSeconds, workDir
//   );
//
// Also update renderPipeline.js — the comment on line 61-64 says:
//   "localPath is also passed through — it's the staged image already
//    downloaded locally, used by the optional continuation-motion step,
//    so we never need to extract a frame from Kling's own video output."
//
// That comment is now wrong. localPath is still passed (Kling needs it for
// the fallback Ken Burns path), but the continuation step NO LONGER uses it.
// Update the comment to:
//   "localPath is passed as the Ken Burns fallback source if Kling fails.
//    The continuation-motion step no longer uses localPath — it extracts
//    the actual last frame from Kling's output video instead."

module.exports = { extractLastFrame };
