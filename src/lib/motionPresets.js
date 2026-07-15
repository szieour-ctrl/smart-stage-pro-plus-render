// motionPresets.js — Applies Ken Burns-style motion to a still image.
//
// All motion is computed in motionRenderer.py via floating-point affine
// transforms at full 1920x1080 output resolution (cv2.warpAffine, INTER_CUBIC).
// Node spawns the Python process per clip and reads JSON from stdout.

const path      = require("path");
const { spawn } = require("child_process");

const RENDERER_PY = path.join(__dirname, "motionRenderer.py");

// Auto preset assignment by room type.
// bathroom changed from pan_left → tilt_down (flooring/tile reveal is the
// natural shot for bathrooms; pan_left was arbitrary).
// exterior uses pull_back (drone boom-up feel for establishing shot).
const AUTO_PRESETS = {
  exterior: "pull_back",
  living:   "push_in",
  kitchen:  "pan_right",
  dining:   "float",
  bedroom:  "push_in",
  bathroom: "tilt_down",
  flex:     "float",
  default:  "float",
};

// Default shot duration by room type
// RAISED +0.7s across the board (Sam's feedback, real render — "timing
// on the single frames is about .7s too short and the narration can't
// breathe... bedroom frames are a blip"). Flat increase, not per-type
// tuning — the complaint was general, not room-specific.
const DEFAULT_DURATIONS = {
  exterior: 6.2,
  living:   6.2,
  kitchen:  5.2,
  dining:   4.7,
  bedroom:  5.2,
  bathroom: 3.7,
  flex:     4.7,
  default:  5.2,
};

// All valid preset names — must match motionRenderer.py choices exactly
const VALID_PRESETS = new Set([
  "push_in", "pull_back",
  "pan_left", "pan_right",
  "tilt_up", "tilt_down",
  "drift", "pan_zoom",
  "float", "luxury_parallax", "static",
]);

function resolveDuration(frame) {
  if (frame.durationSeconds) return frame.durationSeconds;
  return DEFAULT_DURATIONS[frame.roomType] || DEFAULT_DURATIONS.default;
}

function resolvePreset(frame) {
  if (frame.motionPreset && frame.motionPreset !== "auto") {
    if (!VALID_PRESETS.has(frame.motionPreset)) {
      console.warn(`[motionPresets] Unknown preset "${frame.motionPreset}", falling back to auto`);
    } else {
      return frame.motionPreset;
    }
  }
  return AUTO_PRESETS[frame.roomType] || AUTO_PRESETS.default;
}

/**
 * Spawn motionRenderer.py for one frame → resolves with:
 *   { path: string, endingZoom: number, duration: number }
 */
function applyMotionPreset(frame, workDir, startZoom = 1.0) {
  return new Promise((resolve, reject) => {
    const preset     = resolvePreset(frame);
    const duration   = resolveDuration(frame);
    const outputPath = require("path").join(
      workDir,
      `clip_${require("path").basename(frame.localPath, require("path").extname(frame.localPath))}.mp4`
    );

    const args = [
      RENDERER_PY,
      "--source",     frame.localPath,
      "--preset",     preset,
      "--duration",   String(duration),
      "--output",     outputPath,
      "--start-zoom", String(startZoom),
      "--fps",        "25",
      "--output-w",   "1920",
      "--output-h",   "1080",
    ];

    let stdout = "";
    let stderr = "";

    const proc = spawn("python3", args);
    proc.stdout.on("data", (d) => { stdout += d.toString(); });
    proc.stderr.on("data", (d) => { stderr += d.toString(); });

    proc.on("close", (code) => {
      if (code !== 0) {
        return reject(
          new Error(`motionRenderer failed (${preset}, exit ${code}): ${stderr.slice(0, 500)}`)
        );
      }
      try {
        const result = JSON.parse(stdout.trim());
        resolve(result);
      } catch (e) {
        reject(new Error(`motionRenderer returned invalid JSON: ${stdout.slice(0, 200)}`));
      }
    });

    proc.on("error", (err) => {
      reject(new Error(`Failed to spawn motionRenderer.py: ${err.message}`));
    });
  });
}

module.exports = {
  applyMotionPreset,
  resolvePreset,
  resolveDuration,
  AUTO_PRESETS,
  DEFAULT_DURATIONS,
  VALID_PRESETS,
};
