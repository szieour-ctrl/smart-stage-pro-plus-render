// motionPresets.js — Applies Ken Burns-style motion to a still image.
//
// ARCHITECTURE CHANGE: zoompan replaced with motionRenderer.py.
//
// zoompan quantized crop-window positions to integer pixels at the 960x540
// working canvas, producing an alternating +1.9px / +3.75px staircase at the
// 1920x1080 output — confirmed as the "shaky" motion on real footage. Raising
// fps or lowering zoom rate does not fix it; the cause is spatial quantization,
// not temporal resolution.
//
// motionRenderer.py computes every transform as a floating-point affine matrix
// at full 1920x1080 output resolution, applied with INTER_CUBIC interpolation.
// No staircase. No zoompan. Node spawns the Python process per clip and reads
// the resulting JSON from stdout.

const path      = require("path");
const { spawn } = require("child_process");

// Path to the Python renderer — lives next to this file in the render service
const RENDERER_PY = path.join(__dirname, "motionRenderer.py");

// Auto preset assignment by room type
const AUTO_PRESETS = {
  exterior: "pull_back",
  living:   "push_in",
  kitchen:  "pan_right",
  dining:   "float",
  bedroom:  "push_in",
  bathroom: "pan_left",
  flex:     "float",
  default:  "float",
};

// Default shot duration by room type
const DEFAULT_DURATIONS = {
  exterior: 5.5,
  living:   5.5,
  kitchen:  4.5,
  dining:   4.0,
  bedroom:  4.5,
  bathroom: 3.0,
  flex:     4.0,
  default:  4.5,
};

function resolveDuration(frame) {
  if (frame.durationSeconds) return frame.durationSeconds;
  return DEFAULT_DURATIONS[frame.roomType] || DEFAULT_DURATIONS.default;
}

function resolvePreset(frame) {
  if (frame.motionPreset && frame.motionPreset !== "auto") {
    return frame.motionPreset;
  }
  return AUTO_PRESETS[frame.roomType] || AUTO_PRESETS.default;
}

/**
 * Spawn motionRenderer.py and return a promise that resolves with:
 *   { path: string, endingZoom: number, duration: number }
 *
 * The Python process prints exactly one JSON line to stdout on success,
 * which we parse here. Any stderr output is captured for error reporting.
 *
 * @param {Object} frame     - Frame descriptor (localPath, roomType, etc.)
 * @param {string} workDir   - Job temp directory for output files
 * @param {number} startZoom - Zoom level carried from the previous clip
 */
function applyMotionPreset(frame, workDir, startZoom = 1.0) {
  return new Promise((resolve, reject) => {
    const preset     = resolvePreset(frame);
    const duration   = resolveDuration(frame);
    const outputPath = path.join(
      workDir,
      `clip_${path.basename(frame.localPath, path.extname(frame.localPath))}.mp4`
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
};
