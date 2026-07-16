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
// RAISED again (Sam's feedback, real render — "Bedroom durations are not
// nearly long enough for narration"): the earlier +0.7s flat bump was
// real and did apply correctly (confirmed against actual rendered clip
// durations), but it didn't fully account for a second-order effect of
// the room-disambiguation fix earlier this session (narrationGen.js's
// REUSABLE_ROOM_TYPES) — correctly-separated bedrooms/bathrooms/etc. now
// mostly end up as SINGLE-clip narration segments instead of merged
// multi-clip groups. A single clip's actual usable narration window is
// its duration minus crossfade overlap on BOTH sides (~1.2s total,
// CROSSFADE_DURATION twice), not the raw clip length — so these room
// types specifically needed more headroom than the flat bump gave them.
// Kitchen/dining/exterior/living aren't in REUSABLE_ROOM_TYPES (rarely
// duplicated per listing), so they're unaffected and left as-is.
const DEFAULT_DURATIONS = {
  exterior: 6.2,
  living:   6.2,
  kitchen:  5.2,
  dining:   4.7,
  bedroom:  6.8,
  bathroom: 5.2,
  flex:     6.2,
  default:  6.5,
};

// All valid preset names — must match motionRenderer.py choices exactly
const VALID_PRESETS = new Set([
  "push_in", "pull_back",
  "pan_left", "pan_right",
  "tilt_up", "tilt_down",
  "drift", "pan_zoom",
  "float", "luxury_parallax", "static",
]);

// FIX (Sam's catch, real render — confirmed via the room-label
// screenshot): most staged photos' real room_type doesn't exactly match
// PRO Plus's fixed ROOM_TYPES list (e.g. Smart Stage PRO's own naming
// might be "Bedroom 2" or something fully custom), so the frontend's
// matchToRoomType() correctly falls back to "Other" for display — but
// that meant resolveDuration() only ever saw roomType: "Other", which
// isn't a DEFAULT_DURATIONS key, silently landing everything on the
// generic default instead of the room-specific tuning (bedroom's 6.8s,
// etc.). frame.roomLabel already carries the REAL text through to
// Railway (confirmed in downloadFrames.js) — scanning it for keywords
// recovers the right duration category even when the exact type didn't
// match, without touching the display label at all.
function inferDurationCategoryFromLabel(label) {
  const l = (label || "").toLowerCase();
  if (l.includes("bed")) return "bedroom";
  if (l.includes("bath")) return "bathroom";
  if (l.includes("kitchen")) return "kitchen";
  if (l.includes("dining")) return "dining";
  if (l.includes("living") || l.includes("family")) return "living";
  if (l.includes("exterior") || l.includes("yard") || l.includes("pool") || l.includes("front")) return "exterior";
  if (l.includes("flex") || l.includes("office") || l.includes("loft") || l.includes("laundry") || l.includes("closet")) return "flex";
  return null;
}

function resolveDuration(frame) {
  if (frame.durationSeconds) return frame.durationSeconds;
  if (DEFAULT_DURATIONS[frame.roomType]) return DEFAULT_DURATIONS[frame.roomType];
  const inferred = inferDurationCategoryFromLabel(frame.roomLabel);
  if (inferred) return DEFAULT_DURATIONS[inferred];
  return DEFAULT_DURATIONS.default;
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
