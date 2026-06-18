// motionPresets.js — Applies Ken Burns-style motion to a still image,
// producing a short video clip per frame. Fully functional — only
// requires FFmpeg, which Railway installs via nixpacks.toml.

const path = require("path");
const ffmpeg = require("fluent-ffmpeg");

// Auto preset assignment by room type — used when the agent leaves
// motionPreset as "auto" rather than picking one manually.
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

// Default shot duration by room type, used when the agent/frontend doesn't
// specify durationSeconds explicitly. Uniform timing across every shot was
// part of what made early tests feel like a slideshow — real walkthroughs
// vary pacing: a hero shot (living room) holds longer, a utility shot
// (bathroom) moves faster.
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

// FFmpeg zoompan filter strings. d = number of frames at given fps
// (we render at 25fps, so duration*25 = d).
//
// IMPORTANT: zoompan is memory-intensive — it builds the zoom/pan frame
// sequence at full resolution before output. Running it at 1920x1080
// caused SIGKILL (out of memory) on Railway's default container size.
// Fix: run zoompan at a smaller intermediate size (960x540), then scale
// up to 1920x1080 as a separate, cheap final step. This cuts zoompan's
// memory footprint to roughly a quarter of the original.
const ZOOMPAN_W = 960;
const ZOOMPAN_H = 540;
const OUTPUT_W = 1920;
const OUTPUT_H = 1080;

function buildZoompanFilter(preset, durationSeconds, startZoom = 1.0) {
  const fps = 20;
  const frames = Math.round(durationSeconds * fps);

  // Zoom rate increased from 0.0015 to 0.004 — the previous rate was too
  // subtle to read as deliberate movement, contributing to the "slideshow"
  // feel. Faster, more confident motion reads as walking through a space
  // rather than slowly looking at a photo.
  //
  // startZoom allows directional continuity between clips — see
  // renderPipeline.js, which now passes the previous clip's ending zoom
  // level so push_in/pull_back chains feel continuous rather than each
  // clip resetting to a default zoom state.
  const maxZoom = Math.min(startZoom + 0.5, 1.8);
  const minZoom = Math.max(startZoom - 0.5, 1.0);

  // IMPORTANT: the final scale=OUTPUT_W:OUTPUT_H with no aspect-ratio
  // flag was force-stretching every motion preset's output to exactly
  // 16:9 regardless of the source image's real aspect ratio — confirmed
  // visually as a "stretched" result on real test footage. Every preset
  // now pads to preserve aspect ratio, matching what `static` already did
  // correctly. zoompan itself still operates at ZOOMPAN_W x ZOOMPAN_H
  // (stretched internally, which is fine — that's just its working
  // canvas), but the FINAL output step now always preserves real aspect
  // ratio via force_original_aspect_ratio + pad, same pattern as static.
  const finalScale = `scale=${OUTPUT_W}:${OUTPUT_H}:force_original_aspect_ratio=decrease,pad=${OUTPUT_W}:${OUTPUT_H}:(ow-iw)/2:(oh-ih)/2`;

  const filters = {
    push_in:   `zoompan=z='min(zoom+0.004,${maxZoom})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,
    pull_back: `zoompan=z='if(lte(zoom,${minZoom}),${maxZoom},max(${minZoom},zoom-0.004))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,
    pan_left:  `zoompan=z=1.25:x='if(lte(x,0),iw*0.15,x-3)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,
    pan_right: `zoompan=z=1.25:x='if(gte(x,iw*0.85),iw*0.15,x+3)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,
    float:     `zoompan=z='1.1+0.03*sin(2*PI*on/${frames})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,
    // luxury_parallax — slow, deliberate diagonal slider drift at a fixed,
    // modest zoom (1.15x, well below push_in's range) so the room reads as
    // settled/finished rather than still moving inward. The diagonal path
    // (combining slight horizontal AND vertical drift) reads more like a
    // real cinematographer's slider/dolly shot than a simple left-right pan.
    // This is the preset used as the SECOND beat after a Kling vacant→staged
    // transformation — see applyContinuationMotion() in klingMotion.js.
    //
    // FIXED: the original version used an absolute pixel offset (iw*0.08)
    // for x/y, which sat outside zoompan's valid crop-window range and got
    // silently clamped to a single fixed position — confirmed visually as
    // "zero movement" on real test footage. This version starts from the
    // CENTERED position (iw/2-(iw/zoom/2), same anchor as every other
    // preset) and drifts a small, bounded percentage of the available
    // margin — the same safe pattern push_in/pan_left/pan_right already
    // use, just applied diagonally instead of along one axis.
    luxury_parallax: `zoompan=z=1.15:x='(iw/2-(iw/zoom/2))+(iw/zoom)*0.08*(on/${frames})':y='(ih/2-(ih/zoom/2))-(ih/zoom)*0.06*(on/${frames})':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,
    static:    `${finalScale},fps=${fps}`,
  };

  return filters[preset] || filters.float;
}

function resolvePreset(frame) {
  if (frame.motionPreset && frame.motionPreset !== "auto") {
    return frame.motionPreset;
  }
  return AUTO_PRESETS[frame.roomType] || AUTO_PRESETS.default;
}

function getEndingZoom(preset, startZoom) {
  // Mirrors the maxZoom/minZoom logic in buildZoompanFilter so the next
  // clip can pick up where this one left off, creating the illusion of
  // continuous forward movement between rooms rather than each shot
  // resetting to a default state.
  if (preset === "push_in") return Math.min(startZoom + 0.5, 1.8);
  if (preset === "pull_back") return Math.max(startZoom - 0.5, 1.0);
  return 1.1; // pan/float/static presets don't carry a meaningful zoom handoff
}

function applyMotionPreset(frame, workDir, startZoom = 1.0) {
  return new Promise((resolve, reject) => {
    const preset = resolvePreset(frame);
    const duration = resolveDuration(frame);
    const filter = buildZoompanFilter(preset, duration, startZoom);
    const outputPath = path.join(workDir, `clip_${path.basename(frame.localPath, path.extname(frame.localPath))}.mp4`);

    ffmpeg(frame.localPath)
      .loop(duration)
      // Pre-scale large source images down before zoompan ever touches them —
      // Cloudinary staged images can be quite high-res, and feeding a huge
      // source into zoompan multiplies its memory footprint unnecessarily.
      .videoFilters([`scale='min(1600,iw)':-2`, filter])
      .outputOptions([
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", "1", // cap FFmpeg's thread/memory usage per job — safer on constrained containers
      ])
      .duration(duration)
      .output(outputPath)
      .on("end", () => resolve({ path: outputPath, endingZoom: getEndingZoom(preset, startZoom), duration }))
      .on("error", (err) => reject(new Error(`FFmpeg motion failed (${preset}): ${err.message}`)))
      .run();
  });
}

module.exports = { applyMotionPreset, resolvePreset, resolveDuration, AUTO_PRESETS, DEFAULT_DURATIONS };
