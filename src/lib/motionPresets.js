// motionPresets.js — Applies Ken Burns-style motion to a still image,
// producing a short video clip per frame.

const path = require("path");
const ffmpeg = require("fluent-ffmpeg");

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

// ─── WORKING CANVAS ────────────────────────────────────────────────────────
// zoompan runs at 960x540 to stay within Railway's memory limits, then the
// result is scaled up to 1920x1080 as a cheap final step.
const ZOOMPAN_W = 960;
const ZOOMPAN_H = 540;
const OUTPUT_W  = 1920;
const OUTPUT_H  = 1080;

// ─── PRE-PROCESS: WHY WE CENTER-CROP TO 16:9 BEFORE ZOOMPAN ───────────────
//
// zoompan's internal viewport is iw/zoom × ih/zoom (a fraction of the SOURCE
// frame). When zoom=1.0, that's the ENTIRE source — which gets stretched to
// fill the 960×540 (16:9) output canvas. If the source is 3:2 (1600×1066),
// zoompan horizontally stretches it by 1.78/1.5 = 18.5% to fill 16:9. This
// was the "stretched/distorted" result confirmed on real footage.
//
// Fix: center-crop the source to 16:9 BEFORE zoompan. Then at any zoom level
// the viewport is always a 16:9 crop from a 16:9 source → no stretch possible.
//
// Filter chain:
//   1. scale='min(1600,iw)':-2        downscale if needed, keep aspect ratio
//   2. crop to 16:9, centered         remove top/bottom (landscape) or
//                                     left/right (portrait) excess pixels
//
// The crop formula min(ih, iw*9/16) safely handles both landscape and portrait:
//   - 1600×1066 (3:2 landscape): crops to 1600×900  (removes 83px top/bottom)
//   - 1024×768  (4:3 landscape): crops to 1024×576  (removes 96px top/bottom)
//   - 1024×1024 (square):        crops to 1024×576  (removes 224px top/bottom)
//   - 768×1024  (portrait):      crops to 768×432   (removes 296px top/bottom)
//
// Note: single quotes inside the crop expression protect commas from being
// misread as FFmpeg filter-chain separators (FFmpeg expression quoting).
// ───────────────────────────────────────────────────────────────────────────
const PRE_PROCESS = [
  `scale='min(1600,iw)':-2`,
  `crop=iw:'min(ih,iw*9/16)':0:'(ih-min(ih,iw*9/16))/2'`,
];

// ─── FPS: 20 → 25 ──────────────────────────────────────────────────────────
// At 20fps, the per-frame crop-window jump is large enough to read as stutter
// rather than smooth motion — confirmed as "shakey" on real footage. 25fps
// reduces each inter-frame jump by 20%. Still within Railway's memory budget
// at the 960×540 zoompan working size.
// ───────────────────────────────────────────────────────────────────────────
const FPS = 25;

// ─── FINAL SCALE ────────────────────────────────────────────────────────────
// zoompan always outputs exactly ZOOMPAN_W×ZOOMPAN_H (960×540, 16:9), so this
// step is always a clean 2× upscale with no distortion. The pad is a safety
// net in case of any upstream rounding that produces a non-exactly-16:9 frame.
const FINAL_SCALE = `scale=${OUTPUT_W}:${OUTPUT_H}:force_original_aspect_ratio=decrease,pad=${OUTPUT_W}:${OUTPUT_H}:(ow-iw)/2:(oh-ih)/2`;

// ─── WHY `on` INSTEAD OF THE `zoom` STATE VARIABLE ─────────────────────────
//
// Previous versions used FFmpeg's internal `zoom` state variable in the z=
// expression (e.g. z='min(zoom+0.004, maxZoom)'). This caused the jumpiness
// at every clip transition:
//
//   zoompan resets its internal `zoom` to 1.0 at the start of EVERY clip,
//   regardless of what startZoom was passed in. So even though renderPipeline
//   correctly tracked carryZoom (e.g. 1.5) and passed it as startZoom, every
//   push_in clip snapped back to 1.0 on frame 0. The viewer saw: last frame
//   of clip A at 1.5× zoom → first frame of clip B snapping to 1.0×. That
//   visible snap was the "jumpy movement" on real footage.
//
//   pull_back had a second jump: if(lte(zoom, minZoom), maxZoom, ...) fired
//   on frame 0 when zoom=1.0 equalled minZoom, adding a second snap on top.
//
// Fix: express zoom as a function of `on` (output frame counter, 0-indexed).
//   `on` counts up from 0 each clip. startZoom is baked in as a JS constant
//   inside the filter string, so the clip always begins at the right value
//   regardless of zoompan's internal initialization.
// ───────────────────────────────────────────────────────────────────────────

function buildZoompanFilter(preset, durationSeconds, startZoom = 1.0) {
  const frames  = Math.round(durationSeconds * FPS);
  const maxZoom = Math.min(startZoom + 0.5, 1.8);
  const minZoom = Math.max(startZoom - 0.5, 1.0);

  // Per-frame zoom delta for push_in and pull_back.
  const pushRate = ((maxZoom - startZoom) / frames).toFixed(6);
  const pullRate = ((maxZoom - minZoom)   / frames).toFixed(6);

  // Centering anchor — positions the crop window at the source center at any
  // zoom level. Safe at all values zoom >= 1.0.
  //   x = iw/2 - (iw/zoom)/2   →  same as  iw/2 - iw/(zoom*2)
  //   y = ih/2 - (ih/zoom)/2
  const cx = `iw/2-(iw/zoom/2)`;
  const cy = `ih/2-(ih/zoom/2)`;

  // Available pan margin = total source size minus the viewport size.
  // With a 16:9 source (guaranteed by PRE_PROCESS) and 16:9 zoompan canvas,
  // the viewport is iw/zoom × ih/zoom. Available horizontal margin = iw*(1-1/zoom).
  const xMargin = `iw*(1-1/zoom)`;
  const yMargin = `ih*(1-1/zoom)`;

  const filters = {
    // push_in: zoom from startZoom → maxZoom, centered. `on` starts at 0.
    push_in: `zoompan=z='min(${startZoom}+${pushRate}*on,${maxZoom})':x='${cx}':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${FPS},${FINAL_SCALE}`,

    // pull_back: zoom from maxZoom → minZoom, centered. No if() reset — pure linear.
    pull_back: `zoompan=z='max(${minZoom},${maxZoom}-${pullRate}*on)':x='${cx}':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${FPS},${FINAL_SCALE}`,

    // pan_left: fixed 1.25× zoom, pan right→left across the available margin.
    // Start at right margin, move 2px/frame leftward, floor at 0.
    pan_left:  `zoompan=z=1.25:x='max(0,${xMargin}-2*on)':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${FPS},${FINAL_SCALE}`,

    // pan_right: fixed 1.25× zoom, pan left→right across the available margin.
    // Start at 0, move 2px/frame rightward, ceiling at available margin.
    pan_right: `zoompan=z=1.25:x='min(${xMargin},2*on)':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${FPS},${FINAL_SCALE}`,

    // float: gentle sine-wave breathing between 1.07×–1.13×. Already `on`-based.
    float: `zoompan=z='1.1+0.03*sin(2*PI*on/${frames})':x='${cx}':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${FPS},${FINAL_SCALE}`,

    // luxury_parallax: slow diagonal slider drift at a fixed modest zoom.
    // Used as the Ken Burns continuation beat that follows a Kling clip.
    //
    // PREVIOUS BUG — x drift exceeded image bounds:
    //   With iw=1600, zoom=1.15: center_x=104px, drift=(1600/1.15)*0.08=111px.
    //   Final right edge = 104+111+1391 = 1606 > 1600 → FFmpeg clamped silently
    //   → zero movement for the last ~20% of the clip.
    //
    // FIX — express drift as a fraction of the AVAILABLE MARGIN:
    //   Available x margin = iw*(1-1/zoom). Drift = margin * 0.35 (35% of
    //   the safe travel range). This is mathematically bounded at any iw.
    //   Same principle for y (25% of available y margin, drifting upward).
    luxury_parallax: `zoompan=z=1.15:x='${cx}+${xMargin}*0.35*(on/${frames})':y='${cy}-${yMargin}*0.25*(on/${frames})':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${FPS},${FINAL_SCALE}`,

    static: `${FINAL_SCALE},fps=${FPS}`,
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
  if (preset === "push_in")   return Math.min(startZoom + 0.5, 1.8);
  if (preset === "pull_back") return Math.max(startZoom - 0.5, 1.0);
  return 1.1;
}

function applyMotionPreset(frame, workDir, startZoom = 1.0) {
  return new Promise((resolve, reject) => {
    const preset     = resolvePreset(frame);
    const duration   = resolveDuration(frame);
    const filter     = buildZoompanFilter(preset, duration, startZoom);
    const outputPath = path.join(workDir, `clip_${path.basename(frame.localPath, path.extname(frame.localPath))}.mp4`);

    ffmpeg(frame.localPath)
      // -loop 1 + -framerate FPS: loop the still image indefinitely at exactly
      // FPS frames/second so zoompan always receives the right input frame rate.
      // Replaces .loop(duration) which set -loop N (a count, not a rate) and
      // left the input framerate unspecified — a mismatch that contributed to
      // stuttering when zoompan's internal fps differed from the looped input.
      .inputOptions(["-loop", "1", "-framerate", `${FPS}`])
      .videoFilters([...PRE_PROCESS, filter])
      .outputOptions([
        "-pix_fmt",    "yuv420p",
        "-movflags",   "+faststart",
        "-threads",    "1",
      ])
      .duration(duration)
      .output(outputPath)
      .on("end",   ()    => resolve({ path: outputPath, endingZoom: getEndingZoom(preset, startZoom), duration }))
      .on("error", (err) => reject(new Error(`FFmpeg motion failed (${preset}): ${err.message}`)))
      .run();
  });
}

module.exports = { applyMotionPreset, resolvePreset, resolveDuration, buildZoompanFilter, AUTO_PRESETS, DEFAULT_DURATIONS };
