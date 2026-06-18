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

// FFmpeg zoompan filter strings. d = number of frames at given fps.
//
// IMPORTANT — zoompan memory: zoompan is memory-intensive at full resolution.
// Fix: run zoompan at 960x540, then scale up to 1920x1080 as a separate step.
// Cuts zoompan memory footprint to ~1/4 of running at output resolution.
const ZOOMPAN_W = 960;
const ZOOMPAN_H = 540;
const OUTPUT_W = 1920;
const OUTPUT_H = 1080;

// ─── WHY `on` INSTEAD OF `zoom` STATE ──────────────────────────────────────
//
// Previous versions used FFmpeg's internal `zoom` state variable in the z=
// expression (e.g. z='min(zoom+0.004, maxZoom)'). This causes TWO bugs:
//
// Bug 1 — JUMPINESS AT EVERY CLIP TRANSITION:
//   zoompan always initializes its internal `zoom` to 1.0 at the start of each
//   clip, regardless of startZoom. So even though renderPipeline.js correctly
//   tracks carryZoom (e.g. 1.5) and passes it as startZoom, every clip snaps
//   back to 1.0 on frame 0. The viewer sees: last frame of clip A at 1.5x →
//   first frame of clip B snapping back to 1.0x. That visible jump was the
//   "jumpy movement" reported in real footage.
//
// Bug 2 — pull_back's if() reset fired immediately:
//   if(lte(zoom, minZoom), maxZoom, ...) evaluates on frame 0 when zoom=1.0,
//   which equals minZoom when startZoom=1.0 — so it immediately snapped to
//   maxZoom=1.5, introducing an additional first-frame jump on top of Bug 1.
//
// FIX: Replace zoom-state expressions with `on` (output frame counter, 0-indexed).
//   `on` is deterministic — it counts up from 0 each clip — which means
//   startZoom can be baked in as a JS constant in the filter string, rather
//   than relying on zoompan to remember it frame-to-frame. No more reset.
// ───────────────────────────────────────────────────────────────────────────

function buildZoompanFilter(preset, durationSeconds, startZoom = 1.0) {
  const fps = 20;
  const frames = Math.round(durationSeconds * fps);

  const maxZoom = Math.min(startZoom + 0.5, 1.8);
  const minZoom = Math.max(startZoom - 0.5, 1.0);

  // finalScale: zoompan outputs exactly ZOOMPAN_W x ZOOMPAN_H (always 16:9),
  // so the scale step is a clean 2x upscale with no distortion. The
  // force_original_aspect_ratio + pad is a safety net for any edge case where
  // the zoompan output isn't exactly 16:9 (shouldn't happen, but protects
  // against a silent stretch if something upstream changes).
  const finalScale = `scale=${OUTPUT_W}:${OUTPUT_H}:force_original_aspect_ratio=decrease,pad=${OUTPUT_W}:${OUTPUT_H}:(ow-iw)/2:(oh-ih)/2`;

  // Centering anchor — positions the crop window at the center of the input
  // for any zoom level. All presets use this as their x/y baseline.
  // x = iw/2 - (viewport_width/2)  where viewport_width = iw/zoom
  //   = iw/2 - iw/(zoom*2)
  // This is safe (always in bounds) as long as zoom >= 1.0.
  const cx = `iw/2-(iw/zoom/2)`;
  const cy = `ih/2-(ih/zoom/2)`;

  const filters = {
    // push_in: zoom from startZoom up to maxZoom linearly.
    // Uses `on` so frame 0 starts AT startZoom, not at zoompan's default 1.0.
    push_in: `zoompan=z='min(${startZoom}+${(maxZoom - startZoom) / frames}*on,${maxZoom})':x='${cx}':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,

    // pull_back: zoom from maxZoom down to minZoom linearly.
    // Replaces the if() reset expression — that expression jumped to maxZoom
    // on frame 0 (because initial zoom=1.0 triggered the lte condition),
    // creating a double-jump bug on top of the zoom-state reset issue.
    pull_back: `zoompan=z='max(${minZoom},${maxZoom}-${(maxZoom - minZoom) / frames}*on)':x='${cx}':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,

    // pan_left: fixed zoom, pan from right to left across the frame.
    // x starts at iw*0.25 (right-biased start) and decreases by 2px/frame.
    // Safe range: x stays between 0 and iw*0.25 well within the 1.25x viewport.
    pan_left:  `zoompan=z=1.25:x='max(0,iw*0.25-2*on)':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,

    // pan_right: fixed zoom, pan from left to right across the frame.
    // x starts at 0 and increases by 2px/frame, capped at iw*0.25 (safe max).
    pan_right: `zoompan=z=1.25:x='min(iw*0.25,2*on)':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,

    // float: gentle sine-wave breathing — zoom oscillates between 1.07–1.13.
    // Already used `on` correctly in the old version, kept as-is.
    float: `zoompan=z='1.1+0.03*sin(2*PI*on/${frames})':x='${cx}':y='${cy}':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,

    // luxury_parallax: slow diagonal slider drift at a fixed modest zoom.
    // Used as the Ken Burns continuation beat after a Kling clip.
    //
    // PREVIOUS BUG — out-of-bounds drift:
    //   With iw=1600, zoom=1.15: viewport width = 1600/1.15 = 1391px.
    //   Center x = 1600/2 - 1391/2 = 104px. Drift = 1391 * 0.08 = 111px.
    //   Final x = 104 + 111 = 215px. Right edge = 215 + 1391 = 1606 > 1600.
    //   FFmpeg silently clamped, producing zero movement for the last ~20% of
    //   the clip. Fixed by reducing drift multipliers so the viewport stays
    //   safely inside the image at any reasonable input size.
    //
    // NEW VERSION: drift is expressed as a fraction of the AVAILABLE MARGIN
    //   (the space between center and the safe edge), so it's self-bounding
    //   regardless of input resolution. Max drift = 35% of available margin.
    //   Available horizontal margin = iw/2 - (iw/zoom/2) = center x value.
    //   So max drift = center_x * 0.35. Always safe.
    luxury_parallax: `zoompan=z=1.15:x='(iw/2-(iw/zoom/2))*(1+0.35*(on/${frames}))':y='(ih/2-(ih/zoom/2))*(1-0.25*(on/${frames}))':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},${finalScale}`,

    static: `${finalScale},fps=${fps}`,
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
  // Mirrors the zoom math in buildZoompanFilter so renderPipeline.js can
  // track carryZoom accurately and pass it to the next clip as startZoom.
  if (preset === "push_in")   return Math.min(startZoom + 0.5, 1.8);
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
      // Pre-scale large source images before zoompan — high-res Cloudinary
      // images multiply zoompan's memory footprint. Scale to max 1600px wide
      // first, then zoompan operates on the smaller canvas.
      .videoFilters([`scale=min(1600\\,iw):-2`, filter])
      .outputOptions([
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", "1",
      ])
      .duration(duration)
      .output(outputPath)
      .on("end", () => resolve({ path: outputPath, endingZoom: getEndingZoom(preset, startZoom), duration }))
      .on("error", (err) => reject(new Error(`FFmpeg motion failed (${preset}): ${err.message}`)))
      .run();
  });
}

module.exports = { applyMotionPreset, resolvePreset, resolveDuration, buildZoompanFilter, AUTO_PRESETS, DEFAULT_DURATIONS };
