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

function buildZoompanFilter(preset, durationSeconds) {
  const fps = 20; // reduced from 25 — fewer frames to hold in memory, still smooth for Ken Burns motion
  const frames = Math.round(durationSeconds * fps);

  const filters = {
    push_in:   `zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},scale=${OUTPUT_W}:${OUTPUT_H}`,
    pull_back: `zoompan=z='if(lte(zoom,1.0),1.5,max(1.0,zoom-0.0015))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},scale=${OUTPUT_W}:${OUTPUT_H}`,
    pan_left:  `zoompan=z=1.2:x='if(lte(x,0),iw/2,x-2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},scale=${OUTPUT_W}:${OUTPUT_H}`,
    pan_right: `zoompan=z=1.2:x='if(gte(x,iw),iw/2,x+2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},scale=${OUTPUT_W}:${OUTPUT_H}`,
    float:     `zoompan=z='1.05+0.02*sin(2*PI*on/${frames})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${ZOOMPAN_W}x${ZOOMPAN_H}:fps=${fps},scale=${OUTPUT_W}:${OUTPUT_H}`,
    static:    `scale=${OUTPUT_W}:${OUTPUT_H}:force_original_aspect_ratio=decrease,pad=${OUTPUT_W}:${OUTPUT_H}:(ow-iw)/2:(oh-ih)/2,fps=${fps}`,
  };

  return filters[preset] || filters.float;
}

function resolvePreset(frame) {
  if (frame.motionPreset && frame.motionPreset !== "auto") {
    return frame.motionPreset;
  }
  return AUTO_PRESETS[frame.roomType] || AUTO_PRESETS.default;
}

function applyMotionPreset(frame, workDir) {
  return new Promise((resolve, reject) => {
    const preset = resolvePreset(frame);
    const filter = buildZoompanFilter(preset, frame.durationSeconds);
    const outputPath = path.join(workDir, `clip_${path.basename(frame.localPath, path.extname(frame.localPath))}.mp4`);

    ffmpeg(frame.localPath)
      .loop(frame.durationSeconds)
      // Pre-scale large source images down before zoompan ever touches them —
      // Cloudinary staged images can be quite high-res, and feeding a huge
      // source into zoompan multiplies its memory footprint unnecessarily.
      .videoFilters([`scale='min(1600,iw)':-2`, filter])
      .outputOptions([
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", "1", // cap FFmpeg's thread/memory usage per job — safer on constrained containers
      ])
      .duration(frame.durationSeconds)
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`FFmpeg motion failed (${preset}): ${err.message}`)))
      .run();
  });
}

module.exports = { applyMotionPreset, resolvePreset, AUTO_PRESETS };
