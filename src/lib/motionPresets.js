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
// (we render at 25fps, so duration*25 = d). Output is 1920x1080 —
// downstream assembly handles reframing to 9:16 / 1:1.
function buildZoompanFilter(preset, durationSeconds) {
  const fps = 25;
  const frames = Math.round(durationSeconds * fps);

  const filters = {
    push_in:   `zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=1920x1080:fps=${fps}`,
    pull_back: `zoompan=z='if(lte(zoom,1.0),1.5,max(1.0,zoom-0.0015))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=1920x1080:fps=${fps}`,
    pan_left:  `zoompan=z=1.2:x='if(lte(x,0),iw/2,x-2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=1920x1080:fps=${fps}`,
    pan_right: `zoompan=z=1.2:x='if(gte(x,iw),iw/2,x+2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=1920x1080:fps=${fps}`,
    float:     `zoompan=z='1.05+0.02*sin(2*PI*on/${frames})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=1920x1080:fps=${fps}`,
    static:    `scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=${fps}`,
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
      .videoFilters(filter)
      .outputOptions(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
      .duration(frame.durationSeconds)
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`FFmpeg motion failed (${preset}): ${err.message}`)))
      .run();
  });
}

module.exports = { applyMotionPreset, resolvePreset, AUTO_PRESETS };
