// verticalMotionMap.js — which Ken Burns preset the native 9:16 pass uses
// for a room whose 16:9 clip is an AI Motion (Kling) clip.
//
// AI Motion jobs now build BOTH formats: 16:9 gets the Kling clips, 9:16
// renders the same rooms as native Ken Burns (no second paid Kling call).
// This maps each Kling preset to the closest-feeling Ken Burns move.
// Anything unlisted returns null → caller falls back to the room-type
// auto preset.

const KLING_TO_KEN_BURNS = {
  // forward / dolly feel
  cinematic_push: "push_in",
  parallax_push: "push_in",
  rack_focus: "push_in",
  // vertical moves
  crane_up: "tilt_up",
  crane_down: "tilt_down",
  // reveals / pull-backs
  drone_boom_up: "pull_back",
  pull_back_wide: "pull_back",
  // orbit-style sweeps → push + lateral (textbook Ken Burns)
  orbit_arc: "pan_zoom",
  pan_zoom_reveal: "pan_zoom",
  // diagonal drifts
  corner_to_corner_drift: "drift",
  luxury_drift: "drift",
  floating_camera_drift: "drift",
  // lateral glide
  architectural_glide: "pan_right",
  // ambient loops → gentle breathing
  fireplace_flicker: "float",
  water_motion: "float",
  living_room_ambient: "float",
  outdoor_breeze: "float",
  // time-of-day transitions can't be reproduced in Ken Burns; a slow push keeps the energy
  day_to_twilight: "push_in",
  twilight_to_day: "push_in",
  day_to_twilight_timelapse: "push_in",
  twilight_to_day_timelapse: "push_in",
  twilight_night_day_timelapse: "push_in",
};

// Presets that exist in the JS allow-list but not in motionRenderer.py's argparse choices.
const RENDERER_ALIASES = {};

function mapKlingPresetToKenBurns(klingPreset) {
  return KLING_TO_KEN_BURNS[klingPreset] || null;
}

// Any Ken Burns name (or Kling name, in Reveal end motions) → a name the renderer accepts, or null.
function toVerticalKenBurns(name) {
  if (!name) return null;
  if (RENDERER_ALIASES[name]) return RENDERER_ALIASES[name];
  if (KLING_TO_KEN_BURNS[name]) return KLING_TO_KEN_BURNS[name];
  return name;
}

module.exports = { mapKlingPresetToKenBurns, toVerticalKenBurns, KLING_TO_KEN_BURNS };
