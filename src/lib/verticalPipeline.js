// verticalPipeline.js — builds the native 9:16 clip set for a job.
//
// Runs AFTER the 16:9 clips exist and mirrors them 1:1 (same order, same
// per-clip duration, same Reveal structure) so the narration/music timeline
// computed from the 16:9 clips lines up exactly on the 9:16 cut. Each room is
// re-rendered from the FULL-RESOLUTION photo with a 9:16 window anchored on the
// subject focal point — never a crop of the 16:9 master.
//
//   standard Ken Burns room → same preset, native vertical motion table
//   AI Motion (Kling) room  → nearest Ken Burns preset (no second Kling call)
//   Room Reveal             → Original opener (same window) → wipe → staged continuation
//
// Spec per frame (recorded by renderPipeline.js while building the 16:9 clips):
//   { kind: "kb" }
//   { kind: "kling", clipPath }                         (16:9 clip, used for its real duration)
//   { kind: "reveal", presetKey, openerMotion, endMotion, continuationDuration }

const path = require("path");
const fs = require("fs");
const { applyMotionPreset, resolveDuration, VALID_PRESETS } = require("./motionPresets");
const { mapKlingPresetToKenBurns, toVerticalKenBurns } = require("./verticalMotionMap");
const { findFocalPoint } = require("./focalPoint");
const { buildRevealClip, REVEAL_OPENER_DURATION, probeDuration, mapWithConcurrencyLimit } = require("./assemble");

const V_CANVAS = { w: 1080, h: 1920 };

function safeVerticalPreset(name) {
  const mapped = toVerticalKenBurns(name);
  return mapped && VALID_PRESETS.has(mapped) ? mapped : "push_in";
}

async function computeFocals(localFrames, workDir) {
  return mapWithConcurrencyLimit(localFrames, 4, (frame, i) =>
    findFocalPoint(frame.localPath, workDir, { roomLabel: frame.roomLabel, tag: String(i) })
  );
}

async function renderVerticalClips({ jobId, localFrames, specs, workDir, focals }) {
  const vdir = path.join(workDir, "v9x16");
  fs.mkdirSync(vdir, { recursive: true });

  const clipPaths = [];
  let carryZoom = 1.0;

  for (let i = 0; i < localFrames.length; i++) {
    const frame = localFrames[i];
    const spec = specs[i] || { kind: "kb" };
    const focal = focals[i] || { fx: 0.5, fy: 0.5 };
    const vBase = { ...frame, vertical: true, focal };

    if (spec.kind === "reveal") {
      const opener = await applyMotionPreset(
        { ...vBase, localPath: frame.beforeLocalPath, motionPreset: spec.openerMotion,
          durationSeconds: REVEAL_OPENER_DURATION, labelText: "Original" },
        vdir, 1.0
      );
      const cont = await applyMotionPreset(
        { ...vBase, motionPreset: safeVerticalPreset(spec.endMotion), durationSeconds: spec.continuationDuration },
        vdir, 1.0
      );
      const revealPath = await buildRevealClip(
        opener.path, cont.path, spec.presetKey, vdir,
        `reveal_v_${i}.mp4`, spec.continuationDuration, V_CANVAS
      );
      clipPaths.push(revealPath);
      carryZoom = 1.0;
      continue;
    }

    let preset, duration;
    if (spec.kind === "kling") {
      preset = mapKlingPresetToKenBurns(frame.klingMotionPreset) || "auto";
      // Kling's real rendered length (snapped, integer seconds) is what the
      // narration timeline was built from — mirror it exactly.
      duration = await probeDuration(spec.clipPath);
    } else {
      preset = frame.motionPreset && frame.motionPreset !== "auto" ? safeVerticalPreset(frame.motionPreset) : "auto";
      duration = resolveDuration(frame);
    }

    // Distinct output name per clip (vdir is shared by all vertical clips).
    const r = await applyMotionPreset(
      { ...vBase, localPath: frame.localPath, motionPreset: preset, durationSeconds: duration },
      vdir, carryZoom
    );
    carryZoom = r.endingZoom;
    clipPaths.push(r.path);
  }
  console.log(`[${jobId}] Native 9:16: rendered ${clipPaths.length} vertical clips.`);
  return clipPaths;
}

module.exports = { computeFocals, renderVerticalClips, V_CANVAS };
