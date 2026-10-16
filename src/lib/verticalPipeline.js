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

// Pipelined renderer: enqueue(i, spec) is called by renderPipeline.js the moment the
// 16:9 clip for frame i exists, so vertical clips render CONCURRENTLY with the rest of the
// 16:9 loop (and with narration/music) instead of after everything else. Vertical clips
// are still rendered strictly in order among themselves (zoom continuity carries across
// standard clips). finish() resolves with the ordered clip paths.
function createVerticalRenderer({ jobId, localFrames, workDir, focalPromise }) {
  const vdir = path.join(workDir, "v9x16");
  fs.mkdirSync(vdir, { recursive: true });

  const clipPaths = [];
  let carryZoom = 1.0;
  let chain = Promise.resolve();
  let aborted = false;

  async function renderOne(i, spec) {
    if (aborted) return;
    const focals = await focalPromise;
    const frame = localFrames[i];
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
      clipPaths[i] = await buildRevealClip(
        opener.path, cont.path, spec.presetKey, vdir,
        `reveal_v_${i}.mp4`, spec.continuationDuration, V_CANVAS
      );
      carryZoom = 1.0;
      return;
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
    const r = await applyMotionPreset(
      { ...vBase, localPath: frame.localPath, motionPreset: preset, durationSeconds: duration },
      vdir, carryZoom
    );
    carryZoom = r.endingZoom;
    clipPaths[i] = r.path;
  }

  return {
    enqueue(i, spec) {
      chain = chain.then(() => renderOne(i, spec || { kind: "kb" }));
      chain.catch(() => {}); // surfaced by finish(); avoids an unhandled-rejection warning meanwhile
    },
    abort() { aborted = true; },
    async finish() {
      await chain;
      console.log(`[${jobId}] Native 9:16: rendered ${clipPaths.length} vertical clips.`);
      return clipPaths;
    },
  };
}

module.exports = { computeFocals, createVerticalRenderer, V_CANVAS };
