// ltxMotion.js — AI-generated camera motion via LTX 2.3 Fast (through
// fal.ai), used for the CONTINUATION phase of the Ken Burns Reveal
// architecture (assemble.js's REVEAL_PRESETS) and as a standalone AI
// Motion option on single images.
//
// SCOPE — this is deliberately narrower than klingMotion.js:
//
//   - Only the CONTINUATION-phase category migrates to LTX here — the
//     Cinematic LTX Prompt Pack's 10 motions (source: Sam's
//     End_Frame_Generation_and_PRO_Plus_Changes doc), all of which are
//     pure camera-move-or-ambient-animation on a SINGLE, already-real
//     image — no transformation, no end_image_url, no invented room
//     content. Same compliance category as klingMotion.js's
//     SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS.
//
//   - Hero Transformation (vacant→staged morph), Time Lapse, and Exterior
//     landscape TRANSFORMATION all stay on Kling (klingMotion.js,
//     unchanged) — real-render testing this session found Kling's
//     quality still wins for genuine start→end transformations; LTX's
//     advantage is specifically in single-image continuation motion,
//     where cost is meaningfully lower ($0.06/s vs $0.084/s Kling O3 —
//     confirmed directly against the fal.ai playground's displayed rate,
//     not marketing copy; an earlier $0.04/s figure cited from a search
//     result was wrong) with
//     comparable or better quality.
//
//   - room_reveal is EXCLUDED from this build entirely (not just gated —
//     omitted from LTX_MOTION_TEMPLATES below). Sam's explicit call: "NO
//     Open Plan LTX right now." klingMotion.js's own room_reveal preset
//     stays the open-plan option for now; LTX's equivalent isn't wired up
//     until open-plan LTX continuation gets its own real-render testing.
//
// Two use cases, one shared generation function:
//   1. REVEAL_PRESETS continuation — called from renderPipeline.js's
//      reveal branch when the user's selected End Motion is an LTX
//      preset instead of a Ken Burns one. Starts from the ORIGINAL
//      staged image URL (frame.remoteImageUrl) at zoom 1.0 — same
//      starting point Ken Burns continuation already uses, per
//      buildRevealClip's existing design (the continuation phase always
//      restarts fresh from the raw staged image, regardless of what the
//      opener/wipe did). Result gets concatenated onto the Ken Burns
//      wipe clip via the shared concatTwoClips() from klingMotion.js.
//   2. Standalone AI Motion — any single image, no Room Reveal required,
//      mirrors klingMotion.js's SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS
//      usage pattern exactly.
//
// Falls back to Ken Burns on any failure — same "premium enhancement,
// not a dependency" principle as Kling.

const fs = require("fs");
const path = require("path");
const axios = require("axios");
const ffmpeg = require("fluent-ffmpeg");
const { fal } = require("@fal-ai/client");
const { concatTwoClips } = require("./klingMotion");

const OUTPUT_W = 1920;
const OUTPUT_H = 1080;

let configured = false;

function ensureConfigured() {
  if (configured) return;
  if (!process.env.FAL_KEY) {
    throw new Error("FAL_KEY not set — cannot use LTX AI motion");
  }
  fal.config({ credentials: process.env.FAL_KEY });
  configured = true;
}

// ── CINEMATIC LTX FAST PROMPT PACK (16 motions, 2 batches) ────────────
// Batch 1 (9): Sam's End_Frame_Generation_and_PRO_Plus_Changes doc.
// Batch 2 (7): Cinematic_LTX_Prompt_Pack_-Addition doc, July 18, 2026 —
// LTX-safe rewrites of the remaining Kling single-image presets, each
// tested and confirmed by Sam against real fal.ai renders before being
// added here (see the batch-2 section below for the "why this works"
// reasoning specific to each).
// Source: Sam's End_Frame_Generation_and_PRO_Plus_Changes doc, verbatim
// Motion Directive + Negative Constraint per preset, combined into one
// prompt string each — same convention klingMotion.js's
// KLING_MOTION_TEMPLATES already uses (one descriptive+negative-
// constraint string per preset, not two separate fields), so both
// providers' templates read the same way for anyone maintaining this
// later. room_reveal is the 10th preset in the source doc — intentionally
// NOT included here (see file header).
//
// Confidence and Gate are preserved as real, separate metadata (not
// baked into the prompt text) because they drive real logic below:
// confidence determines standalone-selection eligibility, gate becomes
// an enforced check where the underlying signal already exists on the
// frame object (isOpenPlan) and an unenforced advisory note where it
// doesn't (fireplace/water/reflective-surface detection isn't built —
// same "let the user pick, don't fake automatic detection" call already
// made for Kling's equivalent gates).

const LTX_MOTION_TEMPLATES = {
  cinematic_push: {
    prompt:
      "The camera performs a smooth, deliberate push-in toward the center of the room, maintaining straight verticals and stable architectural lines. Depth increases naturally as the camera advances, with gentle parallax emerging only from existing room geometry. All architecture, furniture, windows, lighting, and surfaces remain exactly as photographed — no new objects, textures, reflections, or openings appear.",
    confidence: "high",
    safeWhen: "Any interior room with clear depth.",
    gate: null,
  },
  luxury_drift: {
    prompt:
      "The camera glides laterally in a slow, elegant drift, preserving a stable perspective and straight architectural lines. Motion feels refined and controlled, like a slider shot across the room. No new blinds, reflections, textures, or objects appear; the camera stays fully within the photographed space.",
    confidence: "high",
    safeWhen: "Rooms with strong horizontal sightlines.",
    gate: null,
  },
  floating_camera_drift: {
    prompt:
      "The camera floats gently with subtle micro-sway, creating a weightless, ambient motion. Perspective remains stable, and parallax arises only from existing depth in the photographed room. Architecture, windows, lighting, and furniture remain unchanged — no new textures, blinds, reflections, or objects appear.",
    confidence: "medium-high",
    safeWhen: "Rooms with soft lighting and visible depth.",
    gate: { type: "advisory", note: "Avoid highly reflective rooms (mirrors, glossy tile) — not automatically detected, use judgment when picking the source photo." },
  },
  architectural_glide: {
    prompt:
      "The camera performs a smooth horizontal glide along the room's existing architectural sightline, tracking across visible cabinetry, windows, or built-ins with stable perspective and controlled motion. No new openings, extended rooms, or architectural changes appear; motion stays strictly within the photographed boundaries.",
    confidence: "high",
    safeWhen: "Kitchens, hallways, open-plan living spaces.",
    gate: { type: "advisory", note: "Avoid very tight rooms with no lateral space." },
  },
  corner_to_corner_drift: {
    prompt:
      "The camera drifts diagonally from one visible corner toward the opposite corner, maintaining stable geometry and allowing natural parallax from existing room depth. No new space, openings, or architectural features appear; motion remains fully within the photographed room.",
    confidence: "high",
    safeWhen: "Rooms with visible corners and depth.",
    gate: { type: "advisory", note: "Avoid rooms with obstructed corners." },
  },
  living_room_ambient: {
    prompt:
      "The camera holds a stable frame while subtle ambient motion animates only elements already visible — gentle curtain sway, soft plant movement, or natural fireplace flicker. No new motion sources, objects, lighting changes, or reflections appear; architecture and furniture remain unchanged.",
    confidence: "high",
    safeWhen: "Rooms with visible ambient elements (curtains, plants, or a fireplace).",
    gate: { type: "advisory", note: "Motion may read as too subtle in rooms with no ambient elements at all." },
  },
  fireplace_flicker: {
    prompt:
      "The camera remains locked off while the existing fireplace flame flickers naturally, producing soft, realistic light variation within the photographed scene. Only the existing flame animates; no new reflections, lighting changes, or architectural modifications appear.",
    confidence: "high",
    safeWhen: "Rooms with a visible fireplace.",
    // requiresFireplace: no automatic fireplace detection exists in this
    // pipeline — same call already made for Kling's identical gate.
    // Enforced by NOT offering this as an "auto" default; user must
    // explicitly select it, same as picking any other End Motion.
    gate: { type: "advisory", note: "Requires a visible fireplace — no automatic detection; user must confirm the photo actually shows one." },
  },
  water_motion: {
    prompt:
      "The camera holds a stable exterior frame while gentle, natural ripples animate across the existing water surface, preserving realistic reflections and lighting. Landscaping and structure remain fixed; no new plants, reflections, or water features appear.",
    confidence: "high",
    safeWhen: "Pools, ponds, water features.",
    gate: { type: "advisory", note: "Requires visible water — no automatic detection; user must confirm." },
    exteriorOnly: true,
  },
  outdoor_breeze: {
    prompt:
      "The camera remains locked off while a subtle breeze moves through visible trees or landscaping, creating natural, photorealistic motion. Only existing foliage moves; no new plants, shadows, lighting changes, or objects appear.",
    confidence: "high",
    safeWhen: "Exterior shots with visible foliage.",
    gate: { type: "advisory", note: "Requires visible foliage — hardscape-only exteriors won't have much to animate." },
    exteriorOnly: true,
  },

  // ── Second batch (July 18, 2026) — Cinematic_LTX_Prompt_Pack_-Addition
  // doc, all marked "[Tested good]" by Sam against real fal.ai renders,
  // not just written and assumed. These are ALL LTX-SAFE REWRITES of
  // Kling presets that don't translate literally — LTX cannot do a true
  // orbit, true rack focus (depth-of-field), or a true geometry-revealing
  // boom/crane move. Each rewrite deliberately simulates the FEEL of the
  // original technique using only in-frame, non-revealing motion (lateral
  // drift, micro-parallax, micro-tilt) rather than attempting the literal
  // camera move — that's why every single one of these repeats "no new
  // areas revealed beyond what is visible in the source image": that's
  // the actual mechanism keeping them hallucination-safe, not a
  // boilerplate disclaimer. These fully replace the equivalent Kling
  // presets — see klingMotion.js's SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS,
  // where all 7 have been removed as Kling options now that tested LTX
  // equivalents exist at lower cost.
  orbit_arc: {
    prompt:
      "Create a smooth cinematic lateral drift that gently arcs across the scene, maintaining the central subject in frame. Motion should feel like a soft orbit without rotating around the object. All architecture, furniture, and structural lines remain stable and unchanged. No new areas of the room or exterior are revealed beyond what is visible in the source image.",
    confidence: "medium-high",
    safeWhen: "A central subject — kitchen island, dining table, pool — with room around it in frame.",
    gate: null,
  },
  rack_focus: {
    prompt:
      "Create a gentle cinematic push-in with a subtle depth-weighted motion that suggests a soft focus transition. Foreground and background remain clear and stable, with no blur melt or distortion. Architecture, cabinetry, and furniture stay fixed and unchanged. No new areas of the room are revealed.",
    confidence: "medium-high",
    safeWhen: "A clear foreground detail (hardware, fixture, vignette) with the room visible behind it.",
    gate: null,
  },
  drone_boom_up: {
    prompt:
      "Create a smooth upward drift with a slight forward glide, giving a gentle elevated perspective while staying fully within the photographed frame. House structure, rooflines, windows, and landscaping remain stable and unchanged. Do not reveal any new exterior areas beyond what is visible in the source image.",
    confidence: "medium-high",
    safeWhen: "Exterior shots with the full home visible in frame.",
    gate: null,
    exteriorOnly: true,
  },
  crane_up: {
    prompt:
      "Create a smooth vertical upward drift with a slight upward tilt, bringing more emphasis to the upper portion already visible in the photo. Ceiling details, fixtures, and windows remain stable and unchanged. Do not reveal any new room areas beyond what is visible in the source image.",
    confidence: "medium-high",
    safeWhen: "Rooms with visible ceiling detail — chandelier, fan, high windows.",
    gate: null,
  },
  crane_down: {
    prompt:
      "Create a smooth vertical downward drift with a slight downward tilt, bringing more emphasis to the lower portion already visible in the photo. Flooring, rugs, and lower cabinetry remain stable and unchanged. Do not reveal any new room areas beyond what is visible in the source image.",
    confidence: "medium-high",
    safeWhen: "Rooms with visible flooring detail — tilework, a rug.",
    gate: null,
  },
  parallax_push: {
    prompt:
      "Create a gentle cinematic push-in with subtle layered micro-parallax, where foreground elements shift slightly faster than background elements. All architecture, furniture, and structural lines remain stable and undistorted. Motion stays within the photographed frame with no new areas revealed.",
    confidence: "medium-high",
    safeWhen: "Rooms with clear foreground/background depth separation.",
    gate: null,
  },
  pan_zoom_reveal: {
    prompt:
      "Create a smooth lateral pan combined with a gentle push-in, revealing more emphasis on the areas already visible in the photo. Architecture, furniture, and structural lines remain stable and unchanged. Motion stays fully within the photographed frame with no new room areas revealed.",
    confidence: "medium-high",
    safeWhen: "Any room with reasonable width to pan across.",
    gate: null,
  },
};

const VALID_LTX_PRESETS = new Set(Object.keys(LTX_MOTION_TEMPLATES));

// Confidence tiers eligible for STANDALONE selection (no Room Reveal
// pairing required) — Sam's call: "Medium to High confidence movements"
// only. Every preset in the pack above is medium-high or high, so in
// practice this currently allows all 9 — kept as an explicit filter
// rather than hardcoding "all of them," so a future LOW-confidence
// addition to the pack doesn't silently become standalone-selectable
// without a deliberate decision.
const STANDALONE_ELIGIBLE_CONFIDENCE = new Set(["medium-high", "high"]);

function isStandaloneEligible(presetKey) {
  const preset = LTX_MOTION_TEMPLATES[presetKey];
  return !!preset && STANDALONE_ELIGIBLE_CONFIDENCE.has(preset.confidence);
}

// ── SCOPE ENFORCEMENT ──────────────────────────────────────────────────
// Real, enforced checks only — advisory gates (fireplace/water/reflective
// surfaces) are logged, not blocked, since no automatic detection exists
// for them (same reasoning as klingMotion.js's identical gates).
function enforceLtxScopeRules(frame, presetKey) {
  const preset = LTX_MOTION_TEMPLATES[presetKey];
  if (!preset) {
    throw new Error(
      `LTX motion rejected: unknown preset "${presetKey}". Valid presets: ${[...VALID_LTX_PRESETS].join(", ")}.`
    );
  }

  // NO OPEN PLAN LTX RIGHT NOW (Sam, July 18, 2026) — a blanket
  // restriction, not per-preset. Real-render testing this session found
  // LTX reliably drops furniture/detail into ungrouped "pop-in" clusters
  // on dense, spatially-continuous open-plan layouts even at longer
  // durations — a real, unresolved quality problem, not yet a case
  // where LTX is a safe default. Revisit once that's actually fixed and
  // re-tested, not before.
  if (frame.isOpenPlan) {
    throw new Error(
      `LTX motion rejected: open-plan rooms are not yet supported for LTX continuation (frame.isOpenPlan is true). Real testing found LTX drops furniture in as ungrouped clusters on dense open-plan layouts regardless of duration — this is a known, unresolved quality issue, not a conservative default. Use a Ken Burns continuation preset instead, or Kling if AI motion is required on this room.`
    );
  }

  if (preset.exteriorOnly && frame.roomType !== "exterior") {
    throw new Error(
      `LTX motion rejected: preset "${presetKey}" is exterior-only, but frame.roomType is "${frame.roomType}".`
    );
  }

  // Advisory gates are logged, never thrown — the underlying signal
  // (fireplace/water/reflective-surface presence) has no automatic
  // detection in this pipeline. Blocking on something we can't actually
  // check would just be a confusing false rejection; logging it keeps
  // the requirement visible without pretending we verified it.
  if (preset.gate && preset.gate.type === "advisory") {
    console.log(`  [LTX] Advisory for preset "${presetKey}": ${preset.gate.note}`);
  }
}

function buildLtxPrompt(frame, presetKey) {
  if (frame.customPrompt) return frame.customPrompt;
  const preset = LTX_MOTION_TEMPLATES[presetKey];
  if (!preset) {
    throw new Error(`buildLtxPrompt: unknown preset "${presetKey}"`);
  }
  return preset.prompt;
}

// ── DURATION — LTX Fast's real constraint ─────────────────────────────
// LTX Fast's duration is a FIXED ENUM (6/8/10/12/14/16/18/20s), not a
// free-form number like Ken Burns/Kling accept — confirmed directly from
// fal.ai's own docs. The source doc's spec calls for a 4.0s continuation
// (matching Ken Burns' REVEAL_CONTINUATION_DURATION default) — that's
// below LTX's own minimum. Snapping UP to 6s (not down, there's nowhere
// down to go) rather than rejecting the request outright. This is a
// real, meaningful cost implication worth knowing: an LTX continuation
// is ALWAYS at least 6s / $0.24 minimum (audio-off equivalent), never
// the 4.0s / ~$0.16 the original spec assumed — flagged here rather than
// silently absorbed, since it changes the real per-clip cost math.
const VALID_LTX_DURATIONS = [6, 8, 10, 12, 14, 16, 18, 20];
function snapToValidLtxDuration(requestedSeconds) {
  const n = Number(requestedSeconds) || 6;
  return VALID_LTX_DURATIONS.reduce((closest, valid) =>
    Math.abs(valid - n) < Math.abs(closest - n) ? valid : closest
  );
}

// ── LTX GENERATION ──────────────────────────────────────────────────────
// Single-image continuation only — NO end_image_url. There is no
// transformation happening here (Ken Burns' wipe already did that); this
// is pure camera/ambient motion on an already-real, already-staged image,
// same category as Kling's SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS.
async function generateLtxContinuationClip(frame, presetKey, workDir) {
  ensureConfigured();
  enforceLtxScopeRules(frame, presetKey);

  const prompt = buildLtxPrompt(frame, presetKey);
  const requestedDuration = frame.continuationDurationSeconds || 4.0;
  const duration = snapToValidLtxDuration(requestedDuration);

  if (duration !== requestedDuration) {
    console.log(
      `  [LTX] Requested continuation duration ${requestedDuration}s snapped to ${duration}s ` +
        `(LTX Fast only accepts ${VALID_LTX_DURATIONS.join("/")}).`
    );
  }

  console.log(`  [LTX] Submitting job — preset: ${presetKey}, duration: ${duration}s`);

  const LTX_ENDPOINT = "fal-ai/ltx-2.3/image-to-video/fast";

  // Explicit queue submit/status/result polling, NOT fal.subscribe() —
  // same reasoning as klingMotion.js's identical choice: fal.subscribe()'s
  // internal polling has gone silent on a real render before with no
  // JS-catchable error at all. Every fal.ai call in this codebase since
  // that discovery uses this same explicit, loggable pattern.
  const { request_id } = await fal.queue.submit(LTX_ENDPOINT, {
    input: {
      image_url: frame.remoteImageUrl,
      prompt,
      duration: String(duration),
      fps: 25, // matches motionRenderer.py's own fps, so LTX and Ken Burns clips are never accidentally frame-rate-mismatched at the concat/comparison level
    },
  });

  console.log(`  [LTX] Queued — request_id: ${request_id} (recoverable via fal.ai dashboard even if this process dies)`);

  const POLL_INTERVAL_MS = 10000;
  const MAX_POLL_ATTEMPTS = 90; // 15 minute ceiling, matches klingMotion.js's

  let finalStatus = null;
  for (let attempt = 1; attempt <= MAX_POLL_ATTEMPTS; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));

    const status = await fal.queue.status(LTX_ENDPOINT, { requestId: request_id, logs: true });

    console.log(`  [LTX] Poll ${attempt}/${MAX_POLL_ATTEMPTS} — status: ${status.status}`);
    if (status.status === "IN_PROGRESS" && status.logs) {
      status.logs.forEach((log) => console.log(`  [LTX] ${log.message}`));
    }

    if (status.status === "COMPLETED") {
      finalStatus = status;
      break;
    }
  }

  if (!finalStatus) {
    throw new Error(
      `LTX request ${request_id} did not complete within ${(MAX_POLL_ATTEMPTS * POLL_INTERVAL_MS) / 60000} minutes of polling. Check this request_id on the fal.ai dashboard directly.`
    );
  }

  const result = await fal.queue.result(LTX_ENDPOINT, { requestId: request_id });

  const videoUrl = result.data?.video?.url;
  if (!videoUrl) {
    throw new Error("LTX returned no video URL");
  }

  console.log(`  [LTX] Generation complete — downloading clip from fal.ai`);

  const outputPath = path.join(workDir, `ltx_${Date.now()}.mp4`);
  const response = await axios.get(videoUrl, { responseType: "arraybuffer" });
  fs.writeFileSync(outputPath, response.data);

  console.log(`  [LTX] Clip downloaded to ${outputPath} (${response.data.length} bytes)`);

  return { path: outputPath, duration };
}

// ── REVEAL_PRESETS ENTRY POINT ────────────────────────────────────────
// Called from renderPipeline.js's reveal branch in place of
// applyMotionPreset() when the user's End Motion selection is an LTX
// preset. Returns the same {path, endingZoom} shape applyMotionPreset()
// does, so the caller doesn't need separate handling downstream.
async function generateLtxRevealContinuation(frame, presetKey, workDir) {
  const result = await generateLtxContinuationClip(frame, presetKey, workDir);
  // endingZoom is meaningless for an LTX-generated clip (no affine zoom
  // state to hand off) — 1.0 is a safe, neutral value. Nothing downstream
  // currently uses this for anything other than seeding the NEXT Ken
  // Burns clip's startZoom, and reveal beats already reset carryZoom to
  // 1.0 after themselves regardless (see renderPipeline.js).
  return { path: result.path, endingZoom: 1.0, ltxDuration: result.duration };
}

// ── STANDALONE ENTRY POINT WITH FALLBACK ──────────────────────────────
// Mirrors klingMotion.js's applyKlingMotion() exactly — same fallback
// contract, same "premium enhancement, not a dependency" principle.
async function applyLtxMotion(frame, presetKey, workDir, fallbackFn) {
  try {
    const result = await generateLtxContinuationClip(frame, presetKey, workDir);
    console.log(`  [LTX] Clip ready: ${result.path}`);
    return { path: result.path, source: "ltx", endingZoom: 1.0 };
  } catch (err) {
    console.error(`  [LTX] Generation failed, falling back to Ken Burns: ${err.message}`);
    const fallbackResult = await fallbackFn();
    return { ...fallbackResult, source: "ken_burns_fallback" };
  }
}

module.exports = {
  applyLtxMotion,
  generateLtxContinuationClip,
  generateLtxRevealContinuation,
  enforceLtxScopeRules,
  buildLtxPrompt,
  snapToValidLtxDuration,
  isStandaloneEligible,
  LTX_MOTION_TEMPLATES,
  VALID_LTX_PRESETS,
  VALID_LTX_DURATIONS,
};
