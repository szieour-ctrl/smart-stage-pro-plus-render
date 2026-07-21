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
//   - room_reveal was EXCLUDED entirely earlier this session — Sam's
//     original call: "NO Open Plan LTX right now," pending real testing.
//     REVERSED (July 18, 2026, Cinematic_LTX_-Kling_reference doc): a
//     specific, curated set of presets — including room_reveal — is now
//     confirmed safe for open-plan on LTX. The blanket open-plan block
//     is gone; see OPEN_PLAN_SAFE_LTX_PRESETS below for exactly which
//     ones qualify. Everything NOT in that set still blocks on open-plan,
//     including presets that are already fine for single/enclosed rooms
//     (orbit_arc, rack_focus, crane_up/down, parallax_push,
//     pan_zoom_reveal) — the reference doc doesn't clear those for
//     open-plan specifically, so they stay single-room-only until they
//     get their own real open-plan test.
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

// ── TWO-IMAGE CROP WORKFLOW (July 20, 2026, LTX_Prompt_revision doc) ────
// Only 3 presets need this: orbit_arc, micro_dolly_back, open_plan_reveal
// (see requiresTwoImage flag on each below). Per the doc's Section 1: LTX
// has no reference for what's outside the original frame on any move that
// expands the frame boundary (rotation, pull-back, or — per Sam's July 20
// correction — the open-plan reveal's zone-redistribution too). Two-image
// mechanism: crop the SAME source photo down slightly (94% width/height,
// centered) to use as the tight Start Frame; the untouched original photo
// (necessarily wider by comparison, since it's the same image before that
// crop) is the End Frame. Both frames are guaranteed to share identical
// lighting/color/style because they're literally the same photograph —
// satisfies the doc's "both images share consistent lighting, color
// grade, and style" checklist item with no separate generation step.
// Standard Cloudinary URL transformation insertion — frame.remoteImageUrl
// is confirmed to be a Cloudinary delivery URL (renderPipeline.js already
// depends on this for Kling). Inserting the transformation segment right
// after "/upload/" is Cloudinary's documented URL-transform convention.
const TWO_IMAGE_CROP_TRANSFORM = "c_crop,g_center,w_0.94,h_0.94";

function buildCroppedStartUrl(remoteImageUrl) {
  if (!remoteImageUrl.includes("/upload/")) {
    // Not a recognizable Cloudinary delivery URL — can't safely insert a
    // transformation segment. Caller falls back to using the same URL for
    // both start and end rather than throwing, since a failed crop still
    // produces a renderable (if less ideal) two-image call.
    return remoteImageUrl;
  }
  return remoteImageUrl.replace("/upload/", `/upload/${TWO_IMAGE_CROP_TRANSFORM}/`);
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

// FLAME_CLAUSE — shared suffix appended to every non-ambient camera-motion
// preset below (v3 pack, LTX_Prompt_revision doc, July 20, 2026, tested by
// Sam across 6 real renders: 3 with a real fireplace, 3 with none — zero
// hallucinations in either direction). Two things this deliberately does,
// confirmed as intentional with Sam and NOT an accidental hallucination
// exception to the rest of this file's "no new objects" language:
//   1. Fixes the actual bug this was written for — camera-motion presets'
//      own "everything stays fixed/unchanged" negative constraints were
//      unintentionally freezing an already-lit fireplace's flame along
//      with genuinely-static things (furniture, architecture). This clause
//      carves out "already-present dynamic elements keep moving naturally"
//      from "no NEW objects/geometry/reflections" — the actual protection
//      against hallucination is untouched.
//   2. Also allows flame INJECTION when no flame is visible in an existing
//      fireplace opening. Confirmed with Sam this is intentional, not a
//      hallucination-policy violation: AB 723 governs photo disclosure of
//      STAGING alterations; this is video, showing a real, physical
//      feature of the home (the fireplace itself) as lit, which is a
//      different category from inventing a feature that doesn't exist.
//      Real-render-tested specifically for the failure mode this could
//      cause — 3 no-fireplace renders confirmed LTX did NOT invent a
//      fireplace where none existed.
const FLAME_CLAUSE =
  " If a fireplace or fire pit flame is visible, it should appear with subtle, natural flicker exactly as photographed. If no flame is visible, include a small, photorealistic flame inside the existing fireplace opening, shown with subtle, natural flicker, without altering any surrounding architecture.";

const LTX_MOTION_TEMPLATES = {
  cinematic_push: {
    prompt:
      "Perform a smooth, deliberate push-in toward the center of the room, moving inward without touching boundary geometry. All architectural lines remain stable and unchanged. No new objects, openings, reflections, or extended surfaces appear." + FLAME_CLAUSE,
    confidence: "high",
    safeWhen: "Any interior room with clear depth.",
    gate: null,
  },
  luxury_drift: {
    prompt:
      "Perform an ultra-slow lateral drift at extremely low velocity, staying fully inside the photographed geometry. Maintain straight verticals and stable perspective. No new blinds, reflections, textures, or objects appear." + FLAME_CLAUSE,
    confidence: "high",
    safeWhen: "Rooms with strong horizontal sightlines.",
    gate: null,
  },
  floating_camera_drift: {
    // REVISED (July 21, 2026 — real render feedback: "barely moves"). Sam's
    // exact wording, verbatim — not the shared FLAME_CLAUSE constant, since
    // this replaces the whole prompt string as given, flame sentence
    // included. Adds explicit "gentle, noticeable sway" (the prior wording's
    // motion was too subtle to read as intentional) while adding an
    // explicit no-roll/no-tilt constraint, since the fix for "barely moves"
    // can't be allowed to reopen the kind of roll problem corner_to_corner_
    // drift hit separately in the same test round.
    prompt:
      "Perform a slow, buoyant floating drift with a gentle, noticeable sway that moves the viewer through the space without introducing any camera roll or tilt. Maintain perfectly level horizons, stable verticals, and natural parallax only from existing geometry. No new textures, reflections, or objects appear.\nIf a fireplace or fire pit flame is visible, it should appear with subtle, natural flicker exactly as photographed. If no flame is visible, include a small, photorealistic flame inside the existing fireplace opening, shown with subtle, natural flicker, without altering any surrounding architecture.",
    confidence: "medium-high",
    safeWhen: "Rooms with soft lighting and visible depth.",
    gate: { type: "advisory", note: "Avoid highly reflective rooms (mirrors, glossy tile) — not automatically detected, use judgment when picking the source photo." },
  },
  architectural_glide: {
    prompt:
      "Perform a smooth horizontal glide along the room's architectural sightline. Parallax arises only from existing geometry. No new openings, extended rooms, or architectural changes appear." + FLAME_CLAUSE,
    confidence: "high",
    safeWhen: "Kitchens, hallways, open-plan living spaces.",
    gate: { type: "advisory", note: "Avoid very tight rooms with no lateral space." },
  },
  corner_to_corner_drift: {
    // REVISED (July 21, 2026 — real render feedback: "severe roll"). Sam's
    // exact wording, verbatim — not the shared FLAME_CLAUSE constant. Root
    // cause of the roll: the prior wording ("drift from one visible corner
    // toward the opposite corner") gave LTX a diagonal destination with no
    // explicit ban on rotating to get there, which it apparently read as
    // license to roll/tilt the camera along the way. This version keeps
    // the lateral-drift feel but explicitly rules out diagonal roll/camera
    // tilt and demands level horizons/stable verticals instead of
    // describing a corner-to-corner destination at all.
    prompt:
      "Perform an ultra-slow lateral drift that gently shifts the viewer's perspective across the room without introducing any diagonal roll or camera tilt. Maintain perfectly level horizons, stable verticals, and natural parallax only from existing geometry. No new space, openings, or architectural features appear.\nIf a fireplace or fire pit flame is visible, allow subtle, natural flicker. If no flame is visible, include a small, photorealistic flame inside the existing fireplace opening, shown with subtle, natural flicker, without altering any surrounding architecture.",
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
      "Perform a smooth lateral drift that gently arcs across the scene, simulating an orbit without rotating around the subject. No new areas of the room are revealed beyond what is visible in the source image." + FLAME_CLAUSE,
    confidence: "medium-high",
    safeWhen: "A central subject — kitchen island, dining table, pool — with room around it in frame.",
    gate: null,
    // NEW (July 20, 2026, LTX_Prompt_revision doc) — Orbit/Arc is one of
    // only 3 presets that require the two-image crop workflow. Wide arc
    // rotation reveals background area LTX has no reference for on a
    // single image — see buildCroppedStartUrl's comment above for the
    // full mechanism (tight cropped start + full original as end).
    requiresTwoImage: true,
    cropTransformation: TWO_IMAGE_CROP_TRANSFORM,
  },
  rack_focus: {
    prompt:
      "Perform a gentle cinematic push-in with a soft depth-weighted emphasis shift. Foreground and background remain clear and stable. No blur melt, distortion, or new areas of the room are revealed." + FLAME_CLAUSE,
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
      "Perform an ultra-slow upward drift with a slight upward tilt, emphasizing the upper portion already visible. No new ceiling planes, no perspective exaggeration, and no new room areas are revealed." + FLAME_CLAUSE,
    confidence: "medium-high",
    safeWhen: "Rooms with visible ceiling detail — chandelier, fan, high windows.",
    gate: null,
  },
  crane_down: {
    prompt:
      "Perform an ultra-slow downward drift with a slight downward tilt, emphasizing the lower portion already visible. No new flooring, cabinetry, or extended geometry is revealed." + FLAME_CLAUSE,
    confidence: "medium-high",
    safeWhen: "Rooms with visible flooring detail — tilework, a rug.",
    gate: null,
  },
  parallax_push: {
    prompt:
      "Perform a micro push-in with layered micro-parallax. Foreground shifts slightly faster than background while all architecture remains stable. No new areas of the room are revealed." + FLAME_CLAUSE,
    confidence: "medium-high",
    safeWhen: "Rooms with clear foreground/background depth separation.",
    gate: null,
  },
  pan_zoom_reveal: {
    prompt:
      "Perform an ultra-slow lateral pan paired with a micro zoom-out. Maintain all architectural lines exactly as photographed. No widened field of view, no new geometry, and no inferred depth." + FLAME_CLAUSE,
    confidence: "medium-high",
    safeWhen: "Any room with reasonable width to pan across.",
    gate: null,
  },

  // Batch 3 REVISED (July 18, 2026) — Sam's own real disconnect catch:
  // the single "room_reveal" guess above (written from the reference
  // doc's one-line description, never tested) wasn't hallway-safe enough
  // — a generic "pull back and widen" instruction has no way to guarantee
  // it won't expose hallway depth or a doorway edge in an open-plan space
  // that has one nearby. Replaced entirely with 3 purpose-built,
  // explicitly hallway-constrained movements from LTX_Micro_movement_
  // prompts.docx — each one names the hallway-exposure risk directly and
  // constrains against it, rather than relying on a generic "no new
  // geometry" instruction to cover a specific, known failure mode.
  // UNTESTED — same flag as before: on paper from the doc, not yet
  // confirmed against a real render the way the batch-1/2 presets were.
  // Batch 3 REVISED (July 20, 2026, LTX_Prompt_revision v3 doc) — replaces
  // the earlier, much longer hallway-guardrail phrasing (Batch 3 REVISED,
  // July 18) with the v3 pack's tighter wording, plus the flame clause.
  // STILL UNTESTED (Sam's real-render testing this session covered the
  // FLAME_CLAUSE specifically on other presets, not these 3 movements
  // themselves) — flag carried forward from the prior revision, not
  // cleared by this change.
  micro_zoom_out: {
    prompt:
      "Perform a gentle micro zoom-out that breathes outward while staying strictly inside the photographed boundaries. No new ceiling, flooring, corners, cabinetry, or hallway entrances appear." + FLAME_CLAUSE,
    confidence: "medium-high",
    safeWhen: "Open-plan spaces, especially those with a hallway or corridor nearby that must not be exposed.",
    gate: null,
    openPlanOnly: true,
  },
  micro_dolly_back: {
    prompt:
      "Perform a smooth micro dolly-back that moves backward without exposing any new areas of the room. No hallway depth, doorway edges, or extended wall planes appear." + FLAME_CLAUSE,
    confidence: "medium-high",
    safeWhen: "Open-plan spaces, especially those with a hallway or corridor nearby that must not be exposed.",
    gate: null,
    openPlanOnly: true,
    // NEW (July 20, 2026, LTX_Prompt_revision doc) — one of the 3 presets
    // requiring the two-image crop workflow (see buildCroppedStartUrl).
    requiresTwoImage: true,
    cropTransformation: TWO_IMAGE_CROP_TRANSFORM,
  },
  open_plan_reveal: {
    prompt:
      "Perform a subtle open-plan reveal that redistributes focus across already visible zones without widening the frame. No new corners, walls, or hallway geometry appear." + FLAME_CLAUSE,
    confidence: "medium-high",
    safeWhen: "Open-plan spaces with continuous kitchen-dining-living sightlines.",
    gate: null,
    openPlanOnly: true,
    // NEW (July 20, 2026) — originally shipped as "no crop" in the v3 doc;
    // Sam's explicit follow-up correction same day added Open-Plan Reveal
    // to the cropped/two-image tier alongside Orbit/Arc and Micro Dolly
    // Back (3 total now, not 2 — the doc's "only TWO movements require
    // cropping" line is superseded by this correction).
    requiresTwoImage: true,
    cropTransformation: TWO_IMAGE_CROP_TRANSFORM,
  },
};

const VALID_LTX_PRESETS = new Set(Object.keys(LTX_MOTION_TEMPLATES));

// NEW (July 18, 2026, Cinematic_LTX_-Kling_reference doc) — replaces the
// blanket open-plan block that used to reject ALL LTX presets on
// open-plan frames outright. This is now a curated allowlist: exactly
// the 7 presets the reference doc marks "✅ Safe for Both Kling + LTX"
// under its OPEN-PLAN SPACES section. Everything not in this set still
// blocks on open-plan — including presets already proven fine for
// single/enclosed rooms (orbit_arc, rack_focus, crane_up/down,
// parallax_push, pan_zoom_reveal, drone_boom_up) — because the reference
// doc doesn't clear those for the open-plan case specifically. Being
// safe in a smaller enclosed room doesn't automatically mean safe across
// a larger, multi-zone open-plan space with more geometry to hallucinate
// around; treating "cleared for single-room" as "cleared for open-plan"
// would be exactly the kind of unearned generalization this whole
// pack has been careful to avoid everywhere else.
const OPEN_PLAN_SAFE_LTX_PRESETS = new Set([
  "cinematic_push",
  "luxury_drift",
  "architectural_glide",
  "corner_to_corner_drift",
  "living_room_ambient",
  "fireplace_flicker",
  "micro_zoom_out",
  "micro_dolly_back",
  "open_plan_reveal",
]);

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
function enforceLtxScopeRules(frame, presetKey, jobId) {
  const jobPrefix = jobId ? `[${jobId}] ` : "";
  const preset = LTX_MOTION_TEMPLATES[presetKey];
  if (!preset) {
    throw new Error(
      `LTX motion rejected: unknown preset "${presetKey}". Valid presets: ${[...VALID_LTX_PRESETS].join(", ")}.`
    );
  }

  // REVERSED (July 18, 2026, Cinematic_LTX_-Kling_reference doc) — this
  // used to be a blanket rejection of ANY LTX preset on open-plan frames.
  // The reference doc's real safety analysis clears a specific 7-preset
  // subset for open-plan (OPEN_PLAN_SAFE_LTX_PRESETS above); everything
  // else still blocks there, including presets already fine for single
  // rooms — see that constant's comment for why "safe in a small room"
  // doesn't imply "safe in a large multi-zone space."
  if (frame.isOpenPlan && !OPEN_PLAN_SAFE_LTX_PRESETS.has(presetKey)) {
    throw new Error(
      `LTX motion rejected: preset "${presetKey}" is not cleared for open-plan rooms (frame.isOpenPlan is true). Cleared open-plan presets: ${[...OPEN_PLAN_SAFE_LTX_PRESETS].join(", ")}. Use one of those, a Ken Burns continuation preset, or Kling if this specific motion is required on this room.`
    );
  }

  // These 3 are the inverse case — restricted TO open-plan spaces, unsafe
  // in enclosed single rooms (each one's own hallway-safety constraints
  // only make sense where there's real open-plan continuity to work
  // with). Mirrors klingMotion.js's identical OPEN_PLAN_ONLY_PRESETS
  // restriction on its own room_reveal preset — same underlying risk,
  // same rule, enforced independently per engine/preset.
  if (preset.openPlanOnly && !frame.isOpenPlan) {
    throw new Error(
      `LTX motion rejected: "${presetKey}" is restricted to open-plan spaces (frame.isOpenPlan must be true) — its hallway-safety constraints assume real open-plan continuity that an enclosed single room doesn't have. Use a different preset for this room, or mark it as Open Plan if that's genuinely accurate.`
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
    console.log(`  ${jobPrefix}[LTX] Advisory for preset "${presetKey}": ${preset.gate.note}`);
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
// NEW (July 20, 2026 — real render showed "Unprocessable Entity" with no
// further detail on 2 separate real LTX attempts, both otherwise correctly
// dispatched). @fal-ai/client's ApiError typically carries the REAL reason
// for a 422 in .body (fal.ai's own validation error JSON) — err.message
// alone is often just the generic HTTP status phrase, which tells us
// nothing about WHICH field or value fal.ai actually rejected. This pulls
// every plausible location that detail could live in, so the next real
// failure surfaces the actual cause instead of another generic string.
function extractFalErrorDetail(err) {
  const parts = [];
  if (err.status) parts.push(`status=${err.status}`);
  if (err.body) {
    try {
      parts.push(`body=${JSON.stringify(err.body)}`);
    } catch {
      parts.push(`body=${String(err.body)}`);
    }
  }
  if (err.response?.data) {
    try {
      parts.push(`response.data=${JSON.stringify(err.response.data)}`);
    } catch {
      parts.push(`response.data=${String(err.response.data)}`);
    }
  }
  return parts.length > 0 ? parts.join(" | ") : "(no additional detail available on the error object)";
}

async function generateLtxContinuationClip(frame, presetKey, workDir, jobId) {
  // NEW (July 19, 2026 — real diagnostic gap found): every log line below
  // used to be unprefixed, unlike every other log line in renderPipeline.js
  // (which all carry [jobId]). That made it genuinely ambiguous whether an
  // LTX call ran at all when someone filtered Railway logs by job ID — the
  // lines would fire but not show up in a job-ID-filtered view. jobPrefix
  // is blank if no jobId was passed (backward compatible), so this never
  // breaks existing call sites that do not have one handy.
  const jobPrefix = jobId ? `[${jobId}] ` : "";
  ensureConfigured();
  enforceLtxScopeRules(frame, presetKey, jobId);

  const prompt = buildLtxPrompt(frame, presetKey);
  const requestedDuration = frame.continuationDurationSeconds || 4.0;
  const duration = snapToValidLtxDuration(requestedDuration);

  if (duration !== requestedDuration) {
    console.log(
      `  [LTX] Requested continuation duration ${requestedDuration}s snapped to ${duration}s ` +
        `(LTX Fast only accepts ${VALID_LTX_DURATIONS.join("/")}).`
    );
  }

  console.log(`  ${jobPrefix}[LTX] Submitting job — preset: ${presetKey}, duration: ${duration}s`);

  const LTX_ENDPOINT = "fal-ai/ltx-2.3/image-to-video/fast";

  // Explicit queue submit/status/result polling, NOT fal.subscribe() —
  // same reasoning as klingMotion.js's identical choice: fal.subscribe()'s
  // internal polling has gone silent on a real render before with no
  // JS-catchable error at all. Every fal.ai call in this codebase since
  // that discovery uses this same explicit, loggable pattern.
  const preset = LTX_MOTION_TEMPLATES[presetKey];

  // NEW (July 20, 2026 — real bug: user reported aspect errors requiring a
  // full restart). fal.ai's aspect_ratio param defaults to "auto" when
  // omitted, which infers the ratio from the source image — this codebase
  // (concat, Ken Burns matching, final assembly) assumes 16:9 everywhere,
  // so any non-16:9 source silently broke the rest of the pipeline. Now
  // forced explicitly on every LTX call instead of left on auto. The
  // matching frontend-side message ("AI Movements can only be rendered
  // 16:9") is a separate UI fix — see handoff notes, frontend repo pending.
  const inputPayload = {
    image_url: preset && preset.requiresTwoImage
      ? buildCroppedStartUrl(frame.remoteImageUrl)
      : frame.remoteImageUrl,
    prompt,
    duration: duration,
    fps: 25, // matches motionRenderer.py's own fps, so LTX and Ken Burns clips are never accidentally frame-rate-mismatched at the concat/comparison level
    aspect_ratio: "16:9",
    // NEW (July 20, 2026 — real bug: LTX audio was rendering on every
    // clip despite this pipeline being audio-off/silent by design, narration
    // and music are mixed in separately by assemble.js). fal.ai's schema
    // defaults generate_audio to true; this was never set before, so every
    // LTX clip has been generating (and billing for) audio nobody wanted.
    generate_audio: false,
  };

  // Two-image workflow (July 20, 2026, LTX_Prompt_revision doc) — only
  // orbit_arc, micro_dolly_back, and open_plan_reveal set requiresTwoImage.
  // end_image_url = the ORIGINAL, uncropped staged image — wider by
  // comparison to the cropped start frame above, giving LTX real content
  // to reference for whatever the move would otherwise reveal beyond the
  // tight start frame's boundaries, per the doc's two-image mechanism.
  if (preset && preset.requiresTwoImage) {
    inputPayload.end_image_url = frame.remoteImageUrl;
  }

  let request_id;
  try {
    ({ request_id } = await fal.queue.submit(LTX_ENDPOINT, {
      // FIX (July 20, 2026 — confirmed via real fal.ai error body, not
      // a guess): fal.ai's schema does a strict literal-type match
      // against the duration enum (6/8/10/.../20) — sending it as a
      // STRING here ("14" instead of 14) failed validation with
      // "Input should be 6, 8, 10, 12, 14, 16, 18 or 20" even though
      // "14" and 14 look identical in a log line. This is the exact,
      // sole cause of every LTX "Unprocessable Entity" this session —
      // dispatch/routing was correct the whole time; this one type
      // mismatch was silently killing every real call at the result-fetch
      // step (fal.ai's queue reports COMPLETED regardless, since the
      // job itself ran — it's fetching the actual result that exposes
      // the input it was never valid for).
      input: inputPayload,
    }));
  } catch (err) {
    throw new Error(`LTX submit failed: ${err.message} | ${extractFalErrorDetail(err)}`);
  }

  console.log(`  ${jobPrefix}[LTX] Queued — request_id: ${request_id} (recoverable via fal.ai dashboard even if this process dies)`);

  const POLL_INTERVAL_MS = 10000;
  const MAX_POLL_ATTEMPTS = 90; // 15 minute ceiling, matches klingMotion.js's

  let finalStatus = null;
  for (let attempt = 1; attempt <= MAX_POLL_ATTEMPTS; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));

    let status;
    try {
      status = await fal.queue.status(LTX_ENDPOINT, { requestId: request_id, logs: true });
    } catch (err) {
      throw new Error(`LTX status poll failed (request_id ${request_id}): ${err.message} | ${extractFalErrorDetail(err)}`);
    }

    console.log(`  ${jobPrefix}[LTX] Poll ${attempt}/${MAX_POLL_ATTEMPTS} — status: ${status.status}`);
    if (status.status === "IN_PROGRESS" && status.logs) {
      status.logs.forEach((log) => console.log(`  ${jobPrefix}[LTX] ${log.message}`));
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

  let result;
  try {
    result = await fal.queue.result(LTX_ENDPOINT, { requestId: request_id });
  } catch (err) {
    // This is the exact call that threw "Unprocessable Entity" with no
    // further detail on 2 real attempts (July 20, 2026) — the status
    // poll reported COMPLETED both times, so whatever fal.ai objects to
    // is only surfacing at result-fetch time, not submit or poll time.
    throw new Error(`LTX result fetch failed (request_id ${request_id}, queue reported COMPLETED): ${err.message} | ${extractFalErrorDetail(err)}`);
  }

  const videoUrl = result.data?.video?.url;
  if (!videoUrl) {
    throw new Error(`LTX returned no video URL (request_id ${request_id}). Full result: ${JSON.stringify(result).slice(0, 500)}`);
  }

  console.log(`  ${jobPrefix}[LTX] Generation complete — downloading clip from fal.ai`);

  const outputPath = path.join(workDir, `ltx_${Date.now()}.mp4`);
  const response = await axios.get(videoUrl, { responseType: "arraybuffer" });
  fs.writeFileSync(outputPath, response.data);

  console.log(`  ${jobPrefix}[LTX] Clip downloaded to ${outputPath} (${response.data.length} bytes)`);

  // videoUrl added (July 20, 2026) — purely additive, doesn't change
  // anything for existing callers (generateLtxRevealContinuation,
  // applyLtxMotion just destructure the fields they already use). Needed
  // so a test route can hand back a real, clickable fal.ai URL instead of
  // only a local file path that isn't reachable from outside the
  // container.
  return { path: outputPath, duration, videoUrl };
}

// ── REVEAL_PRESETS ENTRY POINT ────────────────────────────────────────
// Called from renderPipeline.js's reveal branch in place of
// applyMotionPreset() when the user's End Motion selection is an LTX
// preset. Returns the same {path, endingZoom} shape applyMotionPreset()
// does, so the caller doesn't need separate handling downstream.
async function generateLtxRevealContinuation(frame, presetKey, workDir, jobId) {
  const result = await generateLtxContinuationClip(frame, presetKey, workDir, jobId);
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
async function applyLtxMotion(frame, presetKey, workDir, fallbackFn, jobId) {
  const jobPrefix = jobId ? `[${jobId}] ` : "";
  try {
    const result = await generateLtxContinuationClip(frame, presetKey, workDir, jobId);
    console.log(`  ${jobPrefix}[LTX] Clip ready: ${result.path}`);
    return { path: result.path, source: "ltx", endingZoom: 1.0 };
  } catch (err) {
    console.error(`  ${jobPrefix}[LTX] Generation failed, falling back to Ken Burns: ${err.message}`);
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
  OPEN_PLAN_SAFE_LTX_PRESETS,
  buildCroppedStartUrl,
  TWO_IMAGE_CROP_TRANSFORM,
};
