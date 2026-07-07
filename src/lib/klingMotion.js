// klingMotion.js — AI-generated camera motion via Kling (through fal.ai),
// using the official @fal-ai/client SDK rather than raw HTTP requests.
//
// IMPORTANT — SCOPE RESTRICTION (read before modifying):
// The real safety boundary is NOT "interior vs exterior" and NOT "known
// pair vs single image" on its own — it's whether the alteration is a
// disclosed, enumerated-category change (anything AB 723 §10140.8(b)(1)
// already covers: furniture, fixtures, walls, flooring, hardscape,
// landscape, facade, floor plans) vs. content with no photographic ground
// truth at all and no disclosed framing. See enforceScopeRules() below,
// which is a hard runtime check, not just a comment.
//
// Validated known-pair use cases (both endpoints are real, disclosed
// images — Kling interpolates, doesn't invent):
//   1. Interior vacant→staged interpolation (start/end frame, same room)
//   2. Exterior day/twilight transitions (start/end frame) — see
//      KLING_MOTION_TEMPLATES for the day_to_twilight / twilight_to_day /
//      timelapse variants
//
// Exterior also permits single-image motion (no end frame) — drone_boom_up,
// water_motion, and the generic exterior default.
//
// RESOLVED — single-image INTERIOR presets (orbit_arc, rack_focus,
// fireplace_flicker) are also permitted, via
// SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS below. Worked through with Sam
// against the actual AB 723 text and MetroList's MLS Rules (11.6.1,
// 12.10(f)): these are the same kind of disclosed, enumerated-category
// alteration as a virtual pool, a wall removal, or added landscaping —
// not a different risk just because the invented content shows up via a
// camera move instead of a static add-on. A virtual pool isn't internally
// marked "this part is fake" inside the photo either — the only mechanism
// either case has is the same one: a watermark/label, the AB 723
// compliance page, and (per MLS Rule 12.10(f)) identification in Public
// Remarks. If that mechanism is sufficient for a pool, it's sufficient for
// a camera move that reveals an unphotographed cabinet run or animates an
// existing fireplace's flame. Same compliance path, same conclusion.
// (curtain_sway was tested and dropped — not a compliance issue, just no
// real use case in real estate photography, and the motion read as
// exaggerated/windy rather than subtle. Removed entirely below.)
//
// What STAYS blocked for single-image interior: the GENERIC default case
// (no klingMotionPreset set, or any preset not in the allowlist) — i.e.
// converting an empty room into a furnished one directly via Kling, with
// no staged counterpart image at all. That's a categorically different
// case: it bypasses the platform's actual staging pipeline (and its
// review/Generate-Final step) entirely, inventing a full furnished room
// from nothing rather than animating or revealing more of a scene that's
// already been staged, photographed, and disclosed through the normal
// flow. Use a real vacant+staged pair for that, or Ken Burns.
//
// Falls back to Ken Burns (motionPresets.js) on any failure — AI motion
// is a premium enhancement, not a dependency the pipeline requires.

const fs = require("fs");
const path = require("path");
const axios = require("axios");
const ffmpeg = require("fluent-ffmpeg");
const { fal } = require("@fal-ai/client");
const { applyMotionPreset } = require("./motionPresets");

// Must match the output dimensions used in motionPresets.js — kept as a
// separate local constant rather than importing it, since these two
// modules normalize to a shared OUTPUT spec but aren't otherwise coupled.
// If motionPresets.js's OUTPUT_W/OUTPUT_H ever change, update here too.
const OUTPUT_W = 1920;
const OUTPUT_H = 1080;

let configured = false;

function ensureConfigured() {
  if (configured) return;
  if (!process.env.FAL_KEY) {
    throw new Error("FAL_KEY not set — cannot use Kling AI motion");
  }
  fal.config({ credentials: process.env.FAL_KEY });
  configured = true;
}

// ── SCOPE ENFORCEMENT ─────────────────────────────────────────────────
// The real safety boundary isn't "interior vs exterior" — it's whether
// Kling has two REAL, KNOWN endpoints to interpolate between, an exterior
// scene (more tolerant of invented sky/landscape detail), or a single
// interior preset from the allowlist below (disclosed via the same
// mechanism as any other altered image — see file header for the full
// reasoning).
//
//   - Vacant + Staged pair (from Smart Stage PRO staging) → both endpoints
//     are real, disclosed images. Kling fills in the transition between
//     two known states. Low risk, AI Motion freely available here.
//
//   - A single photo with no pair, exterior → permitted. Landscaping/sky
//     have more tolerance for invented detail than interior architecture.
//
//   - A single photo with no pair, interior, named preset in
//     SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS → permitted. These animate
//     camera movement or a dynamic element (flame) within a scene that's
//     already fully visible and disclosed — same compliance category as
//     a virtual pool or a wall removal, not a different risk.
//
//   - A single photo with no pair, interior, NO named preset (or one not
//     in the allowlist) → rejected. This is the generic vacant→furnished
//     case: no staged counterpart exists at all, so Kling would have to
//     invent furniture/layout wholesale rather than animate or extend an
//     already-disclosed scene. Use a real vacant+staged pair, or Ken Burns.

const SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS = new Set([
  "orbit_arc",
  "rack_focus",
  "fireplace_flicker",
  // New Motion Library (July 2026) — camera-move-only, same category as the
  // 3 above: motion through/around a scene that's already fully visible, no
  // invented room content. See KLING_MOTION_TEMPLATES for the reasoning on
  // why room_reveal, living_room_ambient, and corner_to_corner_drift are
  // deliberately NOT included here yet.
  "cinematic_push",
  "luxury_drift",
  "floating_camera_drift",
  "parallax_push",
  "architectural_glide",
  "crane_up",
  "crane_down",
]);

function enforceScopeRules(frame) {
  const hasKnownPair = !!frame.endImageUrl;
  const isExterior = frame.roomType === "exterior";
  const isAllowedSingleImageInteriorPreset =
    !!frame.klingMotionPreset &&
    SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS.has(frame.klingMotionPreset);

  if (hasKnownPair || isExterior || isAllowedSingleImageInteriorPreset) {
    return;
  }

  throw new Error(
    `Kling AI motion rejected: no end image provided for room type "${frame.roomType}", and preset "${frame.klingMotionPreset || "(none — generic default)"}" is not in the single-image interior allowlist (orbit_arc, rack_focus, fireplace_flicker, cinematic_push, luxury_drift, floating_camera_drift, parallax_push, architectural_glide, crane_up, crane_down). The generic interior default requires Kling to invent furniture/layout wholesale rather than interpolate between two known images or animate an already-disclosed scene — this is disabled by design. Use a vacant+staged pair, select one of the allowed single-image presets, or use Ken Burns for single-image interior shots outside that list. See AB 723 scope restriction in klingMotion.js.`
  );
}

// ── INPUT ASPECT RATIO NORMALIZATION ──────────────────────────────────
// Kling infers its output aspect ratio from the input image (confirmed on
// fal.ai's v3/pro docs: "Aspect ratio is inferred from the start image. The
// aspect_ratio field in the UI is ignored by the model" — O3 Standard
// exposes no aspect_ratio parameter either, consistent with the same
// inference behavior). Real estate photos are commonly shot at 3:2
// (e.g. 1176x784), not 16:9 — so without this, Kling faithfully returns a
// 3:2 clip, which then needs scaling/cropping to fit our 1920x1080 canvas
// and visually mismatches the full-frame Ken Burns continuation.
//
// Fixing it here, at the source, means Kling never produces a mismatched
// clip in the first place — same principle as the existing "center-crop
// source to 16:9 before any motion is applied" fix already used for the
// Ken Burns engine, just applied one step earlier in the Kling pipeline.
// Cloudinary applies this transformation on the fly when Kling fetches the
// URL — no extra processing or re-upload needed on our end.

function forceCloudinary16x9(url) {
  if (!url || !url.includes("/upload/")) return url; // not a Cloudinary delivery URL — leave untouched
  return url.replace("/upload/", "/upload/c_fill,ar_16:9,g_auto/");
}

// ── KLING MOTION PRESET TEMPLATES ─────────────────────────────────────
// Named, tested prompt templates beyond the two generic defaults below.
// Selected via frame.klingMotionPreset; falls back to the generic
// interior/exterior prompt when unset or unrecognized — same fallback
// pattern as resolvePreset() in motionPresets.js. Each of these was
// iterated on and verified in the fal.ai Playground before being added
// here (see handoff for testing notes and image-pair requirements).
// Scope/compliance reasoning for which presets are usable on a single
// interior image is in the file header and SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS
// above — resolved, not an open question.

const KLING_MOTION_TEMPLATES = {
  // ── Kling-exclusive moves — impossible to fake with Ken Burns ────────
  orbit_arc:
    "Slow cinematic orbit camera movement arcing around the central feature — a kitchen island, dining table, or pool — sweeping laterally while keeping it centered in frame, photorealistic, no distortion, stable architecture, surrounding cabinetry and furniture remain fixed and undistorted throughout the movement",

  drone_boom_up:
    "Smooth cinematic drone boom-up camera movement, rising upward and slightly forward to reveal the full exterior and surrounding landscaping from an elevated angle, photorealistic, no distortion, house structure and architecture remain completely fixed and unchanged",

  rack_focus:
    "Cinematic rack focus shot, starting with sharp focus on a foreground detail (faucet hardware, fixture, or vignette), then smoothly shifting focus to reveal the room behind it coming into sharp clarity, soft natural depth of field transition, photorealistic, no distortion, structure and cabinetry remain fixed and unchanged",

  fireplace_flicker:
    "Static cinematic shot, camera locked off, with the fireplace flame flickering naturally and realistically, gentle ambient light flicker on surrounding walls, photorealistic, no distortion, room and architecture remain completely fixed and unchanged",

  water_motion:
    "Static cinematic shot, camera locked off, with gentle natural water movement and subtle ripples across the pool surface, photorealistic, no distortion, landscaping and structure remain completely fixed and unchanged",

  // ── Exterior day/twilight transitions — known-pair, like vacant→staged ─
  day_to_twilight:
    "Smooth cinematic camera movement across the exterior as the bright daytime sky gradually deepens into a dusk blue-hour sky with soft pink and purple sunset color near the horizon, interior window lights gradually turning on and glowing warm amber, exterior porch and garage lights turning on, landscape lighting along the walkway and garden beds gradually illuminating, photorealistic, no distortion, house structure, architecture, and landscaping remain completely fixed and unchanged — only sky color, light quality, and lighting fixtures transform",

  twilight_to_day:
    "Smooth cinematic camera movement across the exterior as the dusk blue-hour sky with soft pink and purple sunset color gradually brightens into a clear daytime blue sky, interior window lights gradually turning off, exterior porch and garage lights turning off, landscape lighting along the walkway and garden beds gradually dimming and turning off, photorealistic, no distortion, house structure, architecture, and landscaping remain completely fixed and unchanged — only light quality and color temperature transform",

  day_to_twilight_timelapse:
    "Time-lapse style video, static locked-off camera, exterior daytime sky rapidly transitioning into a dusk blue-hour sky, clouds streaking past, sky color rapidly deepening from bright blue to deep blue with a pink and purple sunset glow near the horizon, interior window lights and exterior porch, garage, and landscape lighting rapidly turning on in sequence as if hours are passing in seconds, photorealistic, no distortion, house structure, architecture, and landscaping remain completely fixed and unchanged — only the sky, light, and time of day transform",

  twilight_to_day_timelapse:
    "Time-lapse style video, static locked-off camera, exterior dusk blue-hour sky with a pink and purple sunset glow rapidly brightening into a clear daytime blue sky, clouds streaking past, interior window lights and exterior porch, garage, and landscape lighting rapidly turning off in sequence as if hours are passing in seconds, photorealistic, no distortion, house structure, architecture, and landscaping remain completely fixed and unchanged — only the sky, light, and time of day transform",

  // Three-phase, 6s, known-pair (twilight start / day end). Lighting timing
  // confirmed against real test footage: interior lights go dark well
  // before full night (people go to bed), while exterior/landscape
  // lighting stays on through the night on timers, then cuts at dawn —
  // two independent lighting timelines in the same clip.
  twilight_night_day_timelapse:
    "Time-lapse style video, static locked-off camera, exterior sky transitioning through three phases: starting at dusk blue-hour with a pink and purple sunset glow near the horizon, deepening into full night with a dark navy-to-black sky, then rapidly brightening into clear daytime blue sky, clouds streaking past throughout. Interior window lights are on at the start during dusk, then turn off at around the 2.5 second mark, leaving only exterior porch, garage, and landscape lighting on for the remainder of the night phase, which then turns off as daylight returns at dawn. Photorealistic, no distortion, house structure, architecture, and landscaping remain completely fixed and unchanged — only the sky, light, and time of day transform",

  // ── New Motion Library (July 2026) — 5 additional camera-move-only presets,
  // same compliance category as orbit_arc/rack_focus/fireplace_flicker above:
  // pure camera movement through/around a scene that's already fully visible
  // and disclosed, no invented furniture, fixtures, or room content. Added to
  // SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS below on that basis.

  cinematic_push:
    "Slow, smooth cinematic push-in camera movement toward the center of the room, gently tightening the framing on the space already shown, photorealistic, no distortion, stable architecture, all visible furniture, fixtures, walls, and windows remain fixed and unchanged throughout the movement",

  luxury_drift:
    "Slow, elegant lateral drift camera movement gliding gently across the room from one side toward the other, refined luxury cinematic feel, photorealistic, no distortion, stable architecture, all visible furniture, fixtures, and architecture remain fixed and undistorted throughout the movement",

  floating_camera_drift:
    "Gentle floating camera movement with subtle drift and micro-sway, as if suspended weightlessly within the room, soft breathing motion, photorealistic, no distortion, stable architecture, all visible furniture, fixtures, and architecture remain fixed and undistorted throughout",

  parallax_push:
    "Cinematic push-in camera movement with layered parallax depth, foreground elements shifting slightly faster than background elements as the camera moves forward to create a sense of dimensional depth, photorealistic, no distortion, stable architecture, all visible furniture, fixtures, and architecture remain fixed and undistorted throughout",

  architectural_glide:
    "Smooth lateral glide camera movement tracking along the sightline, hallway, or open-plan sequence already visible in the photo, emphasizing architectural flow, photorealistic, no distortion, stable architecture, all visible walls, ceilings, fixtures, and furniture remain fixed and undistorted throughout",

  // ── crane_up / crane_down — Kling-quality vertical crane pair, filling a
  // real gap: Ken Burns already has tilt_up/tilt_down (pure FFmpeg vertical
  // pan, no AI generation), and drone_boom_up is Kling's vertical move but
  // exterior-only. There was no AI-generated vertical camera move for
  // single-image interior until now. Same category as cinematic_push/
  // architectural_glide above — camera moves through content already
  // visible in the photo, no invented room content.
  crane_up:
    "Smooth cinematic crane camera movement, rising vertically while tilting slightly upward to bring the upper portion of the room already visible in the photo into clearer view — ceiling details, chandelier, ceiling fan, or high windows — photorealistic, no distortion, stable architecture, all visible fixtures, furniture, and architecture remain fixed and unchanged throughout the movement, do not reveal room area beyond what is visible in the source photo",

  crane_down:
    "Smooth cinematic crane camera movement, descending vertically while tilting slightly downward to bring the lower portion of the room already visible in the photo into clearer view — flooring, tilework, or a rug — photorealistic, no distortion, stable architecture, all visible fixtures, furniture, and architecture remain fixed and unchanged throughout the movement, do not reveal room area beyond what is visible in the source photo",

  // ── room_reveal — TEMPLATE ADDED, DELIBERATELY NOT in
  // SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS below. Unlike the 5 above, this
  // preset's own name implies widening what's visible beyond the original
  // photo's framing, not just moving the camera through/around content
  // already shown — a materially different claim than orbit_arc's "stays
  // centered on one feature." Flagged for Sam's explicit review before
  // enabling for single-image interior use; safe to use today only with a
  // real vacant+staged pair (hasKnownPair bypasses this restriction) or on
  // exterior (isExterior bypasses it too).
  room_reveal:
    "Slow cinematic reveal movement, camera gently pulling back and widening to bring more of the already-visible room into frame, photorealistic, no distortion, stable architecture, all furniture and fixtures remain fixed and unchanged — do not invent new rooms, walls, fixtures, or furniture beyond what is visible in the source photo",

  // ── The following 3 are the exact presets already flagged in Sam's own
  // New Motion Library planning doc as needing fal.ai Playground
  // verification before real use — honoring that flag as-is, not
  // second-guessing it. Templates included so they can be Playground-tested;
  // living_room_ambient and corner_to_corner_drift are deliberately NOT in
  // SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS below until verified.
  // outdoor_breeze needs no allowlist entry — exterior single-image is
  // already permitted for any preset under enforceScopeRules' isExterior
  // check — but the same invented-water-feature risk applies regardless of
  // the scope gate, so verify it in Playground before real use too.
  living_room_ambient:
    "Subtle ambient cinematic motion within the living room — if a lit fireplace is visible, gentle flame flicker; if curtains are visible, a light natural sway; if plants are visible, subtle organic movement — only animating elements already present in the photo, photorealistic, no distortion, stable architecture, all furniture, fixtures, and architecture remain fixed and unchanged, do not add a fireplace, curtains, or plants that are not already visible",

  outdoor_breeze:
    "Gentle outdoor ambient motion — if trees, plants, or landscaping are visible, a subtle natural breeze moving through them; if water features are visible, gentle ripples — only animating elements already present in the photo, photorealistic, no distortion, structure and landscaping remain fixed and unchanged, do not add trees, plants, water features, or landscaping that are not already visible",

  corner_to_corner_drift:
    "Slow diagonal cinematic drift from one corner of the room toward the opposite corner, staying within the room as already framed in the photo, photorealistic, no distortion, stable architecture, all visible furniture, fixtures, and architecture remain fixed and undistorted throughout, do not reveal room area beyond what is visible in the source photo",
};

const VALID_KLING_PRESETS = new Set(Object.keys(KLING_MOTION_TEMPLATES));

// ── PROMPT TEMPLATES ──────────────────────────────────────────────────
// Kept separate per use case since the framing differs meaningfully —
// interior is about furniture appearing, exterior is about lighting/
// landscape transformation with the structure held fixed.
//
// Precedence: customPrompt always wins (full caller override) → named
// klingMotionPreset from the table above → generic room-type default.

function buildPrompt(frame) {
  if (frame.customPrompt) return frame.customPrompt;

  if (frame.klingMotionPreset) {
    if (KLING_MOTION_TEMPLATES[frame.klingMotionPreset]) {
      return KLING_MOTION_TEMPLATES[frame.klingMotionPreset];
    }
    console.warn(
      `[klingMotion] Unknown klingMotionPreset "${frame.klingMotionPreset}", falling back to room-type default`
    );
  }

  const isInterior = !["exterior"].includes(frame.roomType);

  if (isInterior) {
    return "Smooth cinematic push-in camera movement through an empty room as furniture and decor gradually appear, room becomes fully furnished and staged, photorealistic, no distortion, stable architecture, walls and windows remain fixed";
  }

  return "Smooth cinematic camera movement across the exterior as lighting and landscaping gradually transform and improve, photorealistic, no distortion, house structure and architecture remain completely fixed and unchanged";
}

// ── LAST-FRAME EXTRACTION ─────────────────────────────────────────────
// Extracts the actual last frame of Kling's output video as a PNG, so the
// continuation Ken Burns clip can start from the pixel-exact state Kling
// ended on, instead of from the original (pre-transformation) staged image.

/**
 * Extracts the last frame of a video clip as a PNG.
 * Used to seed the Ken Burns continuation clip from Kling's exact end state,
 * so the first frame of Ken Burns is pixel-identical to Kling's last frame.
 *
 * @param {string} videoPath  - Path to the Kling output video on disk.
 * @param {string} workDir    - Temp directory for this job.
 * @returns {Promise<string>} - Path to the extracted PNG.
 */
async function extractLastFrame(videoPath, workDir) {
  const outputPath = path.join(workDir, `lastframe_${path.basename(videoPath, ".mp4")}.png`);
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .inputOptions([
        "-sseof", "-0.1", // seek to 0.1s before end of file — this is an INPUT-side seek
        // option (it affects how ffmpeg reads the source), not an output option. Confirmed via
        // a real "Option sseof cannot be applied to output url" ffmpeg error when this was
        // bundled into outputOptions() instead — ffmpeg parses options positionally, and an
        // input option placed after -i gets misread as belonging to the output file.
      ])
      .outputOptions([
        "-vframes", "1",  // grab exactly one frame
        "-q:v", "2",      // near-lossless quality (PNG ignores this, but safe)
      ])
      .output(outputPath)
      .on("start", (cmd) => console.log(`  [Kling] extractLastFrame ffmpeg command: ${cmd}`))
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`extractLastFrame failed: ${err.message}`)))
      .run();
  });
}

// ── CONTINUATION MOTION ──────────────────────────────────────────────
// After Kling's transformation finishes (vacant becomes staged), the
// clip ends on a static-feeling final frame. This stitches on a few
// extra seconds of Ken Burns motion (push-in, pull-back, float, pan)
// starting from the ACTUAL LAST FRAME of Kling's clip — not the original
// staged image, which is Kling's *starting* frame, not its ending one.
// Using the starting image caused a visible zoom/position jump at the
// stitch point, since Kling's own camera move had already changed the
// composition by the time its clip ended. Extracting the real last frame
// makes the cut pixel-seamless.

function concatTwoClips(firstPath, secondPath, workDir, outputName) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, outputName);

    // Simple hard cut, not a crossfade — Kling's clip already ends on
    // (approximately) the staged image, and the continuation clip starts
    // from that same staged image, so the cut should read as nearly
    // seamless without needing a transition effect to hide a mismatch.
    //
    // IMPORTANT: Kling's native output and our own FFmpeg-rendered clip
    // are NOT guaranteed to share the same resolution, fps, or pixel
    // format — confirmed by a real "Error reinitializing filters! Failed
    // to inject frame into filter network: Invalid argument" failure when
    // concatenating them directly. The fix is to explicitly normalize
    // BOTH inputs to identical specs (scale + pad to OUTPUT_W x OUTPUT_H,
    // fixed fps, yuv420p) as part of this same filter graph, rather than
    // assuming they already match.
    ffmpeg()
      .input(firstPath)
      .input(secondPath)
      .complexFilter([
        // Scale-to-fill + center-crop, not scale-to-fit + pad. Kling's native
        // resolution (e.g. 1176x784) doesn't match our 1920x1080 canvas, and
        // padding to fit left it letterboxed — visually smaller than the
        // continuation clip, which already fills the frame. That mismatch is
        // what read as a jarring zoom jump at the cut: same canvas size, very
        // different effective content size. Crop-to-fill matches the same
        // convention already used elsewhere (the 16:9 pre-crop fix for the
        // Ken Burns stretching bug) — content fills the frame on both sides
        // of the cut, no black bars, no apparent size jump.
        `[0:v]scale=${OUTPUT_W}:${OUTPUT_H}:force_original_aspect_ratio=increase,crop=${OUTPUT_W}:${OUTPUT_H},fps=20,format=yuv420p,setpts=PTS-STARTPTS[v0]`,
        `[1:v]scale=${OUTPUT_W}:${OUTPUT_H}:force_original_aspect_ratio=increase,crop=${OUTPUT_W}:${OUTPUT_H},fps=20,format=yuv420p,setpts=PTS-STARTPTS[v1]`,
        "[v0][v1]concat=n=2:v=1:a=0[outv]",
      ])
      .outputOptions(["-map", "[outv]", "-pix_fmt", "yuv420p"])
      .output(outputPath)
      .on("start", (cmd) => console.log(`  [Kling] concatTwoClips ffmpeg command: ${cmd}`))
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`Continuation concat failed: ${err.message}`)))
      .run();
  });
}

async function applyContinuationMotion(klingClipPath, frame, workDir) {
  // Extract the actual last frame of Kling's clip — this is the
  // pixel-exact state Kling ended on. Ken Burns starts from this image,
  // making the cut between Kling and Ken Burns invisible to the viewer,
  // instead of jumping back to the composition the room started at.
  //
  // Logging on both sides of this call deliberately — this is new,
  // never-before-exercised code, and the Kling polling fix just taught us
  // that any unlogged await is a place a future hang can vanish without a
  // trace. Same principle applied here preemptively.
  console.log(`  [Kling] Extracting last frame from ${klingClipPath}`);
  const lastFramePath = await extractLastFrame(klingClipPath, workDir);
  console.log(`  [Kling] Last frame extracted: ${lastFramePath}`);

  // Default to luxury_parallax — push_in immediately after Kling's own
  // push-in transformation would feel repetitive (same motion twice in a
  // row). A slow parallax drift gives the second beat real contrast and
  // matches the validated "Push → Transform → Parallax" sequence design.
  const continuationPreset = frame.continuationPreset || "luxury_parallax";
  const continuationDuration = frame.continuationDurationSeconds || 3;

  console.log(`  [Kling] Adding ${continuationDuration}s continuation motion (${continuationPreset})`);

  // startZoom = 1.0 is correct here: the extracted last frame already
  // represents Kling's full zoomed/panned composition — Ken Burns doesn't
  // need to compensate for any prior zoom state, it just continues
  // naturally from that still image.
  const continuationResult = await applyMotionPreset(
    {
      localPath: lastFramePath, // Kling's actual last frame, not the original staged image
      motionPreset: continuationPreset,
      durationSeconds: continuationDuration,
    },
    workDir,
    1.0
  );

  // ── TEMPORARY DEBUG: upload both individual clips BEFORE concatenation,
  // so each can be inspected in isolation. This is the fastest way to tell
  // whether the bug lives in the parallax filter itself (motionPresets.js)
  // or only appears after the concat/normalize step. Remove once the
  // continuation feature is confirmed working end to end.
  try {
    const { uploadToCloudinary } = require("./cloudinaryUpload");
    const debugUrls = await uploadToCloudinary(
      { debug_kling_only: klingClipPath, debug_parallax_only: continuationResult.path },
      "debug-continuation"
    );
    console.log(`  [DEBUG] Kling clip alone: ${debugUrls.debug_kling_only}`);
    console.log(`  [DEBUG] Parallax clip alone: ${debugUrls.debug_parallax_only}`);
  } catch (debugErr) {
    console.error(`  [DEBUG] Debug upload failed (non-fatal): ${debugErr.message}`);
  }

  const combinedPath = await concatTwoClips(
    klingClipPath,
    continuationResult.path,
    workDir,
    `kling_continued_${Date.now()}.mp4`
  );

  // ── TEMPORARY DEBUG: upload the combined clip too, right after concat,
  // before it heads into the rest of the pipeline. The two uploads above
  // confirmed each half individually — this one confirms whether the
  // CONCAT step itself produces a correct ~8s combined file, or whether
  // the bug is further downstream (assemble.js) truncating something
  // that was already fine at this point. Remove alongside the other two
  // debug uploads once the full chain is confirmed working.
  try {
    const { uploadToCloudinary } = require("./cloudinaryUpload");
    const combinedDebugUrls = await uploadToCloudinary(
      { debug_combined: combinedPath },
      "debug-continuation"
    );
    console.log(`  [DEBUG] Combined clip (post-concat, pre-pipeline): ${combinedDebugUrls.debug_combined}`);
  } catch (debugErr) {
    console.error(`  [DEBUG] Combined debug upload failed (non-fatal): ${debugErr.message}`);
  }

  return { path: combinedPath, endingZoom: continuationResult.endingZoom };
}

// ── KLING GENERATION ──────────────────────────────────────────────────

async function generateKlingClip(frame, workDir) {
  ensureConfigured();
  enforceScopeRules(frame);

  const prompt = buildPrompt(frame);

  // Kling's API only accepts whole-second duration values (3-15), not
  // decimals. Our room-type defaults (e.g. 5.5s for "living") are tuned
  // for FFmpeg/Ken Burns and need to be rounded for Kling specifically —
  // this caused a 422 "Unprocessable Entity" the first time we tested.
  const rawDuration = frame.durationSeconds || 5;
  const roundedDuration = Math.min(15, Math.max(3, Math.round(rawDuration)));
  const duration = String(roundedDuration);

  console.log(`  [Kling] Submitting job — room: ${frame.roomType}, duration: ${duration}s (requested ${rawDuration}s)`);

  const KLING_ENDPOINT = "fal-ai/kling-video/o3/standard/image-to-video";

  // Submitting explicitly via queue.submit() + our own polling loop, rather
  // than fal.subscribe()'s blocking internal poll. fal.subscribe() submits
  // and waits in one opaque call — if anything in that internal loop dies
  // silently (dropped connection, missed status transition), there's no
  // visibility and no error ever surfaces; the awaited promise just never
  // resolves or rejects. Confirmed twice in testing: Kling completed
  // successfully on fal.ai's own dashboard both times, but the Railway
  // process never logged anything past job submission — no completion, no
  // [Kling] failure fallback, no top-level "Job failed" catch. Total
  // silence with no JS-catchable error means the process likely wasn't
  // failing in JS at all; explicit polling gives us a log line every
  // attempt, so a future stall shows up as a clear gap at a known interval
  // instead of vanishing without a trace.
  const { request_id } = await fal.queue.submit(KLING_ENDPOINT, {
    input: {
      image_url: forceCloudinary16x9(frame.imageUrl),
      end_image_url: frame.endImageUrl ? forceCloudinary16x9(frame.endImageUrl) : undefined,
      prompt,
      duration,
      generate_audio: false, // Mubert handles music separately — avoid conflicting audio tracks
    },
  });

  console.log(`  [Kling] Queued — request_id: ${request_id} (save this — recoverable via fal.ai dashboard even if this process dies)`);

  const POLL_INTERVAL_MS = 10000;
  const MAX_POLL_ATTEMPTS = 90; // 90 * 10s = 15 minute ceiling — well above observed real Kling generation time

  let finalStatus = null;
  for (let attempt = 1; attempt <= MAX_POLL_ATTEMPTS; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));

    const status = await fal.queue.status(KLING_ENDPOINT, {
      requestId: request_id,
      logs: true,
    });

    console.log(`  [Kling] Poll ${attempt}/${MAX_POLL_ATTEMPTS} — status: ${status.status}`);
    if (status.status === "IN_PROGRESS" && status.logs) {
      status.logs.forEach((log) => console.log(`  [Kling] ${log.message}`));
    }

    if (status.status === "COMPLETED") {
      finalStatus = status;
      break;
    }
  }

  if (!finalStatus) {
    throw new Error(
      `Kling request ${request_id} did not complete within ${(MAX_POLL_ATTEMPTS * POLL_INTERVAL_MS) / 60000} minutes of polling. Check this request_id directly on the fal.ai dashboard — the generation may have finished even if polling here gave up.`
    );
  }

  const result = await fal.queue.result(KLING_ENDPOINT, { requestId: request_id });

  const videoUrl = result.data?.video?.url;
  if (!videoUrl) {
    throw new Error("Kling returned no video URL");
  }

  console.log(`  [Kling] Generation complete — downloading clip from fal.ai`);

  // Download the generated clip to local disk so it can flow into the
  // same assembleVideo() pipeline as Ken Burns clips — from this point
  // forward, the rest of the pipeline doesn't know or care whether a
  // clip came from FFmpeg or Kling.
  const outputPath = path.join(workDir, `kling_${Date.now()}.mp4`);
  const response = await axios.get(videoUrl, { responseType: "arraybuffer" });
  fs.writeFileSync(outputPath, response.data);

  console.log(`  [Kling] Clip downloaded to ${outputPath} (${response.data.length} bytes)`);

  return outputPath;
}

// ── ENTRY POINT WITH FALLBACK ────────────────────────────────────────
// AI motion is a premium enhancement, not a hard dependency. Any failure
// (API down, scope rejection, fal.ai account issue) falls back to the
// proven Ken Burns path rather than failing the entire video job.

async function applyKlingMotion(frame, workDir, fallbackFn) {
  let clipPath;

  // Kling generation has its own try/catch — if THIS fails, we have no
  // usable clip at all, so falling back to Ken Burns from scratch is
  // correct here.
  try {
    clipPath = await generateKlingClip(frame, workDir);
    console.log(`  [Kling] Clip ready: ${clipPath}`);
  } catch (err) {
    console.error(`  [Kling] Generation failed, falling back to Ken Burns: ${err.message}`);
    const fallbackResult = await fallbackFn();
    return { ...fallbackResult, source: "ken_burns_fallback" };
  }

  // Continuation motion is a SEPARATE try/catch. If Kling already
  // succeeded (real money already spent, real working clip in hand),
  // a failure in the continuation step should never discard that —
  // it should just skip the continuation and return the Kling clip as-is.
  let endingZoom = 1.0;
  if (frame.addContinuationMotion) {
    try {
      const continued = await applyContinuationMotion(clipPath, frame, workDir);
      clipPath = continued.path;
      endingZoom = continued.endingZoom;
    } catch (err) {
      console.error(`  [Kling] Continuation motion failed, using Kling clip without it: ${err.message}`);
      // clipPath stays as the successful Kling-only result — not discarded.
    }
  }

  return { path: clipPath, source: "kling", endingZoom };
}

module.exports = {
  applyKlingMotion,
  generateKlingClip,
  enforceScopeRules,
  buildPrompt,
  extractLastFrame,
  KLING_MOTION_TEMPLATES,
  VALID_KLING_PRESETS,
  SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS,
};
