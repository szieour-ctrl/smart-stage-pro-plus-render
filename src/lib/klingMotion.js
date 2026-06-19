// klingMotion.js — AI-generated camera motion via Kling (through fal.ai),
// using the official @fal-ai/client SDK rather than raw HTTP requests.
//
// IMPORTANT — SCOPE RESTRICTION (read before modifying):
// This module is intentionally restricted to two use cases validated
// against real Smart Stage PRO listings:
//   1. Interior vacant→staged interpolation (start/end frame, same room)
//   2. Exterior day→twilight / landscape enhancement (start/end frame)
//
// AI motion is NEVER applied to a single still image with no end frame
// for interior rooms — that would mean the model is inventing unseen
// architecture rather than interpolating between two real, disclosed
// images, which is an AB 723 compliance risk. See enforceScopeRules()
// below, which is a hard runtime check, not just a comment.
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
// Kling has two REAL, KNOWN endpoints to interpolate between, or has to
// INVENT content because only one image exists.
//
//   - Vacant + Staged pair (from Smart Stage PRO staging) → both endpoints
//     are real, disclosed images. Kling fills in the transition between
//     two known states. Low risk, AI Motion freely available here.
//
//   - A single photo with no pair (e.g. professional photography uploaded
//     without a staged counterpart) → Kling has nothing to interpolate
//     toward. It must invent what a camera move reveals — what's beyond
//     the frame, what an unseen angle looks like. This is the exact
//     unconstrained-inference risk VideoTour.ai's own docs describe
//     causing hallucinated objects/architecture.
//
// Rule: ANY interior frame without both a start and end image is rejected
// outright, regardless of room type. Exterior is the only case where a
// single image is permitted, since there's no fixed interior architecture
// at risk — landscaping/sky have more tolerance for invented detail than
// a room's walls and layout do. Even then, this is a judgment call worth
// revisiting, not an assumption that exteriors are risk-free.

function enforceScopeRules(frame) {
  const hasKnownPair = !!frame.endImageUrl;
  const isExterior = frame.roomType === "exterior";

  if (!hasKnownPair && !isExterior) {
    throw new Error(
      `Kling AI motion rejected: no end image provided for room type "${frame.roomType}". AI motion on a single interior photo requires Kling to invent unseen content (architecture, layout) rather than interpolate between two known images — this is disabled by design. Use a vacant+staged pair, or use Ken Burns for single-image interior shots. See AB 723 scope restriction in klingMotion.js.`
    );
  }
}

// ── PROMPT TEMPLATES ──────────────────────────────────────────────────
// Kept separate per use case since the framing differs meaningfully —
// interior is about furniture appearing, exterior is about lighting/
// landscape transformation with the structure held fixed.

function buildPrompt(frame) {
  const isInterior = !["exterior"].includes(frame.roomType);

  if (isInterior) {
    return (
      frame.customPrompt ||
      "Smooth cinematic push-in camera movement through an empty room as furniture and decor gradually appear, room becomes fully furnished and staged, photorealistic, no distortion, stable architecture, walls and windows remain fixed"
    );
  }

  return (
    frame.customPrompt ||
    "Smooth cinematic camera movement across the exterior as lighting and landscaping gradually transform and improve, photorealistic, no distortion, house structure and architecture remain completely fixed and unchanged"
  );
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
        `[0:v]scale=${OUTPUT_W}:${OUTPUT_H}:force_original_aspect_ratio=decrease,pad=${OUTPUT_W}:${OUTPUT_H}:(ow-iw)/2:(oh-ih)/2,fps=20,format=yuv420p,setpts=PTS-STARTPTS[v0]`,
        `[1:v]scale=${OUTPUT_W}:${OUTPUT_H}:force_original_aspect_ratio=decrease,pad=${OUTPUT_W}:${OUTPUT_H}:(ow-iw)/2:(oh-ih)/2,fps=20,format=yuv420p,setpts=PTS-STARTPTS[v1]`,
        "[v0][v1]concat=n=2:v=1:a=0[outv]",
      ])
      .outputOptions(["-map", "[outv]", "-pix_fmt", "yuv420p"])
      .output(outputPath)
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
      image_url: frame.imageUrl,
      end_image_url: frame.endImageUrl || undefined,
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

module.exports = { applyKlingMotion, generateKlingClip, enforceScopeRules, buildPrompt, extractLastFrame };
