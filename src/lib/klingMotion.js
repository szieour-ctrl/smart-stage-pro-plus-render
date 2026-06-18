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

// ── CONTINUATION MOTION ──────────────────────────────────────────────
// After Kling's transformation finishes (vacant becomes staged), the
// clip ends on a static-feeling final frame. This stitches on a few
// extra seconds of Ken Burns motion (push-in, pull-back, float, pan)
// starting from the REAL staged image — not an extracted video frame,
// which would risk compounding any minor artifacts from the Kling
// generation itself. Uses the proven, already-tested Ken Burns engine
// rather than trusting Kling's own multi-shot prompting to get a second
// camera move right.

function concatTwoClips(firstPath, secondPath, workDir, outputName) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, outputName);

    // Simple hard cut, not a crossfade — Kling's clip already ends on
    // (approximately) the staged image, and the continuation clip starts
    // from that same staged image, so the cut should read as nearly
    // seamless without needing a transition effect to hide a mismatch.
    ffmpeg()
      .input(firstPath)
      .input(secondPath)
      .complexFilter([
        "[0:v]setpts=PTS-STARTPTS[v0]",
        "[1:v]setpts=PTS-STARTPTS[v1]",
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
  // Continuation always uses the real staged image as its source — frame
  // here refers to the original frame object, which already has the
  // local downloaded staged image path available from downloadFrames.js.
  // Default to luxury_parallax — push_in immediately after Kling's own
  // push-in transformation would feel repetitive (same motion twice in a
  // row). A slow parallax drift gives the second beat real contrast and
  // matches the validated "Push → Transform → Parallax" sequence design.
  const continuationPreset = frame.continuationPreset || "luxury_parallax";
  const continuationDuration = frame.continuationDurationSeconds || 3;

  console.log(`  [Kling] Adding ${continuationDuration}s continuation motion (${continuationPreset})`);

  const continuationResult = await applyMotionPreset(
    {
      localPath: frame.localPath, // the real staged image, downloaded locally
      motionPreset: continuationPreset,
      durationSeconds: continuationDuration,
    },
    workDir,
    1.0
  );

  const combinedPath = await concatTwoClips(
    klingClipPath,
    continuationResult.path,
    workDir,
    `kling_continued_${Date.now()}.mp4`
  );

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

  const result = await fal.subscribe("fal-ai/kling-video/o3/standard/image-to-video", {
    input: {
      image_url: frame.imageUrl,
      end_image_url: frame.endImageUrl || undefined,
      prompt,
      duration,
      generate_audio: false, // Mubert handles music separately — avoid conflicting audio tracks
    },
    logs: true,
    onQueueUpdate: (update) => {
      if (update.status === "IN_PROGRESS") {
        update.logs?.forEach((log) => console.log(`  [Kling] ${log.message}`));
      }
    },
  });

  const videoUrl = result.data?.video?.url;
  if (!videoUrl) {
    throw new Error("Kling returned no video URL");
  }

  // Download the generated clip to local disk so it can flow into the
  // same assembleVideo() pipeline as Ken Burns clips — from this point
  // forward, the rest of the pipeline doesn't know or care whether a
  // clip came from FFmpeg or Kling.
  const outputPath = path.join(workDir, `kling_${Date.now()}.mp4`);
  const response = await axios.get(videoUrl, { responseType: "arraybuffer" });
  fs.writeFileSync(outputPath, response.data);

  return outputPath;
}

// ── ENTRY POINT WITH FALLBACK ────────────────────────────────────────
// AI motion is a premium enhancement, not a hard dependency. Any failure
// (API down, scope rejection, fal.ai account issue) falls back to the
// proven Ken Burns path rather than failing the entire video job.

async function applyKlingMotion(frame, workDir, fallbackFn) {
  try {
    let clipPath = await generateKlingClip(frame, workDir);
    let endingZoom = 1.0;

    // Optional: continue with Ken Burns motion after Kling's transformation
    // settles, so the video doesn't go static the instant staging finishes.
    if (frame.addContinuationMotion) {
      const continued = await applyContinuationMotion(clipPath, frame, workDir);
      clipPath = continued.path;
      endingZoom = continued.endingZoom;
    }

    return { path: clipPath, source: "kling", endingZoom };
  } catch (err) {
    console.error(`  [Kling] Failed, falling back to Ken Burns: ${err.message}`);
    const fallbackResult = await fallbackFn();
    return { ...fallbackResult, source: "ken_burns_fallback" };
  }
}

module.exports = { applyKlingMotion, generateKlingClip, enforceScopeRules, buildPrompt };
