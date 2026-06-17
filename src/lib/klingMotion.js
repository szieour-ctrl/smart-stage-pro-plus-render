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
const { fal } = require("@fal-ai/client");

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
// Hard rule, not a default: interior AI motion REQUIRES both a start
// and end image (real vacant + real staged). A single-image interior
// request is rejected outright rather than silently falling back —
// callers should be using Ken Burns for that case in the first place.
// Exterior is more permissive since there's no fixed interior
// architecture at risk of being misrepresented.

function enforceScopeRules(frame) {
  const isInterior = !["exterior"].includes(frame.roomType);

  if (isInterior && !frame.endImageUrl) {
    throw new Error(
      `Kling AI motion rejected: interior room type "${frame.roomType}" requires both a start and end image (vacant + staged). Single-image AI motion on interiors is disabled by design — see AB 723 scope restriction in klingMotion.js.`
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

// ── KLING GENERATION ──────────────────────────────────────────────────

async function generateKlingClip(frame, workDir) {
  ensureConfigured();
  enforceScopeRules(frame);

  const prompt = buildPrompt(frame);
  const duration = String(frame.durationSeconds || 5);

  console.log(`  [Kling] Submitting job — room: ${frame.roomType}, duration: ${duration}s`);

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
    const clipPath = await generateKlingClip(frame, workDir);
    return { path: clipPath, source: "kling", endingZoom: 1.0 };
  } catch (err) {
    console.error(`  [Kling] Failed, falling back to Ken Burns: ${err.message}`);
    const fallbackResult = await fallbackFn();
    return { ...fallbackResult, source: "ken_burns_fallback" };
  }
}

module.exports = { applyKlingMotion, generateKlingClip, enforceScopeRules, buildPrompt };
