// lib/endFrame.js — Railway
//
// PRO Plus "End Frame": replaces the LAST frame in a render with an
// AI-edited version (fal-ai/flux-2-pro/edit) that bakes the property's
// street address + a CTA line directly into the closing shot, so the
// final clip does double duty as a title-card without needing a separate
// ffmpeg overlay pass.
//
// REVISED (this session) — original draft hand-rolled a signed Cloudinary
// multipart upload and a raw fal queue-submit/poll loop. package.json
// already lists "cloudinary": "^2.3.0" and "@fal-ai/client": "^1.2.0" as
// dependencies, so this version uses those SDKs directly instead —
// simpler, less surface area for a signing bug, and fal.subscribe()
// already does the submit/poll/fetch cycle internally. No new npm
// dependency needed for this file.
//
// ── WHY THIS LIVES HERE, NOT AS A NETLIFY BACKGROUND FUNCTION ──────────
// autoSelect (see autoSelect-background.js / check-autoSelect.js) needed
// the dispatch/poll split because it runs at PLAN time, before a job
// exists, from a synchronous Netlify function with a hard 26s ceiling.
// End Frame runs DURING the actual render, on Railway, which already has
// no such ceiling and already makes long-running calls in-process (Kling,
// LTX, ElevenLabs, Claude Vision) — so this is just one more step in
// processRenderJob, not a second job type. No Netlify or netlify.toml
// changes are needed for this feature at all.
//
// ── FEATURE FLAG — KILL SWITCH ──────────────────────────────────────────
// Set END_FRAME_ENABLED=true (or leave unset) in Railway's environment
// to disable this instantly — no code change or redeploy needed, just an
// env var flip + restart. Defaults OFF until Sam confirms it's ready to
// leave on for real jobs. When off, applyEndFrame() is a pure no-op that
// returns the original frame unchanged.
//
// ── BILLING ──────────────────────────────────────────────────────────
// Deliberately NOT metered. No Image/video quota debit, no
// narration_attempts-style audit row, nothing shown in the UI as a line
// item. Cost is real (~$0.03-0.05/video at typical listing-photo
// resolution via fal-ai/flux-2-pro/edit's per-megapixel pricing) but
// small enough that Sam is absorbing it as a silent quality upgrade
// rather than adding pricing-page complexity. If this changes later, the
// hook point is here: this function would need to return a cost figure
// for video-job.js to charge at download time, same pattern as Ken
// Burns' flat 1-Image charge.
//
// ── FAILURE HANDLING ─────────────────────────────────────────────────
// Any failure at any step (Cloudinary upload, fal call, download) falls
// back to the ORIGINAL frame, untouched — the render always completes
// with the plain closing shot rather than failing the whole job. Same
// "fail loud, not fail hard" principle already used for Kling/LTX
// fallbacks in renderPipeline.js.
//
// ── REQUIRES ─────────────────────────────────────────────────────────
// - Railway env vars: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY,
//   CLOUDINARY_API_SECRET (already used elsewhere in this codebase),
//   FAL_KEY (already confirmed present), END_FRAME_ENABLED (new — kill
//   switch, add when ready to test).
// - No new npm dependencies — cloudinary and @fal-ai/client are already
//   in package.json.

const fs = require("fs");
const path = require("path");
const axios = require("axios");
const cloudinary = require("cloudinary").v2;
const { fal } = require("@fal-ai/client");

// Idempotent — safe to call even if another module (lib/cloudinaryUpload.js)
// already configured cloudinary elsewhere in this process.
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key:    process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

fal.config({ credentials: process.env.FAL_KEY });

function isEndFrameEnabled() {
  return String(process.env.END_FRAME_ENABLED || "").toLowerCase() === "true";
}

// ── uploadFrameToCloudinary — returns a public secure_url Flux can fetch.
async function uploadFrameToCloudinary(localPath) {
  const result = await cloudinary.uploader.upload(localPath, {
    folder: "end-frame-source",
    resource_type: "image",
  });
  if (!result || !result.secure_url) {
    throw new Error("End Frame: Cloudinary upload did not return secure_url");
  }
  return result.secure_url;
}

// ── buildCtaPrompt — full address (street + city/state), unlike the
// narration CTA which deliberately drops city/state for spoken-audio
// reasons (awkward to hear read aloud). That rule doesn't apply here —
// this is TEXT ON AN IMAGE, meant to be read, not heard.
//
// REVISED (this session, second pass) — Sam ran a hand-written prompt
// directly in the fal.ai playground and confirmed it reliably produces
// both the address AND the CTA line, with the gradient/typography look
// he wants. That confirmed-working prompt is now the default template
// below, verbatim in structure, just parameterized for {address} and
// {cta}. The earlier "two lines, first line/second line" phrasing this
// replaces was dropping the CTA line in testing — this version's much
// more explicit gradient-opacity math and typography description is
// almost certainly why it holds up where the shorter version didn't.
//
// END_FRAME_CTA_TEMPLATE and END_FRAME_CTA_TEXT remain env vars so Sam
// can tune either the full prompt or just the CTA copy without a deploy.
function buildCtaPrompt(address) {
  const ctaText = process.env.END_FRAME_CTA_TEXT || "Schedule Your Private Showing";
  const template = process.env.END_FRAME_CTA_TEMPLATE ||
    'Enhance this real-estate exterior photo with a professional marketing overlay. ' +
    'Preserve all architectural details, lighting, and landscaping exactly as captured. ' +
    'Do not modify the home\u2019s structure, colors, or sky. ' +
    'Add a soft dark gradient across the lower third of the image, fading upward ' +
    'from roughly 80% opacity at the bottom to 0% at mid-frame. ' +
    'Keep the gradient smooth and cinematic, ensuring natural light remains visible. ' +
    'Overlay clean, modern sans-serif text centered within the gradient area: ' +
    'Line 1: "{address}" \u2014 bold, white, crisp edges, balanced spacing. ' +
    'Line 2: "{cta}" \u2014 regular weight, white, slightly smaller, ' +
    'with refined letter spacing for a premium look. ' +
    'Typography should appear professional and contemporary, evoking high-end ' +
    'real-estate branding. Avoid decorative or serif styles. ' +
    'Maintain full-frame composition and natural color grading. ' +
    'Apply subtle warmth and contrast for a polished, inviting finish.';
  return template.replace("{address}", address || "This Home").replace("{cta}", ctaText);
}

// ── submitFluxEdit — fal.subscribe() handles submit + poll + fetch
// internally; no manual queue/status/response URL plumbing needed.
//
// FIX (this session — real 422 "Unprocessable Entity" from a live test):
// flux-2-pro/edit's actual input schema takes the source image as
// image_urls (a PLURAL array), not image_url (a singular string) — this
// model supports up to 9 reference images per its multi-reference design,
// so even a single-image edit still goes through the array field.
// Confirmed against fal's own published API example for this exact model.
// The original singular image_url key silently doesn't match any known
// field, which is exactly what surfaces as a 422 rather than a clearer
// "unknown parameter" message.
async function submitFluxEdit(imageUrl, prompt) {
  const result = await fal.subscribe("fal-ai/flux-2-pro/edit", {
    input: { image_urls: [imageUrl], prompt },
    logs: false,
  });
  const outUrl = result?.data?.images?.[0]?.url;
  if (!outUrl) {
    throw new Error(`End Frame: fal edit returned no image URL (${JSON.stringify(result?.data)})`);
  }
  return outUrl;
}

function downloadToFile(url, destPath) {
  return axios.get(url, { responseType: "stream", timeout: 30000 }).then(
    (res) =>
      new Promise((resolve, reject) => {
        const writer = fs.createWriteStream(destPath);
        res.data.pipe(writer);
        writer.on("finish", resolve);
        writer.on("error", reject);
      })
  );
}

// ── applyEndFrame — the only export most callers need.
// frame: one entry from localFrames (must have .localPath — the file
//        downloadFrames.js already pulled to disk).
// address: job.address (already fetched by video-job.js when narration
//        is on — same field, reused here; End Frame can run independently
//        of narration, so pass whatever address is available, may be null).
// workDir: the job's temp working directory (edited file is written here
//        so it gets cleaned up by processRenderJob's existing finally
//        block — no separate cleanup needed).
// jobId: for logging only.
//
// Returns a frame object — either the original (flag off, missing data,
// or any failure) or a shallow copy with .localPath repointed at the
// Flux-edited image and .endFrameApplied: true (informational, not read
// by billing since this is unmetered).
async function applyEndFrame({ frame, address, workDir, jobId }) {
  if (!isEndFrameEnabled()) {
    console.log(`[${jobId}] End Frame: skipped — END_FRAME_ENABLED is not "true" in this environment.`);
    return frame;
  }
  if (!frame || !frame.localPath) {
    console.log(`[${jobId}] End Frame: skipped — no frame/localPath available to edit.`);
    return frame;
  }
  // NEW (this session — real gap found via live test): a null/empty
  // address used to fall through to a generic "This Home" placeholder
  // baked into the actual closing frame. Now skips entirely instead —
  // video-job.js was fixed to always fetch the real address (previously
  // gated behind wantsNarration), so hitting this path now should be
  // rare and worth knowing about via a clear log line, not a silently
  // wrong result on a real customer's video.
  if (!address) {
    console.log(`[${jobId}] End Frame: skipped — no address available (this shouldn't happen after the video-job.js fix; check the calling job's address field).`);
    return frame;
  }

  console.log(`[${jobId}] End Frame: enabled, starting Flux edit on closing frame (${frame.localPath})...`);

  try {
    const sourceUrl = await uploadFrameToCloudinary(frame.localPath);
    console.log(`[${jobId}] End Frame: uploaded source frame to Cloudinary (${sourceUrl}).`);
    const prompt = buildCtaPrompt(address);
    const editedUrl = await submitFluxEdit(sourceUrl, prompt);
    console.log(`[${jobId}] End Frame: Flux edit complete (${editedUrl}), downloading...`);

    const editedLocalPath = path.join(workDir, `end-frame-${Date.now()}.jpg`);
    await downloadToFile(editedUrl, editedLocalPath);

    console.log(`[${jobId}] End Frame applied — closing frame replaced with Flux-edited version.`);
    return { ...frame, localPath: editedLocalPath, endFrameApplied: true };
  } catch (err) {
    console.error(`[${jobId}] End Frame failed (non-fatal — falling back to plain closing frame): ${err.message}`);
    return frame;
  }
}

module.exports = { applyEndFrame, isEndFrameEnabled };
