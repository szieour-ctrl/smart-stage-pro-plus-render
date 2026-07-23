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
// Set END_FRAME_ENABLED=false (or leave unset) in Railway's environment
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
// REVISED AGAIN (this session, real render evidence): asking Flux to
// AUTHOR the address/CTA text from a written prompt produced a real
// spelling error ("Schednle" instead of "Schedule") — text generation
// inside an image-diffusion edit isn't reliable the way a real font
// renderer is. Sam's fix: stop asking Flux to write the words at all.
// Render the exact text deterministically (ffmpeg drawtext — same
// technique already proven in assemble.js's dormant renderClosingCardClip,
// reused here rather than adding a new canvas/PIL dependency) as its own
// transparent-background PNG, then hand FLUX TWO images — the photo and
// the text overlay — and ask it to blend/light the overlay onto the
// photo, not write new characters. Flux still does what it's actually
// good at (the lighting integration, glow, cinematic polish Sam wanted —
// the whole reason this isn't just a flat Sharp/canvas composite), it
// just never touches spelling.
//
// Sam's second catch: text sitting right at the image's perimeter is
// exactly where a blending model's fidelity is weakest (edge regions get
// least faithful treatment in most diffusion blending). The overlay is
// positioned in the image's MIDDLE THIRD vertically instead of hugging
// the bottom edge — see OVERLAY_BAND_* constants below — specifically to
// keep it out of that higher-risk zone.
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
// Set END_FRAME_ENABLED=false (or leave unset) in Railway's environment
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
// Any failure at any step (overlay render, Cloudinary upload, fal call,
// download) falls back to the ORIGINAL frame, untouched — the render
// always completes with the plain closing shot rather than failing the
// whole job. Same "fail loud, not fail hard" principle already used for
// Kling/LTX fallbacks in renderPipeline.js.
//
// ── REQUIRES ─────────────────────────────────────────────────────────
// - Railway env vars: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY,
//   CLOUDINARY_API_SECRET (already used elsewhere in this codebase),
//   FAL_KEY (already confirmed present), END_FRAME_ENABLED (new — kill
//   switch, add when ready to test).
// - No new npm dependencies — cloudinary, @fal-ai/client, and
//   fluent-ffmpeg (for the overlay PNG) are already in package.json.

const fs = require("fs");
const path = require("path");
const axios = require("axios");
const cloudinary = require("cloudinary").v2;
const { fal } = require("@fal-ai/client");
const ffmpeg = require("fluent-ffmpeg");

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

// Same escaping assemble.js's escapeDrawtext already uses for the dormant
// closing-card system — duplicated here (not exported there) rather than
// pulling this whole module into a cross-file dependency for four lines.
function escapeDrawtext(text) {
  return text.replace(/\\/g, "\\\\\\\\").replace(/:/g, "\\:").replace(/'/g, "\u2019");
}

// ── OVERLAY BAND POSITION — Sam's call: middle third vertically, not
// hugging the bottom edge. Perimeter regions are where a blending model's
// fidelity is weakest, so keeping the text well inside the frame reduces
// (does not eliminate) the risk of Flux softening or shifting it during
// the blend, independent of the zoom-cropping issue (already solved
// separately by forcing motionPreset="static" on this frame downstream).
const OVERLAY_BAND_Y_START_FRAC = 0.35;
const OVERLAY_BAND_HEIGHT_FRAC  = 0.30;
const OVERLAY_CANVAS_W = 1920;
const OVERLAY_CANVAS_H = 1080;

// ── uploadImageToCloudinary — generic upload, used for both the source
// frame and the text-overlay PNG. Folder kept as a param so the two don't
// collide/get confused in the Cloudinary media library.
async function uploadImageToCloudinary(localPath, folder) {
  const result = await cloudinary.uploader.upload(localPath, {
    folder,
    resource_type: "image",
  });
  if (!result || !result.secure_url) {
    throw new Error(`End Frame: Cloudinary upload (${folder}) did not return secure_url`);
  }
  return result.secure_url;
}

// ── renderTextOverlayPng — the address+CTA text, deterministic, exact
// spelling guaranteed (ffmpeg drawtext is a real font renderer, not a
// generative model). Transparent canvas (rgba) so Flux receives a clean
// text card to blend, not a second photo. Same gradient-band-plus-two-
// stacked-drawtext-calls approach already proven in assemble.js's
// renderClosingCardClip — just positioned centrally per Sam's perimeter-
// risk call, and rendered as a single still frame instead of a video clip.
function renderTextOverlayPng(address, ctaText, workDir) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, `end-frame-overlay-${Date.now()}.png`);
    const bandY = Math.round(OVERLAY_CANVAS_H * OVERLAY_BAND_Y_START_FRAC);
    const bandH = Math.round(OVERLAY_CANVAS_H * OVERLAY_BAND_HEIGHT_FRAC);

    const filterParts = [
      // Transparent canvas, not a second photo — Flux blends this onto
      // the real frame, so anything outside the gradient band must stay
      // fully transparent. Explicit bracket labels throughout — same
      // filter_complex convention already proven in assemble.js's
      // renderClosingCardClip, not a new pattern.
      `[0:v]format=rgba[fmted]`,
      `[fmted]drawbox=x=0:y=${bandY}:w=${OVERLAY_CANVAS_W}:h=${bandH}:color=black@0.6:t=fill[boxed]`,
      `[boxed]drawtext=text='${escapeDrawtext(address)}':fontcolor=white:fontsize=54:borderw=2:bordercolor=black@0.6:x=(w-text_w)/2:y=${bandY + Math.round(bandH * 0.32)}[addrtext]`,
      `[addrtext]drawtext=text='${escapeDrawtext(ctaText)}':fontcolor=white:fontsize=40:borderw=2:bordercolor=black@0.6:x=(w-text_w)/2:y=${bandY + Math.round(bandH * 0.62)}[outv]`,
    ];

    const OVERLAY_TIMEOUT_MS = 20000;
    let settled = false;
    const command = ffmpeg()
      .input(`color=c=black@0.0:s=${OVERLAY_CANVAS_W}x${OVERLAY_CANVAS_H}`)
      .inputOptions(["-f", "lavfi"])
      .complexFilter(filterParts);

    const timeoutHandle = setTimeout(() => {
      if (settled) return;
      settled = true;
      console.warn(`[endFrame] Overlay PNG render timed out after ${OVERLAY_TIMEOUT_MS}ms — killing process.`);
      try { command.kill("SIGKILL"); } catch (err) { /* best-effort */ }
      reject(new Error("End Frame: overlay PNG render timed out"));
    }, OVERLAY_TIMEOUT_MS);

    command
      .outputOptions(["-map", "[outv]", "-frames:v", "1", "-pix_fmt", "rgba"])
      .output(outputPath)
      .on("end", () => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutHandle);
        resolve(outputPath);
      })
      .on("error", (err) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutHandle);
        reject(new Error(`End Frame: overlay PNG render failed: ${err.message}`));
      })
      .run();
  });
}

// ── buildBlendPrompt — REPLACES the old buildCtaPrompt, which asked Flux
// to author the address/CTA text itself from a written description (real
// evidence: produced "Schednle" instead of "Schedule" — text generation
// inside an image edit isn't reliable). Now Flux receives the exact,
// pre-rendered text as a SECOND reference image and is asked only to
// blend/light it — the wording is never in Flux's hands.
//
// END_FRAME_BLEND_TEMPLATE remains an env var so Sam can tune the
// blending/lighting instruction without a deploy; the text itself is now
// controlled separately via END_FRAME_CTA_TEXT (unchanged from before).
function buildBlendPrompt() {
  return process.env.END_FRAME_BLEND_TEMPLATE ||
    'The first image is a real-estate exterior photo. The second image is a ' +
    'text overlay graphic with a dark gradient band and white text on a ' +
    'transparent background. Blend the second image onto the first exactly ' +
    'as composed — do not move, resize, re-letter, or paraphrase any of its ' +
    'text, and do not add, remove, or change any words. Integrate it ' +
    'naturally into the photo\u2019s own lighting: soften the gradient\u2019s edges ' +
    'to feel like part of the scene, add a subtle warm glow consistent with ' +
    'the photo\u2019s existing light sources, and keep the overall look cinematic ' +
    'and premium. Preserve all architectural details, lighting, and ' +
    'landscaping in the photo exactly as captured \u2014 do not modify the ' +
    'home\u2019s structure, colors, or sky.';
}

// ── submitFluxEdit — fal.subscribe() handles submit + poll + fetch
// internally; no manual queue/status/response URL plumbing needed.
//
// CHANGED (this session) — now takes an ARRAY of image URLs (photo +
// overlay) instead of a single one, per Sam's request to use Flux's
// multi-reference support (up to 9 images) for blending rather than
// authoring. image_urls (plural) was already the correct field name from
// the earlier 422 fix — this just adds a second real entry to it instead
// of always being an array-of-one.
async function submitFluxEdit(imageUrls, prompt) {
  const result = await fal.subscribe("fal-ai/flux-2-pro/edit", {
    input: { image_urls: imageUrls, prompt },
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

  const ctaText = process.env.END_FRAME_CTA_TEXT || "Schedule Your Private Showing";
  console.log(`[${jobId}] End Frame: enabled, rendering text overlay for closing frame (${frame.localPath})...`);

  try {
    const overlayLocalPath = await renderTextOverlayPng(address, ctaText, workDir);
    console.log(`[${jobId}] End Frame: overlay PNG rendered (${overlayLocalPath}).`);

    const [sourceUrl, overlayUrl] = await Promise.all([
      uploadImageToCloudinary(frame.localPath, "end-frame-source"),
      uploadImageToCloudinary(overlayLocalPath, "end-frame-overlay"),
    ]);
    console.log(`[${jobId}] End Frame: uploaded source (${sourceUrl}) and overlay (${overlayUrl}) to Cloudinary.`);

    const prompt = buildBlendPrompt();
    const editedUrl = await submitFluxEdit([sourceUrl, overlayUrl], prompt);
    console.log(`[${jobId}] End Frame: Flux blend complete (${editedUrl}), downloading...`);

    const editedLocalPath = path.join(workDir, `end-frame-${Date.now()}.jpg`);
    await downloadToFile(editedUrl, editedLocalPath);

    console.log(`[${jobId}] End Frame applied — closing frame replaced with Flux-blended version.`);
    return { ...frame, localPath: editedLocalPath, endFrameApplied: true };
  } catch (err) {
    console.error(`[${jobId}] End Frame failed (non-fatal — falling back to plain closing frame): ${err.message}`);
    return frame;
  }
}

module.exports = { applyEndFrame, isEndFrameEnabled };
