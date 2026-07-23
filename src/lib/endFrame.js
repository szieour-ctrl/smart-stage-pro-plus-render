// lib/endFrame.js — Railway
//
// PRO Plus "End Frame": replaces the LAST frame in a render with a
// version that has the property's street address + a CTA line baked
// directly into the closing shot, so the final clip does double duty as
// a title-card.
//
// ── HISTORY (this session — three real approaches tried, in order) ────
// 1. Flux (fal-ai/flux-2-pro/edit) authoring the text itself from a
//    written prompt. Real render evidence: produced a spelling error
//    ("Schednle" instead of "Schedule") — text generation inside an
//    image-diffusion edit isn't reliable the way a real font renderer is.
// 2. Flux given the EXACT text as a second reference image (multi-
//    reference blend, Flux's own supported mode) and asked only to
//    blend/light it, not author it. Real render evidence: STILL
//    corrupted the text — word duplication this time ("Bnt Bent",
//    "Your You") instead of letter substitution. Different failure
//    mode, same root cause: Flux regenerates whatever's in the text
//    region through its own diffusion process regardless of whether the
//    words came from a prompt or a reference image — there's no actual
//    preserve-exactly mode being invoked either way.
// 3. THIS VERSION (Sam's call after two independent failures): Flux is
//    removed from the text step entirely. The address/CTA overlay is
//    rendered deterministically via ffmpeg drawtext (a real font
//    renderer — same technique already proven in assemble.js's dormant
//    renderClosingCardClip, reused here rather than adding a new canvas/
//    PIL dependency) and composited directly onto the photo with
//    ffmpeg's overlay filter — a true alpha composite, not a generative
//    pass. Guarantees exact, correct text every time. Costs the
//    "lighting woven into the text" polish Flux was adding (the
//    motivating reason for trying Flux at all — see prior session
//    notes), in exchange for the address never being wrong. If Sam wants
//    photo-level mood/lighting enhancement back later, that's a
//    separate, lower-stakes Flux call on the PHOTO ONLY (no text
//    involved) that could run before this compositing step — not built
//    here since it wasn't asked for, just noting the door is open.
//
// No fal.ai or Cloudinary round-trip needed anymore for this feature —
// everything happens locally via ffmpeg, which is faster and removes an
// entire class of network/API failure modes this file used to have to
// handle.
//
// ── WHY THIS LIVES HERE, NOT AS A NETLIFY BACKGROUND FUNCTION ──────────
// End Frame runs DURING the actual render, on Railway, which already
// makes long-running calls in-process (Kling, LTX, ElevenLabs, Claude
// Vision) — so this is just one more step in processRenderJob, not a
// second job type. No Netlify or netlify.toml changes needed.
//
// ── FEATURE FLAG — KILL SWITCH ──────────────────────────────────────────
// Set END_FRAME_ENABLED=false (or leave unset) in Railway's environment
// to disable this instantly — no code change or redeploy needed, just an
// env var flip + restart. Defaults OFF until Sam confirms it's ready to
// leave on for real jobs. When off, applyEndFrame() is a pure no-op that
// returns the original frame unchanged.
//
// ── BILLING ──────────────────────────────────────────────────────────
// Deliberately NOT metered — no Image/video quota debit, no audit row,
// nothing shown in the UI as a line item. Now that there's no fal.ai call
// at all, the real cost of this feature is effectively zero (just local
// ffmpeg CPU time already being spent on every other clip in the render).
//
// ── FAILURE HANDLING ─────────────────────────────────────────────────
// Any failure (overlay render, composite) falls back to the ORIGINAL
// frame, untouched — the render always completes with the plain closing
// shot rather than failing the whole job. Same "fail loud, not fail
// hard" principle already used for Kling/LTX fallbacks in
// renderPipeline.js.
//
// ── REQUIRES ─────────────────────────────────────────────────────────
// - Railway env var: END_FRAME_ENABLED (kill switch, add when ready to
//   test). CLOUDINARY_*/FAL_KEY are no longer needed by this file.
// - No new npm dependencies — fluent-ffmpeg is already in package.json.

const fs = require("fs");
const path = require("path");
const ffmpeg = require("fluent-ffmpeg");

function isEndFrameEnabled() {
  return String(process.env.END_FRAME_ENABLED || "").toLowerCase() === "true";
}

// Same escaping assemble.js's escapeDrawtext already uses for the dormant
// closing-card system — duplicated here (not exported there) rather than
// pulling this whole module into a cross-file dependency for four lines.
function escapeDrawtext(text) {
  return text.replace(/\\/g, "\\\\\\\\").replace(/:/g, "\\:").replace(/'/g, "\u2019");
}

// ── OVERLAY BAND POSITION — kept in the middle third vertically (not
// hugging the bottom edge), left over from the Flux-blend attempt where
// this mattered for a different reason (perimeter regions had weaker
// blend fidelity). That specific reason no longer applies now that
// there's no blending at all — this is a plain, position-agnostic
// composite — but the middle-third placement still reads fine visually,
// so it's left as-is rather than moved without a reason to. Easy to
// change if Sam wants a more traditional bottom-anchored placement.
const OVERLAY_BAND_Y_START_FRAC = 0.35;
const OVERLAY_BAND_HEIGHT_FRAC  = 0.30;
const OVERLAY_CANVAS_W = 1920;
const OVERLAY_CANVAS_H = 1080;

// ── renderTextOverlayPng — the address+CTA text, deterministic, exact
// spelling guaranteed (ffmpeg drawtext is a real font renderer, not a
// generative model). Transparent canvas (rgba) so it composites cleanly
// onto the photo. Same gradient-band-plus-two-drawtext-calls approach
// already proven in assemble.js's renderClosingCardClip, just rendered
// as a single still frame instead of a video clip.
function renderTextOverlayPng(address, ctaText, workDir) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, `end-frame-overlay-${Date.now()}.png`);
    const bandY = Math.round(OVERLAY_CANVAS_H * OVERLAY_BAND_Y_START_FRAC);
    const bandH = Math.round(OVERLAY_CANVAS_H * OVERLAY_BAND_HEIGHT_FRAC);

    const filterParts = [
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

// ── compositeOverlayOntoPhoto — the entire "edit" step now. A true alpha
// composite via ffmpeg's overlay filter: the transparent-background text
// PNG is laid directly on top of the photo's own pixels, unchanged
// everywhere outside the gradient band. No model in the loop, so there is
// nothing left that can misspell, duplicate, or shift the text — it's
// exactly the PNG renderTextOverlayPng() produced, every time.
//
// FIX (this session — real render: text appeared tiny, badly placed, and
// flickered during playback). Root cause: the overlay PNG is rendered at
// a fixed 1920x1080 canvas, but the source photo downloadFrames.js pulls
// down is at its NATIVE resolution — often much larger (Sam's own test
// image earlier this session was 6000x3996). The old overlay call
// composited at fixed pixel coordinates 0:0 with no scaling step at all,
// so on a 6000px-wide photo the 1920px overlay only covered a small
// corner of the frame — "small" and "bad location" are the same bug.
// Thin text rendered at that tiny a fraction of the real frame size is
// also exactly the kind of fine, high-frequency detail that shimmers
// under video compression, which explains the flicker too — almost
// certainly one root cause producing all three symptoms, not three
// separate bugs. Fixed with ffmpeg's scale2ref filter, which scales the
// overlay to match the SECOND input's real dimensions in one step — no
// separate ffprobe call needed to learn the photo's size first.
function compositeOverlayOntoPhoto(photoPath, overlayPath, workDir) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, `end-frame-${Date.now()}.jpg`);
    const COMPOSITE_TIMEOUT_MS = 20000;
    let settled = false;

    const command = ffmpeg()
      .input(photoPath)
      .input(overlayPath)
      .complexFilter([
        // scale2ref: input [1:v] (overlay) scaled to match [0:v]'s (photo)
        // real width/height, output labeled [ovl] — the photo itself
        // passes through unchanged as [ref].
        `[1:v][0:v]scale2ref=w=iw:h=ih[ovl][ref]`,
        `[ref][ovl]overlay=0:0[outv]`,
      ]);

    const timeoutHandle = setTimeout(() => {
      if (settled) return;
      settled = true;
      console.warn(`[endFrame] Composite timed out after ${COMPOSITE_TIMEOUT_MS}ms — killing process.`);
      try { command.kill("SIGKILL"); } catch (err) { /* best-effort */ }
      reject(new Error("End Frame: composite timed out"));
    }, COMPOSITE_TIMEOUT_MS);

    command
      .outputOptions(["-map", "[outv]", "-frames:v", "1", "-q:v", "2"])
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
        reject(new Error(`End Frame: composite failed: ${err.message}`));
      })
      .run();
  });
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
// composited image and .endFrameApplied: true (informational, not read
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
  // A null/empty address used to fall through to a generic "This Home"
  // placeholder baked into the actual closing frame. Skips entirely
  // instead — video-job.js was fixed to always fetch the real address
  // (previously gated behind wantsNarration), so hitting this path now
  // should be rare and worth knowing about via a clear log line, not a
  // silently wrong result on a real customer's video.
  if (!address) {
    console.log(`[${jobId}] End Frame: skipped — no address available (this shouldn't happen after the video-job.js fix; check the calling job's address field).`);
    return frame;
  }

  const ctaText = process.env.END_FRAME_CTA_TEXT || "Schedule Your Private Showing";
  console.log(`[${jobId}] End Frame: enabled, rendering text overlay for closing frame (${frame.localPath})...`);

  try {
    const overlayLocalPath = await renderTextOverlayPng(address, ctaText, workDir);
    console.log(`[${jobId}] End Frame: overlay PNG rendered (${overlayLocalPath}).`);

    const compositedPath = await compositeOverlayOntoPhoto(frame.localPath, overlayLocalPath, workDir);
    console.log(`[${jobId}] End Frame applied — closing frame composited with exact text overlay (${compositedPath}), no AI involved in the text.`);

    return { ...frame, localPath: compositedPath, endFrameApplied: true };
  } catch (err) {
    console.error(`[${jobId}] End Frame failed (non-fatal — falling back to plain closing frame): ${err.message}`);
    return frame;
  }
}

module.exports = { applyEndFrame, isEndFrameEnabled };
