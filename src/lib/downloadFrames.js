// downloadFrames.js — Downloads Cloudinary image URLs to local disk
// This module is fully functional today — no external API key required.

const fs = require("fs");
const path = require("path");
const axios = require("axios");

async function downloadFrames(frames, workDir) {
  const localFrames = [];

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    const ext = path.extname(new URL(frame.imageUrl).pathname) || ".jpg";
    const localPath = path.join(workDir, `frame_${i}${ext}`);

    const response = await axios.get(frame.imageUrl, { responseType: "arraybuffer" });
    fs.writeFileSync(localPath, response.data);

    let beforeLocalPath = null;
    // FIX (July 14, 2026 — real test failure): this used to download
    // beforeUrl unconditionally whenever isBeforeAfter was true, with no
    // check for whether anything downstream would actually USE it. A
    // plain Ken Burns video with useRevealEffect unchecked never touches
    // the before image at all — but this was fetching it anyway, wasting
    // a request, and in this case failing the ENTIRE render when that
    // (unused) image happened to 404. Now only fetched when something
    // will actually consume it: Kling's before/after morph (useAiMotion)
    // or the explicit reveal-effect opt-in (useRevealEffect).
    const beforeImageNeeded = frame.isBeforeAfter && frame.beforeUrl && (frame.useAiMotion || frame.useRevealEffect);
    if (beforeImageNeeded) {
      try {
        const beforeExt = path.extname(new URL(frame.beforeUrl).pathname) || ".jpg";
        beforeLocalPath = path.join(workDir, `frame_${i}_before${beforeExt}`);
        const beforeResponse = await axios.get(frame.beforeUrl, { responseType: "arraybuffer" });
        fs.writeFileSync(beforeLocalPath, beforeResponse.data);
      } catch (err) {
        // NEW — a missing/broken before image should degrade this ONE
        // frame gracefully (falls back to single-image treatment
        // downstream — Kling animates from imageUrl alone, Ken Burns
        // skips the reveal branch), not take down the whole render the
        // way an unhandled failure here did before this fix.
        console.warn(`Frame ${i}: before image failed to download (${err.message}) — falling back to single-image treatment for this frame.`);
        beforeLocalPath = null;
      }
    }

    localFrames.push({
      localPath,
      beforeLocalPath,
      remoteImageUrl: frame.imageUrl, // preserved for Kling, which fetches from a public URL itself
      remoteBeforeUrl: frame.beforeUrl || null,
      isBeforeAfter: !!frame.isBeforeAfter,
      // FIX (July 14, 2026 — real test failure): useRevealEffect was
      // missing from this returned object entirely — meaning
      // renderPipeline.js's reveal-effect gate (frame.useRevealEffect)
      // always read undefined here, regardless of what the user actually
      // checked. Same bug class as klingMotionPreset/addContinuationMotion
      // below (already fixed once in this exact file) — an explicit
      // property picklist silently drops anything not named in it.
      useRevealEffect: !!frame.useRevealEffect,
      roomType: frame.roomType || "default",
      // NEW (July 14, 2026 — footage-grounded narration) — real display
      // name, not the small backend-coded vocabulary roomType uses.
      roomLabel: frame.roomLabel || null,
      motionPreset: frame.motionPreset || "auto",
      // FIX (July 2026): klingMotionPreset was never in this file's returned
      // object at all — same bug class as the continuation-motion fields
      // below (an explicit property picklist silently drops anything not
      // named), just never caught for this field until a real test job
      // showed every Kling frame rejected with "(none — generic default)"
      // despite video-job.js correctly sending klingMotionPreset in the
      // dispatch payload, and despite renderPipeline.js correctly forwarding
      // frame.klingMotionPreset onward. Both of those fixes were necessary
      // but not sufficient — this file, one layer earlier, was dropping the
      // field before either of them ever saw it.
      klingMotionPreset: frame.klingMotionPreset || null,
      durationSeconds: frame.durationSeconds || 4.5,
      sequenceOrder: frame.sequenceOrder ?? i,
      useAiMotion: !!frame.useAiMotion,
      customPrompt: frame.customPrompt || null,
      // These three were missing entirely until this fix — the
      // addContinuationMotion flag from the original request was being
      // silently dropped here, so it never reached renderPipeline.js or
      // klingMotion.js no matter how correctly those files were written.
      addContinuationMotion: !!frame.addContinuationMotion,
      continuationPreset: frame.continuationPreset || null,
      continuationDurationSeconds: frame.continuationDurationSeconds || null,
    });
  }

  // Ensure playback order matches what the agent set, regardless of
  // the order frames arrived in the request payload.
  localFrames.sort((a, b) => a.sequenceOrder - b.sequenceOrder);

  return localFrames;
}

module.exports = { downloadFrames };
