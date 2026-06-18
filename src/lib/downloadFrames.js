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
    if (frame.isBeforeAfter && frame.beforeUrl) {
      const beforeExt = path.extname(new URL(frame.beforeUrl).pathname) || ".jpg";
      beforeLocalPath = path.join(workDir, `frame_${i}_before${beforeExt}`);
      const beforeResponse = await axios.get(frame.beforeUrl, { responseType: "arraybuffer" });
      fs.writeFileSync(beforeLocalPath, beforeResponse.data);
    }

    localFrames.push({
      localPath,
      beforeLocalPath,
      remoteImageUrl: frame.imageUrl, // preserved for Kling, which fetches from a public URL itself
      remoteBeforeUrl: frame.beforeUrl || null,
      isBeforeAfter: !!frame.isBeforeAfter,
      roomType: frame.roomType || "default",
      motionPreset: frame.motionPreset || "auto",
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
