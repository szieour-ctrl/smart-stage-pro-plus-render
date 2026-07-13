// renderPipeline.js — Orchestrates the full video render process
//
// Pipeline: download frames → apply motion → generate music (parallel)
//           → assemble clips → mix audio → render formats → upload → notify
//
// Each step is its own module so they can be built and tested independently.
// Right now, applyMotion, generateMusic, and assembleVideo are stubs that
// return placeholder data — replace each stub once the corresponding
// dependency (FFmpeg test, Mubert API key) is ready.

const path = require("path");
const fs = require("fs");
const os = require("os");
const axios = require("axios");

const { downloadFrames } = require("./lib/downloadFrames");
const { applyMotionPreset, resolveDuration } = require("./lib/motionPresets");
const { applyKlingMotion } = require("./lib/klingMotion");
const { generateMusic } = require("./lib/musicGen");
const { assembleVideo, buildBeforeAfterClip } = require("./lib/assemble");
const { uploadToCloudinary } = require("./lib/cloudinaryUpload");
const { notifyWebhook } = require("./lib/notify");

async function processRenderJob(job) {
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), `job-${job.jobId}-`));

  // NEW (bug 2g — refund logic): declared here, not inside the try block,
  // so it's still readable in the catch block below if a failure happens
  // partway through the frame loop — informational on the failure path
  // (the failure refund is a full-charge refund regardless, per the locked
  // design: no usable video means no charge is defensible either way), but
  // useful for diagnosing exactly which frames had already rendered via
  // Kling (real cost already incurred) before the failure occurred.
  let klingFrameOutcomes = [];

  try {
    console.log(`[${job.jobId}] Starting render. ${job.frames.length} frames.`);

    // ── Step 1: Download all frame images to local disk ──────────────────
    const localFrames = await downloadFrames(job.frames, workDir);
    console.log(`[${job.jobId}] Downloaded ${localFrames.length} frames.`);

    // ── Step 2: Apply motion preset to each frame → individual clips ─────
    // Runs while music generates in parallel (Step 3 kicks off immediately).
    const totalDuration = localFrames.reduce((sum, f) => sum + resolveDuration(f), 0);

    const musicPromise = generateMusic({
      durationSeconds: Math.ceil(totalDuration),
      musicStyle: job.musicStyle || "default",
      workDir,
    });

    const clipPaths = [];
    let carryZoom = 1.0; // tracks ending zoom of the previous clip for continuity

    // Tracks, per Kling-requested frame, whether it actually rendered via
    // Kling or silently fell back to Ken Burns. sequenceOrder is passed
    // through from video-job.js's dispatch payload — falls back to array
    // index if a frame is somehow missing it, so this never throws, it just
    // degrades to position-based matching.

    for (let i = 0; i < localFrames.length; i++) {
      const frame = localFrames[i];
      let clipPath;

      if (frame.useAiMotion) {
        // Premium AI motion via Kling — interior requires both vacant+staged
        // URLs (enforced inside klingMotion.js), exterior allows single or
        // paired images. Falls back to standard Ken Burns on any failure.
        //
        // IMPORTANT: Kling fetches images itself from a public URL — it
        // needs the original Cloudinary URLs (frame.remoteImageUrl /
        // frame.remoteBeforeUrl), not the local disk paths downloadFrames.js
        // already pulled down for the Ken Burns/FFmpeg path.
        //
        // localPath is also passed through — it's the staged image already
        // downloaded locally, used by the optional continuation-motion step
        // (Ken Burns push-in/parallax that plays after Kling's transformation
        // settles), so we never need to extract a frame from Kling's own
        // video output.
        const result = await applyKlingMotion(
          {
            imageUrl: frame.isBeforeAfter ? frame.remoteBeforeUrl : frame.remoteImageUrl,
            endImageUrl: frame.isBeforeAfter ? frame.remoteImageUrl : undefined,
            roomType: frame.roomType,
            // BUG FIX (July 2026): this object never included the preset
            // field at all, in any prior version of this file — confirmed
            // via a real test job where every frame was rejected with
            // "(none — generic default)" despite video-job.js correctly
            // sending klingMotionPreset in the dispatch payload (bug 2e's
            // fix). That fix was necessary but not sufficient: this object
            // construction, independently, dropped the field before it ever
            // reached klingMotion.js's enforceScopeRules()/buildPrompt(),
            // both of which read frame.klingMotionPreset specifically. This
            // means NO custom Kling preset has ever actually been honored in
            // production until this fix — every Kling-bound frame without a
            // known pair silently hit the generic-default rejection and fell
            // back to Ken Burns, regardless of what preset the user selected.
            klingMotionPreset: frame.klingMotionPreset,
            durationSeconds: resolveDuration(frame),
            customPrompt: frame.customPrompt,
            localPath: frame.localPath,
            addContinuationMotion: !!frame.addContinuationMotion,
            continuationPreset: frame.continuationPreset,
            continuationDurationSeconds: frame.continuationDurationSeconds,
          },
          workDir,
          () => applyMotionPreset(frame, workDir, carryZoom)
        );
        clipPath = result.path;
        carryZoom = result.endingZoom;

        // result.source is "kling" (real Kling output) or
        // "ken_burns_fallback" (applyKlingMotion() caught a failure and
        // fell back) — set in klingMotion.js's applyKlingMotion(). This is
        // exactly the distinction video-notify.js needs to know which
        // billed frames to refund.
        klingFrameOutcomes.push({
          sequenceOrder: frame.sequenceOrder !== undefined ? frame.sequenceOrder : i,
          outcome: result.source,
        });
      } else if (frame.isBeforeAfter && frame.beforeLocalPath) {
        // Flagship feature: vacant room holds, then wipes into staged version.
        // Build each side's motion clip first, then composite the wipe.
        // Before/After clips don't participate in cross-room zoom continuity —
        // they're a self-contained reveal, so always start at a fresh zoom.
        const vacantResult = await applyMotionPreset(
          { ...frame, localPath: frame.beforeLocalPath, motionPreset: "pull_back", durationSeconds: 3 },
          workDir,
          1.0
        );
        const stagedResult = await applyMotionPreset(
          { ...frame, motionPreset: "push_in", durationSeconds: 4 },
          workDir,
          1.0
        );
        clipPath = await buildBeforeAfterClip(
          vacantResult.path,
          stagedResult.path,
          workDir,
          `beforeafter_${path.basename(frame.localPath, path.extname(frame.localPath))}.mp4`
        );
        carryZoom = 1.0; // reset after a before/after beat
      } else {
        const result = await applyMotionPreset(frame, workDir, carryZoom);
        clipPath = result.path;
        carryZoom = result.endingZoom;
      }
      clipPaths.push(clipPath);
    }
    console.log(`[${job.jobId}] Motion applied to ${clipPaths.length} clips.`);

    // ── Step 3: Wait for music ─────────────────────────────────────────
    const musicPath = await musicPromise;
    console.log(`[${job.jobId}] Music ready: ${musicPath}`);

    // ── Step 4: Assemble clips + music (+ optional narration) into final formats
    const outputs = await assembleVideo({
      clipPaths,
      musicPath,
      narrationUrl: job.narrationAudioUrl || null,
      formats: job.formats || ["16x9", "9x16"],
      workDir,
    });
    console.log(`[${job.jobId}] Assembled ${Object.keys(outputs).length} formats.`);

    // ── Step 5: Upload finished videos to Cloudinary ──────────────────────
    const urls = await uploadToCloudinary(outputs, job.projectId);
    console.log(`[${job.jobId}] Uploaded to Cloudinary.`);

    // ── Step 6: Notify Netlify the job is complete ───────────────────────
    await notifyWebhook({
      jobId: job.jobId,
      status: "complete",
      urls,
      klingFrameOutcomes,
    });

    console.log(`[${job.jobId}] Done.`);
  } catch (err) {
    console.error(`[${job.jobId}] Render failed:`, err);
    await notifyWebhook({
      jobId: job.jobId,
      status: "failed",
      error: err.message,
      klingFrameOutcomes,
    });
    throw err;
  } finally {
    // Always clean up temp files, even on failure
    fs.rm(workDir, { recursive: true, force: true }, (err) => {
      if (err) console.error(`Cleanup failed for ${workDir}:`, err.message);
    });
  }
}

module.exports = { processRenderJob };
