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
const { assembleVideo, buildBeforeAfterClip, computeClipTimeline, extractMidpointFrame, probeDuration, mapWithConcurrencyLimit, FFMPEG_CONCURRENCY_LIMIT } = require("./lib/assemble");
const { generateNarration, groupContiguousByRoom } = require("./lib/narrationGen");
const { uploadToCloudinary } = require("./lib/cloudinaryUpload");
const { notifyWebhook } = require("./lib/notify");

// ── NARRATION VOICE LIBRARY ───────────────────────────────────────────
// MUST stay in sync with NARRATION_VOICES in build-video-demo.html.
// Real voice_ids, already confirmed working (Male — Audiobook Narrator,
// Female — Adeline/Conversational — see the July 13 voice-selection
// conversation). Not "Default" category voices, which expire Dec 31,
// 2026 — these are permanent Voice Library picks.
const NARRATION_VOICE_LIBRARY = {
  "voice_male_1":   "pVYHFs8oaIDPWJxvmXWW",
  "voice_female_1": "5l5f8iK3YPeGga21rQIX",
};

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

    // NEW (narration grouping + intro/outro build): when narration is on,
    // clip 1 and the last clip in the queue get extra duration so there's
    // real room for an intro/outro narration beat later, instead of that
    // narration being crammed into whatever the room's normal hold time
    // happens to be. Duration-only mechanic for now — actual intro
    // (address/location) and outro (CTA/sign-off) SCRIPT CONTENT is
    // separate, not-yet-built work; Sam's call was best-practices help
    // content for photo ordering, not an auto-generated title card. This
    // just reserves the time so that future work has somewhere to land.
    //
    // Whole-second padding, not a fractional value — Kling's API only
    // accepts whole-second durations (3-15, see klingMotion.js's
    // buildKlingRequest), and while it does round fractional requests
    // itself, staying whole here avoids an unnecessary round-trip through
    // that rounding logic for the common case.
    // RAISED from 3 to 5 (Sam's feedback, real render — outro cut off
    // mid-sentence): mixAudio's NARRATION_END_BUFFER_SECONDS (2s) is now
    // subtracted from the last group's usable duration when its word
    // budget is calculated (see groupContiguousByRoom in narrationGen.js)
    // — so net real extra room for the outro was only 3-2=1s at the old
    // padding value, too thin for a full closing line. 5s padding now
    // nets ~3s of real usable extra time after that subtraction.
    const NARRATION_INTRO_OUTRO_PADDING_SECONDS = 5;
    if (job.wantsNarration && localFrames.length > 0) {
      const first = localFrames[0];
      first.durationSeconds = resolveDuration(first) + NARRATION_INTRO_OUTRO_PADDING_SECONDS;
      // Guard against localFrames.length === 1: first and last would be
      // the SAME object reference, so padding both would silently double
      // it on a single-clip video. Only pad the last clip separately when
      // there's genuinely more than one.
      if (localFrames.length > 1) {
        const last = localFrames[localFrames.length - 1];
        last.durationSeconds = resolveDuration(last) + NARRATION_INTRO_OUTRO_PADDING_SECONDS;
      }
      console.log(`[${job.jobId}] Narration on — padded clip 1${localFrames.length > 1 ? " and last clip" : ""} by ${NARRATION_INTRO_OUTRO_PADDING_SECONDS}s for future intro/outro room.`);
    }

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
        // FIX (July 14, 2026 — real test failure): this used to gate on
        // frame.isBeforeAfter alone — the ORIGINAL flag from the dispatch
        // payload, set before any download was even attempted. When the
        // before image 404'd, downloadFrames.js correctly nulled
        // beforeLocalPath (the LOCAL file our own Ken Burns/continuation
        // path reads), but isBeforeAfter and remoteBeforeUrl were never
        // touched — so THIS code, independently, still told Kling to
        // fetch the same broken URL directly (Kling fetches images
        // itself, from the public URL, not from our local disk). Kling
        // correctly rejected it: "Unprocessable Entity." Two different
        // consumers of "is there really a usable before image" fell out
        // of sync — downloadFrames.js's local check got fixed, this
        // remote one didn't. Now gated on beforeLocalPath specifically,
        // the one field that reflects whether the download ACTUALLY
        // succeeded, not just whether one was originally expected.
        const hasRealBeforeImage = frame.isBeforeAfter && !!frame.beforeLocalPath;
        const result = await applyKlingMotion(
          {
            imageUrl: hasRealBeforeImage ? frame.remoteBeforeUrl : frame.remoteImageUrl,
            endImageUrl: hasRealBeforeImage ? frame.remoteImageUrl : undefined,
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
          // FIX (July 14, 2026 — real test failure): was passing the
          // original `frame` straight through, unchanged — meaning
          // frame.motionPreset still held whatever KLING preset was
          // selected (e.g. "cinematic_push" for Hero Transformation).
          // Ken Burns has its own, entirely different preset vocabulary
          // and doesn't recognize Kling preset names — motionPresets.js
          // correctly caught this and fell back to "auto" anyway, but
          // logged a confusing "Unknown preset" warning every time,
          // making a real, intentional fallback look like a bug. Now
          // explicit: a Kling failure always means Ken Burns renders
          // with auto, stated directly instead of arrived at via a
          // failed lookup.
          () => applyMotionPreset({ ...frame, motionPreset: "auto" }, workDir, carryZoom)
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
      } else if (frame.useRevealEffect && frame.isBeforeAfter && frame.beforeLocalPath) {
        // CHANGE (found during before/after-pair conversation): this used
        // to fire off isBeforeAfter alone — meaning ANY Ken Burns room
        // with a real pair on file got this more elaborate reveal treatment
        // automatically, whether the user wanted it or not (isBeforeAfter
        // is auto-detected from real gallery data, not a deliberate
        // choice). Now requires the separate, explicit useRevealEffect
        // opt-in too — a user who just wants plain single-image Ken Burns
        // on a room that happens to have a pair on file gets exactly that.
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

    // ── Step 2.5: Generate footage-grounded narration (July 14, 2026) ────
    // Runs HERE, not before rendering — see narrationGen.js's header
    // comment for the full reasoning. Needs the REAL rendered clips
    // (their real durations, which can differ from what was requested —
    // Kling has returned 5s when 4.5s was asked for in real logs) to
    // compute accurate timeline positions and extract real footage to
    // ground the script in. Wrapped in its own try/catch: a narration
    // failure should never take down a video that otherwise rendered
    // fine, same principle as Kling falling back to Ken Burns rather
    // than failing the whole job.
    let narrationSegments = [];
    let narrationFullScript = null;
    if (job.wantsNarration) {
      try {
        const realDurations = await Promise.all(clipPaths.map(probeDuration));
        const timeline = computeClipTimeline(realDurations);

        // FIX (July 14, 2026 — real test failure): was Promise.all,
        // unbounded — see assemble.js's mapWithConcurrencyLimit header
        // comment for the full reasoning (a real test job threw a
        // resource-exhaustion-flavored ffmpeg error here, "Resource
        // temporarily unavailable," extracting frames for all 9 clips
        // simultaneously).
        const framePaths = await mapWithConcurrencyLimit(
          clipPaths, FFMPEG_CONCURRENCY_LIMIT, (clip, i) => extractMidpointFrame(clip, realDurations[i], workDir, i)
        );

        // CHANGE (narration coherence fix, per July 14 handoff + Sam's
        // direct feedback on real output): was strict 1:1 — one segment
        // per clip, one frame per segment. Replaced with contiguous
        // room-based grouping (groupContiguousByRoom in narrationGen.js)
        // — multiple angles/crops of the same room, ordered back-to-back
        // by the user, now become ONE longer narration segment spanning
        // all of them, with up to 4 representative frames sent to the
        // same vision call instead of one call per clip. This is what
        // actually fixes the ~10-word-per-segment cramping: a 3-angle
        // bedroom run at ~4s/clip now gets ~12s of real word budget
        // instead of 3 separate 4s budgets that each cut off mid-thought.
        const timelineSegments = groupContiguousByRoom(localFrames, framePaths, timeline);

        const voiceId = NARRATION_VOICE_LIBRARY[job.voiceKey];
        if (!voiceId) {
          throw new Error(`Voice "${job.voiceKey}" is not configured with a real ElevenLabs voice_id.`);
        }

        const narrationResult = await generateNarration({
          address: job.address || null,
          timelineSegments,
          voiceId,
          workDir,
          anthropicKey: process.env.ANTHROPIC_API_KEY,
          elevenLabsKey: process.env.ELEVENLABS_API_KEY,
        });

        narrationSegments = narrationResult.segments;
        narrationFullScript = narrationResult.fullScript;
        console.log(`[${job.jobId}] Narration generated: ${narrationSegments.length} segments.`);
      } catch (err) {
        console.error(`[${job.jobId}] Narration generation failed (non-fatal to the video itself): ${err.message}`);
        narrationSegments = [];
        narrationFullScript = null;
      }
    }

    // ── Step 3: Wait for music ─────────────────────────────────────────
    const musicPath = await musicPromise;
    console.log(`[${job.jobId}] Music ready: ${musicPath}`);

    // ── Step 4: Assemble clips + music (+ optional narration) into final formats
    const outputs = await assembleVideo({
      clipPaths,
      musicPath,
      narrationSegments,
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
      // NEW (July 14, 2026) — the actual generated script, so Netlify can
      // store it and show it alongside the finished video (see Sam's
      // idea: display the script at the result screen, since it was
      // never shown anywhere before this rebuild either).
      narrationScript: narrationFullScript,
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

