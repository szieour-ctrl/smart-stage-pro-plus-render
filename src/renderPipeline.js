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
const { generateLtxRevealContinuation, applyLtxMotion, isStandaloneEligible, LTX_MOTION_TEMPLATES } = require("./lib/ltxMotion");
const { generateMusic } = require("./lib/musicGen");
const { assembleVideo, buildRevealClip, REVEAL_PRESETS, REVEAL_OPENER_DURATION, REVEAL_WIPE_DURATION, REVEAL_CONTINUATION_DURATION, computeClipTimeline, extractMidpointFrame, probeDuration, mapWithConcurrencyLimit, FFMPEG_CONCURRENCY_LIMIT } = require("./lib/assemble");
const { generateNarration, groupContiguousByRoom } = require("./lib/narrationGen");
const { uploadToCloudinary } = require("./lib/cloudinaryUpload");
const { notifyWebhook } = require("./lib/notify");
// NEW (this session) — End Frame: replaces the closing shot with a
// Flux-edited version (address + CTA baked into the image itself).
// Feature-flagged and unbilled — see lib/endFrame.js's header comment
// for the full kill-switch/billing/fallback reasoning.
const { isEndFrameEnabled } = require("./lib/endFrame");

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

// NEW (July 18, 2026 — Sam's request, after a real render where the flat
// +0.8s breathing-room padding silently didn't apply and the only way to
// notice was manually diffing clip durations across log lines). This
// probes the REAL rendered file — not an echoed-back request value — and
// logs a hard, impossible-to-miss [err]-tagged line if the actual output
// doesn't match what was asked for. Deliberately does NOT throw: a
// padding mismatch means a tighter narration window, not a broken video,
// so the render should still complete — same "fail loud, not fail hard"
// principle Kling's fallback and narration's own try/catch already use
// elsewhere in this file. Tolerance (0.3s) covers normal fps-quantization
// and encoder rounding, not a real dropped-padding bug — motionRenderer.py
// rounds frame count via round(duration*fps), and buildRevealClip's xfade
// crossfade timing has its own small rounding, so exact-to-the-millisecond
// matches were never realistic even when everything is working correctly.
const CLIP_DURATION_TOLERANCE_SECONDS = 0.3;
async function verifyClipDuration(jobId, label, clipPath, expectedDuration) {
  try {
    const actualDuration = await probeDuration(clipPath);
    const diff = Math.abs(actualDuration - expectedDuration);
    if (diff > CLIP_DURATION_TOLERANCE_SECONDS) {
      console.error(
        `[${jobId}] [PADDING MISMATCH] ${label}: expected ${expectedDuration.toFixed(2)}s, ` +
        `actual rendered duration ${actualDuration.toFixed(2)}s (off by ${diff.toFixed(2)}s). ` +
        `Narration padding for this clip did NOT reach the rendered file — check that the ` +
        `deployed renderPipeline.js/assemble.js actually match what's in the repo.`
      );
    }
  } catch (e) {
    // Never let a verification failure itself break the render.
    console.error(`[${jobId}] [PADDING MISMATCH CHECK FAILED] ${label}: could not probe ${clipPath}: ${e.message}`);
  }
}

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
  // Mirrors klingFrameOutcomes above — same reasoning: distinguishes
  // real LTX renders from ken_burns_fallback outcomes, for the same
  // refund/diagnostic purposes.
  let ltxFrameOutcomes = [];

  try {
    console.log(`[${job.jobId}] Starting render. ${job.frames.length} frames.`);

    // ── Step 1: Download all frame images to local disk ──────────────────
    const localFrames = await downloadFrames(job.frames, workDir);
    console.log(`[${job.jobId}] Downloaded ${localFrames.length} frames.`);

    // ── Step 1.5 REMOVED (this session, real architecture correction) ──
    // Previously this step replaced the LAST ROOM FRAME's image with an
    // edited version and forced it onto a plain Ken Burns path, skipping
    // whatever motion it would naturally get. Sam's explicit correction:
    // the last room clip is the MOST IMPORTANT clip in the video and must
    // never be replaced, motion-downgraded, or otherwise treated
    // differently — it plays exactly as it would with End Frame disabled
    // entirely, including its own normal narration and spoken CTA. The
    // address/CTA card is now a genuinely separate, additional clip
    // appended AFTER the main video finishes (narration and all) — see
    // the `closingCard` param on the assembleVideo call in Step 4 below,
    // and lib/endFrame.js's kill-switch check (isEndFrameEnabled) used
    // there. Nothing in this frame-processing loop touches End Frame at
    // all anymore.

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
    // RAISED from 5 to 7 (July 18, 2026 — same reasoning as the outro
    // bump below: Sam's call that the opening and closing frames are the
    // two that matter most and need real margin, not technically-
    // sufficient math). At the observed pace FLOOR (~110 wpm), the
    // intro's content (one room/exterior detail + a brief location
    // welcome, ~17 words) needs ~9.3s of actual speech. The old 5s
    // padding left almost no margin above that for a shorter room type
    // (e.g. dining's 4.7s base) — technically enough on an average read,
    // not genuinely safe on a slow one. 7s gives real headroom across
    // every room type's base duration, not just the longer ones.
    // CORRECTED to 9 (July 18, 2026 — my first correction, 6s, was still
    // wrong: it assumed the opening room gets an ordinary ~10-14 word
    // description like a typical mid-video room. Checking against Sam's
    // actual real script showed the opener ran 25 words on its own —
    // exterior/establishing shots legitimately warrant a fuller
    // description than a single interior room does. Add the welcome
    // phrase (~7 words) on top of a real 25-word opener and the intro
    // needs to cover ~32 words, not meaningfully less than the outro's
    // load — hence landing at the same order of magnitude.
    // LOWERED to 6 (this session — Sam's explicit request: take 3s away
    // from intro and give it to the CTA, confirming the rebalancing
    // flagged as an option two rounds ago. Sam's own real-world
    // comparison showed the intro consistently finishing narration with
    // ~6s of unused time before the next clip started, while the outro
    // kept needing more room across several rounds of fixes — this
    // moves 3 of that surplus over rather than growing total video
    // length further.
    const NARRATION_INTRO_PADDING_SECONDS = 6;
    // RAISED to 15 (July 18, 2026 — Sam's call: stop budgeting the two
    // frames that matter most — the user's chosen opening and closing
    // shots — against average-case timing math, and give them enough
    // slack to work even in the worst case). 12s was technically enough
    // against the AVERAGE measured natural pace (~150 wpm across all
    // renders so far), but the real range has been 111-174 wpm depending
    // on sentence content — budgeting against the average left almost no
    // margin on a naturally slower read. At the observed FLOOR (~110
    // wpm), the outro's real content (one room/exterior detail + the
    // full address + an original CTA, ~28 words) needs ~15.3s of actual
    // speech. 15s of padding gives real margin above that, not just
    // technical sufficiency — the goal is this segment fitting at
    // NATURAL pace (speed=1.0) without ever needing speed correction to
    // bail it out, since correction has already been observed to
    // silently fail to shorten anything in at least one real case.
    // CORRECTED to 9 (July 18, 2026 — Sam's real math check: 15s was
    // wrong, derived from the pace FLOOR the same way the old global
    // rate was, and it showed up as ~7.6s of unintended dead air on a
    // real script). Base room durations already fit an ordinary one-
    // sentence description reasonably (confirmed: a segment with NO
    // extra padding landed almost exactly on target on a real script).
    // This padding should only cover the EXTRA content beyond that — an
    // address (~7 words) plus an original CTA (~10 words) ≈ 17 extra
    // words. At SPEAKING_RATE_WORDS_PER_MINUTE (130), that's ~7.8s of
    // real extra speech, plus a small margin — not a full re-derivation
    // of the whole segment's duration from zero.
    // RAISED to 10.5 (this session — Sam's explicit request: +1.5s more
    // for the CTA specifically, on top of everything else this segment
    // already gets — the address-shortening (street only, no city/state)
    // and the CTA-specific pace correction in narrationGen.js's
    // wordTargets computation). All three land together: less content to
    // say, a more realistic pace assumption for that content, and more
    // real time to say it in.
    // RAISED to 12 (this session — Sam's explicit request, second round:
    // +1.5s more again, on top of the previous +1.5s raise. Sam's own
    // real-world comparison: the intro (9s padding) finishes narration
    // with ~6s of unused time before the next clip starts, while the
    // outro has been consistently tight across multiple rounds of fixes
    // (address shortening, CTA-specific pace correction, dropping
    // unfittable closing lines) — the two are not currently symmetric,
    // and outro genuinely needs more room, not just better budgeting.
    // RAISED to 17 (this session — Sam's explicit request: +5s more,
    // paired with -3s taken from intro above — a real rebalance, not
    // just growing both numbers independently).
    const NARRATION_OUTRO_PADDING_SECONDS = 17;
    if (job.wantsNarration && localFrames.length > 0) {
      const first = localFrames[0];
      first.durationSeconds = resolveDuration(first) + NARRATION_INTRO_PADDING_SECONDS;
      // FIX (July 18, 2026 — real render, narration cut off ~2-3s before
      // the video's true end): frame.durationSeconds is only read by the
      // standard (non-reveal) branch of the frame loop below. The Reveal
      // Presets branch ALWAYS renders at the fixed REVEAL_OPENER_DURATION/
      // REVEAL_CONTINUATION_DURATION phase lengths and never looked at
      // frame.durationSeconds at all — so when the padded clip landed on
      // a Room Reveal frame, the padding was silently discarded and the
      // clip rendered at its bare ~5.1s length instead of the intended
      // ~10-11s. Confirmed from a real render log: the outro segment got
      // only a 2.62s narration window (5.12s bare clip minus crossfade/
      // end-buffer overhead) instead of the ~7.6s a properly-padded clip
      // would have given it — a 20-word closing line needed 7.39s even at
      // max speed correction, so most of it played past the video's end.
      // introOutroPaddingSeconds is a separate field specifically so the
      // reveal branch can add it to REVEAL_CONTINUATION_DURATION without
      // needing to reverse-engineer it back out of durationSeconds.
      first.introOutroPaddingSeconds = NARRATION_INTRO_PADDING_SECONDS;
      // Guard against localFrames.length === 1: first and last would be
      // the SAME object reference, so padding both would silently double
      // it on a single-clip video. Only pad the last clip separately when
      // there's genuinely more than one.
      if (localFrames.length > 1) {
        const last = localFrames[localFrames.length - 1];
        last.durationSeconds = resolveDuration(last) + NARRATION_OUTRO_PADDING_SECONDS;
        last.introOutroPaddingSeconds = NARRATION_OUTRO_PADDING_SECONDS;
        // NEW (Sam's request — outro end motion): flags the last clip for
        // resolvePreset() in motionPresets.js, which defaults it to the
        // calm "float" preset instead of whatever directional preset its
        // room type would normally get — only takes effect if the user
        // left it on "auto"; an explicit user choice still wins. Note:
        // this doesn't apply when the last clip is a Reveal Preset frame,
        // since the reveal branch always passes an explicit endMotion,
        // never "auto" — isOutro's float fallback is a non-issue there.
        last.isOutro = true;
      }
      console.log(`[${job.jobId}] Narration on — padded clip 1 by ${NARRATION_INTRO_PADDING_SECONDS}s (intro)${localFrames.length > 1 ? ` and last clip by ${NARRATION_OUTRO_PADDING_SECONDS}s (outro — room+address+CTA needs real room)` : ""}.`);
    }

    // NEW (Sam's request, July 18, 2026 — real render, CTA still cut off
    // even after the reveal-padding fix above: "Come see it in person,
    // and make this home..." got truncated mid-sentence, and the opening
    // clip had time to spare while others were squeezed). Sam's call:
    // stop trying to shave narration to fit ever-tighter windows via
    // ElevenLabs speed correction, and instead give EVERY clip — not
    // just the padded first/last — flat extra breathing room. Stacks on
    // top of the +5s intro/outro padding above (so first/last end up
    // with +5.8s total, everything else +0.8s). In a 20-frame video that's
    // 20 * 0.8 = 16s of added total runtime — an explicit, deliberate
    // tradeoff Sam chose, not something to re-optimize away later without
    // asking him first.
    //
    // Runs AFTER the intro/outro block on purpose: resolveDuration(frame)
    // returns frame.durationSeconds if already set (motionPresets.js),
    // so first/last correctly stack on top of their existing +5s instead
    // of this loop overwriting it.
    const NARRATION_CLIP_PADDING_SECONDS = 0.8;
    if (job.wantsNarration) {
      for (const frame of localFrames) {
        frame.durationSeconds = resolveDuration(frame) + NARRATION_CLIP_PADDING_SECONDS;
        // Separate field (mirrors introOutroPaddingSeconds) so the Reveal
        // Presets branch — which never reads frame.durationSeconds — can
        // still add this to its continuation phase.
        frame.narrationClipPaddingSeconds = NARRATION_CLIP_PADDING_SECONDS;
      }
      console.log(`[${job.jobId}] Narration breathing-room padding: +${NARRATION_CLIP_PADDING_SECONDS}s applied to all ${localFrames.length} clips.`);
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
        // CHANGE (July 17, 2026 — Reveal Presets architecture replaces the
        // old hardcoded pull_back(vacant)/push_in(staged) pairing). Same
        // gating as before (explicit useRevealEffect opt-in, not just a
        // known pair) — see the original comment this replaces for why
        // that gate exists. What changed: the vacant side's motion, the
        // wipe transition, and the staged side's motion are now all
        // driven by REVEAL_PRESETS[frame.revealPreset] + frame.endMotion
        // instead of two fixed constants. Durations are also no longer
        // 3s/4s — they're the fixed REVEAL_OPENER_DURATION (1.5s) /
        // REVEAL_CONTINUATION_DURATION (4.0s) phase lengths locked for
        // this architecture (see assemble.js's buildRevealClip header
        // comment for the full 3-phase timing math).
        //
        // Before/After clips don't participate in cross-room zoom
        // continuity — they're a self-contained reveal, so always start
        // at a fresh zoom (1.0) on both phases.
        const presetKey = REVEAL_PRESETS[frame.revealPreset] ? frame.revealPreset : "classic_reveal";
        const preset = REVEAL_PRESETS[presetKey];

        // Server-side clamp, step 1 — frame.endMotion arrives from the
        // client (build-video-demo.html), which already filters its End
        // Motion dropdown to preset.allowedEndMotions, but this is the
        // authoritative backend and shouldn't trust that filtering held
        // (stale client build, direct API call, etc.).
        let endMotion = preset.allowedEndMotions.includes(frame.endMotion)
          ? frame.endMotion
          : preset.allowedEndMotions[0];

        // Server-side clamp, step 2 — THE REAL GATING RULE (Sam, July 19,
        // 2026, after two wrong attempts at this same feature): "if the
        // user selects Ken Burns (engine) ALL AI is gated/locked... If the
        // user wants a PREMIUM AI Movement they select the AI Motion
        // (Engine), all Ken Burns Standard movements are gated/Locked."
        // This is a hard, unconditional rule — which engine is active
        // decides the ENTIRE category of continuation available, not just
        // a preference. preset.allowedEndMotions above lists BOTH
        // namespaces (every continuation this preset identity supports
        // across either engine); this second clamp is what actually
        // enforces "only ONE namespace is reachable for this frame,"
        // matching frame.revealEngine — this is also the real billing
        // enforcement point, not just a UI nicety, since a Ken-Burns-
        // engine frame that somehow ended up with an LTX endMotion would
        // render real paid AI motion on a clip the user was told was free.
        //
        // frame.revealEngine (not frame.motion directly) — a purpose-built
        // field sent specifically for reveal frames, "ken_burns" or "ltx".
        // Using frame.motion directly here would have been a real bug:
        // that raw string is never persisted through video-job.js's
        // frameRows/Railway-dispatch layer at all (confirmed by grep —
        // only motionPreset-family fields survive that round trip), so
        // it would always read undefined on the real backend object.
        const isLtxNamespace = (key) => !!LTX_MOTION_TEMPLATES[key];
        // DIAGNOSTIC (July 19, 2026 — confirmed real bug from Sam's first
        // real render: frame.revealEngine arrived as null/undefined,
        // silently forcing this reveal to Ken Burns with ZERO trace
        // anywhere in the logs — no error, no [LTX] line, nothing. The
        // fal.ai dashboard showed no activity and the video still
        // rendered successfully, which is exactly what made it invisible.
        // Loud now, matching the [PADDING MISMATCH] pattern elsewhere in
        // this file — a missing/invalid revealEngine on a reveal frame is
        // always worth knowing about immediately, not discovering by
        // noticing an empty fal.ai dashboard after the fact.
        if (frame.revealEngine !== "ltx" && frame.revealEngine !== "ken_burns") {
          console.error(
            `  [${job.jobId}] [REVEAL ENGINE MISSING] frame ${i}: revealEngine is "${frame.revealEngine}" (expected "ltx" or "ken_burns"). ` +
            `Defaulting to Ken Burns (the safe/free side) — this frame will NOT call LTX even if the user selected AI Motion. ` +
            `Check that video-job.js's reveal_engine column exists and the frameRows insert/read-back is wired correctly.`
          );
        }
        if (frame.revealEngine === "ltx") {
          if (!isLtxNamespace(endMotion)) {
            const firstLtx = preset.allowedEndMotions.find(isLtxNamespace);
            endMotion = firstLtx || endMotion; // no LTX option exists for this preset+room combo — rare, but don't crash
          }
        } else {
          // Ken Burns engine (or anything else — default to the safe,
          // free side of the gate) — AI Motion is locked out entirely.
          if (isLtxNamespace(endMotion)) {
            endMotion = preset.allowedEndMotions.find((key) => !isLtxNamespace(key)) || "push_in";
          }
        }

        // FIX (July 18, 2026) — see the padding block's comment above for
        // the full story. Only the continuation phase gets the extra
        // time — the opener is a brief vacant hold, stretching THAT for
        // outro narration room would make no visual sense; the
        // continuation (staged room, real motion) is the phase an intro/
        // outro narration beat should actually play over. Stacks both
        // padding sources: introOutroPaddingSeconds (first/last only,
        // +5s) and narrationClipPaddingSeconds (every clip, +0.8s).
        const continuationDuration = REVEAL_CONTINUATION_DURATION
          + (frame.introOutroPaddingSeconds || 0)
          + (frame.narrationClipPaddingSeconds || 0);

        // NEW (July 20, 2026 — Sam's request): burns a small "Original"
        // badge into the opener clip only — the vacant/before image shown
        // before the wipe into the staged continuation. labelText is a
        // no-op everywhere else in this file (continuation phase, standalone
        // clips) since only this call site sets it.
        const openerResult = await applyMotionPreset(
          { ...frame, localPath: frame.beforeLocalPath, motionPreset: preset.openerMotion, durationSeconds: REVEAL_OPENER_DURATION, labelText: "Original" },
          workDir,
          1.0
        );

        // NEW (July 18, 2026) — LTX Fast continuation option. endMotion
        // now comes from one of TWO namespaces (Ken Burns preset names in
        // motionRenderer.py, or LTX preset names in LTX_MOTION_TEMPLATES —
        // confirmed no name collisions between the two lists). Dispatch on
        // which one it belongs to.
        //
        // continuationDuration above is the DESIRED/padded value (a
        // flexible float). Ken Burns respects it exactly. LTX Fast cannot
        // — its duration is a fixed enum (6/8/10.../20s) — so LTX snaps
        // UP to the nearest valid value internally and returns what it
        // ACTUALLY rendered at. That real value, not the original
        // request, is what has to flow into buildRevealClip's trim below
        // — using the pre-snap number would silently discard whatever
        // extra seconds LTX actually generated (and were actually paid
        // for), and would also make verifyClipDuration fire a false
        // [PADDING MISMATCH] on every single LTX reveal clip.
        let continuationResult;
        let actualContinuationDuration = continuationDuration;
        const isLtxEndMotion = !!LTX_MOTION_TEMPLATES[endMotion];
        // TEMPORARY DIAGNOSTIC (July 19, 2026) — every static check of this
        // code path has come back clean (revealEngine confirmed correct,
        // both clamps confirmed logically sound, LTX_MOTION_TEMPLATES
        // confirmed to contain the expected key, renderPipeline.js and
        // ltxMotion.js confirmed matching deployed code) — yet zero [LTX]
        // log activity on a real render where all of that should have
        // resulted in at least an attempted call. Printing the actual
        // runtime values directly rather than continuing to reason about
        // what they "should" be. Remove once this mystery is resolved.
        console.log(`  [DIAGNOSTIC] frame ${i}: presetKey="${presetKey}" revealEngine="${frame.revealEngine}" endMotion(final)="${endMotion}" isLtxEndMotion=${isLtxEndMotion} LTX_MOTION_TEMPLATES_has_key=${Object.prototype.hasOwnProperty.call(LTX_MOTION_TEMPLATES, endMotion)} total_LTX_keys=${Object.keys(LTX_MOTION_TEMPLATES).length}`);

        if (isLtxEndMotion) {
          try {
            const ltxResult = await generateLtxRevealContinuation(
              { ...frame, continuationDurationSeconds: continuationDuration },
              endMotion,
              workDir,
              job.jobId
            );
            continuationResult = ltxResult;
            actualContinuationDuration = ltxResult.ltxDuration;
          } catch (err) {
            // No LTX-preset-to-Ken-Burns-preset mapping exists (the two
            // libraries' names don't correspond 1:1), so this can't fall
            // back to "the same motion via Ken Burns" the way Kling's own
            // fallback does. Falls back to push_in — a safe, always-
            // available default — same "premium enhancement, never a hard
            // dependency" principle as every other AI motion fallback in
            // this file.
            console.error(`  [${job.jobId}] [Reveal] LTX continuation "${endMotion}" failed, falling back to Ken Burns push_in: ${err.message}`);
            continuationResult = await applyMotionPreset(
              { ...frame, motionPreset: "push_in", durationSeconds: continuationDuration },
              workDir,
              1.0
            );
            actualContinuationDuration = continuationDuration;
          }
        } else {
          continuationResult = await applyMotionPreset(
            { ...frame, motionPreset: endMotion, durationSeconds: continuationDuration },
            workDir,
            1.0
          );
        }

        clipPath = await buildRevealClip(
          openerResult.path,
          continuationResult.path,
          presetKey,
          workDir,
          `reveal_${path.basename(frame.localPath, path.extname(frame.localPath))}.mp4`,
          actualContinuationDuration
        );
        carryZoom = 1.0; // reset after a reveal beat

        if (job.wantsNarration) {
          const expectedRevealDuration = REVEAL_OPENER_DURATION + actualContinuationDuration - REVEAL_WIPE_DURATION;
          await verifyClipDuration(job.jobId, `frame ${i} (reveal, ${presetKey}${isLtxEndMotion ? ", LTX" : ""})`, clipPath, expectedRevealDuration);
        }
      } else if (frame.ltxMotionPreset && LTX_MOTION_TEMPLATES[frame.ltxMotionPreset]) {
        // NEW (July 18, 2026) — standalone LTX AI Motion, no Room Reveal
        // pairing required. Same category as klingMotion.js's
        // SINGLE_IMAGE_INTERIOR_ALLOWED_PRESETS: pure camera/ambient
        // motion on an already-real, already-staged single image. Only
        // Medium-High/High confidence presets from the Cinematic LTX
        // Prompt Pack are exposed here at all (Sam's explicit scope call)
        // — ltxMotion.js's isStandaloneEligible() is the enforcement
        // point the frontend dropdown is expected to match, but this is
        // still checked server-side too, same "don't trust the client
        // alone" principle as the Reveal Presets' endMotion clamp.
        if (!isStandaloneEligible(frame.ltxMotionPreset)) {
          console.error(`  [${job.jobId}] [LTX] Rejected standalone use of "${frame.ltxMotionPreset}" — below the Medium-High confidence floor for standalone selection. Falling back to Ken Burns auto.`);
          const result = await applyMotionPreset({ ...frame, motionPreset: "auto" }, workDir, carryZoom);
          clipPath = result.path;
          carryZoom = result.endingZoom;
        } else {
          const result = await applyLtxMotion(
            frame,
            frame.ltxMotionPreset,
            workDir,
            () => applyMotionPreset({ ...frame, motionPreset: "auto" }, workDir, carryZoom),
            job.jobId
          );
          clipPath = result.path;
          carryZoom = result.endingZoom;

          ltxFrameOutcomes.push({
            sequenceOrder: frame.sequenceOrder !== undefined ? frame.sequenceOrder : i,
            outcome: result.source,
          });
        }
      } else {
        const result = await applyMotionPreset(frame, workDir, carryZoom);
        clipPath = result.path;
        carryZoom = result.endingZoom;

        // Kling frames are deliberately excluded from this check — Kling's
        // own returned duration routinely differs from what was requested
        // (confirmed in real logs, e.g. "returned 5s when 4.5s was asked
        // for"), which is expected vendor behavior, not a padding bug.
        // This branch only ever runs for standard Ken Burns clips, so no
        // extra guard is needed here beyond job.wantsNarration.
        if (job.wantsNarration) {
          await verifyClipDuration(job.jobId, `frame ${i} (standard, ${frame.motionPreset || "auto"})`, clipPath, frame.durationSeconds);
        }
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
    // CHANGE (this session, real architecture correction): the closing
    // card is back — but not as the old live-overlay version this
    // comment used to describe (which really was removed for causing
    // hangs). This is assembleVideo's own already-built closingCard
    // path (see appendClosingCardWithAudio in assemble.js), which existed
    // in code but was never actually invoked because nothing passed this
    // param in. It runs fully decoupled from the main mix (after mixAudio,
    // in isolation, with its own short music-only stinger) — exactly the
    // "after the last clip and its narration finish, separate card, music
    // only" behavior Sam asked for. Uses the ORIGINAL last frame's own
    // photo as the card's background (localFrames is never mutated for
    // End Frame anymore — the last room clip plays completely normally,
    // untouched, with its own narration and spoken CTA intact).
    const lastFrame = localFrames[localFrames.length - 1];
    const closingCard = (isEndFrameEnabled() && job.address && lastFrame)
      ? {
          stillImagePath: lastFrame.localPath,
          addressLine: job.address,
          ctaLine: process.env.END_FRAME_CTA_TEXT || "Schedule Your Private Showing",
        }
      : null;
    if (isEndFrameEnabled() && !closingCard) {
      console.log(`[${job.jobId}] End Frame: skipped — ${!job.address ? "no address available" : "no frames to build a card from"}.`);
    }

    const outputs = await assembleVideo({
      clipPaths,
      musicPath,
      narrationSegments,
      formats: job.formats || ["16x9", "9x16"],
      workDir,
      closingCard,
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
      ltxFrameOutcomes,
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
      ltxFrameOutcomes,
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

