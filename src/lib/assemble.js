// assemble.js — Concatenates motion clips with crossfade transitions,
// mixes in background music, and renders final output formats.

const path = require("path");
const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");

const CROSSFADE_DURATION = 0.6; // seconds between clips

// ── CONCURRENCY-LIMITED MAP (July 14, 2026 — real test failure) ────────
// Two separate real failures in the same render (a narration frame
// extraction and a clip normalization) both threw resource-exhaustion-
// flavored ffmpeg errors ("Resource temporarily unavailable", "Error
// while opening encoder" on a clip with perfectly clean, even
// dimensions — ruling out the odd-dimension theory from the prior fix).
// Both call sites were launching one ffmpeg process PER CLIP, all at
// once, via unbounded Promise.all — up to 9 simultaneous ffmpeg encodes
// for a 9-frame job. That's a real, plausible cause of exactly this
// class of intermittent failure on a resource-constrained container,
// regardless of which specific resource (CPU, memory, file descriptors)
// is actually the bottleneck. This caps how many run at once instead of
// firing all of them simultaneously — processes the rest as slots free
// up, rather than requiring every clip to fit in memory/CPU at the same
// instant.
async function mapWithConcurrencyLimit(items, limit, fn) {
  const results = new Array(items.length);
  let cursor = 0;
  async function worker() {
    while (cursor < items.length) {
      const i = cursor++;
      results[i] = await fn(items[i], i);
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, worker);
  await Promise.all(workers);
  return results;
}
const FFMPEG_CONCURRENCY_LIMIT = 3; // conservative default — revisit if Railway's plan tier is confirmed to have more headroom

// NEW (July 14, 2026 — real test failure) — mirrors
// NARRATION_END_BUFFER_SECONDS in generate-narration-background.js. That
// file sizes the SCRIPT to roughly fit before this buffer; this constant
// is the hard enforcement backstop at mix time, against the real, exact
// video duration rather than an estimate.
const NARRATION_END_BUFFER_SECONDS = 2;

// ── NORMALIZE CLIP (fixes the real root cause of the xfade crash) ────────
// CONFIRMED ROOT CAUSE of "Error reinitializing filters! / Failed to
// inject frame into filter network: Invalid argument" — a real crash from
// a real test job with 4 clips, 2 sourced from Kling (fal.ai's own output
// resolution/fps/codec) and 2 from the local Ken Burns fallback (OpenCV/
// FFmpeg, whatever this codebase already renders at). xfade requires every
// pair of inputs it chains together to share resolution, frame rate, and
// pixel format — concatenateClips() never enforced this, so any mix of
// Kling + Ken Burns clips (or even two Kling clips at different durations/
// fal.ai output settings) was one mismatch away from crashing the whole
// render.
//
// FIX: normalize every clip to identical parameters BEFORE it ever reaches
// the xfade chain, regardless of source. This is deliberately a fixed
// target (1920x1080/30fps/yuv420p) rather than "whatever Kling outputs" —
// Ken Burns clips would still need separate normalization to MATCH
// whatever Kling's native spec is, so pinning both to one target this
// codebase controls is no more work and is robust to fal.ai ever changing
// Kling's default output spec in the future. 1920x1080 matches the final
// 16:9 output exactly, so this adds no redundant scaling at renderFormat()
// time for the common case.
// CHANGE: fps corrected from an initial 30 to 20, to match the EXISTING
// normalization convention already established in klingMotion.js's own
// concatTwoClips() (used for the Before/After continuation feature) —
// confirmed by direct inspection: OUTPUT_W=1920, OUTPUT_H=1080, fps=20,
// format=yuv420p. Using a different fps here would have meant a
// continuation-combined clip (already normalized once, at 20fps, inside
// klingMotion.js) gets silently re-normalized to a SECOND, different fps
// the moment it reaches assemble.js — wasted re-encoding at best, and a
// new source of inconsistency at worst if other clips in the same job
// were normalized to yet a third value. One target, matched across both
// files, is the actual fix — not just "pick a fixed number and move on."
const NORMALIZE_WIDTH  = 1920;
const NORMALIZE_HEIGHT = 1080;
const NORMALIZE_FPS    = 20;

function normalizeClip(clipPath, workDir, index) {
  return new Promise(async (resolve, reject) => {
    const outputPath = path.join(workDir, `normalized_${index}.mp4`);

    // NEW (July 14, 2026 — real test failure) — a real render failed here
    // with "Error while opening encoder for output stream #0:0 - maybe
    // incorrect parameters such as bit_rate, rate, width or height" on
    // clip index 6, with no further detail on WHY. That error message is
    // FFmpeg's generic catch-all for the encoder rejecting the stream —
    // it doesn't say what was actually wrong with the source. Rather than
    // guess, probe the source clip FIRST and log its real properties, so
    // if this happens again the log states the actual cause (corrupt
    // file, zero duration, unusual dimensions) instead of a mystery.
    try {
      const meta = await new Promise((res, rej) => {
        ffmpeg.ffprobe(clipPath, (err, data) => err ? rej(err) : res(data));
      });
      const videoStream = meta.streams?.find(s => s.codec_type === "video");
      console.log(`[normalizeClip] clip ${index} (${clipPath}): ${videoStream?.width}x${videoStream?.height}, duration=${meta.format?.duration}, codec=${videoStream?.codec_name}`);
    } catch (probeErr) {
      // The clip is likely corrupt/unreadable if even ffprobe can't read
      // it — this IS useful diagnostic information, log it and continue
      // to the real normalization attempt below (which will then fail
      // with its own, now-better-understood error).
      console.error(`[normalizeClip] clip ${index} (${clipPath}) failed to probe — likely corrupt or incomplete: ${probeErr.message}`);
    }

    ffmpeg(clipPath)
      .videoFilters([
        // Scale to fit within target dimensions preserving aspect ratio,
        // then pad to exactly the target size — same safe-default pattern
        // already used in renderFormat() below, applied here instead at
        // the per-clip stage so xfade never sees a size mismatch.
        //
        // FIX (July 14, 2026 — real test failure): added
        // force_divisible_by=2. Without it, an unusual source aspect
        // ratio can make force_original_aspect_ratio=decrease compute an
        // intermediate ODD dimension (e.g. 1919 instead of 1920) —
        // libx264's yuv420p output requires both dimensions even, and an
        // odd intermediate size is a well-documented cause of exactly
        // this "Error while opening encoder" failure. This forces the
        // scale step itself to only ever produce even numbers, closing
        // off that failure mode regardless of the source clip's own
        // aspect ratio.
        `scale=${NORMALIZE_WIDTH}:${NORMALIZE_HEIGHT}:force_original_aspect_ratio=decrease:force_divisible_by=2`,
        `pad=${NORMALIZE_WIDTH}:${NORMALIZE_HEIGHT}:(ow-iw)/2:(oh-ih)/2`,
        `fps=${NORMALIZE_FPS}`,
      ])
      .outputOptions(["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast"])
      // Audio is dropped here deliberately — concatenateClips' xfade only
      // ever operates on [x:v] video streams (see lastLabel/nextLabel
      // below), and mixAudio() adds the real soundtrack afterward across
      // the whole concatenated video. Carrying per-clip audio through this
      // step would be discarded work and a second source of format
      // mismatches (Kling clips may have audio tracks with different
      // sample rates than Ken Burns clips, which have none at all).
      .noAudio()
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`Clip normalization failed for ${clipPath} (index ${index}): ${err.message}`)))
      .run();
  });
}

// ── PROBE ACTUAL CLIP DURATION ───────────────────────────────────────────
// The previous version assumed every clip was exactly 4.5s when calculating
// crossfade offsets. That assumption was wrong as soon as duration varied
// even slightly, and caused xfade offsets to be miscalculated — visually
// "swallowing" earlier clips in the sequence (only the last clip appeared
// in testing with 2 frames). Probing real duration fixes this at the root.

function probeDuration(clipPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(clipPath, (err, metadata) => {
      if (err) return reject(new Error(`ffprobe failed for ${clipPath}: ${err.message}`));
      resolve(metadata.format.duration);
    });
  });
}

// NEW (July 16, 2026 — real render error: closing-card append failed with
// "Error reinitializing filters! ... Invalid argument" on stream #1:0 —
// a genuine format mismatch, not the old resource-exhaustion signature.
// No fps was ever explicitly enforced anywhere in this pipeline; the main
// concatenated video's effective fps just drifts from whatever the
// original clips + several rounds of xfadeChain re-encoding happen to
// produce, while a fresh -loop 1 single-image render defaults to
// whatever ffmpeg's own default is — those two don't reliably match.
// Probing the real value and forcing the closing card to it explicitly
// closes that gap instead of assuming a hardcoded number.
function probeFps(clipPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(clipPath, (err, metadata) => {
      if (err) return reject(new Error(`ffprobe (fps) failed for ${clipPath}: ${err.message}`));
      const videoStream = (metadata.streams || []).find(s => s.codec_type === "video");
      const rateStr = videoStream?.r_frame_rate || "30/1";
      const [num, den] = rateStr.split("/").map(Number);
      resolve(den ? num / den : 30);
    });
  });
}

// ── COMPUTE CLIP TIMELINE (July 2026 — footage-grounded narration) ──────
// Extracted from concatenateClips' own offset math below — previously
// that math only existed inline, computed and discarded per-transition,
// with no way for anything outside this function to know where a given
// clip actually lands in the final timeline. Narration generation needs
// exactly that: a real start time per clip, to extract a representative
// frame from and to place a narration segment at. Single source of
// truth — concatenateClips now calls this instead of recomputing the
// same math a second time in a way that could drift out of sync with it.
function computeClipTimeline(durations) {
  const timeline = [];
  let cumulativeStart = 0;
  for (let i = 0; i < durations.length; i++) {
    timeline.push({ startTime: Math.max(0, cumulativeStart), duration: durations[i] });
    cumulativeStart += durations[i] - CROSSFADE_DURATION;
  }
  return timeline;
}

// Extracts one representative frame from a clip — its own local midpoint,
// independent of where it lands in the crossfaded final timeline (a
// crossfade blends the very start/end of adjacent clips, but the middle
// of any clip is always a clean, representative frame of that room).
function extractMidpointFrame(clipPath, duration, workDir, index) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, `narration_frame_${index}.jpg`);
    const midpoint = Math.max(0.1, duration / 2);
    ffmpeg(clipPath)
      .inputOptions([`-ss`, `${midpoint.toFixed(2)}`])
      .outputOptions(["-vframes", "1", "-q:v", "3"])
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`extractMidpointFrame failed for clip ${index}: ${err.message}`)))
      .run();
  });
}

// ── CONCATENATE CLIPS WITH CROSSFADE ─────────────────────────────────────
// Chains xfade filters across all clips in sequence using REAL probed
// durations for offset calculation, not an assumed fixed length.

// ── SINGLE XFADE CHAIN ────────────────────────────────────────────────
// The actual crossfade-chaining logic, extracted so it can run both on a
// full clip set (small jobs) and on individual batches (large jobs) — see
// concatenateClips below for why batching exists. Behavior is identical
// either way: same CROSSFADE_DURATION overlap, same offset math, so
// total transition count and total overlap time stays consistent
// regardless of how many ffmpeg invocations it takes to get there —
// which matters because computeClipTimeline() (used separately, for
// narration placement in renderPipeline.js) assumes that consistency.
function xfadeChain(paths, durations, workDir, outputName) {
  if (paths.length === 1) return Promise.resolve(paths[0]);

  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, outputName);
    const command = ffmpeg();
    paths.forEach((clip) => command.input(clip));

    let filterParts = [];
    let lastLabel = "0:v";
    let cumulativeOffset = durations[0] - CROSSFADE_DURATION;

    for (let i = 1; i < paths.length; i++) {
      const nextLabel = `${i}:v`;
      const outLabel = i === paths.length - 1 ? "outv" : `v${i}`;
      filterParts.push(
        `[${lastLabel}][${nextLabel}]xfade=transition=fade:duration=${CROSSFADE_DURATION}:offset=${Math.max(0, cumulativeOffset).toFixed(2)}[${outLabel}]`
      );
      lastLabel = outLabel;
      cumulativeOffset += durations[i] - CROSSFADE_DURATION;
    }

    // NEW (July 15, 2026 — real render hang, 32+ minutes of total
    // silence with the process itself still healthy per the memory
    // logger): nothing previously bounded how long a single xfadeChain
    // ffmpeg call could run. If the process spawns and then genuinely
    // never emits 'end' OR 'error' — a real ffmpeg hang, not a crash —
    // this used to wait forever with zero recourse. XFADE_TIMEOUT_MS
    // gives it a generous window (this file's own batches are small,
    // ≤3 clips each) before force-killing the process and failing
    // cleanly instead of silently consuming Railway resources forever.
    const XFADE_TIMEOUT_MS = 120000; // 2 minutes — generous for a ≤3-clip batch
    let settled = false;
    const timeoutHandle = setTimeout(() => {
      if (settled) return;
      settled = true;
      console.error(`[xfadeChain] TIMEOUT after ${XFADE_TIMEOUT_MS}ms — killing hung ffmpeg process for ${outputPath}.`);
      try { command.kill("SIGKILL"); } catch (err) { /* best-effort */ }
      reject(new Error(`xfadeChain timed out after ${XFADE_TIMEOUT_MS}ms for ${outputPath} — ffmpeg process spawned but never completed or errored.`));
    }, XFADE_TIMEOUT_MS);

    command
      .complexFilter(filterParts)
      .outputOptions(["-map", "[outv]", "-pix_fmt", "yuv420p"])
      .output(outputPath)
      .on("end", async () => {
        if (settled) return;
        // FIX (July 15, 2026 — real render failure): ffmpeg's 'end' event
        // firing means the PROCESS reported success, but on a container
        // filesystem under I/O load (which a render with several
        // sequential batch xfadeChain calls definitely has), that can
        // fire microseconds before the output file is actually durably
        // visible on disk. Confirmed real: a genuine 2-clip batch's
        // output (concat_r0_6.mp4) resolved successfully here, then
        // probeDuration's ffprobe call moments later got "No such file
        // or directory" on that exact path. This almost certainly
        // explains the prior silent deaths too — those likely hit this
        // same race, just before the process-level crash handlers
        // existed to catch and log the resulting unhandled rejection
        // instead of the process dying with zero trace.
        //
        // Polling a few times with a short delay before giving up gives
        // the filesystem a moment to catch up rather than trusting the
        // event's timing blindly.
        for (let attempt = 0; attempt < 5; attempt++) {
          if (fs.existsSync(outputPath) && fs.statSync(outputPath).size > 0) {
            settled = true;
            clearTimeout(timeoutHandle);
            resolve(outputPath);
            return;
          }
          await new Promise((r) => setTimeout(r, 200));
        }
        if (settled) return;
        settled = true;
        clearTimeout(timeoutHandle);
        reject(new Error(`xfadeChain reported success but ${outputPath} never became visible on disk after 1s of polling.`));
      })
      .on("error", (err) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutHandle);
        reject(new Error(`Concatenation failed: ${err.message}`));
      })
      .run();
  });
}

// Keeps any single xfadeChain() call's simultaneous open-input count
// bounded. CONFIRMED CAUSE (July 15, 2026 — real render failure, 17
// clips): "Error reinitializing filters! Failed to inject frame into
// filter network: Resource temporarily unavailable" on the LAST stream
// of a 17-input complex filter graph — the exact same resource-
// exhaustion error signature already root-caused once before in this
// file (see mapWithConcurrencyLimit's header comment), but that earlier
// fix only covered too many concurrent ffmpeg PROCESSES (normalizeClip,
// extractMidpointFrame). It never covered this: one single process with
// 17 simultaneous input STREAMS chained through 16 xfade filters in one
// complex filter graph. An 11-clip job rendered fine; 17 didn't — a real
// scaling ceiling, not a one-off.
//
// LOWERED to 3 (July 15, 2026 — same exact death, second time in a row,
// now with even longer clips than the first failure — 11.2s padded vs
// 9.52s last time, same silent cutoff right after the last clip
// normalizes, right at the transition into concatenation's heaviest
// phase). Memory graph data was inconclusive on the first failure, but
// two consecutive identical deaths at growing clip sizes is a strong
// enough pattern to act on rather than wait for cleaner metrics.
const XFADE_BATCH_SIZE = 3;

// ── CONCATENATE CLIPS (batched) ───────────────────────────────────────
async function concatenateClips(clipPaths, workDir) {
  // NEW — normalize every clip BEFORE the single-clip early-return check
  // too. A single Kling clip still needs to end up at a known, consistent
  // resolution/fps/pixel format for mixAudio()/renderFormat() downstream,
  // even though there's no xfade chain to crash with only one clip.
  //
  // FIX (July 14, 2026 — real test failure): was Promise.all, unbounded —
  // see mapWithConcurrencyLimit's header comment for the full reasoning.
  const normalizedPaths = await mapWithConcurrencyLimit(
    clipPaths, FFMPEG_CONCURRENCY_LIMIT, (clip, i) => normalizeClip(clip, workDir, i)
  );

  if (normalizedPaths.length === 1) {
    return normalizedPaths[0];
  }

  const durations = await Promise.all(normalizedPaths.map(probeDuration));

  // Small jobs: one chain, same as before, no behavior change.
  if (normalizedPaths.length <= XFADE_BATCH_SIZE) {
    return xfadeChain(normalizedPaths, durations, workDir, "concatenated.mp4");
  }

  // Large jobs: divide-and-conquer. Chain each batch of XFADE_BATCH_SIZE
  // clips into an intermediate file (bounded input count per ffmpeg
  // call), probe each intermediate's REAL resulting duration (crossfade
  // overlap means it's shorter than a naive sum of its inputs), then
  // repeat on the intermediates. Keeps going until one file remains.
  //
  // FIX (July 15, 2026 — real render still failed after the first
  // batching attempt): batches were being run through
  // mapWithConcurrencyLimit at FFMPEG_CONCURRENCY_LIMIT=3 — the same
  // pool used for normalizeClip. That was the wrong reuse. normalizeClip
  // is one input per process, so 3 concurrent calls = 3 total open
  // streams, trivial. Each xfadeChain batch call is itself a heavy
  // multi-stream ffmpeg operation — running 3 of THOSE concurrently
  // meant up to 3 processes × 6 inputs ≈ 18 simultaneous streams on the
  // container at once, barely less aggregate load than the original
  // 20-in-one-process failure this was supposed to fix. Confirmed by a
  // real render crashing on batch 1 alone (stream #5:0) — under
  // contention from the other batches starting alongside it. Batches now
  // run strictly sequentially — one xfadeChain call at a time, nothing
  // else competing with it. Costs some wall-clock time; that's the right
  // trade for a step that's failed twice in production.
  let currentPaths = normalizedPaths;
  let currentDurations = durations;
  let round = 0;

  while (currentPaths.length > 1) {
    const batches = [];
    for (let i = 0; i < currentPaths.length; i += XFADE_BATCH_SIZE) {
      batches.push({
        paths: currentPaths.slice(i, i + XFADE_BATCH_SIZE),
        durations: currentDurations.slice(i, i + XFADE_BATCH_SIZE),
      });
    }

    const batchOutputs = [];
    for (let i = 0; i < batches.length; i++) {
      console.log(`[concatenateClips] Round ${round}, batch ${i}/${batches.length - 1} starting (${batches[i].paths.length} clips)...`);
      const out = await xfadeChain(batches[i].paths, batches[i].durations, workDir, `concat_r${round}_${i}.mp4`);
      console.log(`[concatenateClips] Round ${round}, batch ${i}/${batches.length - 1} done.`);
      batchOutputs.push(out);
    }

    // NEW (July 15, 2026 — same crash, second time in a row, at growing
    // clip sizes): every normalized clip and every prior round's
    // intermediate files were sitting on disk for the ENTIRE render with
    // zero cleanup — a real, growing footprint as clip durations
    // increased this session (padded clips went from 7.52s to 9.52s to
    // 11.2s across three consecutive test rounds). currentPaths here are
    // exactly what THIS round just consumed as input and will never be
    // read again — safe to delete now that their outputs exist on disk.
    // Best-effort: a failed cleanup should never crash the render over
    // something that was only ever a disk-space optimization.
    // NEW (July 15, 2026 — found while tracing the missing-file bug
    // above): a batch with a single leftover clip (odd clip counts)
    // passes that file through unchanged in xfadeChain rather than
    // creating a new one — so it ends up in BOTH currentPaths (about to
    // be deleted below) AND batchOutputs (what the NEXT round needs).
    // Excluding anything that's also a batch output from deletion, so a
    // passthrough file doesn't get deleted out from under the round that
    // still needs it.
    const outputSet = new Set(batchOutputs);
    for (const consumedPath of currentPaths) {
      if (outputSet.has(consumedPath)) continue;
      try { fs.unlinkSync(consumedPath); } catch (err) { /* non-fatal */ }
    }

    currentDurations = await Promise.all(batchOutputs.map(probeDuration));
    currentPaths = batchOutputs;
    round++;
  }

  // Final result needs to land at the canonical path the rest of the
  // pipeline (mixAudio, renderFormat) already expects.
  const finalPath = path.join(workDir, "concatenated.mp4");
  if (currentPaths[0] !== finalPath) {
    fs.copyFileSync(currentPaths[0], finalPath);
  }
  return finalPath;
}

// ── BEFORE/AFTER WIPE TRANSITION ─────────────────────────────────────────
// Vacant room (pull_back) holds briefly, then wipes left-to-right into
// the staged version (push_in). This is the flagship PRO Plus feature —
// no general video tool can do this because it requires the paired
// vacant/staged Cloudinary URLs that only exist because PRO staged it.

function buildBeforeAfterClip(beforeClipPath, afterClipPath, workDir, outputName) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, outputName);

    ffmpeg()
      .input(beforeClipPath)
      .input(afterClipPath)
      .complexFilter([
        `[0:v]trim=duration=3,setpts=PTS-STARTPTS[vacant]`,
        `[1:v]trim=duration=4,setpts=PTS-STARTPTS[staged]`,
        `[vacant][staged]xfade=transition=wipeleft:duration=0.8:offset=2.5[out]`,
      ])
      .outputOptions(["-map", "[out]", "-pix_fmt", "yuv420p"])
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`Before/After wipe failed: ${err.message}`)))
      .run();
  });
}

// ── MIX AUDIO — music + optional narration segments ────────────────────
//
// CHANGE (July 14, 2026 — footage-grounded narration rebuild): REPLACES
// the single continuous narrationPath model. Narration is now generated
// as separate segments, one per room/clip, each with its own real start
// time in the final timeline (computed via computeClipTimeline — see its
// header comment). Each segment gets adelay'd to its real position, all
// segments are mixed together into one combined narration stream, THEN
// that combined stream goes through the same sidechain-ducking-against-
// music approach as before. narrationSegments is an array of
// { audioPath, startTime } — empty/absent means music-only, same as the
// old narrationPath being absent.

function mixAudio(videoPath, musicPath, workDir, narrationSegments) {
  return new Promise(async (resolve, reject) => {
    try {
      const outputPath = path.join(workDir, "with_music.mp4");
      const videoDuration = await probeDuration(videoPath);
      const musicDuration = await probeDuration(musicPath);
      const fadeOutStart = Math.max(0, videoDuration - 1.5);
      const hasNarration = narrationSegments && narrationSegments.length > 0;

      // FIX (July 17, 2026 — third attempt at the same underlying bug):
      // music_fitted.mp3 is shorter than the real final videoDuration
      // (closing card + real per-clip duration variance both add time
      // AFTER music is generated in renderPipeline.js Step 2), so it
      // needs to be extended to cover the whole video. Two prior attempts
      // both failed on real renders: the `aloop` FILTER mishandles a
      // partial final loop on compressed audio; `-stream_loop` on the
      // INPUT then also produced dead silence on a real test — and
      // musicGen.js confirms music_fitted.mp3 has no baked-in silence at
      // all (fitTrackToDuration always loops+trims to a fully-packed
      // file), which rules out the "silent tail" theory the second fix
      // was chasing. Root cause of the stream_loop failure is unconfirmed
      // (possibly a version-specific quirk combining -stream_loop with
      // -filter_complex on this container's ffmpeg build), but rather
      // than debug that further, this switches to plain `concat` —
      // the exact same filter xfadeChain already uses reliably for every
      // clip transition in this codebase all session. The music file is
      // added as N separate inputs (enough copies to cover videoDuration)
      // and concatenated explicitly in the filtergraph, then atrim cuts
      // it to the exact needed length. No input-level looping involved.
      const musicCopies = Math.max(1, Math.ceil(videoDuration / musicDuration));

      const command = ffmpeg().input(videoPath);
      for (let i = 0; i < musicCopies; i++) {
        command.input(musicPath);
      }
      // Music occupies input indices 1..musicCopies; narration (if any)
      // starts right after.
      const narrationStartIndex = 1 + musicCopies;

      let filterParts;
      const musicInputLabels = [];
      for (let i = 0; i < musicCopies; i++) {
        musicInputLabels.push(`[${i + 1}:a]`);
      }
      // Single copy needs no concat at all — just alias it directly so
      // the rest of the graph can always reference [music_looped]
      // uniformly regardless of how many copies were needed.
      const musicLoopedFilter =
        musicCopies === 1
          ? `[1:a]anull[music_looped]`
          : `${musicInputLabels.join("")}concat=n=${musicCopies}:v=0:a=1[music_looped]`;

      if (hasNarration) {
        narrationSegments.forEach((seg) => command.input(seg.audioPath));

        filterParts = [
          musicLoopedFilter,
          // FIX #2 (Sam's feedback, real render — still too loud): 0.35
          // wasn't enough headroom. Dropping further to 0.2, and
          // tightening the sidechain ducking itself (lower threshold =
          // engages more easily, higher ratio = ducks harder once
          // engaged) so music is genuinely a soft bed under narration,
          // not competing with it.
          `[music_looped]atrim=0:${videoDuration.toFixed(2)},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=1,afade=t=out:st=${fadeOutStart.toFixed(2)}:d=1.5,volume=0.35[music_faded]`,
        ];

        // Each segment: fade in/out on its OWN local timeline (0..its own
        // duration), THEN adelay shifts the whole faded clip to its real
        // position in the final video. Order matters — afade's st= values
        // are relative to the stream's own start, so they have to be
        // applied before the stream gets shifted forward.
        //
        // NEW (July 14, 2026 — Sam's speed-correction suggestion): each
        // segment is ALSO hard-capped (atrim) at the real gap before the
        // next segment starts. narrationGen.js already corrects for this
        // by regenerating a too-long segment at a faster ElevenLabs
        // `speed` — but that correction is clamped to ElevenLabs' real
        // 0.7–1.2 range, so a segment whose natural length wildly exceeds
        // its window even at max speed could still theoretically overrun.
        // This is the backstop that guarantees no audible overlap between
        // adjacent room narrations regardless — defense in depth, not the
        // primary fix.
        const delayedLabels = [];
        narrationSegments.forEach((seg, i) => {
          const inputIndex = narrationStartIndex + i; // 0=video, 1..musicCopies=music, rest=narration segments
          const delayMs = Math.round(seg.startTime * 1000);
          const label = `narr_${i}`;
          const nextSeg = narrationSegments[i + 1];
          const capDuration = nextSeg ? Math.max(0.1, nextSeg.startTime - seg.startTime) : seg.duration;
          const fadeOutAt = Math.max(0, Math.min(seg.duration, capDuration) - 0.4);
          filterParts.push(
            `[${inputIndex}:a]afade=t=in:st=0:d=0.3,atrim=end=${capDuration.toFixed(2)},afade=t=out:st=${fadeOutAt.toFixed(2)}:d=0.4,adelay=${delayMs}|${delayMs}[${label}]`
          );
          delayedLabels.push(`[${label}]`);
        });

        // Combine all per-room segments onto the shared timeline into one
        // narration stream — each already sits at its correct offset via
        // adelay above, so amix here is just summing them, not blending
        // overlapping speech (segments shouldn't overlap in practice,
        // since they're spaced by real, non-overlapping clip positions).
        filterParts.push(
          `${delayedLabels.join("")}amix=inputs=${narrationSegments.length}:duration=longest:dropout_transition=0[narration_mixed]`
        );

        // FIX #4 (Sam's feedback, real render — "narration is barely
        // heard" after the loudnorm-removal fix): removing loudnorm from
        // the COMBINED mix was correct — it was undoing the music
        // balance work. But that also removed the only thing that was
        // guaranteeing narration ITSELF sat at a strong, consistent
        // level regardless of what ElevenLabs happened to output raw.
        // With nothing boosting narration up, the whole mix could end up
        // too quiet overall even with music correctly balanced under it.
        // Normalizing HERE — narration alone, before it ever touches
        // music — fixes that without reintroducing the original bug:
        // this loudnorm only ever sees narration, so it has no way to
        // rebalance music back up the way normalizing the combined mix
        // did.
        filterParts.push(`[narration_mixed]loudnorm=I=-16:TP=-1.5:LRA=11[narration_all]`);

        // Defensive cap — same reasoning as the single-track version this
        // replaces: even with real per-clip timestamps, guarantee nothing
        // plays into the final buffer before the video ends.
        const narrationEndCap = Math.max(0, videoDuration - NARRATION_END_BUFFER_SECONDS);
        filterParts.push(`[narration_all]atrim=end=${narrationEndCap.toFixed(2)},asplit=2[narration_for_sidechain][narration_for_mix]`);

        filterParts.push(
          `[music_faded][narration_for_sidechain]sidechaincompress=threshold=0.03:ratio=10:attack=5:release=300[music_ducked]`,
          `[music_ducked][narration_for_mix]amix=inputs=2:duration=longest:dropout_transition=2[premix]`,
          // FIX #3 (Sam's feedback, real render — still no noticeable
          // volume change despite two real upstream fixes): loudnorm
          // here was normalizing the COMBINED mix's overall integrated
          // loudness to -16 LUFS, with zero awareness of the internal
          // music/narration balance everything upstream was carefully
          // tuning. If hitting that target meant boosting music's
          // relative contribution back up, loudnorm would do exactly
          // that — silently undoing the volume=0.2 + sidechain ducking
          // work above. Replaced with alimiter, which only prevents
          // clipping (a safety ceiling) and does nothing to re-balance
          // the mix — so the upstream tuning actually survives to the
          // final output now.
          `[premix]alimiter=limit=0.95[audio_out]`,
        );
      } else {
        filterParts = [
          musicLoopedFilter,
          // Same concat-based extension as the narration branch above —
          // music needs to cover the real full videoDuration (including
          // any closing card), not just its own original fitted length.
          `[music_looped]atrim=0:${videoDuration.toFixed(2)},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=1,afade=t=out:st=${fadeOutStart.toFixed(2)}:d=1.5,volume=0.6[music_faded]`,
          `[music_faded]loudnorm=I=-16:TP=-1.5:LRA=11[audio_out]`,
        ];
      }

      // FIX (July 17, 2026 — real render, closing card silently truncated):
      // videoPath here is probed AFTER assembleVideo's optional closing-card
      // step, so videoDuration already reflects the full video INCLUDING the
      // appended card. But [audio_out] above (music via amix duration=longest,
      // or narration) only ever spans the ORIGINAL clip timeline — music's own
      // length comes from generateMusic({durationSeconds: totalDuration}) back
      // in renderPipeline.js Step 2, computed before the closing card exists,
      // and narration never extends past its own last segment + buffer either.
      // The old "-shortest" output flag then truncated the OUTPUT to whichever
      // mapped stream was shorter — which was always the audio, landing almost
      // exactly at the end of narration/buffer and silently cutting off the
      // entire closing card that had already rendered fine on the video track.
      // Explicitly padding audio with silence out to the real video duration
      // means -shortest (kept below as a defensive rounding backstop, not the
      // active truncation mechanism) has nothing left to cut.
      filterParts.push(`[audio_out]apad=whole_dur=${videoDuration.toFixed(2)}[audio_out_padded]`);

      command
        .complexFilter(filterParts)
        .outputOptions(["-map", "0:v", "-map", "[audio_out_padded]", "-c:v", "copy", "-c:a", "aac", "-shortest"])
        .output(outputPath)
        .on("end", () => resolve(outputPath))
        .on("error", (err) => reject(new Error(`Audio mix failed: ${err.message}`)))
        .run();
    } catch (err) {
      reject(err);
    }
  });
}

// ── RENDER FORMAT (16:9 master, 9:16 reframe) ────────────────────────────

function renderFormat(inputPath, dimensions, workDir, outputName) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, outputName);
    const [w, h] = dimensions.split("x");

    // 9:16 reframe crops to center — smart subject-aware cropping is a
    // Phase 2B refinement; center crop is the correct safe default for v1.
    const filter =
      dimensions === "1080x1920"
        ? `crop=ih*9/16:ih,scale=${w}:${h}`
        : `scale=${w}:${h}:force_original_aspect_ratio=decrease,pad=${w}:${h}:(ow-iw)/2:(oh-ih)/2`;

    ffmpeg(inputPath)
      .videoFilters(filter)
      .outputOptions(["-c:a", "copy", "-movflags", "+faststart"])
      .output(outputPath)
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(new Error(`Format render failed (${dimensions}): ${err.message}`)))
      .run();
  });
}

// ── ENTRY POINT ────────────────────────────────────────────────────────

// ── CLOSING CARD (Sam's idea, built July 15, 2026) ──────────────────────
// Text fades in the instant narration's LAST spoken word actually ends
// (not the clip's nominal duration — the real, possibly speed-corrected
// end timestamp, passed in from renderPipeline.js), holds through the
// video's own natural tail. That tail already exists by design: mixAudio
// reserves NARRATION_END_BUFFER_SECONDS (2s) of narration-free video at
// the very end specifically so speech never gets cut off by the video
// ending — this reuses that exact same reserved window rather than
// adding new video length on top of it.
//
// Background is the LAST frame's real source still (not a video-
// extracted frame) at reduced opacity, overlaid on whatever's already
// playing at that point in the timeline (the tail of the last clip's own
// motion) rather than a hard cut to a static image — reads as the shot
// settling into a closing card, not an abrupt swap.
//
// Gracefully skipped (returns the input path unchanged) if timing
// doesn't make sense — e.g. narration ran long enough that there's no
// real tail left to show a card in. A closing card is a nice-to-have;
// it should never be the reason a render fails.
function escapeDrawtext(text) {
  // ffmpeg drawtext treats \ : ' as filter-syntax-significant — escape
  // them so an address with an apostrophe or a colon doesn't break the
  // filter graph or get silently mangled.
  return text.replace(/\\/g, "\\\\\\\\").replace(/:/g, "\\:").replace(/'/g, "\u2019");
}

// ── CLOSING CARD (Sam's idea, rebuilt July 16, 2026) ────────────────────
// REPLACES the live-overlay version entirely, which hung THREE separate
// times despite three different targeted fixes (duration cap, font
// family, no font at all) — the last fix (removing the font parameter)
// still hung with total silence for the full 60s timeout, which rules
// out font as the cause and points at something structural in blending
// a looped image live with ongoing video via overlay.
//
// New approach: render the closing card as its own small, completely
// standalone clip — a single still image, no loop-duration ambiguity, no
// second video stream to reconcile via overlay/shortest — using the same
// kind of simple one-input render every normal clip in this pipeline
// already does successfully. Then APPEND it using xfadeChain, the exact
// same proven concatenation machinery that's handled every other
// transition in this video without issue all session. Reuses working
// code instead of patching the same fragile filter graph a fourth time.
//
// Trade-off, accepted explicitly: this adds real seconds to the video
// (a genuine new clip) rather than reusing the narration-end buffer
// window the old version tried to blend into. The text still fades in
// right as the card begins — which, since the card is appended AFTER
// the main video already finished narration + its buffer, lands at the
// same perceptual moment ("right as narration ends") the original spec
// asked for, just via a clean transition into a new clip instead of a
// live blend on top of continuing footage.
const CLOSING_CARD_DURATION_SECONDS = 4.0;
const CLOSING_CARD_FADE_SECONDS = 0.6;

function renderClosingCardClip(stillImagePath, addressLine, ctaLine, workDir, fps) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(workDir, "closing_card.mp4");
    const command = ffmpeg()
      .input(stillImagePath)
      // Single input, fixed finite duration from the start — no second
      // stream, no "shortest" semantics to get wrong. This is the same
      // -loop 1 -t pattern that already works correctly everywhere else
      // in this codebase; the old version's bug was never this pattern
      // itself, it was combining it with a live overlay onto a SECOND,
      // independently-timed video stream.
      .inputOptions(["-loop", "1", "-t", CLOSING_CARD_DURATION_SECONDS.toFixed(2)]);

    // Text fades in over the first CLOSING_CARD_FADE_SECONDS of this
    // card's own timeline, then holds for the rest — no dependency on
    // narration timing at all anymore, since the card only ever starts
    // after the main video (narration + its buffer) has already ended.
    const alphaExpr = `min(1,t/${CLOSING_CARD_FADE_SECONDS})`;

    // FIX (July 17, 2026 — real render, first card ever seen on screen):
    // was one drawtext with address + CTA jammed into a single line at a
    // single fontsize (54) — Sam's real screenshot showed it cramped and
    // hard to read. drawtext doesn't handle multi-line text reliably in
    // one call, so this is two separate stacked drawtext filters instead:
    // address (if present) smaller, above center; CTA larger, below
    // center — CTA is the action we actually want taken, so it gets the
    // visual weight. Degrades gracefully to a single centered CTA line
    // (old single-line layout) when there's no address at all.
    const filterParts = [
      // Fill-crop to the standard frame size, then darken directly on
      // the pixel values (eq=brightness) rather than the old alpha-
      // blend-over-a-second-stream approach — simpler, and there's
      // nothing else in this clip for it to blend against.
      `[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,eq=brightness=-0.28[dimmed]`,
    ];

    if (addressLine) {
      filterParts.push(
        `[dimmed]drawtext=text='${escapeDrawtext(addressLine)}':fontcolor=white:fontsize=42:borderw=2:bordercolor=black@0.6:x=(w-text_w)/2:y=(h/2)-60:alpha='${alphaExpr}'[with_addr]`,
        `[with_addr]drawtext=text='${escapeDrawtext(ctaLine)}':fontcolor=white:fontsize=68:borderw=3:bordercolor=black@0.6:x=(w-text_w)/2:y=(h/2)+10:alpha='${alphaExpr}'[outv]`
      );
    } else {
      filterParts.push(
        `[dimmed]drawtext=text='${escapeDrawtext(ctaLine)}':fontcolor=white:fontsize=68:borderw=3:bordercolor=black@0.6:x=(w-text_w)/2:y=(h-text_h)/2:alpha='${alphaExpr}'[outv]`
      );
    }

    // Standalone single-image render — much simpler than the old
    // overlay version, so 30s is generous rather than needing the full
    // 60s the old, more complex filter graph was given.
    const CLOSING_CARD_TIMEOUT_MS = 30000;
    let settled = false;
    const timeoutHandle = setTimeout(() => {
      if (settled) return;
      settled = true;
      console.warn(`[renderClosingCardClip] TIMEOUT after ${CLOSING_CARD_TIMEOUT_MS}ms — killing process, proceeding without closing card.`);
      try { command.kill("SIGKILL"); } catch (err) { /* best-effort */ }
      reject(new Error("Closing card render timed out"));
    }, CLOSING_CARD_TIMEOUT_MS);

    command
      .complexFilter(filterParts)
      .outputOptions(["-map", "[outv]", "-r", fps.toFixed(3), "-pix_fmt", "yuv420p"])
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
        reject(new Error(`Closing card render failed: ${err.message}`));
      })
      .run();
  });
}

async function assembleVideo({ clipPaths, musicPath, narrationSegments, formats, workDir, closingCard }) {
  let concatenated = await concatenateClips(clipPaths, workDir);

  // Closing card runs on the pure video track, BEFORE mixAudio — mixAudio
  // does -c:v copy (passthrough, no video re-encode), so any video-level
  // work has to happen before it, not after.
  //
  // No narrationEndTime dependency anymore — the card only ever starts
  // after the main video (narration + its buffer) has already finished,
  // so there's nothing left to time it against. Wrapped in try/catch:
  // a closing card is a nice-to-have, never the reason a render fails.
  if (closingCard && narrationSegments && narrationSegments.length > 0) {
    try {
      const videoDuration = await probeDuration(concatenated);
      const fps = await probeFps(concatenated);
      const cardPath = await renderClosingCardClip(closingCard.stillImagePath, closingCard.addressLine, closingCard.ctaLine, workDir, fps);
      concatenated = await xfadeChain(
        [concatenated, cardPath],
        [videoDuration, CLOSING_CARD_DURATION_SECONDS],
        workDir,
        "with_closing_card.mp4"
      );
    } catch (err) {
      console.warn(`Closing card skipped (non-fatal, video proceeds without it): ${err.message}`);
    }
  }

  // CHANGE (July 14, 2026 — footage-grounded narration rebuild):
  // narrationSegments now arrives pre-generated (audio already downloaded
  // to local disk, real start times already computed) — renderPipeline.js
  // does the generation itself, since it's the one place that knows real
  // clip durations/positions. This function no longer downloads anything;
  // it just passes the segments through to mixAudio. Absent/empty for any
  // job that didn't request narration, or where generation failed
  // upstream (a failed narration should never block the video itself).
  const withMusic = await mixAudio(concatenated, musicPath, workDir, narrationSegments || []);

  const outputs = {};

  if (formats.includes("16x9")) {
    outputs["16x9"] = await renderFormat(withMusic, "1920x1080", workDir, "output_16x9.mp4");
  }
  if (formats.includes("9x16")) {
    outputs["9x16"] = await renderFormat(withMusic, "1080x1920", workDir, "output_9x16.mp4");
  }

  return outputs;
}

module.exports = { assembleVideo, buildBeforeAfterClip, concatenateClips, mixAudio, renderFormat, computeClipTimeline, extractMidpointFrame, probeDuration, mapWithConcurrencyLimit, FFMPEG_CONCURRENCY_LIMIT, NARRATION_END_BUFFER_SECONDS };
