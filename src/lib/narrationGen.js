// narrationGen.js — Footage-grounded, timestamp-synced narration.
//
// REPLACES the Netlify-side generate-narration-background.js model
// entirely (that file generated ONE continuous script from room LABELS
// alone, before any rendering happened — Claude never saw the actual
// footage, and the script had no real relationship to clip timing beyond
// a rough duration estimate). Sam's explicit direction (July 14, 2026):
// generic voiceover isn't worth shipping — narration needs to genuinely
// reflect what's in the footage, tastefully, not as a mechanical
// second-by-second description.
//
// This runs HERE, in Railway, AFTER motion clips are rendered — not in
// Netlify before rendering — because it needs two things that don't
// exist until then: the real rendered footage to look at, and each
// clip's real position/duration in the final timeline (computeClipTimeline
// in assemble.js). One Claude vision call sees one representative frame
// per room (assemble.js's extractMidpointFrame) and writes a script
// broken into per-room segments; each segment gets its own ElevenLabs
// call and its own real timestamp; assemble.js's mixAudio places each
// at its correct position instead of playing one track from t=0.

const https = require("https");
const fs = require("fs");
const path = require("path");
const { probeDuration, NARRATION_END_BUFFER_SECONDS } = require("./assemble");

const SPEAKING_RATE_WORDS_PER_MINUTE = 135; // LOWERED from 150 (Sam's feedback, real render — narration almost universally hit MAX_SPEED and still ran over): extra safety margin on top of the content-ambition fix below, which is the primary cause.
const MIN_SEGMENT_WORDS = 6; // even a 3-second bathroom shot gets a real sentence, not one word
const MIN_SPEED = 0.7;
const MAX_SPEED = 1.2; // ElevenLabs' real supported range — confirmed via voice_settings.speed
const MAX_FRAMES_PER_GROUP = 4; // cap on how many representative stills go to one vision call per group
// NEW (Sam's feedback — "there is never a breath between clips"): segments
// were timed to fit right up to the instant the NEXT one starts, with zero
// deliberate silence built in. Subtracting this from the available window
// (below) guarantees real, audible breathing room between rooms instead of
// narration butting up against itself.
const SEGMENT_BREATHING_ROOM_SECONDS = 0.5;

function wordBudgetForSegment(durationSeconds) {
  // Slightly tighter than the whole-video version this replaces (no
  // separate end-buffer subtraction needed per segment — mixAudio's
  // overall NARRATION_END_BUFFER_SECONDS backstop still guards the very
  // end of the whole video; individual segments just need to roughly fit
  // their own clip).
  const words = Math.round((durationSeconds / 60) * SPEAKING_RATE_WORDS_PER_MINUTE);
  return Math.max(MIN_SEGMENT_WORDS, words);
}

// Room-type tags where a listing having MULTIPLE distinct instances is
// common — unlike "Kitchen" or "Primary Bedroom," which are typically
// singular per listing. CONFIRMED REAL BUG (Sam's feedback, a genuine
// 4-bedroom listing): 3 secondary bedrooms all tagged generic "Bedroom"
// (only the primary gets its own distinct tag in the UI — see
// ROOM_TYPES in build-video-demo.html) got merged into one narrated
// group when shot back-to-back, since groupContiguousByRoom had no way
// to tell "3 angles of 1 bedroom" apart from "1 angle each of 3
// different bedrooms" — both look identical as contiguous same-label
// runs. Disambiguating by occurrence order below fixes it for the
// common case (one photo per room) at the cost of over-splitting the
// rarer case (someone genuinely shoots multiple angles of the SAME
// secondary bedroom) — a soft failure (slightly more, smaller segments)
// that's clearly preferable to the hard failure this replaces
// (confidently wrong/skipped room content).
const REUSABLE_ROOM_TYPES = new Set(["Bedroom", "Bathroom", "Office", "Flex Room", "Other"]);

function disambiguateRoomLabels(localFrames) {
  const seenCounts = {};
  return localFrames.map((frame) => {
    const raw = frame.roomLabel || frame.roomType || "this room";
    if (!REUSABLE_ROOM_TYPES.has(raw)) return raw;
    seenCounts[raw] = (seenCounts[raw] || 0) + 1;
    return seenCounts[raw] === 1 ? raw : `${raw} ${seenCounts[raw]}`;
  });
}

// ── GROUP CONTIGUOUS CLIPS BY ROOM ───────────────────────────────────────
// REPLACES strict 1:1 segment-per-clip mapping (the root cause of the
// fragmentation Sam flagged after hearing real output — a 3-4.5s clip
// only budgets ~10 words, not enough for a complete thought even with
// speed correction working correctly). Users naturally order multiple
// angles/crops of the same room back-to-back — video_job_frames already
// carries room_label per clip, so a contiguous run of matching labels is
// a free, reliable grouping boundary. A run of length 1 just becomes a
// single-clip group, handled identically to a multi-clip one — no special
// case needed anywhere downstream.
//
// localFrames: the same array renderPipeline.js already has (room_label
// per frame). framePaths/timeline: parallel arrays, same length/order,
// already produced by extractMidpointFrame + computeClipTimeline.
//
// Returns: [{ index, roomLabel, framePaths: [...], startTime, duration }]
// — same shape generateSegmentedScript/generateNarration already expect,
// except framePath (singular) is now framePaths (array).
function groupContiguousByRoom(localFrames, framePaths, timeline) {
  const disambiguatedLabels = disambiguateRoomLabels(localFrames);
  const groups = [];
  for (let i = 0; i < localFrames.length; i++) {
    const roomLabel = disambiguatedLabels[i];
    const prior = groups[groups.length - 1];
    if (prior && prior.roomLabel === roomLabel) {
      prior.framePaths.push(framePaths[i]);
      // Group duration = span from the group's own start to the end of
      // its own last clip (used for the word-budget prompt below). This
      // is deliberately NOT the same as availableWindow (computed later
      // in generateNarration from consecutive groups' real startTimes) —
      // that one accounts for how much real silence exists before the
      // NEXT group starts talking, which is what actually gates whether
      // a too-long read needs speed correction.
      prior.duration = (timeline[i].startTime + timeline[i].duration) - prior.startTime;
    } else {
      groups.push({
        roomLabel,
        framePaths: [framePaths[i]],
        startTime: timeline[i].startTime,
        duration: timeline[i].duration,
      });
    }
  }

  // FIX (Sam's feedback, real render — outro cut off mid-sentence after
  // "schedule your Priv..."): mixAudio's NARRATION_END_BUFFER_SECONDS
  // hard-trims the ENTIRE narration track 2s before the video's real
  // end, regardless of what any individual segment needed. The final
  // group's word budget was being calculated against its full padded
  // clip duration, with no awareness that buffer eats back into exactly
  // the room the intro/outro padding was meant to create — so the CTA
  // script routinely got written long enough to need time that the
  // buffer then hard-cuts. Subtracting the buffer here means the LAST
  // group's script is written to what will actually survive to air,
  // not what fits before a truncation nobody told it about.
  if (groups.length > 0) {
    const lastGroup = groups[groups.length - 1];
    lastGroup.duration = Math.max(1, lastGroup.duration - NARRATION_END_BUFFER_SECONDS);
  }

  // Cap representative frames per group — a 5-angle primary bedroom
  // shouldn't send 5 full images to one vision call. Keep first, last,
  // and evenly spaced frames in between for real angle coverage rather
  // than just the first N in sequence.
  groups.forEach((g) => {
    if (g.framePaths.length > MAX_FRAMES_PER_GROUP) {
      const step = (g.framePaths.length - 1) / (MAX_FRAMES_PER_GROUP - 1);
      const keepIndices = [...new Set(
        Array.from({ length: MAX_FRAMES_PER_GROUP }, (_, k) => Math.round(k * step))
      )];
      g.framePaths = keepIndices.map((idx) => g.framePaths[idx]);
    }
  });

  return groups.map((g, i) => ({ index: i, ...g }));
}

// ── SEGMENTED, VISION-GROUNDED SCRIPT ────────────────────────────────────
// segments: [{ index, roomLabel, framePaths: [...], duration }] — one or
// more representative frames per segment (see groupContiguousByRoom).
// Returns: [{ index, text }] in the same order.
function generateSegmentedScript(address, segments, apiKey) {
  return new Promise((resolve, reject) => {
    const segmentDescriptions = segments
      .map((s, i) => `Group ${i + 1}: "${s.roomLabel}" — ${s.framePaths.length} frame(s) shown for this group — target ${wordBudgetForSegment(s.duration)} words max (this group plays for about ${s.duration.toFixed(1)}s total in the final video).`)
      .join("\n");

    const promptText = `You are writing narration for a real estate video tour. You are shown one or more still frames per group, grouped in the order they appear in the video.

IMPORTANT: frames are grouped because they were tagged with the same room-type label (e.g. "Bathroom") and appear consecutively. Most of the time that means they genuinely ARE the same physical room shown from different angles or crops — but room-type tags don't guarantee that. Occasionally a group will actually contain two distinct rooms of the same type placed back-to-back (e.g. a primary bath, then a guest bath). Look at what's actually different across the frames in each group and judge for yourself:
- If they're clearly the same space (same fixtures, same finishes, same layout from a different angle) — narrate it as one continuous room description, as usual.
- If they're clearly DIFFERENT rooms that just share a type — say so distinctly in the narration (e.g. "...and down the hall, a second full bath offers..."), don't blend two different rooms into one generic description as if they were one.

Address: ${address || "this property"}

${segmentDescriptions}

IMPORTANT — exterior accuracy: for any group whose room label suggests an EXTERIOR shot (exterior, yard, backyard, front yard, pool, patio, curb appeal), be cautious about ONE specific thing: this platform sometimes adds outdoor furniture/staging (a firepit, patio dining set, planters) or enhances landscaping (fresh lawn, trimmed hedges, seasonal flowers) that may not reflect the property's real, permanent condition. Do NOT describe added outdoor furniture or specific landscaping details (lawn condition, plantings, hedges) as if they're confirmed, permanent features of the property. Time of day and lighting (golden hour warmth, twilight glow) ARE safe to describe normally — that's just atmosphere, not a factual claim about the property itself. Stick to the home's own architectural features (roofline, siding material, window style) plus safe lighting/mood description; just avoid asserting specific outdoor furnishings or landscaping as real. Interior rooms don't need this caution; describe what you actually see in interiors normally.

Write ONE short narration segment per GROUP (not per frame). Pick exactly ONE specific, vivid detail per group — not a list, not room type plus finishes plus layout plus a value statement. One real, grounded observation (a material, a light quality, a single standout feature) said naturally, the way someone would mention the one thing that actually caught their eye walking through — not an inventory of the room. Say the room's name once when you start describing it — don't re-announce it for every frame that's clearly still the same space — but if the group turns out to span more than one real room (see above), make sure narration reflects that rather than silently describing only one of them.

This is NOT a mechanical, second-by-second description of each image, and it is NOT a features list — it should read like someone who toured the home and mentioned the ONE thing worth noting about each room in passing, tastefully, in a warm, professional, conversational tone, then moved on. The segments together should feel like one continuous, cohesive walkthrough — each one can build on the last — not a series of disconnected blurbs, and never more than ONE sentence, two at the absolute most, per group.

Rules:
- Third person only (never "I" or "my listing").
- Never invent square footage, bedroom/bathroom counts, or amenities not visible in the photos.
- ONE sentence per group. Two only if both are short. This is the single most important constraint — a segment that tries to name the room, list what's in it, AND editorialize about value will not fit its window and will get cut off mid-sentence. Pick the one detail that matters most and say only that.
- Respect each group's word target — it's timed to that group's real length; going over means it gets cut off mid-sentence.
- The FINAL group's segment must still include one real, specific detail about that room — exactly like every other group. Don't drop the description just because it's last. If (and only if) there's genuinely room left in that group's word budget after the detail, it may close with a brief, natural phrase inviting a showing, tacked onto the end of that same sentence (e.g. "...ready for its next owner to call home."). The room detail always comes first and is never cut to make space for the closing phrase — if the budget is tight, skip the closing phrase entirely rather than shortening the detail. There is already a separate closing card with its own call-to-action shown after this video ends, so this closing phrase is a nice-to-have flourish here, not the segment's main job.
- Return ONLY a JSON array, nothing else — no markdown fences, no prose before or after. Exact shape: [{"index": 0, "text": "..."}, {"index": 1, "text": "..."}, ...] with exactly ${segments.length} entries, one per group, in the order shown.`;

    const content = [{ type: "text", text: promptText }];
    segments.forEach((s, i) => {
      s.framePaths.forEach((framePath) => {
        const imageData = fs.readFileSync(framePath).toString("base64");
        content.push({
          type: "image",
          source: { type: "base64", media_type: "image/jpeg", data: imageData },
        });
      });
    });

    const bodyStr = JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 2000,
      messages: [{ role: "user", content }],
    });

    const req = https.request({
      hostname: "api.anthropic.com",
      path: "/v1/messages",
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(bodyStr),
      },
    }, (res) => {
      let data = "";
      res.on("data", (c) => (data += c));
      res.on("end", () => {
        try {
          const parsed = JSON.parse(data);
          const text = parsed?.content?.find((b) => b.type === "text")?.text;
          if (!text) return reject(new Error(`Claude returned no script text: ${data.slice(0, 300)}`));
          // Strip markdown fences defensively — the prompt explicitly
          // forbids them, but models occasionally add them anyway.
          const cleaned = text.trim().replace(/^```(json)?/i, "").replace(/```$/, "").trim();
          const segmentsOut = JSON.parse(cleaned);
          if (!Array.isArray(segmentsOut)) throw new Error("Claude's response was not a JSON array");
          resolve(segmentsOut);
        } catch (e) {
          reject(new Error(`Claude segmented-script response parse error: ${e.message}. Raw: ${data.slice(0, 500)}`));
        }
      });
    });
    req.on("error", reject);
    req.write(bodyStr);
    req.end();
  });
}

// ── PER-SEGMENT TTS (ElevenLabs) ─────────────────────────────────────────
// speed is optional — omitted on the first attempt (ElevenLabs' own
// default, 1.0). See generateNarration's correction pass below for when
// a second call with an adjusted speed is actually needed.
function generateSegmentAudio(text, voiceId, apiKey, speed) {
  return new Promise((resolve, reject) => {
    const bodyStr = JSON.stringify({
      text,
      model_id: "eleven_v3",
      ...(speed ? { voice_settings: { speed } } : {}),
    });
    const req = https.request({
      hostname: "api.elevenlabs.io",
      path: `/v1/text-to-speech/${voiceId}`,
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
        "Content-Length": Buffer.byteLength(bodyStr),
      },
    }, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        if (res.statusCode !== 200) {
          return reject(new Error(`ElevenLabs error (status ${res.statusCode}): ${Buffer.concat(chunks).toString("utf8").slice(0, 300)}`));
        }
        resolve(Buffer.concat(chunks));
      });
    });
    req.on("error", reject);
    req.write(bodyStr);
    req.end();
  });
}

// ── ORCHESTRATOR ──────────────────────────────────────────────────────
// timelineSegments: [{ index, roomLabel, framePaths, duration, startTime }]
// Returns: { segments: [{ audioPath, startTime, duration, text }], fullScript: string }
// `duration` in the returned segments is the REAL rendered audio length
// (after any speed correction), NOT the target clip duration — assemble.js
// needs the real value to fade/cap each segment correctly; using the
// target duration there was the exact bug that motivated this fix (a
// segment that ran long would get faded out too early, cutting off
// speech, before ever reaching the separate overlap problem).
//
// Throws on any failure — caller (renderPipeline.js) treats a narration
// failure as non-fatal to the video itself, same principle as Kling
// falling back to Ken Burns rather than failing the whole job.
async function generateNarration({ address, timelineSegments, voiceId, workDir, anthropicKey, elevenLabsKey }) {
  const scriptSegments = await generateSegmentedScript(address, timelineSegments, anthropicKey);

  const results = [];
  for (let i = 0; i < timelineSegments.length; i++) {
    const seg = timelineSegments[i];
    const scriptEntry = scriptSegments.find((s) => s.index === seg.index);
    if (!scriptEntry || !scriptEntry.text) continue; // skip a missing segment rather than fail the whole narration

    // NEW (July 14, 2026 — Sam's speed-correction suggestion): the real
    // available window before the NEXT segment's audio starts — not this
    // clip's own nominal duration. Two segments back-to-back with no gap
    // between them is exactly the overlap risk this whole fix closes.
    const nextSeg = timelineSegments[i + 1];
    const availableWindow = Math.max(1, (nextSeg ? (nextSeg.startTime - seg.startTime) : seg.duration) - SEGMENT_BREATHING_ROOM_SECONDS);

    let audioBuffer = await generateSegmentAudio(scriptEntry.text, voiceId, elevenLabsKey);
    let audioPath = path.join(workDir, `narration_seg_${seg.index}.mp3`);
    fs.writeFileSync(audioPath, audioBuffer);
    let realDuration = await probeDuration(audioPath);
    let appliedSpeed = 1.0; // NEW — tracked outside the if-block below so the wpm log after it can always reference it safely, whether or not correction actually ran

    // Only correct when the real read is actually TOO LONG for its
    // window — a segment that finishes early is never a problem (silence
    // after it is fine; the next segment still starts exactly on time via
    // adelay in assemble.js). Clamped to ElevenLabs' real supported
    // range; if even MAX_SPEED can't make it fit, accept the overrun
    // rather than distort the voice further — assemble.js's per-segment
    // cap (see mixAudio) is the final backstop for that rare case.
    if (realDuration > availableWindow) {
      appliedSpeed = Math.min(MAX_SPEED, Math.max(MIN_SPEED, realDuration / availableWindow));
      console.warn(`Narration segment ${seg.index}: ${realDuration.toFixed(2)}s ran over its ${availableWindow.toFixed(2)}s window — regenerating at speed=${appliedSpeed.toFixed(2)}.`);
      audioBuffer = await generateSegmentAudio(scriptEntry.text, voiceId, elevenLabsKey, appliedSpeed);
      fs.writeFileSync(audioPath, audioBuffer);
      realDuration = await probeDuration(audioPath);
    }

    // NEW (Sam's question — is one voice's pace better matched than the
    // other, and what's the actual wpm): ElevenLabs has no direct wpm
    // setting, only a speed MULTIPLIER on top of each voice's own
    // inherent pace (baked in from its training audio) — so different
    // voices genuinely can sound faster/slower at the identical speed
    // value. Rather than guess, log the REAL measured wpm per segment —
    // actual word count over actual final audio duration, correcting for
    // the speed multiplier already applied — so real data accumulates
    // across renders instead of a one-off estimate.
    const wordCount = scriptEntry.text.trim().split(/\s+/).length;
    const measuredWpm = (wordCount / realDuration) * 60;
    console.log(`[narration wpm] segment ${seg.index}: voiceId=${voiceId} words=${wordCount} duration=${realDuration.toFixed(2)}s speed=${appliedSpeed.toFixed(2)} → ${measuredWpm.toFixed(0)} wpm`);

    results.push({ audioPath, startTime: seg.startTime, duration: realDuration, availableWindow, text: scriptEntry.text });
  }

  return {
    segments: results,
    fullScript: results.map((r) => r.text).join(" "),
  };
}

module.exports = { generateNarration, wordBudgetForSegment, groupContiguousByRoom };
