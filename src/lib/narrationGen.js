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

// CORRECTED to 130 (July 18, 2026 — Sam caught a real reasoning error:
// budgeting off the pace FLOOR (115) was the wrong fix. Removing speed
// correction means duration padding and the word ceiling both need to
// agree with EACH OTHER, not each separately hedge against worst case —
// stacking two conservative assumptions doesn't add safety, it just
// creates large unintended dead air. Verified against a real script:
// segment 5 (14 words, base duration only, no extra padding) landed
// almost exactly on target — proof the base per-room durations were
// already reasonably calibrated. The problem was specifically the
// intro/outro EXTRA padding (see renderPipeline.js), sized off 115 wpm
// instead of something close to real observed average pace (~150 wpm
// this session). 130 is a deliberate middle: a little under true average
// (so a script at the ceiling still tends to finish with a small natural
// margin, since real delivery usually runs faster than 130), but not the
// pessimistic floor that was manufacturing multi-second gaps.
const SPEAKING_RATE_WORDS_PER_MINUTE = 130;
const MIN_SEGMENT_WORDS = 6; // even a 3-second bathroom shot gets a real sentence, not one word
// MIN_SPEED/MAX_SPEED removed (July 18, 2026) — were only ever used by
// the now-removed speed-correction call, confirmed permanently inert on
// eleven_v3. generateSegmentAudio() below still accepts an optional
// `speed` param, unused for now, in case a future model swap actually
// supports it — but nothing in this file passes one anymore.
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

  // FIX (July 18, 2026 — real render, Sam's report: narration routinely
  // ran over even on ordinary mid-video segments, not just the outro; and
  // "there's supposed to be a breathing pause between clips" wasn't
  // actually happening). Root cause, confirmed against assemble.js's real
  // numbers: this function was handing generateSegmentedScript's word-
  // budget PROMPT the group's raw `duration` (for a single-clip group,
  // just that clip's own rendered length) — but the ENFORCEMENT check in
  // generateNarration() below computed something ELSE: the real gap to
  // the NEXT group's startTime, minus SEGMENT_BREATHING_ROOM_SECONDS.
  // Those two numbers were never the same number. computeClipTimeline()
  // in assemble.js spaces consecutive clips CROSSFADE_DURATION (0.6s)
  // apart specifically BECAUSE of the crossfade overlap — so the real gap
  // between one segment's start and the next is always 0.6s tighter than
  // a clip's raw duration, before SEGMENT_BREATHING_ROOM_SECONDS (0.5s)
  // is even subtracted on top. That's up to 1.1s of real-world tightening
  // the word budget never saw — Claude was routinely told it had ~1s more
  // room than it actually got, wrote to that generous budget, and then
  // got cut off against the real one. Computing availableWindow ONCE here
  // — the same value used for both the prompt AND the correction check
  // below — makes the two impossible to disagree again.
  groups.forEach((g, i) => {
    const nextStart = i < groups.length - 1 ? groups[i + 1].startTime : null;
    g.availableWindow = Math.max(1, (nextStart !== null ? nextStart - g.startTime : g.duration) - SEGMENT_BREATHING_ROOM_SECONDS);
  });

  return groups.map((g, i) => ({ index: i, ...g }));
}

// ── SEGMENTED, VISION-GROUNDED SCRIPT ────────────────────────────────────
// segments: [{ index, roomLabel, framePaths: [...], duration }] — one or
// more representative frames per segment (see groupContiguousByRoom).
// Returns: [{ index, text }] in the same order.
function generateSegmentedScript(address, segments, apiKey) {
  return new Promise((resolve, reject) => {
    // REBUILT (July 20, 2026 — real evidence the placeholder-token
    // version wasn't good enough either): the previous fix asked Claude
    // to write a literal {{ADDRESS}} token instead of the real address.
    // A real render showed Claude just... didn't use it — wrote the full,
    // correct street address directly into the sentence anyway, token
    // never appearing anywhere in the output. That's the SECOND
    // instruction-compliance approach to fail against real evidence (the
    // first was "cut the room detail if it doesn't fit," which also
    // didn't hold). Removing the dependency on compliance entirely this
    // time, not trying a third wording: the final group's response now
    // has TWO separate fields — "text" (room/exterior detail only) and
    // "closing" (the CTA only) — and BOTH are explicitly told to contain
    // NO address/location content whatsoever. The real address is
    // inserted as our own fixed, deterministic sentence BETWEEN them in
    // code, after generation (see the resolve() call below) — there's
    // nothing left for Claude to get right or wrong about it, because
    // it's never asked to touch it at all, not even via a token.
    const addressSentence = address ? `This is ${address}.` : "";
    const addressWordCount = addressSentence ? addressSentence.split(/\s+/).filter(Boolean).length : 0;

    const segmentDescriptions = segments
      .map((s, i) => {
        const marker = segments.length === 1
          ? " — THIS IS BOTH THE OPENING AND CLOSING GROUP (the only group in this video)"
          : i === 0 ? " — THIS IS THE OPENING GROUP" : (i === segments.length - 1 ? " — THIS IS THE CLOSING GROUP" : "");
        const isFinal = i === segments.length - 1;
        const rawWordTarget = wordBudgetForSegment(s.availableWindow);
        // Reserves real speaking time for the address sentence we insert
        // in code — that sentence costs real seconds when spoken even
        // though Claude never writes it, so the ceiling given to Claude
        // has to be honest about how much of the window is actually
        // available for what it's writing.
        const wordTarget = isFinal ? Math.max(10, rawWordTarget - addressWordCount) : rawWordTarget;
        const targetNote = isFinal
          ? `target ${wordTarget} words max combined across "text" and "closing" (a separate fixed address sentence gets inserted between them afterward — already accounted for, don't write about it)`
          : `target ${wordTarget} words max`;
        return `Group ${i + 1}: "${s.roomLabel}"${marker} — ${s.framePaths.length} frame(s) shown for this group — ${targetNote} (this group has about ${s.availableWindow.toFixed(1)}s of real speaking room before the next group starts).`;
      })
      .join("\n");

    const promptText = `You are writing narration for a real estate video tour. You are shown one or more still frames per group, grouped in the order they appear in the video.

IMPORTANT: frames are grouped because they were tagged with the same room-type label (e.g. "Bathroom") and appear consecutively. Most of the time that means they genuinely ARE the same physical room shown from different angles or crops — but room-type tags don't guarantee that. Occasionally a group will actually contain two distinct rooms of the same type placed back-to-back (e.g. a primary bath, then a guest bath). Look at what's actually different across the frames in each group and judge for yourself:
- If they're clearly the same space (same fixtures, same finishes, same layout from a different angle) — narrate it as one continuous room description, as usual.
- If they're clearly DIFFERENT rooms that just share a type — say so distinctly in the narration (e.g. "...and down the hall, a second full bath offers..."), don't blend two different rooms into one generic description as if they were one.

Address: ${address || "this property"} (for your own context only — see the FINAL group rule below for exactly how/whether to reference this; do not assume every group should mention it)

${segmentDescriptions}

IMPORTANT — exterior accuracy: for any group whose room label suggests an EXTERIOR shot (exterior, yard, backyard, front yard, pool, patio, curb appeal), be cautious about ONE specific thing: this platform sometimes adds outdoor furniture/staging (a firepit, patio dining set, planters) or enhances landscaping (fresh lawn, trimmed hedges, seasonal flowers) that may not reflect the property's real, permanent condition. Do NOT describe added outdoor furniture or specific landscaping details (lawn condition, plantings, hedges) as if they're confirmed, permanent features of the property. Time of day and lighting (golden hour warmth, twilight glow) ARE safe to describe normally — that's just atmosphere, not a factual claim about the property itself. Stick to the home's own architectural features (roofline, siding material, window style) plus safe lighting/mood description; just avoid asserting specific outdoor furnishings or landscaping as real. Interior rooms don't need this caution; describe what you actually see in interiors normally.

Write ONE short narration segment per GROUP (not per frame). Pick exactly ONE specific, vivid detail per group — not a list, not room type plus finishes plus layout plus a value statement. One real, grounded observation (a material, a light quality, a single standout feature) said naturally, the way someone would mention the one thing that actually caught their eye walking through — not an inventory of the room. Say the room's name once when you start describing it — don't re-announce it for every frame that's clearly still the same space — but if the group turns out to span more than one real room (see above), make sure narration reflects that rather than silently describing only one of them.

This is NOT a mechanical, second-by-second description of each image, and it is NOT a features list — it should read like someone who toured the home and mentioned the ONE thing worth noting about each room in passing, tastefully, in a warm, professional, conversational tone, then moved on. The segments together should feel like one continuous, cohesive walkthrough — each one can build on the last — not a series of disconnected blurbs, and never more than ONE sentence, two at the absolute most, per group.

Rules:
- Third person only (never "I" or "my listing").
- Never invent square footage, bedroom/bathroom counts, or amenities not visible in the photos.
- ONE sentence per group. Two only if both are short. This is the single most important constraint — a segment that tries to name the room, list what's in it, AND editorialize about value will not fit its window and will get cut off mid-sentence. Pick the one detail that matters most and say only that.
- Each group's word target is a HARD CEILING, not a suggestion. There is no correction step after this — whatever you write plays at natural pace in exactly the time given. Going even slightly over means the audio gets cut off mid-word, which sounds like a broken, amateur edit. When in doubt, write UNDER the target, never over it — a slightly short segment just means a beat of natural silence, which is fine; an over-budget one is a real, audible defect.
- Vary how each segment opens. Do NOT start consecutive (or most) segments with the same phrase (e.g. "Here we have," "This room features") — that reads as a stutter when segments play back to back. Vary sentence structure across the whole script the way a real person naturally would, not a template being refilled per room.
- THE OPENING GROUP IS DIFFERENT FROM AN ORDINARY GROUP: a video needs a strong opening. It must (1) give one real, grounded observation about the room or exterior actually shown — same as any other group — combined naturally with (2) a brief, warm welcome that references the property's general LOCATION (city, or street + city informally — e.g. "Welcome to Main Street in Roseville," or "This beautiful Roseville home..."). Keep it informal and brief — do NOT recite the complete formal address (street number, city, state, zip) as if reading a mailing label here; that full, precise address belongs only at the close (see below), not here.
- THE FINAL GROUP HAS A DIFFERENT RESPONSE SHAPE FROM EVERY OTHER GROUP — instead of one "text" field, it needs TWO fields: {"index": N, "text": "...", "closing": "..."}. "text" = one real, grounded observation about the room/exterior actually shown, same as any ordinary group — a single detail, not a list. "closing" = a strong, original call to action inviting a showing. NEITHER field should mention the address, street name, city, state, or any part of the location — say NOTHING about where the property is in either field. The complete, exact address gets inserted automatically BETWEEN "text" and "closing" after you respond (a separate, fixed sentence you never see or write) — so the two fields you write will be read as: [your text] + [inserted address sentence] + [your closing], and need to flow naturally into that gap without referencing it directly. Combined word count for "text" + "closing" together must stay within this group's target above.
- For the CTA ("closing" field) specifically: do NOT write the literal phrase "schedule your private showing" (or close variants like "schedule your private tour/viewing") — that exact instruction already appears as on-screen text on the closing card that follows this segment, so saying it verbatim here is pure repetition. Write something that earns the invitation instead — a genuine, warm reason to come see it in person, in your own words, not the boilerplate line the viewer is about to read anyway.
- If the FINAL group's image looks like a branded closing card rather than an ordinary room/exterior photo — visible overlaid text (an address, a tagline), a darkened or gradient-scrimmed background rather than a naturally-lit room — do NOT describe it as if it were a normal room in the "text" field, and do NOT read the on-screen text aloud verbatim like you're narrating a sign. Instead, "text" can be a brief, natural transition line (no address, no room description) and "closing" the original CTA — the inserted address sentence between them still applies exactly the same way.
- Return ONLY a JSON array, nothing else — no markdown fences, no prose before or after. Every entry except the final group uses {"index": N, "text": "..."}; the final group entry uses {"index": N, "text": "...", "closing": "..."} as described above. Exactly ${segments.length} entries, one per group, in the order shown.`;

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
          // FIX (July 21, 2026 — real render failure): a real response came
          // back with a full reasoning preamble BEFORE the JSON array
          // ("Looking at the frames carefully:\n\n- Group 1: ... Group 2:
          // ..."), despite the prompt's explicit "Return ONLY a JSON array,
          // nothing else — no prose before or after." JSON.parse(cleaned)
          // failed immediately ("Unexpected token L... at position 0") since
          // `cleaned` started with "Looking", not "[". This is the SAME
          // category of problem this file has hit twice before (the address
          // token and the "cut room detail" instruction-compliance
          // failures) — asking Claude to comply with a strict output shape
          // isn't reliable, so stop depending on it entirely rather than
          // trying a differently-worded instruction a third time. Instead
          // of assuming `cleaned` IS the array, find the JSON array
          // EMBEDDED in it — from the first "[" to the last "]" — and parse
          // only that substring. This works whether Claude adds prose
          // before, after, both, or neither.
          const arrayStart = cleaned.indexOf("[");
          const arrayEnd = cleaned.lastIndexOf("]");
          if (arrayStart === -1 || arrayEnd === -1 || arrayEnd < arrayStart) {
            throw new Error(`No JSON array found in Claude's response. Raw: ${cleaned.slice(0, 500)}`);
          }
          const jsonSlice = cleaned.slice(arrayStart, arrayEnd + 1);
          const segmentsOut = JSON.parse(jsonSlice);
          if (!Array.isArray(segmentsOut)) throw new Error("Claude's response was not a JSON array");
          // REBUILT (July 20, 2026) — no more token substitution (the
          // previous approach Claude simply didn't use, writing the real
          // address directly instead — confirmed on a real render). The
          // final segment now arrives as separate "text"/"closing" fields
          // that never contained any address content in the first place
          // (nothing to substitute, nothing to strip) — this just joins
          // them with our own fixed, always-correct address sentence
          // built earlier in this function. Every other segment is
          // untouched, using its "text" field exactly as before.
          if (segmentsOut.length > 0) {
            const last = segmentsOut[segmentsOut.length - 1];
            if (typeof last.closing === "string") {
              const detail = (last.text || "").trim();
              const closing = last.closing.trim();
              last.text = [detail, addressSentence, closing].filter(Boolean).join(" ");
              delete last.closing;
            }
          }
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

  // FIX (July 20, 2026 — real render, COMPLETE narration failure: first
  // clip silent, clip 1's script spoken over clip 2, CTA never attempted).
  // Root cause: generateSegmentedScript's prompt labels groups for Claude
  // as "Group 1", "Group 2", ... (1-based, for human readability) but never
  // states what numeric convention the JSON "index" field should use.
  // Claude naturally mirrored the 1-based labels it was shown, while
  // seg.index here is 0-based (assigned by groupContiguousByRoom's
  // .map((g, i) => ({ index: i, ...g }))). Matching by that value
  // (scriptSegments.find(s => s.index === seg.index)) meant: seg.index=0
  // never matched anything (Claude wrote index:1 for the first group) →
  // silently skipped, no narration on the real first clip. seg.index=1
  // matched Claude's index:1 entry — but that entry's TEXT was written
  // for "Group 1" (the actual first room) → the real second clip spoke
  // the first room's narration. The true final group (index N-1, the one
  // carrying the "closing" CTA field) never matched either, for the same
  // reason, so the CTA silently vanished too. Same fix already proven for
  // the address-token problem on July 20: stop depending on Claude
  // returning a specific numeric convention we never actually specified —
  // match by ARRAY POSITION instead, which is correct by construction
  // (generateSegmentedScript's own prompt requires "Exactly N entries, one
  // per group, in the order shown").
  if (scriptSegments.length !== timelineSegments.length) {
    console.error(
      `[NARRATION MISMATCH] Claude returned ${scriptSegments.length} script segments, ` +
      `expected ${timelineSegments.length} (one per group). Matching by position — ` +
      `any group beyond the shorter length will have no narration this render.`
    );
  }

  const results = [];
  for (let i = 0; i < timelineSegments.length; i++) {
    const seg = timelineSegments[i];
    const scriptEntry = scriptSegments[i]; // position-based match — NOT scriptSegments.find(s => s.index === seg.index)
    if (!scriptEntry || !scriptEntry.text) {
      // LOUD now (was a bare `continue` with zero logging before) — a
      // missing segment used to vanish with no trace anywhere in the
      // logs, which is exactly why this bug took this long to pin down.
      console.error(
        `[NARRATION SEGMENT MISSING] position ${i} (roomLabel: "${seg.roomLabel}", ` +
        `startTime: ${seg.startTime}s) has no usable script text — this clip will play with no narration.`
      );
      continue;
    }

    // FIX (July 18, 2026) — this used to recompute availableWindow here,
    // independently from what generateSegmentedScript's prompt was told.
    // Now reads seg.availableWindow, set once in groupContiguousByRoom —
    // the exact same number the word-budget prompt targeted. See that
    // function's comment for the full diagnosis (a real render where
    // narration routinely overran, confirmed against assemble.js's
    // CROSSFADE_DURATION/SEGMENT_BREATHING_ROOM_SECONDS math).
    const availableWindow = seg.availableWindow;

    let audioBuffer = await generateSegmentAudio(scriptEntry.text, voiceId, elevenLabsKey);
    let audioPath = path.join(workDir, `narration_seg_${seg.index}.mp3`);
    fs.writeFileSync(audioPath, audioBuffer);
    let realDuration = await probeDuration(audioPath);

    // REMOVED (July 18, 2026) — this used to regenerate at a corrected
    // ElevenLabs `speed` value when a segment ran over its window.
    // Confirmed via ElevenLabs' own docs ("Speed is not available for the
    // Eleven v3 model.") plus a real log showing speed=1.20 producing the
    // EXACT SAME duration as the uncorrected take: this call has never
    // once actually shortened anything on eleven_v3, the model this
    // codebase uses and Sam has explicitly chosen to keep for its
    // expressiveness. It was burning a second ElevenLabs API call and
    // real latency per overrun segment for zero effect. There is now NO
    // correction step — SPEAKING_RATE_WORDS_PER_MINUTE (lowered to the
    // observed pace floor for exactly this reason) and the per-segment
    // padding are the entire defense against overrun. This log line is
    // what's left: an honest signal that the budget itself needs
    // attention if it ever fires, not a claim that anything was fixed.
    if (realDuration > seg.availableWindow) {
      console.error(
        `[NARRATION OVERRUN — NO CORRECTION AVAILABLE] segment ${seg.index}: ${realDuration.toFixed(2)}s ` +
        `ran over its ${seg.availableWindow.toFixed(2)}s window. eleven_v3 does not support speed correction, ` +
        `so this segment will play past its window uncorrected. If this fires often, the word budget ` +
        `(SPEAKING_RATE_WORDS_PER_MINUTE) is still too generous for this voice/content, not a one-off.`
      );
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
    console.log(`[narration wpm] segment ${seg.index}: voiceId=${voiceId} words=${wordCount} duration=${realDuration.toFixed(2)}s → ${measuredWpm.toFixed(0)} wpm`);

    results.push({ audioPath, startTime: seg.startTime, duration: realDuration, availableWindow, text: scriptEntry.text });
  }

  return {
    segments: results,
    fullScript: results.map((r) => r.text).join(" "),
  };
}

module.exports = { generateNarration, wordBudgetForSegment, groupContiguousByRoom };
