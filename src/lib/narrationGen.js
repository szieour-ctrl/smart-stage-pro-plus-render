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
const { probeDuration } = require("./assemble");

const SPEAKING_RATE_WORDS_PER_MINUTE = 150;
const MIN_SEGMENT_WORDS = 6; // even a 3-second bathroom shot gets a real sentence, not one word
const MIN_SPEED = 0.7;
const MAX_SPEED = 1.2; // ElevenLabs' real supported range — confirmed via voice_settings.speed

function wordBudgetForSegment(durationSeconds) {
  // Slightly tighter than the whole-video version this replaces (no
  // separate end-buffer subtraction needed per segment — mixAudio's
  // overall NARRATION_END_BUFFER_SECONDS backstop still guards the very
  // end of the whole video; individual segments just need to roughly fit
  // their own clip).
  const words = Math.round((durationSeconds / 60) * SPEAKING_RATE_WORDS_PER_MINUTE);
  return Math.max(MIN_SEGMENT_WORDS, words);
}

// ── SEGMENTED, VISION-GROUNDED SCRIPT ────────────────────────────────────
// segments: [{ index, roomLabel, framePath, duration }]
// Returns: [{ index, text }] in the same order.
function generateSegmentedScript(address, segments, apiKey) {
  return new Promise((resolve, reject) => {
    const segmentDescriptions = segments
      .map((s, i) => `Frame ${i + 1}: "${s.roomLabel}" — target ${wordBudgetForSegment(s.duration)} words max (this segment plays for about ${s.duration.toFixed(1)}s in the final video).`)
      .join("\n");

    const promptText = `You are writing narration for a real estate video tour. You are shown one still frame from each room, in the order they appear in the video.

Address: ${address || "this property"}

${segmentDescriptions}

Write ONE short narration segment per frame, grounded in what you actually see in that specific photo — real details (finishes, light, layout, a genuine standout feature), not generic room-type filler. This is NOT a mechanical, second-by-second description of the image — it should read like someone who toured the home and picked out what's genuinely worth mentioning, tastefully, in a warm, professional, conversational tone. The segments together should feel like one continuous, cohesive walkthrough — each one can build on the last — not a series of disconnected blurbs.

Rules:
- Third person only (never "I" or "my listing").
- Never invent square footage, bedroom/bathroom counts, or amenities not visible in the photo.
- Respect each segment's word target — it's timed to that specific clip's length; going over means it gets cut off mid-sentence.
- The FINAL segment should end with a brief, natural closing line inviting a showing — still within that segment's own word limit.
- Return ONLY a JSON array, nothing else — no markdown fences, no prose before or after. Exact shape: [{"index": 0, "text": "..."}, {"index": 1, "text": "..."}, ...] with exactly ${segments.length} entries, one per frame, in the order shown.`;

    const content = [{ type: "text", text: promptText }];
    segments.forEach((s) => {
      const imageData = fs.readFileSync(s.framePath).toString("base64");
      content.push({
        type: "image",
        source: { type: "base64", media_type: "image/jpeg", data: imageData },
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
// timelineSegments: [{ index, roomLabel, framePath, duration, startTime }]
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
    const availableWindow = nextSeg ? (nextSeg.startTime - seg.startTime) : seg.duration;

    let audioBuffer = await generateSegmentAudio(scriptEntry.text, voiceId, elevenLabsKey);
    let audioPath = path.join(workDir, `narration_seg_${seg.index}.mp3`);
    fs.writeFileSync(audioPath, audioBuffer);
    let realDuration = await probeDuration(audioPath);

    // Only correct when the real read is actually TOO LONG for its
    // window — a segment that finishes early is never a problem (silence
    // after it is fine; the next segment still starts exactly on time via
    // adelay in assemble.js). Clamped to ElevenLabs' real supported
    // range; if even MAX_SPEED can't make it fit, accept the overrun
    // rather than distort the voice further — assemble.js's per-segment
    // cap (see mixAudio) is the final backstop for that rare case.
    if (realDuration > availableWindow) {
      const correctionSpeed = Math.min(MAX_SPEED, Math.max(MIN_SPEED, realDuration / availableWindow));
      console.warn(`Narration segment ${seg.index}: ${realDuration.toFixed(2)}s ran over its ${availableWindow.toFixed(2)}s window — regenerating at speed=${correctionSpeed.toFixed(2)}.`);
      audioBuffer = await generateSegmentAudio(scriptEntry.text, voiceId, elevenLabsKey, correctionSpeed);
      fs.writeFileSync(audioPath, audioBuffer);
      realDuration = await probeDuration(audioPath);
    }

    results.push({ audioPath, startTime: seg.startTime, duration: realDuration, availableWindow, text: scriptEntry.text });
  }

  return {
    segments: results,
    fullScript: results.map((r) => r.text).join(" "),
  };
}

module.exports = { generateNarration, wordBudgetForSegment };
