"use strict";
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs"), os = require("os"), path = require("path");
const { execFileSync } = require("child_process");
const { voiceShapeFilter } = require("../src/lib/agentMarketingRender");

test("no shaping means no filter, and values are clamped", () => {
  assert.strictEqual(voiceShapeFilter(undefined), "");
  assert.strictEqual(voiceShapeFilter({ semitones: 0, bass_db: 0 }), "");
  assert.match(voiceShapeFilter({ semitones: -40, bass_db: 99 }), /bass=g=8/);
  assert.match(voiceShapeFilter({ semitones: 5 }), /^$/);
});
test("the pitch drop lowers the voice and keeps the length", () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "vs-"));
  const src = path.join(d, "a.wav"), out = path.join(d, "b.wav");
  execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=200:duration=4:sample_rate=44100", src]);
  execFileSync("ffmpeg", ["-y", "-v", "error", "-i", src, "-af", voiceShapeFilter({ semitones: -2, bass_db: 3 }).replace(/,$/, ""), out]);
  const dur = (f) => Number(execFileSync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f]).toString());
  assert.ok(Math.abs(dur(out) - dur(src)) < 0.06, `${dur(src)} vs ${dur(out)}`);
  // measured pitch: count zero crossings
  const zc = (f) => { const b = execFileSync("ffmpeg", ["-v", "error", "-i", f, "-f", "s16le", "-ac", "1", "-"], { maxBuffer: 1 << 26 }); let c = 0, p = 0; for (let i = 0; i < b.length - 1; i += 2) { const v = b.readInt16LE(i); if ((p < 0) !== (v < 0)) c++; p = v; } return c / 2 / (b.length / 2 / 44100); };
  const ratio = zc(out) / zc(src);
  assert.ok(Math.abs(ratio - Math.pow(2, -2 / 12)) < 0.03, `pitch ratio ${ratio}`);
});
