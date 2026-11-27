// Lip-sync alignment: build a clip whose mouth area moves a KNOWN time before its audio, and check the measurement finds that time.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs"), os = require("os"), path = require("path");
const { spawn } = require("child_process");
const A = require("../src/lib/agentMarketingLipAlign");

// Speech-like pattern: bursts of voice with gaps, from a fixed seed.
function bursts(seconds, seed) {
  let s = seed; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const on = []; let t = 0.3;
  while (t < seconds - 0.4) { const d = 0.08 + rnd() * 0.17, gap = 0.06 + rnd() * 0.14, amp = 0.4 + rnd() * 0.6; on.push({ a: t, b: t + d, amp }); t += d + gap; }
  return on;
}
const level = (on, t) => { for (const o of on) if (t >= o.a && t < o.b) return o.amp; return 0; };

function makeClip(file, { seconds = 8, lead = 0.12, seed = 7 } = {}) {
  const on = bursts(seconds, seed), SR = 16000, N = Math.floor(seconds * SR);
  const pcm = Buffer.alloc(N * 2);
  let ns = 12345;
  for (let i = 0; i < N; i++) { ns = (ns * 1103515245 + 12345) >>> 0; const noise = ((ns >>> 8) / 8388608) - 1; pcm.writeInt16LE(Math.round(noise * level(on, i / SR) * 12000), i * 2); }
  const wav = path.join(path.dirname(file), "a.wav");
  const hdr = Buffer.alloc(44); hdr.write("RIFF", 0); hdr.writeUInt32LE(36 + pcm.length, 4); hdr.write("WAVEfmt ", 8); hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(1, 22); hdr.writeUInt32LE(SR, 24); hdr.writeUInt32LE(SR * 2, 28); hdr.writeUInt16LE(2, 32); hdr.writeUInt16LE(16, 34); hdr.write("data", 36); hdr.writeUInt32LE(pcm.length, 40);
  fs.writeFileSync(wav, Buffer.concat([hdr, pcm]));
  // 25 fps, 160x160: the lower-middle box changes with the voice level of (t + lead): the mouth moves `lead` seconds BEFORE the voice.
  const W = 160, H = 160, frames = Math.floor(seconds * 25);
  return new Promise((resolve, reject) => {
    const ff = spawn("ffmpeg", ["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", `${W}x${H}`, "-r", "25", "-i", "pipe:0", "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", file]);
    let err = ""; ff.stderr.on("data", (d) => (err += d)); ff.on("close", (c) => (c ? reject(new Error(err)) : resolve(file)));
    const buf = Buffer.alloc(W * H * frames);
    for (let k = 0; k < frames; k++) {
      const tt = k / 25 + lead, v = level(on, tt) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 6.5 * tt));   // the mouth keeps moving while the voice is on
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const inBox = x > 40 && x < 120 && y > 96 && y < 140;
        buf[k * W * H + y * W + x] = inBox ? Math.round(60 + v * 150 + ((x * 7 + y * 13) % 9)) : 90;
      }
    }
    ff.stdin.end(buf);
  });
}

test("measures a mouth that leads the audio by 120 ms, 60 ms and 0 ms", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "lipalign-"));
  for (const lead of [0.12, 0.06, 0]) {
    const clip = await makeClip(path.join(dir, `c${lead}.mp4`), { lead, seed: 3 + Math.round(lead * 100) });
    const r = await A.measureLead(clip);
    assert.ok(r, `a result for lead ${lead}`);
    assert.ok(Math.abs(r.leadSec - lead) <= 0.04, `lead ${lead}: measured ${r.leadSec} (corr ${r.corr})`);
    assert.ok(r.corr >= A.MIN_CORR);
  }
});

test("no usable speech gives null, not a guess", () => {
  assert.equal(A.leadFromSeries(null, null), null);
  assert.equal(A.leadFromSeries(new Array(10).fill(0), new Array(100).fill(0)), null);
  const flat = new Array(100).fill(1), env = new Array(800).fill(0.2);
  assert.equal(A.leadFromSeries(flat, env), null);
});
