// agentMarketingLipAlign.js — measures how far a lip-sync clip's mouth runs AHEAD of the clip's own audio (Nov 27, 2026).
//
// HeyGen's mouth moves slightly before the audio it was given, and by an amount that is not the same on every clip. The finished video uses SSP's own
// narration, not the clip's sound, so the clip must be started that much later or the voice seems to lag. Instead of one fixed guess, each clip is measured:
// the movement in the lower middle of the frame (the mouth) is compared with the loudness of the clip's audio, and the shift that lines them up is the lead.
// No clip, no audio, no clear speech: null, and the caller keeps its default.
"use strict";

const { execFile } = require("child_process");
const FPS = 25;
const MAX_LEAD = 0.3, MIN_LEAD = -0.1;         // seconds searched (negative = mouth behind the audio)
const MIN_CORR = 0.3;                          // below this the match is not trusted
const run = (cmd, args) => new Promise((resolve, reject) => execFile(cmd, args, { encoding: "buffer", maxBuffer: 1 << 28 }, (e, so, se) => (e ? reject(new Error(`${cmd} failed: ${String(se).slice(-300)}`)) : resolve(so))));

// Mean absolute change between neighbouring frames of the mouth area, one value per frame at 25 fps.
async function mouthMotion(clip) {
  const W = 48, H = 32;
  const raw = await run("ffmpeg", ["-v", "error", "-i", clip, "-an", "-vf", `fps=${FPS},crop=iw*0.5:ih*0.35:iw*0.25:ih*0.55,scale=${W}:${H}:flags=area,format=gray`, "-f", "rawvideo", "-"]);
  const n = Math.floor(raw.length / (W * H));
  const m = new Array(n).fill(0);
  for (let k = 1; k < n; k++) {
    let s = 0; const a = k * W * H, b = (k - 1) * W * H;
    for (let i = 0; i < W * H; i++) s += Math.abs(raw[a + i] - raw[b + i]);
    m[k] = s / (W * H);
  }
  return m;
}

// Loudness of the clip's audio every 5 ms (RMS over 20 ms), so the search can move in 5 ms steps.
async function audioEnvelope(clip) {
  const SR = 8000, HOP = SR / 200, WIN = SR / 50;
  const raw = await run("ffmpeg", ["-v", "error", "-i", clip, "-vn", "-ac", "1", "-ar", String(SR), "-f", "s16le", "-"]);
  const n = Math.floor(raw.length / 2);
  if (n < SR) return null;
  const x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = raw.readInt16LE(i * 2) / 32768;
  const env = [];
  for (let s = 0; s + WIN <= n; s += HOP) { let e = 0; for (let i = s; i < s + WIN; i++) e += x[i] * x[i]; env.push(Math.sqrt(e / WIN)); }
  return env;   // 200 values per second
}

const smooth3 = (a) => a.map((v, i) => (a[i - 1] === undefined ? v : (a[i - 1] + 2 * v + (a[i + 1] === undefined ? v : a[i + 1])) / 4));
function zscore(a) { const m = a.reduce((s, v) => s + v, 0) / a.length; const sd = Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length) || 1; return a.map((v) => (v - m) / sd); }

/**
 * leadFromSeries(motion, env) → { leadSec, corr } | null. Pure: `motion` is one value per frame (25 fps), `env` is 200 values per second.
 * lead > 0 means the mouth moves before the audio does, so the mouth at time t matches the audio at t + lead.
 */
function leadFromSeries(motion, env) {
  if (!motion || !env || motion.length < 25 || env.length < 400) return null;
  const mv = zscore(smooth3(motion.slice(1)));                          // frame 0 has no previous frame
  const ev = env;
  const best = { corr: -2, lag: 0 };
  const corrs = [];
  for (let lag = Math.round(MIN_LEAD * 200); lag <= Math.round(MAX_LEAD * 200); lag++) {
    const a = [], b = [];
    for (let k = 0; k < mv.length; k++) {
      const idx = (k + 0.5) * (200 / FPS) + lag;                     // a change between frames k and k+1 happens half a frame after frame k; shifted by the lag
      const lo = Math.floor(idx);
      if (lo < 0 || lo + 1 >= ev.length) continue;
      const f = idx - lo;
      a.push(mv[k]); b.push(ev[lo] * (1 - f) + ev[lo + 1] * f);
    }
    if (a.length < 20) continue;
    const bz = zscore(b);
    let r = 0; for (let i = 0; i < a.length; i++) r += a[i] * bz[i];
    r /= a.length;
    corrs.push({ lag, r });
    if (r > best.corr) { best.corr = r; best.lag = lag; }
  }
  if (!corrs.length || best.corr < MIN_CORR) return null;
  return { leadSec: Math.round(best.lag * 5) / 1000, corr: Math.round(best.corr * 1000) / 1000 };
}

async function measureLead(clip) {
  const [motion, env] = await Promise.all([mouthMotion(clip), audioEnvelope(clip)]);
  return leadFromSeries(motion, env);
}

module.exports = { measureLead, leadFromSeries, mouthMotion, audioEnvelope, MAX_LEAD, MIN_LEAD, MIN_CORR };
