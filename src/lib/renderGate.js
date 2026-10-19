// renderGate.js — one shared limit on heavy renders (videos AND Reels).
//
// Why (Oct 18, 2026): the video pipeline (/render) and the Reel pipeline
// (/render-reel) run on one Railway instance (8 GB) with no knowledge of each
// other. A video plus two Reels at once exhausted memory and ffmpeg failed to
// open its encoder in BOTH jobs ("Error while opening encoder for output
// stream"). This gate makes extra renders wait their turn (FIFO) instead of
// all running at once.
//
// RENDER_MAX_CONCURRENT (env, default 2): heavy jobs allowed to run at once.
// A job holds its slot for its whole run (including time spent waiting on
// Kling/ElevenLabs) — simple and safe; tune the number from Railway metrics.
// Not for Smart Correct (/correct-batch), which is a different, lighter path.

const MAX = Math.max(1, Number(process.env.RENDER_MAX_CONCURRENT) || 2);

let running = 0;
const runningLabels = new Set();
const waiting = []; // { label, start, onStart }

function pump() {
  while (running < MAX && waiting.length) {
    const w = waiting.shift();
    running++;
    w.start();
  }
}

// Runs fn() when a slot is free. onQueued(position) fires immediately if the
// job has to wait (position 1 = next up); onStart() fires when it begins.
// Resolves/rejects with fn's result; the slot is always released.
function runExclusive(label, fn, { onQueued, onStart } = {}) {
  return new Promise((resolve, reject) => {
    const start = () => {
      runningLabels.add(label);
      console.log(`[renderGate] start ${label} (running ${running}/${MAX}, waiting ${waiting.length})`);
      if (onStart) { try { onStart(); } catch (_) {} }
      Promise.resolve()
        .then(fn)
        .then(resolve, reject)
        .finally(() => { running--; runningLabels.delete(label); pump(); });
    };
    if (running < MAX) { running++; start(); return; }
    waiting.push({ label, start });
    console.log(`[renderGate] queued ${label} at position ${waiting.length} (running ${running}/${MAX})`);
    if (onQueued) { try { onQueued(waiting.length); } catch (_) {} }
  });
}

// Where a job stands: { state: "running" | "waiting" | "none", ahead } where
// "ahead" counts every render that must finish before this one starts.
function position(label) {
  if (runningLabels.has(label)) return { state: "running", ahead: 0 };
  const i = waiting.findIndex((w) => w.label === label);
  if (i === -1) return { state: "none", ahead: 0 };
  return { state: "waiting", ahead: running + i };
}

function stats() { return { running, waiting: waiting.length, max: MAX }; }

module.exports = { runExclusive, stats, position };
