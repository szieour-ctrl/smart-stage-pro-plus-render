// agentMarketingExcerpt.js — cuts the exact narration excerpt for a scene, for the optional lip-sync clips (Nov 25, 2026).
// The narration is SSP's own ElevenLabs track; this only copies the seconds from `start` to `end` into a short mp3. It changes no word and no timing,
// and calls no outside service other than downloading the (presigned) narration link it is given.
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFile } = require("child_process");

const MAX_EXCERPT_SEC = 40;
const MAX_START_SEC = 600;

function check({ url, start, end }, { allowFile = false } = {}) {
  const s = Number(start), e = Number(end);
  if (!Number.isFinite(s) || !Number.isFinite(e) || s < 0 || s > MAX_START_SEC || e <= s) return "start and end must be seconds with end after start";
  if (e - s > MAX_EXCERPT_SEC) return `an excerpt can be at most ${MAX_EXCERPT_SEC} seconds`;
  if (typeof url !== "string" || !url) return "a narration link is required";
  if (!allowFile) {
    let u; try { u = new URL(url); } catch { return "the narration link is not a valid address"; }
    if (u.protocol !== "https:" || !/(^|\.)amazonaws\.com$/i.test(u.hostname)) return "the narration link must be a secure storage link";
  }
  return null;
}

/** cutExcerpt({ url, start, end }, deps?) -> Buffer (mp3). deps.fetchBuffer(url) is injectable; tests may pass a local file path with allowFile. */
async function cutExcerpt({ url, start, end }, { fetchBuffer, allowFile = false } = {}) {
  const problem = check({ url, start, end }, { allowFile });
  if (problem) { const e = new Error(problem); e.status = 400; throw e; }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "amx-"));
  try {
    const src = path.join(dir, "narration.mp3"), out = path.join(dir, "excerpt.mp3");
    if (allowFile && fs.existsSync(url)) fs.copyFileSync(url, src);
    else {
      const buf = fetchBuffer ? await fetchBuffer(url) : Buffer.from((await require("axios").get(url, { responseType: "arraybuffer", timeout: 60000 })).data);
      fs.writeFileSync(src, buf);
    }
    await new Promise((resolve, reject) => execFile("ffmpeg", ["-y", "-v", "error", "-i", src, "-ss", Number(start).toFixed(3), "-t", (Number(end) - Number(start)).toFixed(3), "-vn", "-ar", "44100", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "128k", out], { maxBuffer: 1 << 24 }, (err, so, se) => (err ? reject(new Error(`ffmpeg failed: ${String(se).slice(-300)}`)) : resolve())));
    return fs.readFileSync(out);
  } finally { fs.rm(dir, { recursive: true, force: true }, () => {}); }
}

module.exports = { cutExcerpt, check, MAX_EXCERPT_SEC };
