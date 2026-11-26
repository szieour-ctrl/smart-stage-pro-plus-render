"use strict";
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");
const X = require("../src/lib/agentMarketingExcerpt");

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "amxt-"));
const src = path.join(dir, "n.mp3");
execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=10", "-c:a", "libmp3lame", "-b:a", "128k", src]);
const dur = (f) => Number(execFileSync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f]).toString());

test("cuts the exact seconds asked for", async () => {
  const buf = await X.cutExcerpt({ url: src, start: 1.5, end: 6.0 }, { allowFile: true });
  const f = path.join(dir, "o.mp3"); fs.writeFileSync(f, buf);
  assert.ok(Math.abs(dur(f) - 4.5) < 0.1, `got ${dur(f)}`);
});
test("refuses bad ranges and links that are not secure storage links", async () => {
  await assert.rejects(X.cutExcerpt({ url: src, start: 5, end: 5 }, { allowFile: true }), /end after start/);
  await assert.rejects(X.cutExcerpt({ url: src, start: 0, end: 90 }, { allowFile: true }), /at most 40/);
  await assert.rejects(X.cutExcerpt({ url: "http://example.com/a.mp3", start: 0, end: 3 }), /secure storage link/);
  await assert.rejects(X.cutExcerpt({ url: "https://example.com/a.mp3", start: 0, end: 3 }), /secure storage link/);
  assert.strictEqual(X.check({ url: "https://b.s3.us-west-2.amazonaws.com/x?sig=1", start: 0, end: 3 }), null);
});
