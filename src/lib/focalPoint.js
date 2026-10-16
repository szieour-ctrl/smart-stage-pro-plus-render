// focalPoint.js — Subject-aware crop centre for native 9:16.
//
// A 9:16 window on a 3:2 photo only shows ~37% of its width, so WHERE the
// window sits decides whether the shot shows the room's hero (island, bed,
// sofa grouping, front door) or an empty wall. Claude Vision picks that
// point per photo. Always degrades to dead-centre {0.5,0.5} — never throws,
// never blocks a render.
//
// Env: ANTHROPIC_API_KEY (already used by narrationGen.js),
//      FOCAL_VISION_MODEL (optional, default claude-sonnet-4-6),
//      FOCAL_VISION_ENABLED=false to force centre (kill switch).

const fs = require("fs");
const path = require("path");
const https = require("https");
const { spawn } = require("child_process");

const CENTER = { fx: 0.5, fy: 0.5, source: "center" };
const MODEL = process.env.FOCAL_VISION_MODEL || "claude-sonnet-4-6";
const TIMEOUT_MS = 25000;
const cache = new Map();

// Downscale to ≤1024px JPEG via ffmpeg (no Pillow/sharp in this image).
function downscaleToJpegBase64(imagePath, workDir, tag) {
  return new Promise((resolve, reject) => {
    const out = path.join(workDir, `focal_${tag}.jpg`);
    const p = spawn("ffmpeg", ["-y", "-i", imagePath, "-vf", "scale='min(1024,iw)':-2", "-q:v", "4", out]);
    p.on("error", reject);
    p.on("close", (code) => {
      if (code !== 0) return reject(new Error(`ffmpeg downscale exit ${code}`));
      try { resolve(fs.readFileSync(out).toString("base64")); } catch (e) { reject(e); }
    });
  });
}

function callVision(b64, apiKey, roomLabel) {
  return new Promise((resolve, reject) => {
    const prompt =
      `This is a real-estate photo${roomLabel ? ` of: ${roomLabel}` : ""}. It will be cut into a TALL 9:16 vertical video; ` +
      `the vertical window shows only about the middle third of the width, so choose the point it should be centred on.\n` +
      `Pick the single hero subject a buyer should see (kitchen island, bed, main seating group, fireplace, front door/entry, ` +
      `pool, view window) — not an empty wall, and not a decorative corner.\n` +
      `Answer with ONLY JSON: {"fx": <0-1 horizontal position of that subject, 0=left edge>, "fy": <0-1 vertical position, 0=top>}`;
    const body = JSON.stringify({
      model: MODEL,
      max_tokens: 60,
      messages: [{ role: "user", content: [
        { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
        { type: "text", text: prompt },
      ] }],
    });
    const req = https.request({
      hostname: "api.anthropic.com", path: "/v1/messages", method: "POST",
      headers: {
        "x-api-key": apiKey, "anthropic-version": "2023-06-01",
        "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body),
      },
    }, (res) => {
      let data = "";
      res.on("data", (c) => (data += c));
      res.on("end", () => {
        try {
          const text = JSON.parse(data)?.content?.find((b) => b.type === "text")?.text || "";
          const m = text.match(/\{[^}]*\}/);
          if (!m) return reject(new Error(`no JSON in vision reply: ${text.slice(0, 120)}`));
          const j = JSON.parse(m[0]);
          const fx = Number(j.fx), fy = Number(j.fy);
          if (!Number.isFinite(fx) || !Number.isFinite(fy)) return reject(new Error("non-numeric focal"));
          resolve({ fx: Math.min(Math.max(fx, 0), 1), fy: Math.min(Math.max(fy, 0), 1), source: "vision" });
        } catch (e) { reject(e); }
      });
    });
    req.setTimeout(TIMEOUT_MS, () => req.destroy(new Error("vision timeout")));
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

async function findFocalPoint(imagePath, workDir, { roomLabel, tag } = {}) {
  if (process.env.FOCAL_VISION_ENABLED === "false") return { ...CENTER };
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return { ...CENTER };
  if (cache.has(imagePath)) return cache.get(imagePath);
  try {
    const b64 = await downscaleToJpegBase64(imagePath, workDir, tag || path.basename(imagePath, path.extname(imagePath)));
    const f = await callVision(b64, apiKey, roomLabel);
    cache.set(imagePath, f);
    return f;
  } catch (err) {
    console.warn(`[focalPoint] ${path.basename(imagePath)}: ${err.message} — using centre.`);
    return { ...CENTER };
  }
}

module.exports = { findFocalPoint };
