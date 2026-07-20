// server.js — Smart Stage PRO Plus Render Service
// Runs on Railway. Receives video render jobs from Netlify's video-job.js,
// processes them asynchronously, and calls back to a Netlify webhook
// when complete.
//
// This service NEVER writes to Supabase directly for credit/listing data.
// It only reports job status back via the webhook — Netlify owns all
// Supabase writes for shared tables.

require("dotenv").config();
const express = require("express");
const os = require("os");
const path = require("path");
const fs = require("fs");
const { processRenderJob } = require("./renderPipeline");
const { processCorrectBatch } = require("./lib/correctPipeline");
const { generateLtxContinuationClip } = require("./lib/ltxMotion");

const app = express();
app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT || 3000;

// NEW (July 15, 2026 — three consecutive silent renderer deaths, no
// application-level trace, no Railway-config explanation holding up:
// replica limits at plan max, no healthcheck configured, billing nowhere
// near the cap, memory graph never actually hitting the 8GB replica
// ceiling before dying). Every prior theory has been ruled out by
// checking Railway's own dashboard directly — this isn't more guessing,
// it's making sure the NEXT death actually leaves a trace to diagnose
// from, whatever the real cause turns out to be.
//
// uncaughtException/unhandledRejection: catches anything that would
// otherwise crash the process with zero log output. This won't catch a
// hard SIGKILL (nothing can — that's the OS ending the process with no
// chance for JS to run at all), but it WILL catch a genuine JS-level
// crash (a thrown error nothing awaited/caught upstream, a rejected
// promise nobody handled) that was previously indistinguishable from an
// OOM/platform-level kill in the logs.
process.on("uncaughtException", (err) => {
  console.error("[FATAL uncaughtException]", err.stack || err.message || err);
  // Deliberately NOT calling process.exit() here — let whatever's
  // in-flight finish logging/writing if it can. If the process is truly
  // unrecoverable, Railway's own restart behavior takes over regardless.
});
process.on("unhandledRejection", (reason, promise) => {
  console.error("[FATAL unhandledRejection]", reason instanceof Error ? (reason.stack || reason.message) : reason);
});

// SIGTERM/SIGINT: if RAILWAY (not the OS OOM killer) is the one ending
// this process — a redeploy, a manual restart, a platform-initiated
// graceful shutdown — it sends SIGTERM first, before a harder kill. This
// logs that distinctly from an uncaught JS error, so future logs can
// tell "the platform told me to stop" apart from "something crashed."
["SIGTERM", "SIGINT"].forEach((signal) => {
  process.on(signal, () => {
    console.error(`[PROCESS SIGNAL] Received ${signal} — Railway or the OS is ending this process now. Any render in flight will not complete.`);
    process.exit(0);
  });
});

// Periodic memory snapshot — logs actual Node heap + RSS (total process
// memory, closer to what Railway's own Memory graph measures) every 30s.
// If the real cause is a slow climb toward SOME ceiling rather than a
// sudden spike, this shows it happening in real time in the render logs
// themselves, correlated with exactly which render step was running —
// something Railway's own graph, aggregated over a full hour, couldn't
// show clearly enough to pin down last time.
setInterval(() => {
  const mem = process.memoryUsage();
  console.log(`[MEMORY] rss=${(mem.rss / 1024 / 1024).toFixed(0)}MB heapUsed=${(mem.heapUsed / 1024 / 1024).toFixed(0)}MB heapTotal=${(mem.heapTotal / 1024 / 1024).toFixed(0)}MB external=${(mem.external / 1024 / 1024).toFixed(0)}MB`);
}, 30000).unref(); // .unref() — this timer should never be the reason the process stays alive on its own

// TEMPORARY DIAGNOSTIC (July 7, 2026): SMART_CORRECT_WEBHOOK_URL appears
// correctly set in Railway's dashboard, but process.env.SMART_CORRECT_WEBHOOK_URL
// is resolving as undefined at runtime inside notifyWebhook(), causing every
// Smart Correct batch to fall back to VIDEO_WEBHOOK_URL instead. This dumps
// every env var name containing "WEBHOOK" at startup, so we can see exactly
// what Node actually has access to — removing all guesswork from dashboard
// screenshots, which can't show trailing whitespace or environment-scoping
// issues. Remove this block once the root cause is confirmed and fixed.
console.log("[STARTUP DIAGNOSTIC] Env vars containing 'WEBHOOK':",
  Object.keys(process.env).filter(k => k.includes("WEBHOOK")).map(k => `${k}=[${process.env[k]}]`)
);

// In-memory job tracking for active renders (process-local; Supabase is
// the source of truth for status, this is just for quick local debugging)
const activeJobs = new Map();

// ── AUTH MIDDLEWARE ──────────────────────────────────────────────────────
// Every request to /render must include the shared secret. This is the
// only thing standing between this service and the open internet.

function requireSecret(req, res, next) {
  const provided = req.headers["x-railway-secret"];
  if (!provided || provided !== process.env.RAILWAY_SECRET) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  next();
}

// ── HEALTH CHECK ──────────────────────────────────────────────────────────
// Netlify pings this before queuing a job, to confirm the service is alive.
// No auth required — this endpoint reveals nothing sensitive.

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "smart-stage-pro-plus-render",
    activeJobs: activeJobs.size,
    timestamp: new Date().toISOString(),
  });
});

// ── RENDER ENDPOINT ──────────────────────────────────────────────────────
// Receives a complete job payload, acknowledges immediately, and processes
// asynchronously. This endpoint must respond fast — actual rendering can
// take 30-90+ seconds, far longer than an HTTP client should wait.

app.post("/render", requireSecret, async (req, res) => {
  const job = req.body;

  if (!job.jobId || !job.frames || !Array.isArray(job.frames) || job.frames.length === 0) {
    return res.status(400).json({ error: "Missing jobId or frames array" });
  }

  // Acknowledge immediately — do not make Netlify/agent wait on this request
  res.status(202).json({ accepted: true, jobId: job.jobId });

  activeJobs.set(job.jobId, { startedAt: Date.now() });

  // Process in the background. Errors are caught and reported via webhook —
  // they must never crash the server process.
  processRenderJob(job)
    .catch((err) => {
      console.error(`Job ${job.jobId} failed:`, err.message);
    })
    .finally(() => {
      activeJobs.delete(job.jobId);
    });
});

// ── SMART CORRECT BATCH ENDPOINT ──────────────────────────────────────────
// Smart Connect™ — Module 1/2 deterministic image correction.
// Same shape as /render: acknowledge fast (202), process in the background,
// report the result via a webhook (SMART_CORRECT_WEBHOOK_URL, separate
// from the video webhook so results never collide with jobId-shaped
// video payloads on the Netlify side).

app.post("/correct-batch", requireSecret, async (req, res) => {
  const job = req.body;

  if (!job.batchId || !job.images || !Array.isArray(job.images) || job.images.length === 0) {
    return res.status(400).json({ error: "Missing batchId or images array" });
  }

  // Acknowledge immediately — Module 1/2 correction across a batch can take
  // real time even with parallelism; the Netlify side must not hold this
  // HTTP connection open waiting for it.
  res.status(202).json({ accepted: true, batchId: job.batchId });

  activeJobs.set(job.batchId, { startedAt: Date.now(), type: "smart-correct" });

  processCorrectBatch(job)
    .catch((err) => {
      console.error(`Smart Correct batch ${job.batchId} failed:`, err.message);
    })
    .finally(() => {
      activeJobs.delete(job.batchId);
    });
});

// ── TEMPORARY DIAGNOSTIC — TEST LTX ─────────────────────────────────────
// Added (July 20, 2026) to test the REAL, deployed ltxMotion.js directly —
// not a parallel/reimplemented script that could quietly drift from what
// production actually runs (exactly the kind of gap several of today's
// bugs turned out to be). Same secret-auth as /render and /correct-batch:
// this makes a real, billed fal.ai call, so it can't be left open.
//
// Usage (GET, so it's testable from a browser address bar or Postman,
// no request body needed):
//   /test-ltx?imageUrl=<url>&preset=<presetKey>&duration=<seconds>
//     &x-railway-secret header required, same as /render
//
// Bypasses the entire render pipeline on purpose — no Ken Burns opener,
// no wipe, no narration, no job/webhook plumbing. Just the one thing
// actually being tested: does the real LTX call succeed end to end.
//
// Remove this route once LTX is confirmed reliably working in real
// renders and no longer needs this kind of direct, isolated testing.
app.get("/test-ltx", requireSecret, async (req, res) => {
  const { imageUrl, preset, duration, roomType, isOpenPlan } = req.query;

  if (!imageUrl || !preset) {
    return res.status(400).json({
      error: "Missing required query params.",
      required: ["imageUrl", "preset"],
      optional: ["duration (seconds, snaps to nearest valid LTX value)", "roomType (default: 'default')", "isOpenPlan ('true' or omit)"],
      example: "/test-ltx?imageUrl=https://res.cloudinary.com/.../staged.jpg&preset=cinematic_push&duration=6",
    });
  }

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "test-ltx-"));

  try {
    // Minimal frame object — only the fields generateLtxContinuationClip
    // and enforceLtxScopeRules actually read. Real production frames
    // carry far more (reveal state, motion presets, etc.); none of that
    // is relevant here.
    const frame = {
      remoteImageUrl: imageUrl,
      continuationDurationSeconds: duration ? Number(duration) : 6,
      isOpenPlan: isOpenPlan === "true",
      roomType: roomType || "default",
    };

    const result = await generateLtxContinuationClip(frame, preset, workDir, "test-ltx-route");

    res.json({
      success: true,
      preset,
      requestedDuration: duration ? Number(duration) : 6,
      actualDuration: result.duration,
      videoUrl: result.videoUrl,
      note: "videoUrl is hosted directly by fal.ai — open it in a browser to watch the clip. Full [LTX] log detail for this request is also in this service's regular logs, same as a real render.",
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  } finally {
    // Clean up the local download even on success — this route only
    // needs to confirm the real API call works and hand back fal.ai's
    // own hosted URL, not keep a local copy on this container.
    fs.rm(workDir, { recursive: true, force: true }, () => {});
  }
});

app.listen(PORT, () => {
  console.log(`Smart Stage PRO Plus render service listening on port ${PORT}`);
});
