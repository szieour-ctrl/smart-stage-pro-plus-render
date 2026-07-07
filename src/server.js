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
const { processRenderJob } = require("./renderPipeline");
const { processCorrectBatch } = require("./lib/correctPipeline");

const app = express();
app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT || 3000;

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

app.listen(PORT, () => {
  console.log(`Smart Stage PRO Plus render service listening on port ${PORT}`);
});
