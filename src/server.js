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

const app = express();
app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT || 3000;

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

app.listen(PORT, () => {
  console.log(`Smart Stage PRO Plus render service listening on port ${PORT}`);
});
