// notify.js — Calls the Netlify webhook to report job completion or failure.
// Fully functional today — no external dependency beyond the webhook URL
// and shared secret, both set as Railway environment variables.

const axios = require("axios");

async function notifyWebhook(payload, webhookUrlOverride) {
  // Backward compatible: every existing call site (renderPipeline.js) calls
  // this with just one argument, so webhookUrlOverride is undefined and
  // behavior is byte-for-byte identical to before this change.
  // New callers (correctPipeline.js) pass SMART_CORRECT_WEBHOOK_URL
  // explicitly, so a Smart Correct batch result never gets POSTed to the
  // video webhook receiver, which expects a jobId-shaped payload, not a
  // batchId-shaped one.
  const targetUrl = webhookUrlOverride || process.env.VIDEO_WEBHOOK_URL;
  if (!targetUrl) {
    console.warn("No webhook URL configured — skipping notification. Job result:", payload);
    return;
  }

  try {
    await axios.post(targetUrl, payload, {
      headers: {
        "Content-Type": "application/json",
        "x-webhook-secret": process.env.WEBHOOK_SECRET || "",
      },
      timeout: 15000,
    });
  } catch (err) {
    // Log but don't throw — a failed notification shouldn't crash the
    // render process. The job status in Supabase may need manual review
    // if this keeps happening; check Railway logs.
    console.error("Webhook notification failed:", err.message);
  }
}

module.exports = { notifyWebhook };
