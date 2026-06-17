// notify.js — Calls the Netlify webhook to report job completion or failure.
// Fully functional today — no external dependency beyond the webhook URL
// and shared secret, both set as Railway environment variables.

const axios = require("axios");

async function notifyWebhook(payload) {
  if (!process.env.VIDEO_WEBHOOK_URL) {
    console.warn("VIDEO_WEBHOOK_URL not set — skipping notification. Job result:", payload);
    return;
  }

  try {
    await axios.post(process.env.VIDEO_WEBHOOK_URL, payload, {
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
