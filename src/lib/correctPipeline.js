// correctPipeline.js — Orchestrates a Smart Connect(TM) batch correction job
//
// Pipeline: write each image's base64 to local disk → spawn smartCorrect.py
//           per image IN PARALLEL (this is the whole point of doing this on
//           Railway rather than a Netlify function — real compute, real
//           parallelism) → read corrected results back as base64 → notify
//           Netlify's webhook with the full batch result.
//
// Unlike renderPipeline.js, there is no Cloudinary upload step here. Per
// the established rule (no previews/drafts/intermediate outputs go to
// Cloudinary, only finals), corrected images from THIS step are drafts —
// they only reach Cloudinary later, if/when the user picks SSC and hits
// "Generate Corrected Final" (generate-corrected-final.js, Netlify side).
// This function's job is just to run Module 1/2 and hand base64 back.
//
// Sam's decision (July 7, 2026): user waits for the FULL batch before
// seeing any result — no progressive per-image display. This function
// still processes images in parallel internally (Promise.all) to keep
// total wait time reasonable; it just doesn't report partial progress.

const path = require("path");
const fs = require("fs");
const os = require("os");
const { spawn } = require("child_process");

const { notifyWebhook } = require("./notify");

const CORRECT_PY = path.join(__dirname, "smartCorrect.py");

function base64ToExt(mimeType) {
  if (!mimeType) return ".jpg";
  if (mimeType.includes("png")) return ".png";
  if (mimeType.includes("webp")) return ".webp";
  return ".jpg";
}

/**
 * Spawns smartCorrect.py for one image. Resolves with the parsed JSON
 * result plus the corrected image re-read as base64. Never rejects on a
 * per-image failure — resolves with { status: "error", error } instead,
 * so one bad image in a batch doesn't kill the other 19.
 */
function correctOneImage(image, workDir) {
  return new Promise((resolve) => {
    const ext = base64ToExt(image.mimeType);
    const sourcePath = path.join(workDir, `src_${image.id}${ext}`);
    const outputPath = path.join(workDir, `out_${image.id}${ext}`);

    try {
      fs.writeFileSync(sourcePath, Buffer.from(image.imageBase64, "base64"));
    } catch (err) {
      console.error(`[correctOneImage] ${image.id}: failed to write source image: ${err.message}`);
      return resolve({ id: image.id, status: "error", error: `Failed to write source image: ${err.message}` });
    }

    const args = [CORRECT_PY, "--source", sourcePath, "--output", outputPath];
    let stdout = "";
    let stderr = "";

    const proc = spawn("python3", args);
    proc.stdout.on("data", (d) => { stdout += d.toString(); });
    proc.stderr.on("data", (d) => { stderr += d.toString(); });

    proc.on("close", (code) => {
      if (code !== 0) {
        console.error(`[correctOneImage] ${image.id}: smartCorrect.py exited ${code}. stderr: ${stderr.slice(0, 800)}`);
        return resolve({
          id: image.id,
          status: "error",
          error: `smartCorrect.py failed (exit ${code}): ${stderr.slice(0, 500)}`,
        });
      }
      try {
        const parsed = JSON.parse(stdout.trim());
        const correctedBase64 = fs.readFileSync(outputPath).toString("base64");
        resolve({
          id: image.id,
          status: "done",
          correctedBase64,
          modulesApplied: parsed.modulesApplied,
          modulesSkipped: parsed.modulesSkipped,
          perspectiveCorrectionDegrees: parsed.perspectiveCorrectionDegrees,
        });
      } catch (err) {
        console.error(`[correctOneImage] ${image.id}: failed to read/parse output. stdout: ${stdout.slice(0, 300)} | err: ${err.message}`);
        resolve({ id: image.id, status: "error", error: `Failed to read corrected output: ${err.message}` });
      }
    });

    proc.on("error", (err) => {
      console.error(`[correctOneImage] ${image.id}: failed to spawn python3: ${err.message}`);
      resolve({ id: image.id, status: "error", error: `Failed to spawn smartCorrect.py: ${err.message}` });
    });
  });
}

async function processCorrectBatch(job) {
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), `smart-correct-${job.batchId}-`));

  try {
    console.log(`[${job.batchId}] Starting Smart Correct batch. ${job.images.length} images.`);

    // Real parallelism — this is the reason batch correction runs on
    // Railway instead of sequentially in a Netlify function. All images
    // process concurrently; total wait is closer to the slowest single
    // image than the sum of all of them.
    const results = await Promise.all(
      job.images.map((image) => correctOneImage(image, workDir))
    );

    const doneCount = results.filter(r => r.status === "done").length;
    const errorCount = results.filter(r => r.status === "error").length;
    console.log(`[${job.batchId}] Batch complete — ${doneCount} done, ${errorCount} errored.`);

    await notifyWebhook({
      batchId: job.batchId,
      status: "done",
      results,
    }, process.env.SMART_CORRECT_WEBHOOK_URL);

    console.log(`[${job.batchId}] Notified Netlify webhook.`);

  } catch (err) {
    console.error(`[${job.batchId}] Smart Correct batch failed:`, err);
    await notifyWebhook({
      batchId: job.batchId,
      status: "error",
      error: err.message,
    }, process.env.SMART_CORRECT_WEBHOOK_URL);
    throw err;
  } finally {
    fs.rm(workDir, { recursive: true, force: true }, (err) => {
      if (err) console.error(`Cleanup failed for ${workDir}:`, err.message);
    });
  }
}

module.exports = { processCorrectBatch };
