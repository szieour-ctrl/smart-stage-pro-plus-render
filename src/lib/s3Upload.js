// s3Upload.js — Uploads finished video files to S3.
//
// MIGRATED (Aug 25, 2026): replaces cloudinaryUpload.js.
//
// PREFIX: smart-stage-video-finals/ — deliberately NOT smart-stage-finals/
// (that prefix is the staged-IMAGE product's finished output, covered by
// the bucket's public-read policy — see upload-original.js's comment).
// Video needs the opposite: NOT in the public-read policy at all. Per
// Sam's clarification, the old Cloudinary type:"authenticated" + signed
// URL setup wasn't just general security hardening — it's the actual
// mechanism that prevents a user from obtaining the finished video
// before their credit is deducted. A plain public URL would defeat that
// gate entirely (grab the URL, never get charged). So this prefix must
// stay out of any public bucket policy, and the ONLY way to a working
// URL must be a presigned S3 URL minted server-side, from inside the
// same gated endpoints that used to call Cloudinary's signVideoUrl()
// (video-job.js, compliance-page.js) — after their existing credit
// check, not before. This file only produces the S3 key/object; it does
// NOT return anything directly viewable — see uploadToS3's return value
// note below.

const fs = require("fs");
const path = require("path");
const { S3Client, PutObjectCommand } = require("@aws-sdk/client-s3");

const s3 = new S3Client({
  region: process.env.S3_REGION,
  credentials: {
    accessKeyId: process.env.S3_ACCESS_KEY_ID,
    secretAccessKey: process.env.S3_SECRET_ACCESS_KEY,
  },
});

const S3_BUCKET = process.env.S3_BUCKET_NAME;

async function uploadFileToS3(localPath, key, contentType) {
  const body = fs.readFileSync(localPath);
  await s3.send(
    new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: key,
      Body: body,
      ContentType: contentType,
    })
  );
  // Returns the S3 KEY, not a URL — this prefix is private, so there is
  // no working public URL to hand back. Callers (renderPipeline.js) store
  // this key; video-job.js / compliance-page.js presign it into a
  // temporary URL only after their credit-check gate passes.
  return key;
}

// Same call shape as the old uploadToCloudinary(outputs, projectId) —
// one call per output format, returns { [format]: s3Key } now instead
// of { [format]: url }.
async function uploadToS3(outputs, projectId) {
  const keys = {};

  for (const [format, localPath] of Object.entries(outputs)) {
    const key = `smart-stage-video-finals/${projectId}/video_${format}_${Date.now()}.mp4`;
    keys[format] = await uploadFileToS3(localPath, key, "video/mp4");
  }

  return keys;
}

// Same signature/behavior as the old uploadAudioToCloudinary(localPath, projectId, label).
async function uploadAudioToS3(localPath, projectId, label) {
  const key = `smart-stage-video-finals/${projectId}/narration/${label}_${Date.now()}.mp3`;
  return uploadFileToS3(localPath, key, "audio/mpeg");
}

// NEW (this session — /test-motion diagnostic route): simple public
// upload for one-off test/scratch files. NOT for real finished videos —
// those must stay private behind the gated key/presign flow above (see
// this file's header comment for why). This instead targets the
// already-public smart-stage-scratch/ prefix (same one used for
// temporary per-render crop/resize files, with its own 1-2 day S3
// lifecycle expiry already confirmed set up), and hands back a real,
// directly-clickable URL — no credit-check gate needed for a throwaway
// test clip nobody is being billed for.
async function uploadScratchFile(localPath, filename, contentType) {
  const key = `smart-stage-scratch/${filename}`;
  await uploadFileToS3(localPath, key, contentType);
  return `https://${S3_BUCKET}.s3.${process.env.S3_REGION}.amazonaws.com/${key}`;
}

module.exports = { uploadToS3, uploadAudioToS3, uploadScratchFile };
