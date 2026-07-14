// cloudinaryUpload.js — Uploads finished video files to Cloudinary.
// Fully functional today — uses the same Cloudinary account already
// configured for Smart Stage PRO image storage.

const cloudinary = require("cloudinary").v2;

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key:    process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

async function uploadToCloudinary(outputs, projectId) {
  const urls = {};

  for (const [format, localPath] of Object.entries(outputs)) {
    const result = await cloudinary.uploader.upload(localPath, {
      resource_type: "video",
      folder: `smart-stage-pro-plus/${projectId}`,
      public_id: `video_${format}_${Date.now()}`,
      overwrite: false,
    });
    urls[format] = result.secure_url;
  }

  return urls;
}

// NEW (July 14, 2026 — footage-grounded narration rebuild) — narration
// audio is generated Railway-side now (see narrationGen.js), so it needs
// its own upload path here too, for the same reason video outputs do:
// a durable record of what was actually generated. resource_type "raw"
// since this is just storage for a stable URL, not something Cloudinary
// needs to transform (matches the same choice made in the Netlify-side
// version this replaces, upload-staged.js's signed-upload pattern —
// this repo already has the real Cloudinary SDK available, so no need
// for that pattern's native-https workaround here).
async function uploadAudioToCloudinary(localPath, projectId, label) {
  const result = await cloudinary.uploader.upload(localPath, {
    resource_type: "raw",
    folder: `smart-stage-pro-plus/${projectId}/narration`,
    public_id: `${label}_${Date.now()}`,
    overwrite: false,
  });
  return result.secure_url;
}

module.exports = { uploadToCloudinary, uploadAudioToCloudinary };
