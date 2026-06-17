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

module.exports = { uploadToCloudinary };
