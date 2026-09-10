// imagePrep.js — Railway
//
// Shared helper for preparing an S3 image URL for a motion-generation API
// (Kling via fal.ai, LTX via fal.ai) — both aspect cropping AND file-size
// guarding, replacing two separate Cloudinary on-the-fly URL transforms
// that silently stopped working after the S3 migration (August 2026):
//
//   1. klingMotion.js's forceCloudinary16x9() used Cloudinary's
//      `c_fill,ar_16:9,g_auto` to crop real-estate photos (often 3:2) to
//      16:9 before Kling ever saw them. Its `/upload/` guard clause meant
//      it silently no-op'd on S3 URLs — no error, just full uncropped
//      images sent to Kling from that point on.
//   2. ltxMotion.js's buildCroppedStartUrl() used a similar
//      `c_crop,g_center,w_0.94,h_0.94` transform for its two-image
//      workflow (orbit_arc, micro_dolly_back, open_plan_reveal) — same
//      silent no-op on S3.
//   3. Separately, ANY single-image LTX call (not just the two-image
//      presets) was sending full-resolution S3 Finals — up to 12,000px,
//      upscale-image.js's own ceiling — directly to fal.ai with no size
//      guard at all. Confirmed via a real render log: fal.ai's LTX
//      endpoint rejected a source image for exceeding its 7,340,032-byte
//      (7MB) limit, silently falling back to Ken Burns for that clip.
//
// KNOWN BEHAVIORAL DIFFERENCE from the old Cloudinary crop: Cloudinary's
// `g_auto` used content-aware/ML gravity to pick the smartest crop region
// automatically. This does a plain CENTER crop instead — no ML gravity
// detection available without Cloudinary. For real estate interiors this
// is usually fine (the subject is rarely at the frame edge), but if crops
// start looking obviously off-center on certain shots, that's the reason,
// and center-weighted heuristics (e.g. biasing toward the lower-middle
// where floors/furniture usually sit) would be the next thing to try —
// not a sign this function is broken.
//
// Output objects are uploaded to a SCRATCH prefix (smart-stage-scratch/)
// which must stay public (fal.ai's servers fetch by URL, can't
// authenticate to our bucket) — see the bucket policy PublicReadMedia
// statement. These are ephemeral, only needed for the duration of one
// render — set an S3 Lifecycle rule expiring this prefix after 1-2 days
// so it doesn't quietly accumulate storage cost forever (not done here;
// a bucket-level console setting, see handoff notes).

const https = require("https");
const crypto = require("crypto");
const sharp = require("sharp");
const { S3Client, PutObjectCommand } = require("@aws-sdk/client-s3");

const s3 = new S3Client({
  region: process.env.S3_REGION,
  credentials: {
    accessKeyId: process.env.S3_ACCESS_KEY_ID,
    secretAccessKey: process.env.S3_SECRET_ACCESS_KEY,
  },
});

// fal.ai's actual documented ceiling is 7,340,032 bytes (7MB) — staying
// meaningfully under it (6MB) rather than exactly at it, since JPEG
// re-encoding at a given quality setting doesn't hit an exact byte count.
const MAX_BYTES = 6 * 1024 * 1024;
const JPEG_QUALITY_STEPS = [85, 75, 65, 55, 45]; // tried in order until under MAX_BYTES

function downloadBuffer(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return downloadBuffer(res.headers.location).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode} fetching ${url}`));
        res.resume();
        return;
      }
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => resolve(Buffer.concat(chunks)));
      res.on("error", reject);
    }).on("error", reject);
  });
}

// Prepares a single image for a motion API: optional crop, then JPEG
// re-encode stepped down in quality until under MAX_BYTES, uploaded to
// the scratch prefix. Returns the new public URL.
//
// cropTo16x9: true replicates forceCloudinary16x9's old Kling behavior
// (center crop to 16:9 — Cloudinary's content-aware gravity isn't
// available, see file header).
// cropPercent: e.g. 0.94 replicates ltxMotion.js's old
// TWO_IMAGE_CROP_TRANSFORM (crop to N% width/height, centered, no aspect
// change — LTX gets its own explicit aspect_ratio:"16:9" parameter
// separately, so this crop was never about aspect ratio in the first
// place, just a tighter "start frame" for its two-image workflow).
// Leave both unset for a plain size-guard pass with no cropping — the
// single-image LTX case (the one that actually failed in the real log
// this was built from) needs exactly that: never cropped under
// Cloudinary either, just guarded against fal.ai's 7MB ceiling.
//
// UPDATED (Sep 10, 2026 — klingMotion.js's two-image crop workflow for
// orbit_arc/pan_zoom_reveal, ported from ltxMotion.js). cropTo16x9 and
// cropPercent now COMPOSE when both are passed together — cropPercent
// crops the ALREADY-16:9-cropped frame, not the original photo — instead
// of being mutually exclusive (previously an if/else-if: passing both
// would have silently applied only cropTo16x9 and dropped cropPercent
// with no error, no log, nothing — exactly the class of silent bug this
// file's own header calls out about the old Cloudinary transforms it
// replaced). No existing caller passes both at once today, so this is
// purely additive — cropTo16x9-only and cropPercent-only behavior are
// both byte-for-byte unchanged.
async function prepareImageForMotionAPI(sourceUrl, { cropTo16x9 = false, cropPercent = null, jobId = "unknown" } = {}) {
  const bucket = process.env.S3_BUCKET_NAME;
  const region = process.env.S3_REGION;
  if (!bucket || !region) throw new Error("S3_BUCKET_NAME or S3_REGION not configured");

  const original = await downloadBuffer(sourceUrl);
  let pipeline = sharp(original);
  const meta = await sharp(original).metadata();

  // Tracks the CURRENT working dimensions through the pipeline, not the
  // original photo's — cropPercent, when it runs after cropTo16x9, needs
  // to crop relative to the frame cropTo16x9 just produced, or the two
  // extract() calls below would be computed off the wrong base and either
  // crop too little/much or attempt a region larger than what's actually
  // left after the first crop.
  let currentWidth = meta.width;
  let currentHeight = meta.height;

  if (cropTo16x9) {
    const targetRatio = 16 / 9;
    const currentRatio = currentWidth / currentHeight;
    if (currentRatio > targetRatio) {
      // wider than 16:9 — crop left/right
      const newWidth = Math.round(currentHeight * targetRatio);
      const left = Math.round((currentWidth - newWidth) / 2);
      pipeline = pipeline.extract({ left, top: 0, width: newWidth, height: currentHeight });
      currentWidth = newWidth;
    } else if (currentRatio < targetRatio) {
      // taller than 16:9 — crop top/bottom
      const newHeight = Math.round(currentWidth / targetRatio);
      const top = Math.round((currentHeight - newHeight) / 2);
      pipeline = pipeline.extract({ left: 0, top, width: currentWidth, height: newHeight });
      currentHeight = newHeight;
    }
    // else already 16:9 — no crop needed
  }

  if (cropPercent) {
    // Runs AFTER cropTo16x9 above when both are requested — currentWidth/
    // currentHeight already reflect that crop, so this correctly narrows
    // the 16:9 frame itself (chained sharp .extract() calls crop relative
    // to the pipeline's current state, not the original image), rather
    // than computing an independent 94% crop of the raw photo that then
    // gets discarded by a second, unrelated 16:9 crop.
    const newWidth = Math.round(currentWidth * cropPercent);
    const newHeight = Math.round(currentHeight * cropPercent);
    const left = Math.round((currentWidth - newWidth) / 2);
    const top = Math.round((currentHeight - newHeight) / 2);
    pipeline = pipeline.extract({ left, top, width: newWidth, height: newHeight });
    currentWidth = newWidth;
    currentHeight = newHeight;
  }

  let outputBuffer = null;
  for (const quality of JPEG_QUALITY_STEPS) {
    outputBuffer = await pipeline.clone().jpeg({ quality }).toBuffer();
    if (outputBuffer.length <= MAX_BYTES) break;
  }
  if (outputBuffer.length > MAX_BYTES) {
    console.warn(`[imagePrep] [${jobId}] Still over ${MAX_BYTES} bytes after lowest quality step (${outputBuffer.length} bytes) — proceeding anyway, API may still reject it.`);
  }

  const key = `smart-stage-scratch/${jobId}/${crypto.randomUUID()}.jpg`;
  await s3.send(new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: outputBuffer,
    ContentType: "image/jpeg",
  }));

  const newUrl = `https://${bucket}.s3.${region}.amazonaws.com/${key}`;
  // Log label now reflects cropPercent too, not just cropTo16x9 — with
  // the two composing (see above), a cropPercent-only or combined call
  // was previously logged as plain "sized copy" with no indication any
  // cropping happened at all.
  const cropLabel = (cropTo16x9 || cropPercent) ? "cropped+" : "";
  console.log(`[imagePrep] [${jobId}] Prepared ${cropLabel}sized copy: ${newUrl} (${Math.round(outputBuffer.length / 1024)}KB)`);
  return newUrl;
}

module.exports = { prepareImageForMotionAPI };
