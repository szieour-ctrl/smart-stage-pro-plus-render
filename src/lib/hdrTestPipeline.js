// hdrTestPipeline.js — STANDALONE HDR gain-map test orchestration
//
// Mirrors the shape of correctOneImage() in correctPipeline.js (write
// base64 to disk, spawn a Python script, read results back) but is a
// fully separate file, imported only by the isolated /hdr-gainmap-test
// route in server.js. Nothing here is called from, or calls into,
// correctPipeline.js / smartCorrect.py. Easy to delete cleanly once this
// investigation phase is done.

const path = require("path");
const fs = require("fs");
const os = require("os");
const { spawn } = require("child_process");

const HDR_TEST_PY = path.join(__dirname, "hdrGainMapTest.py");

function base64ToExt(mimeType) {
  if (!mimeType) return ".jpg";
  if (mimeType.includes("png")) return ".png";
  return ".jpg"; // HEIC will arrive with an incorrect .jpg extension here if
                 // mimeType is missing/wrong — fine for now, exiftool and
                 // cv2 both sniff actual file content, not the extension.
}

function fileToBase64(filePath) {
  return fs.readFileSync(filePath).toString("base64");
}

/**
 * Runs hdrGainMapTest.py against one uploaded image. Resolves with the
 * parsed JSON report plus standard/recovered images re-read as base64
 * (recovered may be absent if no gain map was found or extraction failed
 * — that's a normal, reportable outcome, not an error).
 */
function runHdrTest(image, targetDisplayHeadroom, highlightReserve) {
  return new Promise((resolve) => {
    const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "hdr-test-"));
    const ext = base64ToExt(image.mimeType);
    const sourcePath = path.join(workDir, `src${ext}`);
    const outDir = path.join(workDir, "out");

    try {
      fs.writeFileSync(sourcePath, Buffer.from(image.imageBase64, "base64"));
      fs.mkdirSync(outDir);
    } catch (err) {
      fs.rm(workDir, { recursive: true, force: true }, () => {});
      return resolve({ status: "error", error: `Failed to write source image: ${err.message}` });
    }

    const args = [
      HDR_TEST_PY,
      "--source", sourcePath,
      "--output-dir", outDir,
    ];
    // Only pass this flag when the caller explicitly set a value — leaving
    // it off lets hdrGainMapTest.py use its own scene-aware default (the
    // photo's actual decoded headroom) instead of being silently forced
    // back to a fixed number here.
    if (targetDisplayHeadroom !== undefined && targetDisplayHeadroom !== null && targetDisplayHeadroom !== "") {
      args.push("--target-display-headroom", String(targetDisplayHeadroom));
    }
    if (highlightReserve !== undefined && highlightReserve !== null && highlightReserve !== "") {
      args.push("--highlight-reserve", String(highlightReserve));
    }
    let stdout = "";
    let stderr = "";

    const proc = spawn("python3", args);
    proc.stdout.on("data", (d) => { stdout += d.toString(); });
    proc.stderr.on("data", (d) => { stderr += d.toString(); });

    proc.on("close", (code) => {
      if (code !== 0) {
        fs.rm(workDir, { recursive: true, force: true }, () => {});
        return resolve({
          status: "error",
          error: `hdrGainMapTest.py exited ${code}`,
          stderr: stderr.slice(0, 1500),
          stdout: stdout.slice(0, 1500),
        });
      }
      try {
        const report = JSON.parse(stdout.trim());
        const result = { status: "done", report };

        if (report.standardDecode?.output && fs.existsSync(report.standardDecode.output)) {
          result.standardBase64 = fileToBase64(report.standardDecode.output);
        }
        if (report.recoveredDecode?.output && fs.existsSync(report.recoveredDecode.output)) {
          result.recoveredBase64 = fileToBase64(report.recoveredDecode.output);
        }
        resolve(result);
      } catch (err) {
        resolve({
          status: "error",
          error: `Failed to parse hdrGainMapTest.py output: ${err.message}`,
          stdout: stdout.slice(0, 1500),
          stderr: stderr.slice(0, 800),
        });
      } finally {
        fs.rm(workDir, { recursive: true, force: true }, () => {});
      }
    });

    proc.on("error", (err) => {
      fs.rm(workDir, { recursive: true, force: true }, () => {});
      resolve({ status: "error", error: `Failed to spawn hdrGainMapTest.py: ${err.message}` });
    });
  });
}

module.exports = { runHdrTest };
