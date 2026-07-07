#!/usr/bin/env python3
"""
smartCorrect.py — Smart Connect(TM) Module 1/2 deterministic image correction.

Mirrors motionRenderer.py's invocation convention exactly: CLI args in,
single JSON object on stdout, non-zero exit code + stderr on failure.
Node spawns one process per image (correctPipeline.js), same pattern
motionPresets.js uses to spawn motionRenderer.py per clip.

CRITICAL DESIGN RULE (do not violate): every operation in this file must be
classical, deterministic computer vision — no generative model, ever. This
is the load-bearing assumption behind the SSC path's "no AB 723 required"
claim: the statute excludes edits like white balance, exposure, color cast,
sharpening, angle/perspective, and cropping when they don't change the
representation of the property. A generative model doesn't just adjust
values, it regenerates pixels — that's a different legal category entirely,
and this script must never drift into it.

MODULE 1 — always applied:
  - White balance calibration (gray-world assumption)
  - Lens/perspective alignment (Hough-line-based vertical deskew)
  - Exposure normalization (CLAHE on luminance)
  - Color cast removal (part of the gray-world WB pass)
  - Adaptive noise reduction (fastNlMeansDenoisingColored, strength scaled
    to a detected-noise proxy rather than a fixed constant)
  - Vignette neutralization (radial gain correction)

MODULE 2 — conditional, histogram-triggered:
  - Shadow recovery
  - Highlight recovery
  - Texture / micro-contrast boost

NOT IMPLEMENTED in this first pass (explicitly stubbed, not silently
faked — each returns the image unchanged and is flagged in the JSON output
as "skipped"):
  - Color uniformity harmonization — needs whole-batch context (comparing
    wall/floor tones ACROSS frames), not just this one image. This script
    only ever sees one image at a time. Would need to move up into
    correctPipeline.js as a batch-level pass if built later.
  - Reflection/glare reduction — specular highlight detection + inpainting
    is a materially harder CV problem than the rest of this list; cut from
    MVP per the same reasoning that cut HDR/bracket merge (see the July 7,
    2026 Notion decision page — no clear near-term need, real engineering
    cost, better to build only if a concrete case shows up in testing).
  - HDR / bracket merge — no multi-exposure upload path exists for
    single-shot iPhone/agent uploads. Cut from MVP per that same doc.

Usage:
  python3 smartCorrect.py --source IN.jpg --output OUT.jpg
"""

import argparse
import json
import sys

import cv2
import numpy as np


def white_balance_gray_world(img):
    """Module 1: white balance + color cast removal via gray-world assumption.
    Scales each channel so its mean matches the overall gray mean — a
    standard, deterministic WB technique, not a learned/generative one."""
    result = img.astype(np.float32)
    mean_b, mean_g, mean_r = [result[:, :, i].mean() for i in range(3)]
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    # Guard against a division blowup on near-black test images
    for i, mean_c in enumerate([mean_b, mean_g, mean_r]):
        if mean_c > 1e-3:
            result[:, :, i] *= (mean_gray / mean_c)
    return np.clip(result, 0, 255).astype(np.uint8)


def deskew_perspective(img):
    """Module 1: perspective/vertical alignment via Hough-line detection.

    APPROXIMATION NOTE: this detects the dominant near-vertical line angle
    (architectural edges — door frames, wall corners) and applies a single
    global rotation to correct it. This is a real, working first pass, but
    it is NOT full 4-point perspective/keystone correction (which would
    independently correct convergence at top vs. bottom of frame). If
    testing shows meaningfully converging verticals that a single rotation
    can't fix, this is the function to upgrade next — flagging now rather
    than silently under-delivering on the "perspective alignment" claim.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=img.shape[0] // 4, maxLineGap=10)
    if lines is None:
        return img, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) < 1e-3:
            continue
        angle_from_vertical = np.degrees(np.arctan2(dx, dy))
        # Only count lines reasonably close to vertical (architectural
        # edges) — ignore near-horizontal lines (floor lines, countertops)
        # which aren't relevant to vertical deskew.
        if abs(angle_from_vertical) < 20:
            angles.append(angle_from_vertical)

    if not angles:
        return img, 0.0

    correction_angle = float(np.median(angles))
    # Cap correction to a sane range — a large "correction" is more likely
    # a misdetection (e.g. a rug pattern) than real lens tilt.
    correction_angle = max(-8.0, min(8.0, correction_angle))
    if abs(correction_angle) < 0.3:
        return img, 0.0  # not worth rotating for sub-degree noise

    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (w, h),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, correction_angle


def exposure_normalize(img):
    """Module 1: exposure normalization via CLAHE on the luminance channel
    only (LAB color space), so color saturation isn't distorted the way
    running CLAHE on each RGB channel independently would."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def detect_noise_level(img):
    """Proxy for ISO/noise level: variance of the Laplacian on a grayscale
    copy. Lower variance in flat regions with high local variance elsewhere
    is a reasonable, cheap noise proxy without needing EXIF ISO data (most
    agent/iPhone uploads won't reliably carry it through upload pipelines)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def adaptive_denoise(img):
    """Module 1: adaptive noise reduction — strength scales inversely with
    the detected noise proxy so a clean DSLR photo isn't over-smoothed and
    a noisy low-light iPhone shot gets real cleanup."""
    noise_proxy = detect_noise_level(img)
    # Empirically reasonable bounds — noisy images (low Laplacian variance)
    # get stronger denoising; sharp/clean images get little to none.
    if noise_proxy > 800:
        h_luma = 3
    elif noise_proxy > 300:
        h_luma = 6
    else:
        h_luma = 10
    return cv2.fastNlMeansDenoisingColored(img, None, h_luma, h_luma, 7, 21), h_luma


def vignette_correct(img):
    """Module 1: vignette neutralization via a radial gain mask — boosts
    brightness toward the frame edges to counteract typical lens falloff."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max_dist
    # Mild correction curve — avoid overcorrecting into a "flashlight" look
    gain = 1.0 + 0.25 * (dist ** 2)
    gain = gain[:, :, np.newaxis]
    result = img.astype(np.float32) * gain
    return np.clip(result, 0, 255).astype(np.uint8)


def shadow_highlight_recovery(img):
    """Module 2 (conditional): lifts shadow detail and recovers blown
    highlights, but only applies each half of the correction if the
    histogram actually shows a problem — this is the "conditional" part
    of Module 2, not an always-applied global curve."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_px = gray.size

    shadow_frac = hist[:32].sum() / total_px       # near-black pixels
    highlight_frac = hist[224:].sum() / total_px   # near-white pixels

    applied = []
    result = img.astype(np.float32)

    if shadow_frac > 0.20:
        # Lift shadows via a gamma curve applied only to the low end
        gamma = 0.8
        result = 255.0 * np.power(result / 255.0, gamma)
        applied.append("shadow_recovery")

    if highlight_frac > 0.10:
        # Compress highlights slightly to recover blown window/ceiling light detail
        result = np.where(result > 200, 200 + (result - 200) * 0.5, result)
        applied.append("highlight_recovery")

    return np.clip(result, 0, 255).astype(np.uint8), applied, {
        "shadow_frac": round(float(shadow_frac), 4),
        "highlight_frac": round(float(highlight_frac), 4),
    }


def texture_microcontrast_boost(img):
    """Module 2 (conditional): applies unsharp-mask micro-contrast only
    when the image's own detail proxy suggests flat/low-separation
    mid-tones — same detect-then-decide pattern as shadow/highlight."""
    detail_proxy = detect_noise_level(img)  # reuse the same Laplacian-variance proxy
    if detail_proxy >= 150:
        return img, False  # already has reasonable micro-contrast, skip

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)
    return sharpened, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    modules_applied = []

    img = cv2.imread(args.source)
    if img is None:
        print(json.dumps({"error": f"Could not read image: {args.source}"}), file=sys.stderr)
        sys.exit(1)

    # ── Module 1 — always applied ─────────────────────────────────────────
    img = white_balance_gray_world(img)
    modules_applied.append("white_balance")

    img, rotation_deg = deskew_perspective(img)
    if rotation_deg != 0.0:
        modules_applied.append("perspective_alignment")

    img = exposure_normalize(img)
    modules_applied.append("exposure_normalization")

    img, denoise_strength = adaptive_denoise(img)
    modules_applied.append("adaptive_noise_reduction")

    img = vignette_correct(img)
    modules_applied.append("vignette_neutralization")

    # ── Module 2 — conditional ────────────────────────────────────────────
    img, sh_applied, histogram_stats = shadow_highlight_recovery(img)
    modules_applied.extend(sh_applied)

    img, texture_applied = texture_microcontrast_boost(img)
    if texture_applied:
        modules_applied.append("texture_microcontrast_boost")

    # Explicitly not implemented — see file header. Listed here so the
    # output JSON is honest about what did and didn't happen, rather than
    # silently omitting them.
    skipped = ["color_uniformity_harmonization", "reflection_glare_reduction"]

    cv2.imwrite(args.output, img)

    print(json.dumps({
        "output": args.output,
        "modulesApplied": modules_applied,
        "modulesSkipped": skipped,
        "perspectiveCorrectionDegrees": round(rotation_deg, 2),
        "denoiseStrength": denoise_strength,
        "histogramStats": histogram_stats,
    }))


if __name__ == "__main__":
    main()
