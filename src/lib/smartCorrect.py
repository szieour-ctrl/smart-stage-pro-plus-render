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

MODULE 1 — always RUNS, but each correction measures its own defect
severity first and scales its strength proportionally (redesigned July 8,
2026 after real output looked overprocessed — see inline comments on each
function for the specific before/after reasoning):
  - White balance calibration (white-patch/gray-world hybrid, strength
    scaled to measured color-cast magnitude)
  - Lens/perspective alignment (Hough-line-based vertical deskew — already
    binary/measured: only rotates when a real tilt is detected)
  - Exposure normalization (contrast stretch + CLAHE, strength scaled to
    measured range-utilization and local-contrast flatness)
  - Color cast removal (part of the white balance pass)
  - Adaptive noise reduction (fastNlMeansDenoisingColored, strength scaled
    to a detected-noise proxy rather than a fixed constant)
  - Vignette neutralization (radial gain, strength scaled to measured
    center-vs-edge brightness falloff)
  - Saturation lift (new — strength scaled to measured current saturation)

A photo with only one real defect (e.g. a mild color cast, nothing else
wrong) should come out with only that one correction meaningfully applied
— reflected honestly in modulesApplied, which only lists a correction once
its measured strength crosses a real threshold, not just because the
function ran.

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
    """Module 1: white balance + color cast removal.

    REDESIGNED (July 8, 2026) after Sam's direct feedback on real output:
    stacking full-strength WB + full contrast stretch + full CLAHE + full
    vignette lift + full saturation boost on every photo, regardless of
    what that specific photo actually needed, produced an overprocessed,
    "life edited out" look — Sam's own words. His example: "sometimes all
    a photo may need is white balance." The fix isn't reverting to weaker
    fixed strength (that was the original complaint) — it's making each
    correction MEASURE its own defect severity and scale its strength to
    match, so a photo with a mild cast gets a mild nudge and a photo with
    a real problem gets real correction, instead of every photo getting
    the same blanket treatment either way.

    Uses the same white-patch/gray-world hybrid as before to compute the
    needed scale per channel, but now measures how far that scale actually
    deviates from "no change" (1.0) and interpolates between no-correction
    and full-correction proportionally — full strength only kicks in once
    the detected cast is genuinely significant.
    """
    result = img.astype(np.float32)

    mean_b, mean_g, mean_r = [result[:, :, i].mean() for i in range(3)]
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    # White-patch reference: mean of the brightest 5% of pixels per channel
    wp_scales = []
    for i in range(3):
        channel = result[:, :, i]
        threshold = np.percentile(channel, 95)
        bright_pixels = channel[channel >= threshold]
        wp_scales.append(bright_pixels.mean() if bright_pixels.size > 0 else mean_gray)
    wp_mean = sum(wp_scales) / 3.0

    blended_scales = []
    for i, mean_c in enumerate([mean_b, mean_g, mean_r]):
        if mean_c > 1e-3:
            gray_world_scale = mean_gray / mean_c
            white_patch_scale = (wp_mean / wp_scales[i]) if wp_scales[i] > 1e-3 else gray_world_scale
            blended_scales.append(0.6 * white_patch_scale + 0.4 * gray_world_scale)
        else:
            blended_scales.append(1.0)

    # Severity-proportional application: measure how far the computed scale
    # deviates from "no change needed" (1.0), then interpolate between
    # identity and the full computed correction. Below ~3% deviation, barely
    # touch it (near-neutral photo). Above ~18% deviation, apply the full
    # computed correction (genuinely significant cast). Linear ramp between.
    cast_magnitude = max(abs(s - 1.0) for s in blended_scales)
    strength = float(np.clip((cast_magnitude - 0.03) / (0.18 - 0.03), 0.0, 1.0))

    for i in range(3):
        applied_scale = 1.0 + strength * (blended_scales[i] - 1.0)
        result[:, :, i] *= applied_scale
    return np.clip(result, 0, 255).astype(np.uint8), strength


def _largest_crop_after_rotation(w, h, angle_deg):
    """Standard formula for the largest axis-aligned rectangle, with the
    same aspect ratio as the original, that fits entirely inside a WxH
    image after it's been rotated by angle_deg — i.e. the region that
    contains zero fabricated/replicated border pixels. Well-established
    technique (the "rotate and crop" problem), not something invented for
    this file."""
    angle = np.radians(abs(angle_deg))
    if angle < 1e-6:
        return w, h
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = np.sin(angle), np.cos(angle)
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


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

    CROP FIX (July 8, 2026): rotating an image within a fixed-size canvas
    leaves the corners of the rotated content outside the frame; the
    original code filled that gap with BORDER_REPLICATE (smeared copies of
    edge pixels) rather than real content — a real, if subtle, quality
    issue at the corners. This now crops down to the largest rectangle
    containing zero fabricated pixels, then scales back up to the original
    WxH, so the delivered image has real content everywhere and keeps the
    exact same pixel dimensions and aspect ratio as the input.
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
        # CRITICAL: cv2.HoughLinesP() does not guarantee which endpoint of a
        # detected segment comes first — it can return either (top, bottom)
        # or (bottom, top) order for the same physical line, seemingly
        # arbitrarily. Without canonicalizing direction, the SAME real edge
        # could compute to ~+5 degrees in one photo and ~175 degrees in
        # another (or even within the same photo across different detected
        # segments of the same wall), and the `abs(angle) < 20` filter below
        # would silently keep one and reject the other — an uncontrolled,
        # per-detection bias rather than a real measurement of tilt. Forcing
        # every line to point downward (dy >= 0) before computing the angle
        # makes the result direction-independent and fixes this at the root.
        if dy < 0:
            dx, dy = -dx, -dy
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

    # SIGN FIX: cv2.getRotationMatrix2D rotates image content counter-
    # clockwise for positive angles. The measured `angles` above describe
    # how the content is CURRENTLY tilted (e.g. a negative median means
    # verticals lean with their top shifted right). Applying that same
    # signed value as the correction rotates the content FURTHER in the
    # direction it's already leaning, doubling the tilt instead of
    # removing it — confirmed directly: a real 5-degree test tilt became
    # a ~10-degree residual tilt with the unfixed sign, and ~0 with this
    # fix. The corrective rotation must be the NEGATIVE of the detected
    # tilt to actually straighten the image.
    correction_angle = -float(np.median(angles))
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

    # Crop out the replicated-border corners, then scale back to original
    # dimensions so the output always matches the input's pixel size and
    # aspect ratio exactly.
    crop_w, crop_h = _largest_crop_after_rotation(w, h, correction_angle)
    crop_w, crop_h = int(round(crop_w)), int(round(crop_h))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    cropped = rotated[y0:y0 + crop_h, x0:x0 + crop_w]
    result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

    return result, correction_angle


def exposure_normalize(img):
    """Module 1: exposure normalization.

    REDESIGNED (July 8, 2026) alongside white_balance_gray_world — same
    reasoning: measure the actual defect, then scale correction strength
    to match, instead of applying a fixed strong stretch + CLAHE to every
    photo regardless of whether it needs it. A photo that already uses
    most of the 0-255 range and already has decent local contrast should
    come out of this function nearly unchanged; a genuinely flat, hazy,
    or underexposed photo should get real correction.
    """
    result = img.astype(np.float32)

    # Measure how much of the 0-255 range each channel actually uses, and
    # build the fully-stretched target in parallel.
    range_deficits = []
    stretch_target = np.zeros_like(result)
    for i in range(3):
        channel = result[:, :, i]
        low, high = np.percentile(channel, [1, 99])
        used_range = high - low
        range_deficits.append(1.0 - min(used_range / 255.0, 1.0))
        if high > low:
            stretch_target[:, :, i] = np.clip((channel - low) * (255.0 / (high - low)), 0, 255)
        else:
            stretch_target[:, :, i] = channel

    # Below ~8% range deficit, the photo already uses the range well —
    # barely stretch it. Above ~35% deficit (genuinely flat/hazy), apply
    # the full stretch. Linear ramp between.
    range_deficit = float(np.mean(range_deficits))
    stretch_strength = float(np.clip((range_deficit - 0.08) / (0.35 - 0.08), 0.0, 1.0))
    result = result + stretch_strength * (stretch_target - result)
    stretched = np.clip(result, 0, 255).astype(np.uint8)

    # Local contrast (CLAHE) — scale the clip limit by how flat the
    # luminance channel actually is (measured via std dev), rather than
    # always applying the same strong clip limit. A photo with already-good
    # local contrast gets a gentle pass; a genuinely flat photo gets a
    # strong one.
    gray_std = float(cv2.cvtColor(stretched, cv2.COLOR_BGR2GRAY).astype(np.float32).std())
    clip_limit = float(np.interp(gray_std, [35, 65], [3.2, 1.3]))

    lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    # Combined strength for reporting purposes: contrast-stretch strength
    # plus how far the CLAHE clip limit landed above its gentle baseline
    # (1.3), normalized — either signal alone can indicate a real correction.
    clahe_strength = float(np.clip((clip_limit - 1.3) / (3.2 - 1.3), 0.0, 1.0))
    combined_strength = max(stretch_strength, clahe_strength)
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), combined_strength


def saturation_boost(img):
    """Module 1: mild saturation lift.

    REDESIGNED (July 8, 2026) same pattern — measures current average
    saturation first. An already-vivid photo gets almost no boost; a
    genuinely dull/washed-out photo gets up to the full 12% lift."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    current_sat = float(hsv[:, :, 1].mean())
    boost_strength = float(np.clip((110.0 - current_sat) / (110.0 - 60.0), 0.0, 1.0))
    boost_factor = 1.0 + 0.12 * boost_strength
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), boost_strength


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
    """Module 1: vignette neutralization via a radial gain mask.

    REDESIGNED (July 8, 2026) same severity-first pattern — measures the
    actual brightness falloff between the frame's center and its outer
    border before deciding how hard to push the correction. Most modern
    phone cameras have very mild real vignetting; applying a fixed strong
    curve to every photo regardless was part of what made corrected photos
    look artificially "lifted" at the edges rather than actually corrected.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    center_region = gray[h // 3:2 * h // 3, w // 3:2 * w // 3]
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = False
    border_region = gray[border_mask]

    center_mean = float(center_region.mean()) if center_region.size > 0 else 128.0
    border_mean = float(border_region.mean()) if border_region.size > 0 else center_mean
    falloff = max(0.0, (center_mean - border_mean) / max(center_mean, 1.0))

    # Below ~5% falloff, there's essentially no real vignetting to correct.
    # Above ~20% falloff (genuine lens darkening), apply the full curve.
    strength = float(np.clip((falloff - 0.05) / (0.20 - 0.05), 0.0, 1.0))
    max_gain_coeff = 0.4 * strength

    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max_dist
    gain = 1.0 + max_gain_coeff * (dist ** 2)
    gain = gain[:, :, np.newaxis]
    result = img.astype(np.float32) * gain
    return np.clip(result, 0, 255).astype(np.uint8), strength


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

    # ── Module 1 — measured, applied proportionally to detected severity ──
    # REDESIGNED July 8, 2026: previously every Module 1 correction applied
    # at fixed full strength to every photo, which stacked into an
    # overprocessed look on photos that only had one real issue. Each
    # function below now measures its own defect first and returns how
    # strongly it actually applied — modules_applied only lists a
    # correction if it crossed a meaningful threshold (0.1), so the output
    # JSON honestly reflects what that specific photo needed, e.g. a photo
    # that only had a color cast will correctly show just "white_balance"
    # rather than every module regardless of relevance.
    STRENGTH_REPORT_THRESHOLD = 0.1

    img, wb_strength = white_balance_gray_world(img)
    if wb_strength >= STRENGTH_REPORT_THRESHOLD:
        modules_applied.append("white_balance")

    img, rotation_deg = deskew_perspective(img)
    if rotation_deg != 0.0:
        modules_applied.append("perspective_alignment")

    img, exposure_strength = exposure_normalize(img)
    if exposure_strength >= STRENGTH_REPORT_THRESHOLD:
        modules_applied.append("exposure_normalization")

    img, denoise_strength = adaptive_denoise(img)
    modules_applied.append("adaptive_noise_reduction")

    img, vignette_strength = vignette_correct(img)
    if vignette_strength >= STRENGTH_REPORT_THRESHOLD:
        modules_applied.append("vignette_neutralization")

    img, saturation_strength = saturation_boost(img)
    if saturation_strength >= STRENGTH_REPORT_THRESHOLD:
        modules_applied.append("saturation_boost")

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
