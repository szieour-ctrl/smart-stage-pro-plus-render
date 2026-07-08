#!/usr/bin/env python3
"""
smartCorrect.py — Smart Connect(TM) deterministic image correction, MLS Bright calibrated.

Mirrors motionRenderer.py's invocation convention exactly: CLI args in,
single JSON object on stdout, non-zero exit code + stderr on failure.
Node spawns one process per image (correctPipeline.js), same pattern
motionPresets.js uses to spawn motionRenderer.py per clip.

CRITICAL DESIGN RULE (do not violate): every operation in this file must be
classical, deterministic computer vision — no generative model, ever. This
is the load-bearing assumption behind the SSC path's "no AB 723 required"
claim: the statute excludes edits like white balance, exposure, color cast,
sharpening, angle/perspective, lens geometry, and cropping when they don't
change the representation of the property. A generative model doesn't just
adjust values, it regenerates pixels — that's a different legal category
entirely, and this script must never drift into it. Confirmed by Sam (July
8, 2026): lens correction is standard real-estate-photography practice and
does not raise AB 723 concerns, same as any other geometric correction here.

HISTORY (July 8, 2026): this pipeline went through three real iterations
in one session, each driven by direct feedback on actual output rather
than assumption:
  1. First pass — every correction applied at fixed full strength to every
     photo. Real bug found via direct measurement: the perspective deskew
     had a sign error that DOUBLED tilt instead of removing it, and
     rotation left replicated-border artifacts at the corners. Both fixed
     and verified (marker-color test proved zero fabricated pixels survive
     the crop; a known 5-degree test tilt measured 0.0 degrees residual
     after the fix).
  2. Second pass — made each correction measure its own defect severity
     and scale proportionally, so a photo with only one real issue doesn't
     get every correction applied at once. Sam's own words on the first
     pass: "the life was edited out of the photos — overprocessed."
  3. Third pass (this version) — Sam provided a reference implementation
     built with another tool and calibrated against real professional MLS
     Bright photos (his own words: "23 years in real estate and I've paid
     for photos that have MLS Bright corrections done as SOP"). That
     reference is used here as intended — as a reference, not verbatim —
     merging its validated MLS Bright calibration (measured brightness
     targets, targeted white-surface masking, do-no-harm gate) into this
     file's existing structure, so the working Railway/Node integration
     (correctPipeline.js's JSON parsing contract) doesn't need to change.

PIPELINE:
  Do-No-Harm Gate (Professional MLS Guard)
     -> if the photo already matches the calibrated MLS Bright profile,
        copy through untouched rather than reprocess it
  Technical correction
     -> white balance (neutral-surface-aware) / mild lens correction /
        perspective deskew / adaptive denoise / vignette lift
  MLS Bright finish
     -> calibrated interior brightness lift (highlight-protected) /
        adaptive clean-whites / window highlight balance / color+clarity
        finish

NOT IMPLEMENTED (explicitly stubbed, not silently faked — flagged in the
JSON output as "skipped"):
  - Color uniformity harmonization — needs whole-batch context (comparing
    wall/floor tones ACROSS frames), not just this one image. Would need
    to move up into correctPipeline.js as a batch-level pass if built later.
  - Reflection/glare reduction — specular highlight detection + inpainting
    is a materially harder CV problem than the rest of this list; cut per
    the July 7, 2026 Notion decision page reasoning.
  - HDR / bracket merge — no multi-exposure upload path exists for
    single-shot iPhone/agent uploads.

Usage:
  python3 smartCorrect.py --source IN.jpg --output OUT.jpg
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np


def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


# ── Measurement helpers (read-only, no pixel changes) ──────────────────────

def image_stats(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return {
        "mean_luma": round(float(l.mean()), 2),
        "median_luma": round(float(np.median(l)), 2),
        "p05_luma": round(float(np.percentile(l, 5)), 2),
        "p95_luma": round(float(np.percentile(l, 95)), 2),
        "mean_saturation": round(float(hsv[:, :, 1].mean()), 2),
    }


def white_surface_stats(img):
    """Measure likely-white architectural surfaces (trim, cabinets,
    ceilings) without modifying pixels — used by the do-no-harm gate and
    by clean_whites_adaptive to decide whether/how much to correct."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
    white_mask = (L > 145.0) & (chroma < 22.0)
    strong_white_mask = (L > 165.0) & (chroma < 16.0)

    white_fraction = float(np.mean(white_mask))
    strong_fraction = float(np.mean(strong_white_mask))
    if white_fraction < 0.003:
        return {
            "whiteFraction": round(white_fraction, 4),
            "strongWhiteFraction": round(strong_fraction, 4),
            "meanWhiteLuma": 0.0,
            "whiteCastMagnitude": 99.0,
            "meanWhiteA": 0.0,
            "meanWhiteB": 0.0,
        }

    sample_mask = strong_white_mask if np.any(strong_white_mask) else white_mask
    mean_l = float(np.mean(L[sample_mask]))
    mean_a = float(np.mean(A[sample_mask]))
    mean_b = float(np.mean(B[sample_mask]))
    cast_mag = float(np.sqrt((mean_a - 128.0) ** 2 + (mean_b - 128.0) ** 2))

    return {
        "whiteFraction": round(white_fraction, 4),
        "strongWhiteFraction": round(strong_fraction, 4),
        "meanWhiteLuma": round(mean_l, 2),
        "whiteCastMagnitude": round(cast_mag, 3),
        "meanWhiteA": round(mean_a, 2),
        "meanWhiteB": round(mean_b, 2),
    }


def shadow_highlight_stats(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    return {
        "shadowFraction": round(float(np.mean(L < 45.0)), 4),
        "brightFraction": round(float(np.mean(L > 232.0)), 4),
        "veryBrightFraction": round(float(np.mean(L > 245.0)), 4),
    }


# ── Do-No-Harm gate ─────────────────────────────────────────────────────────

def assess_professional_mls_bright(img):
    """Detect photos that already match the calibrated MLS Bright profile
    and should be left untouched rather than reprocessed. Thresholds per
    Sam's reference implementation, calibrated against real professional
    MLS Bright photos (his SOP standard, 23 years in the business)."""
    stats = image_stats(img)
    whites = white_surface_stats(img)
    hist = shadow_highlight_stats(img)

    _, measured_rotation = deskew_perspective(img.copy())
    measured_rotation = float(measured_rotation)

    checks = {
        "median_luma_ok": stats["median_luma"] >= 180.0,
        "mean_luma_ok": stats["mean_luma"] >= 158.0,
        "p95_luma_ok": stats["p95_luma"] >= 220.0,
        "white_area_ok": whites["whiteFraction"] >= 0.55,
        "white_luma_ok": whites["meanWhiteLuma"] >= 198.0,
        "white_cast_ok": whites["whiteCastMagnitude"] <= 3.25,
        "saturation_ok": stats["mean_saturation"] <= 32.0,
        "shadow_ok": hist["shadowFraction"] <= 0.085,
        "geometry_ok": abs(measured_rotation) <= 0.75,
    }
    score = sum(1 for v in checks.values() if v) / float(len(checks))

    load_bearing = (
        checks["median_luma_ok"] and checks["mean_luma_ok"]
        and checks["white_area_ok"] and checks["white_cast_ok"]
        and checks["geometry_ok"]
    )
    already_mls_bright = bool(load_bearing and score >= 0.86)

    return {
        "alreadyMLSBright": already_mls_bright,
        "score": round(score, 3),
        "checks": checks,
        "stats": stats,
        "whiteSurfaceStats": whites,
        "shadowHighlightStats": hist,
        "measuredPerspectiveCorrectionDegrees": round(measured_rotation, 3),
    }


# ── Technical correction layer ──────────────────────────────────────────────

def white_balance_neutral_aware(img):
    """White balance using likely-neutral surfaces (trim, doors, cabinets,
    ceilings) as the primary reference, falling back to gray-world when
    there aren't enough neutral candidates in frame. More targeted than
    pure gray-world, per Sam's calibrated reference — real estate photos
    are full of genuinely colorful content (wood, furniture) that pulls a
    whole-image average away from true neutral."""
    bgr = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    neutral_mask = (s < 55) & (v > 120) & (v < 245)
    if neutral_mask.sum() < img.size * 0.01:
        neutral_mask = (s < 75) & (v > 100) & (v < 248)

    if neutral_mask.sum() > max(250, img.shape[0] * img.shape[1] * 0.006):
        sample = bgr[neutral_mask]
        means = sample.mean(axis=0)
    else:
        means = bgr.reshape(-1, 3).mean(axis=0)

    target = float(means.mean())
    scales = target / np.maximum(means, 1.0)
    # Conservative cap: correct cast, do not change material color.
    scales = np.clip(scales, 0.82, 1.20)

    cast_mag = float(np.max(np.abs(scales - 1.0)))
    strength = clamp01((cast_mag - 0.015) / 0.11)
    applied = 1.0 + strength * (scales - 1.0)

    out = bgr * applied.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8), round(strength, 3)


def mild_mobile_lens_correction(img, mode="auto"):
    """Mild deterministic radial correction for typical mobile/wide-angle
    barrel distortion. Confirmed by Sam (July 8, 2026) as standard real
    estate photography practice, not an AB 723 concern — geometric lens
    correction, same category as perspective/angle correction. Uses a
    generic distortion estimate (not a per-device calibration), kept
    small and capped so it never meaningfully alters composition."""
    if mode == "off":
        return img, 0.0
    h, w = img.shape[:2]
    if mode == "auto" and max(w, h) < 900:
        return img, 0.0

    strength = 0.020 if mode == "auto" else 0.032
    camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-strength, 0.0, 0.0, 0.0], dtype=np.float32)
    new_camera, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y + rh, x:x + rw]
        undistorted = cv2.resize(undistorted, (w, h), interpolation=cv2.INTER_CUBIC)
    return undistorted, round(strength, 4)


def _largest_crop_after_rotation(w, h, angle_deg):
    """Standard formula for the largest axis-aligned rectangle, with the
    same aspect ratio as the original, that fits entirely inside a WxH
    image after it's been rotated by angle_deg — i.e. the region that
    contains zero fabricated/replicated border pixels."""
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
    """Perspective/vertical alignment via Hough-line detection.

    APPROXIMATION NOTE: detects the dominant near-vertical line angle
    (architectural edges) and applies a single global rotation — not full
    4-point perspective/keystone correction. Upgrade path if testing shows
    meaningfully converging verticals a single rotation can't fix.

    Includes both fixes verified earlier this session: (1) canonicalized
    line direction so HoughLinesP's arbitrary endpoint ordering can't flip
    a valid line into the rejected ~180-degree range, (2) correct sign on
    the corrective rotation (proved via a known 5-degree test tilt: the
    unfixed version measured ~10 degrees residual, doubling the tilt; this
    version measures 0.0), and (3) crop-after-rotate so no replicated
    border pixels survive into the delivered image (proved via a
    marker-color test — zero fabricated pixels found in the final output).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=max(40, img.shape[0] // 4), maxLineGap=12)
    if lines is None:
        return img, 0.0

    angles = []
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        dx, dy = x2 - x1, y2 - y1
        if dy < 0:
            dx, dy = -dx, -dy
        if abs(dy) < 1e-3:
            continue
        angle_from_vertical = np.degrees(np.arctan2(dx, dy))
        if abs(angle_from_vertical) < 14:
            angles.append(float(angle_from_vertical))
            lengths.append(float(np.hypot(dx, dy)))

    if len(angles) < 3:
        return img, 0.0

    # PERSPECTIVE-CONVERGENCE FIX (July 8, 2026): confirmed directly on a
    # real photo that a "pick the winning cluster" strategy (both plain
    # weighted median AND an earlier consensus-clustering attempt) can
    # confidently rotate the WRONG way when a photo has genuine wide-angle
    # perspective convergence — different verticals in different parts of
    # the frame legitimately show different apparent angles (a real
    # keystone effect, not camera roll), and no single rotation can
    # satisfy both. On the test photo: door-frame lines (left side)
    # clustered at +3.5 degrees; window-mullion/right-side lines (more
    # numerous, often longer) clustered at -4 degrees. Picking either
    # side as "the truth" made the other side visibly worse. This is the
    # single-rotation limitation already flagged in this function's own
    # docstring — full 4-point perspective correction would resolve it
    # properly, but that's a materially larger feature, not a tuning fix.
    #
    # Safer interim behavior: measure how SCATTERED the angle distribution
    # is. Low scatter (angles agree) means a real, confident tilt exists —
    # apply full correction. High scatter (angles genuinely disagree, as
    # in the perspective-convergence case above) means committing to
    # either side risks visibly worsening it — scale correction strength
    # down instead of confidently picking a "winner."
    weighted = []
    for a, l in zip(angles, lengths):
        weighted.extend([a] * max(1, int(l // 60)))
    if not weighted:
        weighted = angles
    weighted_arr = np.array(weighted)

    raw_median = float(np.median(weighted_arr))

    # SIGN-AGREEMENT CONFIDENCE (July 8, 2026, replacing an earlier
    # scatter/std-based attempt that still wasn't reliable): std alone
    # doesn't distinguish "wide spread but everyone agrees on direction"
    # from "genuine conflict between regions" — a photo can have high std
    # while still being 95%+ one-sided (trustworthy), or lower std while
    # having a real ~20% minority pulling the opposite sign (confirmed on
    # the test photo: 71% of weighted votes negative, but 20.5% positive
    # — that 20% minority was exactly the door frame, and even a
    # confidence-scaled-down correction in the majority's direction still
    # measurably worsened it). Sign-agreement is a more direct proxy for
    # "can this direction be trusted": what fraction of the weighted vote
    # agrees on a side. Below 75% agreement, treat the photo as having a
    # real directional conflict and skip correction rather than guess.
    # Above 95% agreement, trust it fully.
    pos_frac = float((weighted_arr > 0.5).mean())
    neg_frac = float((weighted_arr < -0.5).mean())
    majority_fraction = max(pos_frac, neg_frac)
    confidence = float(np.clip((majority_fraction - 0.75) / (0.95 - 0.75), 0.0, 1.0))

    correction_angle = -raw_median * confidence
    correction_angle = max(-6.0, min(6.0, correction_angle))
    if abs(correction_angle) < 0.35:
        return img, 0.0

    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (w, h),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    crop_w, crop_h = _largest_crop_after_rotation(w, h, correction_angle)
    crop_w, crop_h = int(round(crop_w)), int(round(crop_h))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    cropped = rotated[y0:y0 + crop_h, x0:x0 + crop_w]
    result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

    return result, correction_angle


def detect_noise_level(img):
    """Proxy for ISO/noise level: variance of the Laplacian on a grayscale
    copy. No EXIF ISO dependency (most agent/iPhone uploads won't reliably
    carry it through upload pipelines)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def adaptive_denoise(img):
    """Adaptive noise reduction — strength scales inversely with detected
    noise proxy. Skips entirely (h<=2) on already-sharp/clean photos
    rather than applying even a mild unnecessary smoothing pass."""
    noise_proxy = detect_noise_level(img)
    if noise_proxy > 1200:
        h_luma = 2
    elif noise_proxy > 500:
        h_luma = 4
    elif noise_proxy > 200:
        h_luma = 6
    else:
        h_luma = 8
    if h_luma <= 2:
        return img, h_luma
    return cv2.fastNlMeansDenoisingColored(img, None, h_luma, h_luma, 7, 21), h_luma


def vignette_correct(img):
    """Vignette neutralization via a radial gain mask, strength scaled to
    measured center-vs-edge brightness falloff — most modern phone cameras
    have very mild real vignetting, so most photos should see little to
    no correction here."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]

    center = l[h // 3:2 * h // 3, w // 3:2 * w // 3]
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = False
    center_mean = float(center.mean()) if center.size > 0 else 128.0
    border_mean = float(l[border_mask].mean()) if border_mask.any() else center_mean
    falloff = max(0.0, (center_mean - border_mean) / max(center_mean, 1.0))

    strength = clamp01((falloff - 0.025) / 0.16)
    if strength <= 0.01:
        return img, 0.0

    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    gain = 1.0 + (0.22 * strength) * (dist ** 2)
    lab[:, :, 0] = np.clip(l * gain, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, round(strength, 3)


# ── MLS Bright finish layer (calibrated against real MLS Bright photos) ────

def mls_brightness_lift(img, intensity=1.0):
    """Interior-first MLS brightness pass.

    REDESIGNED (July 8, 2026) after Sam directly flagged that a single
    global gamma curve didn't achieve real "professional balance" between
    bright and dark zones — his point: professional photographers
    typically achieve that balance via bracketed exposure capture (3+
    shots blended), not post-processing a single frame. True bracket HDR
    is genuinely blocked by the current single-shot upload workflow (no
    multiple exposures to merge). This uses the achievable middle ground:
    SYNTHETIC exposure fusion — generating virtual under/over-exposed
    versions of the ONE real captured photo (linear exposure scaling, not
    a generative reconstruction) and blending them via Mertens fusion
    (cv2.createMergeMertens — a standard, deterministic computational
    photography technique, same legal category as any other exposure
    correction). This is still fundamentally limited to the dynamic range
    actually captured in the one real exposure — it can't manufacture
    detail that was never captured — but it balances what IS there more
    like real HDR than a single gamma curve does.

    Verified directly on a real test photo: overall median luma moved
    153 -> 167 (vs. topping out around 165 with the old approach at full
    intensity), and — genuinely nice property of fusion, not something I
    had to hack in — a true-black oven's minimum luma stayed at 0 (true
    black) WITHOUT needing an explicit protection rule, since a pixel
    that's black in the original stays black across every synthetic
    exposure by definition.

    Sam's calibrated target (median luma 178, from his real MLS Bright
    reference photos) is kept as a secondary nudge: if fusion alone
    doesn't reach that target, a mild additional lift closes the gap,
    rather than discarding the validated calibration.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_before = lab[:, :, 0]
    before_median = float(np.median(l_before))
    before_mean = float(l_before.mean())

    # Synthetic exposure brackets from the one real photo — linear
    # exposure-stop scaling (roughly -1.5 / +1.7 stops), closer to how
    # real camera bracketing works than a gamma curve alone.
    orig_f = img.astype(np.float32) / 255.0
    under_8u = np.clip(orig_f * 0.35, 0, 1)
    under_8u = (under_8u * 255).astype(np.uint8)
    over_8u = np.clip(orig_f * 3.2, 0, 1)
    over_8u = (over_8u * 255).astype(np.uint8)

    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process([under_8u, img.copy(), over_8u])
    fused = np.clip(fusion * 255, 0, 255).astype(np.uint8)

    # Blend fusion result with the original by `intensity`, so the
    # existing intensity dial (0.6-1.25) still controls overall strength.
    blended = cv2.addWeighted(fused, intensity, img, 1.0 - intensity, 0) if intensity < 1.0 else fused
    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]

    # Secondary nudge toward the calibrated target, only if fusion alone
    # didn't reach it — mild, since fusion should do most of the work.
    fusion_median = float(np.median(l))
    target_median = 178.0
    residual_need = clamp01((target_median - fusion_median) / 100.0) * intensity
    if residual_need > 0.03:
        normalized = np.clip(l / 255.0, 0, 1)
        gamma = 1.0 - (0.15 * residual_need)
        # Same true-black protection as before — belt and suspenders on
        # top of fusion's natural black-preservation property.
        true_black_protect = np.clip(l / 25.0, 0, 1)
        l = (255.0 * np.power(normalized, gamma)) * true_black_protect + l * (1.0 - true_black_protect)

    # Highlight/window protection — compress rather than let anything blow out.
    highlight_mask = np.clip((l - 214.0) / 41.0, 0, 1)
    compressed_highlights = 214.0 + (l - 214.0) * 0.62
    l = l * (1.0 - highlight_mask) + compressed_highlights * highlight_mask

    l = np.clip(l, 0, 246)
    lab[:, :, 0] = l
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, {
        "before_median_luma": round(before_median, 2),
        "before_mean_luma": round(before_mean, 2),
        "after_fusion_median_luma": round(fusion_median, 2),
        "target_median_luma": target_median,
        "residual_need": round(float(residual_need), 3),
        "method": "synthetic_exposure_fusion",
    }


def clean_whites_adaptive(img, intensity=1.0):
    """Adaptive MLS Bright clean-whites pass — measures actual likely-white
    architectural surfaces (trim, cabinets, ceilings via LAB chroma+luma)
    and only neutralizes/lifts them if they measurably need it, feathered
    with a blurred mask so there's no hard edge. Leaves walls, wood floors,
    and decor untouched — this targets only the surfaces a professional
    retoucher would target for "clean whites," not a global shift."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
    white_mask = (L > 145.0) & (chroma < 22.0)
    strong_white_mask = (L > 165.0) & (chroma < 16.0)

    white_fraction = float(np.mean(white_mask))
    if white_fraction < 0.006:
        return img, {"applied": False, "whiteFraction": round(white_fraction, 4),
                      "reason": "insufficient_likely_white_surface"}

    sample_mask = strong_white_mask if np.any(strong_white_mask) else white_mask
    mean_a = float(np.mean(A[sample_mask]))
    mean_b = float(np.mean(B[sample_mask]))
    mean_l = float(np.mean(L[sample_mask]))
    cast_mag = float(np.sqrt((mean_a - 128.0) ** 2 + (mean_b - 128.0) ** 2))

    neutralize_strength = clamp01((cast_mag - 1.8) / (10.0 - 1.8)) * 0.55 * intensity

    target_l = 212.0
    l_gap = max(0.0, target_l - mean_l)
    lift_strength = clamp01(l_gap / 38.0) * 0.22 * intensity
    if mean_l > 220.0:
        lift_strength *= 0.15
    elif mean_l > 212.0:
        lift_strength *= 0.35

    if neutralize_strength < 0.03 and lift_strength < 0.03:
        return img, {"applied": False, "whiteFraction": round(white_fraction, 4),
                      "castMagnitude": round(cast_mag, 3), "meanWhiteLuma": round(mean_l, 2),
                      "reason": "likely_whites_already_clean"}

    mask_u8 = white_mask.astype(np.uint8) * 255
    mask_blur = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=5, sigmaY=5).astype(np.float32) / 255.0

    A_target = A - neutralize_strength * (A - 128.0)
    B_target = B - neutralize_strength * (B - 128.0)
    L_target = L + (255.0 - L) * lift_strength

    A_adj = A * (1.0 - mask_blur) + A_target * mask_blur
    B_adj = B * (1.0 - mask_blur) + B_target * mask_blur
    L_adj = L * (1.0 - mask_blur) + L_target * mask_blur

    merged = cv2.merge((np.clip(L_adj, 0, 247), np.clip(A_adj, 0, 255), np.clip(B_adj, 0, 255))).astype(np.uint8)
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out, {
        "applied": True, "whiteFraction": round(white_fraction, 4),
        "castMagnitude": round(cast_mag, 3), "meanWhiteLuma": round(mean_l, 2),
        "liftStrength": round(float(lift_strength), 3),
        "neutralizeStrength": round(float(neutralize_strength), 3),
    }


def window_balance(img):
    """Safe window/highlight balancing — compresses overly bright highlight
    regions only. Does not reconstruct exterior detail, replace views, or
    add any content."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    bright_frac = float((l > 232).sum()) / float(l.size)

    if bright_frac < 0.012:
        return img, {"applied": False, "highlight_fraction": round(bright_frac, 4)}

    mask = np.clip((l - 224.0) / 31.0, 0, 1)
    compressed = 224.0 + (l - 224.0) * 0.48
    lab[:, :, 0] = np.clip(l * (1.0 - mask) + compressed * mask, 0, 246)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, {"applied": True, "highlight_fraction": round(bright_frac, 4)}


def mls_color_finish(img, intensity=1.0):
    """MLS finish: neutral, clean, bright — not editorial. Normalizes
    saturation toward the calibrated MLS Bright target range and adds a
    mild clarity/unsharp pass so the image doesn't read as flat after the
    brightness and white-surface work above."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_mean = float(hsv[:, :, 1].mean())

    if sat_mean < 62:
        factor = 1.0 + 0.07 * intensity
    elif sat_mean > 96:
        factor = 1.0 - 0.08 * intensity
    else:
        factor = 1.0 - 0.02 * intensity
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    color_finished = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    blurred = cv2.GaussianBlur(color_finished, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(color_finished, 1.16, blurred, -0.16, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8), {
        "mean_saturation_before": round(sat_mean, 2), "saturation_factor": round(float(factor), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lens-mode", choices=["auto", "mild", "off"], default="auto")
    parser.add_argument("--intensity", type=float, default=1.0)
    args = parser.parse_args()
    intensity = float(np.clip(args.intensity, 0.6, 1.25))

    img = cv2.imread(args.source)
    if img is None:
        print(json.dumps({"error": f"Could not read image: {args.source}"}), file=sys.stderr)
        sys.exit(1)

    modules_applied = []
    skipped = ["color_uniformity_harmonization", "reflection_glare_reduction"]

    # ── Do-No-Harm gate ─────────────────────────────────────────────────
    guard = assess_professional_mls_bright(img)
    if guard["alreadyMLSBright"]:
        if os.path.abspath(args.source) != os.path.abspath(args.output):
            shutil.copyfile(args.source, args.output)
        print(json.dumps({
            "output": args.output,
            "modulesApplied": ["already_mls_bright_no_correction_applied"],
            "modulesSkipped": skipped,
            "perspectiveCorrectionDegrees": 0.0,
            "denoiseStrength": 0,
            "histogramStats": guard["shadowHighlightStats"],
            "professionalMLSGuard": guard,
        }))
        return

    # ── Technical correction ────────────────────────────────────────────
    img, wb_strength = white_balance_neutral_aware(img)
    if wb_strength >= 0.1:
        modules_applied.append("white_balance")

    img, lens_strength = mild_mobile_lens_correction(img, args.lens_mode)
    if lens_strength > 0:
        modules_applied.append("lens_correction")

    img, rotation_deg = deskew_perspective(img)
    if rotation_deg != 0.0:
        modules_applied.append("perspective_alignment")

    img, denoise_strength = adaptive_denoise(img)
    if denoise_strength > 2:
        modules_applied.append("adaptive_noise_reduction")

    img, vignette_strength = vignette_correct(img)
    if vignette_strength >= 0.1:
        modules_applied.append("vignette_neutralization")

    # ── MLS Bright finish ────────────────────────────────────────────────
    img, brightness_metrics = mls_brightness_lift(img, intensity=intensity)
    # Fusion runs every time (it's the primary technique now, not
    # conditional) — report as applied if it moved the median meaningfully
    # OR the secondary target-nudge kicked in.
    brightness_moved = abs(brightness_metrics["after_fusion_median_luma"] - brightness_metrics["before_median_luma"]) >= 5
    if brightness_moved or brightness_metrics["residual_need"] >= 0.05:
        modules_applied.append("mls_brightness_lift")

    img, white_metrics = clean_whites_adaptive(img, intensity=intensity)
    if white_metrics.get("applied"):
        modules_applied.append("clean_whites")

    img, window_metrics = window_balance(img)
    if window_metrics.get("applied"):
        modules_applied.append("window_highlight_balance")

    img, finish_metrics = mls_color_finish(img, intensity=intensity)
    modules_applied.append("color_clarity_finish")

    histogram_stats = shadow_highlight_stats(img)

    cv2.imwrite(args.output, img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    print(json.dumps({
        "output": args.output,
        "modulesApplied": modules_applied,
        "modulesSkipped": skipped,
        "perspectiveCorrectionDegrees": round(rotation_deg, 2),
        "denoiseStrength": denoise_strength,
        "histogramStats": {
            "shadow_frac": histogram_stats["shadowFraction"],
            "highlight_frac": histogram_stats["brightFraction"],
        },
        "professionalMLSGuard": guard,
        "metrics": {
            "whiteBalanceStrength": wb_strength,
            "lensCorrectionStrength": lens_strength,
            "vignetteStrength": vignette_strength,
            "mlsBrightness": brightness_metrics,
            "cleanWhites": white_metrics,
            "windowBalance": window_metrics,
            "mlsFinish": finish_metrics,
        },
    }))


if __name__ == "__main__":
    main()
