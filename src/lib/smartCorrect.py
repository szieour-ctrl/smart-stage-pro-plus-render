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

from hdrRecover import recover_hdr_if_present, looks_like_heic

# Kill switch, matching the existing END_FRAME_ENABLED pattern in this
# codebase — lets HDR recovery be disabled instantly via Railway env var
# without a redeploy, in case something unexpected shows up on real
# customer photos this investigation's test set didn't cover.
HDR_RECOVERY_ENABLED = os.environ.get("HDR_RECOVERY_ENABLED", "true").lower() not in ("false", "0", "")


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


def is_exterior_daylight(img):
    """Detect backyard/patio/pool-style exterior daylight photos so the
    interior-calibrated MLS Bright stack (target median luma 178,
    clean-whites neutralization, window highlight compression) doesn't
    get applied to them — those assumptions are all built for rooms.
    Confirmed directly on a real photo (IMG_8311, pool/patio): running
    the full interior pipeline pushed a deliberately-shaded foreground
    concrete slab toward the same brightness as sunlit concrete, with
    visible tonal banding in the shadow as a side effect.

    PENDING VALIDATION — tuned against a single confirmed exterior photo
    plus seven confirmed interior photos (including several with windows
    showing sky/trees, to check for false positives on those). Vegetation
    coverage was the most reliable signal: 0.164 on the real exterior
    photo vs. 0.0-0.014 across every interior photo tested, including
    window-heavy ones. A close-up patio/pool shot with little visible
    plant life (rare in practice, but possible) would not be caught by
    this heuristic — worth widening the test set before leaning on this
    further."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    green_mask = (H > 33) & (H < 85) & (S > 40) & (V > 40)
    green_frac = float(green_mask.mean())
    blue_sky_mask = (H > 95) & (H < 135) & (S > 45) & (V > 120)
    sky_frac = float(blue_sky_mask.mean())
    return bool(green_frac > 0.06 or (green_frac > 0.03 and sky_frac > 0.05)), {
        "greenFraction": round(green_frac, 3), "skyFraction": round(sky_frac, 3),
    }


def exterior_daylight_correction(img, intensity=1.0):
    """Restrained correction path for exterior daylight photos — per
    Sam's review of a real over-corrected pool/patio photo, exteriors
    should get a small shadow lift and nothing resembling the interior
    MLS Bright target (median luma 178 is appropriate for a room, not
    for deliberately-shaded concrete in daylight). Mild gamma lift only,
    capped low.

    NO PRE-BLUR (patch, pending validation): an earlier version of this
    function pre-blurred shadow regions to guard against banding on flat
    concrete. Confirmed directly on a real photo (IMG_8311) that this
    created a visible light streak instead — a pool-fence mesh panel has
    a fine diagonal lattice pattern, and shadow-weighted blur smooths
    that unevenly (dark mesh lines protected less than the lighter gaps
    between them), which reads as an accentuated streak following the
    lattice. Measured directly: max localized luma spike in that region
    dropped from +24 to +5 once the pre-blur was removed, while the
    concrete's own banding metric moved only marginally (4.435 -> 4.663,
    essentially noise around the original photo's own 4.581) — nowhere
    near the 5.394 the old interior-pipeline bug produced. Tried a
    texture-aware version that only pre-blurred genuinely flat regions;
    it didn't discriminate reliably, since the concrete itself has enough
    natural aggregate texture to trip the same threshold as the mesh.
    Removed rather than ship unproven complexity."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    before_median = float(np.median(l))
    l_pre = l

    normalized = np.clip(l_pre / 255.0, 0, 1)
    gamma = 1.0 - (0.05 * intensity)  # deliberately mild — no 178 target here
    lifted = 255.0 * np.power(normalized, gamma)

    # Keep genuine deep shadow mostly as-is (that's real shade, not a
    # defect) and don't touch bright highlights (sky/sunlit areas).
    mid_mask = np.clip((l_pre - 30.0) / 60.0, 0, 1) * np.clip((200.0 - l_pre) / 60.0, 0, 1)
    l_out = l_pre * (1.0 - mid_mask) + lifted * mid_mask

    lab[:, :, 0] = np.clip(l_out, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, {
        "before_median_luma": round(before_median, 2),
        "after_median_luma": round(float(np.median(l_out)), 2),
        "method": "exterior_mild_shadow_lift",
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

    # WARM-CAST DAMPENING (patch, pending validation): a "neutral" mask
    # built from HSV saturation/value alone can't tell a genuine sensor
    # color cast from a warmly-lit room's actual color — cream/tan walls
    # and trim under tungsten light legitimately average out R > G > B,
    # which reads identically to a cast. Confirmed directly on a real
    # photo (IMG_8198): the neutral-mask pixels measured LAB b +5.7 above
    # neutral (mean B channel 133.7 vs 128 neutral point) — a textbook
    # warm-light signature, not a color-cast signature. Correcting that
    # out at meaningful strength strips the room's real, intentional
    # warmth. Fix: dampen strength heavily when the dominant channel
    # needing correction is the classic warm/tungsten signature (R
    # channel highest of the three). Leave other cast types (green cast
    # from fluorescents, magenta cast from some LEDs/sensor artifacts)
    # corrected at full strength, since those are not a normal "warm
    # room" signature.
    is_warm_cast = means[2] > means[0]  # BGR order: means[2]=R, means[0]=B
    if is_warm_cast:
        strength *= 0.35

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
    """Proxy for ISO/noise level.

    FIXED (July 9, 2026, corrected on the second attempt): the original
    version used a single GLOBAL Laplacian variance, which can't
    distinguish "genuinely noisy/blurry photo" from "photo with naturally
    smooth, low-texture content" (confirmed directly on a real marble
    bathroom photo — global variance measured 79.8, triggering the
    strongest denoise tier, which smoothed away real marble veining/detail
    into a flat, hazy look). A first fix attempt using cv2.blur() on the
    squared Laplacian was ALSO wrong — box-blurring dilutes genuinely sharp
    edge spikes (a single grout line gets averaged against ~600 smooth
    neighboring pixels), making the signal WORSE, not better (measured:
    51.3, even lower than the original 79.8). Correct approach: compute
    variance WITHIN actual small tiles independently (not a blurred
    average), then take a high percentile across tiles — this correctly
    lets a few genuinely sharp regions (grout lines, fixture edges) signal
    "this photo is in focus," even when most tiles are naturally smooth.
    Verified directly: proper per-tile variance on the bathroom photo
    showed a 95th-percentile tile variance of ~458 — well into "clean,
    doesn't need strong denoise" — versus the flawed global/blurred
    approaches which both stayed under 100."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = lap.shape
    tile = 40
    variances = []
    for y in range(0, max(1, h - tile), tile):
        for x in range(0, max(1, w - tile), tile):
            block = lap[y:y + tile, x:x + tile]
            if block.size > 0:
                variances.append(block.var())
    if not variances:
        return float(lap.var())
    return float(np.percentile(variances, 95))


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

    # LUMINANCE-ONLY FUSION FIX (July 9, 2026): confirmed directly on a
    # real bathroom photo that full-RGB Mertens fusion introduces visible
    # blotchy color artifacts on large uniform surfaces (a plain wall) —
    # local chroma variance jumped ~4.5x (max local variance 38.9 -> 168.7)
    # from the fusion step alone, before any other correction ran. Cause:
    # Mertens fusion assigns per-pixel weights based on local
    # well-exposedness/contrast/saturation, and on a subtly-textured
    # uniform surface, neighboring pixels can get slightly different
    # weights — which shows up as visible color mottling, not just
    # brightness variation. Fix: apply the fusion result's LUMINANCE only,
    # keep the original photo's actual color/chroma channels untouched.
    # Verified: wall chroma variance dropped back to near-original levels
    # (41.3 vs 38.9 baseline) while the brightness benefit was fully
    # preserved (identical median luma, 161, with or without this fix).
    lab_fused = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB)
    lab_orig_full = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_hybrid = lab_orig_full.copy()

    # DARK-MATERIAL FUSION PROTECTION (patch, pending validation): Mertens
    # fusion, by construction, treats every dark pixel as underexposed
    # and brightens it via the synthetic over-exposure bracket (orig_f *
    # 3.2) — it has no way to distinguish a genuinely underexposed shadow
    # from a pixel that's dark because the actual material there (dark
    # leather, black granite, dark wood) is supposed to read dark. This
    # is the PRIMARY source of the crush, not the secondary residual-nudge
    # gamma step below — confirmed directly on real photos: a black
    # granite fireplace surround (IMG_8317, orig mean L 85.2) and a dark
    # brown leather couch (IMG_8198, orig mean L 77.5) both got lifted
    # substantially by the fusion step alone, before any other correction
    # ran or the residual nudge fired. Fix: pull the fused luma back
    # toward the ORIGINAL (pre-fusion) luma for pixels that started dark,
    # tapering out by L=140 so genuine underexposed shadow/hallway detail
    # in that upper range still gets the fusion benefit in full.
    orig_l = lab_orig_full[:, :, 0].astype(np.float32)
    fused_l = lab_fused[:, :, 0].astype(np.float32)
    dark_protect = 1.0 - np.clip((orig_l - 40.0) / 100.0, 0.0, 1.0)  # 1.0 at L<=40, 0 by L>=140
    protected_l = fused_l * (1.0 - dark_protect * 0.6) + orig_l * (dark_protect * 0.6)
    lab_hybrid[:, :, 0] = protected_l
    fused = cv2.cvtColor(lab_hybrid, cv2.COLOR_LAB2BGR)

    # Blend fusion result with the original by `intensity`, so the
    # existing intensity dial (0.6-1.25) still controls overall strength.
    blended = cv2.addWeighted(fused, intensity, img, 1.0 - intensity, 0) if intensity < 1.0 else fused
    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]

    # Secondary nudge toward the calibrated target, only if fusion alone
    # didn't reach it — mild, since fusion should do most of the work.
    fusion_median = float(np.median(l))
    target_median = 168.0
    residual_need = clamp01((target_median - fusion_median) / 100.0) * intensity
    if residual_need > 0.03:
        normalized = np.clip(l / 255.0, 0, 1)
        gamma = 1.0 - (0.15 * residual_need)
        # DARK-MATERIAL PROTECTION (patch, pending validation): the old
        # floor here (l / 25.0) only shields true-black pixels (L<25) —
        # anything darker than that from crushing further, but it does
        # nothing for dark furniture/fixtures that sit well above true
        # black. Confirmed directly on a real photo (IMG_8317): dark
        # brown leather and a black granite fireplace surround measured
        # in the L 30-90 range, fully exposed to the gamma lift meant for
        # underexposed shadows — same treatment as a dim hallway, when
        # these are supposed to read as rich, dark material, not a
        # lighting defect. Widened ramp so protection tapers out to L=100
        # instead of L=25, so these materials keep most of their depth
        # while genuine near-black shadow detail still gets lifted.
        dark_material_protect = np.clip(l / 100.0, 0, 1)
        l = (255.0 * np.power(normalized, gamma)) * dark_material_protect + l * (1.0 - dark_material_protect)

    # Highlight/window protection — compress rather than let anything blow out.
    # WIDENED CEILING (patch, pending validation): confirmed on real photos
    # that this compression stacks with window_balance()'s own compression
    # immediately downstream, and together they hard-cap every corrected
    # photo's brightest pixels around L 226-233 — even when the original
    # had legitimate bright content (window light, white trim, lamps)
    # above 245. Measured directly: IMG_8198 had 1.65% of pixels >245 in
    # the original, 0% after correction; IMG_8317 the same (0.58% -> 0%).
    # Narrowed this stage's compression to only engage for genuinely
    # near-blown highlights (L>230 instead of L>214) and softened the
    # ratio so more real highlight detail survives into window_balance().
    highlight_mask = np.clip((l - 230.0) / 25.0, 0, 1)
    compressed_highlights = 230.0 + (l - 230.0) * 0.82
    l = l * (1.0 - highlight_mask) + compressed_highlights * highlight_mask

    l = np.clip(l, 0, 252)
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

    # MAJORITY-WHITE-ROOM FIX (July 9, 2026): this function was designed
    # and tested against kitchens, where white cabinets/trim are a
    # distinguishable MINORITY of the frame against colorful counters,
    # floors, and walls. Confirmed directly on a real marble bathroom
    # photo that this assumption breaks down completely in predominantly-
    # white rooms: whiteFraction measured 80%+ of the ENTIRE frame, so
    # this function was applying its lift/neutralize to almost the whole
    # photo — not a targeted surface correction anymore, just a second
    # (redundant, compounding) global brightness/color pass stacked on
    # top of mls_brightness_lift, vignette, and white balance, which were
    # ALSO already touching the same dominant white content. That
    # stacking is what produced a visibly blown-out, hazy result. Fix:
    # scale strength down as white_fraction grows past what's plausible
    # for "a minority of trim/cabinets" — full strength at <=35% white
    # coverage (a normal amount of trim in a typical room), tapering to
    # near-zero by 65%+ coverage (the room IS predominantly white by
    # design — global corrections already handle it, a second targeted
    # pass on top is redundant and risks exactly this compounding).
    majority_room_factor = clamp01((0.75 - white_fraction) / (0.75 - 0.50))

    # REDUCED CEILING (patch, pending validation): report flagged a real
    # photo (IMG_8301, hallway) as over-lifted and flattened even with
    # adaptive intensity at max (a genuinely dim hallway correctly gets
    # full intensity — the issue is this pass's own ceiling being too
    # strong on top of that, not the do-no-harm gate). Lowered both
    # multipliers modestly (0.55->0.46, 0.22->0.17) so already-white
    # surfaces get corrected without pushing as hard toward flat/bright.
    neutralize_strength = clamp01((cast_mag - 1.8) / (10.0 - 1.8)) * 0.46 * intensity * majority_room_factor

    target_l = 212.0
    l_gap = max(0.0, target_l - mean_l)
    lift_strength = clamp01(l_gap / 38.0) * 0.17 * intensity * majority_room_factor
    if mean_l > 220.0:
        lift_strength *= 0.15
    elif mean_l > 212.0:
        lift_strength *= 0.35

    if neutralize_strength < 0.03 and lift_strength < 0.03:
        reason = "majority_white_room_scaled_down" if white_fraction > 0.55 else "likely_whites_already_clean"
        return img, {"applied": False, "whiteFraction": round(white_fraction, 4),
                      "castMagnitude": round(cast_mag, 3), "meanWhiteLuma": round(mean_l, 2),
                      "majorityRoomFactor": round(majority_room_factor, 3), "reason": reason}

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
    bright_frac = float((l > 240).sum()) / float(l.size)

    # RAISED TRIGGER + SOFTENED RATIO (patch, pending validation): this
    # function runs immediately after mls_brightness_lift(), which already
    # compresses highlights on its own. Confirmed on real photos that the
    # two stages stack, hard-capping corrected output around L 226-233
    # even when the source had legitimate bright content (window light,
    # trim, lamps) well above 245. Old trigger (bright_frac >= 0.012, i.e.
    # just 1.2% of the frame) fired on nearly any photo with a window or
    # lamp in it. Raised the luma threshold and required fraction so this
    # only engages for genuinely overexposed highlight regions, and
    # softened the compression ratio so what does trigger doesn't crush
    # highlight detail mls_brightness_lift already preserved.
    if bright_frac < 0.02:
        return img, {"applied": False, "highlight_fraction": round(bright_frac, 4)}

    mask = np.clip((l - 238.0) / 17.0, 0, 1)
    compressed = 238.0 + (l - 238.0) * 0.75
    lab[:, :, 0] = np.clip(l * (1.0 - mask) + compressed * mask, 0, 252)
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

    # REDUCED STRENGTH + TEXTURE MASK (patch, pending validation): the old
    # unsharp pass (1.16 / -0.16, full strength everywhere) measurably
    # increased edge energy 26-48% on real photos (Laplacian variance,
    # IMG_8198 1009->1273, IMG_8317 1107->1642) and was the biggest single
    # contributor to new near-black pixels that weren't dark in the
    # original — dark undershoot/halos along trim and furniture edges,
    # applied uniformly even on flat walls and ceilings with nothing to
    # sharpen. Fix: halve the strength, and mask it to only apply where
    # local texture already exists (a gray-level co-occurrence proxy via
    # local variance), so flat surfaces stay flat instead of picking up
    # sharpening-induced grain and edge halos.
    gray = cv2.cvtColor(color_finished, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.blur(gray, (9, 9))
    local_sqmean = cv2.blur(gray * gray, (9, 9))
    local_var = np.clip(local_sqmean - local_mean ** 2, 0, None)
    texture_mask = np.clip(local_var / 60.0, 0, 1)[:, :, None]  # 0 on flat surfaces, 1 on real texture

    sharpened_full = cv2.addWeighted(color_finished, 1.08, blurred, -0.08, 0)
    sharpened = color_finished.astype(np.float32) * (1.0 - texture_mask) + sharpened_full.astype(np.float32) * texture_mask
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

    # ── HDR gain-map recovery (added July 30, 2026) ─────────────────────
    # Attempts to recover real highlight detail from an embedded Apple
    # HDR gain map before any other correction runs, so every downstream
    # step (white balance, brightness lift, etc.) works from the actual
    # recovered image instead of the flattened, gain-map-discarded one
    # cv2.imread() alone would produce. Falls through to plain cv2.imread()
    # unchanged for the vast majority of photos with no gain map present —
    # zero behavior change for those. See hdrRecover.py for the full
    # validation history (ceiling, color, and gamut fixes, each confirmed
    # against real photos before this integration).
    hdr_report = {"gainMapPresent": False, "recoveryApplied": False, "enabled": HDR_RECOVERY_ENABLED}
    img = None
    if HDR_RECOVERY_ENABLED:
        img, hdr_report = recover_hdr_if_present(args.source)
        hdr_report["enabled"] = True

    if img is None:
        try:
            img = cv2.imread(args.source)
            if img is None:
                if looks_like_heic(args.source):
                    print(json.dumps({
                        "error": (
                            "This is a genuine HEIC/HEIF file, which this pipeline cannot decode "
                            "in its current form. Please re-select this photo from your Photos "
                            "library (not a HEIC file transferred by cable/AirDrop) and try again."
                        )
                    }), file=sys.stderr)
                else:
                    print(json.dumps({"error": f"Could not read image: {args.source}"}), file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(json.dumps({"error": f"Could not read image: {args.source} ({e})"}), file=sys.stderr)
            sys.exit(1)

    modules_applied = []
    skipped = ["color_uniformity_harmonization", "reflection_glare_reduction"]

    # ── Exterior daylight scene gate (patch, pending validation) ────────
    # Confirmed directly on a real photo (IMG_8311, pool/patio) that the
    # interior-calibrated MLS Bright stack — target median luma 178,
    # clean-whites neutralization, window highlight compression — was
    # being applied to an exterior daylight shot, pushing a deliberately
    # shaded concrete slab toward sunlit brightness with visible tonal
    # banding as a side effect. See is_exterior_daylight() docstring for
    # detection method and known limitations.
    is_exterior, exterior_signals = is_exterior_daylight(img)
    if is_exterior:
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
        img, exterior_metrics = exterior_daylight_correction(img, intensity=intensity)
        if abs(exterior_metrics["after_median_luma"] - exterior_metrics["before_median_luma"]) >= 3:
            modules_applied.append("exterior_daylight_shadow_lift")
        skipped = skipped + ["mls_brightness_lift", "clean_whites", "window_highlight_balance",
                              "vignette_neutralization"]

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
            "sceneProfile": "exterior_daylight",
            "exteriorSignals": exterior_signals,
            "hdrRecovery": hdr_report,
            "metrics": {
                "whiteBalanceStrength": wb_strength,
                "lensCorrectionStrength": lens_strength,
                "exteriorCorrection": exterior_metrics,
            },
        }))
        return

    # ── Do-No-Harm gate ─────────────────────────────────────────────────
    guard = assess_professional_mls_bright(img)
    if guard["alreadyMLSBright"]:
        # If HDR recovery ran, `img` now holds the recovered image — NOT
        # what's sitting in args.source on disk (that's still the
        # original, gain-map-discarded-on-read file). Copying the raw
        # source bytes here would silently throw away a successful
        # recovery the moment the do-no-harm gate decides no further
        # correction is needed — exactly backwards from the point of this
        # integration. Only take the fast raw-copy path when recovery
        # didn't apply; write the actual (possibly recovered) image
        # otherwise.
        if hdr_report.get("recoveryApplied"):
            cv2.imwrite(args.output, img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        elif os.path.abspath(args.source) != os.path.abspath(args.output):
            shutil.copyfile(args.source, args.output)
        print(json.dumps({
            "output": args.output,
            "modulesApplied": (["hdr_gain_map_recovery"] if hdr_report.get("recoveryApplied") else []) + ["already_mls_bright_no_correction_applied"],
            "modulesSkipped": skipped,
            "perspectiveCorrectionDegrees": 0.0,
            "denoiseStrength": 0,
            "histogramStats": guard["shadowHighlightStats"],
            "professionalMLSGuard": guard,
            "hdrRecovery": hdr_report,
        }))
        return

    # ── Technical correction ────────────────────────────────────────────
    if hdr_report.get("recoveryApplied"):
        modules_applied.append("hdr_gain_map_recovery")

    # ── Adaptive intensity scaling (patch, pending validation) ──────────
    # The do-no-harm gate above was previously binary: anything short of
    # near-perfect (score >= 0.86 AND 5 load-bearing checks) got the SAME
    # full-strength correction as a genuinely badly-lit photo. Confirmed
    # on a real photo (IMG_8198, professional guard score 0.333) that this
    # meant an already-reasonably-exposed room got the full brightness
    # lift / clean-whites / sharpening stack meant for underexposed
    # photos, over-correcting it. Fix: scale intensity continuously by
    # how much EXPOSURE correction is actually needed (median/mean/p95
    # luma, shadow fraction) rather than the full 9-point score, since
    # several of the other checks (white area, saturation) reflect real
    # room content — a colorful or trim-light room isn't a defect — not
    # something that should drive correction strength.
    exposure_checks = ["median_luma_ok", "mean_luma_ok", "p95_luma_ok", "shadow_ok"]
    exposure_need = 1.0 - (sum(1 for k in exposure_checks if guard["checks"][k]) / float(len(exposure_checks)))
    adaptive_intensity = float(np.clip(intensity * (0.35 + 0.65 * exposure_need), 0.35, intensity))

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
    img, brightness_metrics = mls_brightness_lift(img, intensity=adaptive_intensity)
    # Fusion runs every time (it's the primary technique now, not
    # conditional) — report as applied if it moved the median meaningfully
    # OR the secondary target-nudge kicked in.
    brightness_moved = abs(brightness_metrics["after_fusion_median_luma"] - brightness_metrics["before_median_luma"]) >= 5
    if brightness_moved or brightness_metrics["residual_need"] >= 0.05:
        modules_applied.append("mls_brightness_lift")

    img, white_metrics = clean_whites_adaptive(img, intensity=adaptive_intensity)
    if white_metrics.get("applied"):
        modules_applied.append("clean_whites")

    img, window_metrics = window_balance(img)
    if window_metrics.get("applied"):
        modules_applied.append("window_highlight_balance")

    img, finish_metrics = mls_color_finish(img, intensity=adaptive_intensity)
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
        "hdrRecovery": hdr_report,
        "metrics": {
            "whiteBalanceStrength": wb_strength,
            "lensCorrectionStrength": lens_strength,
            "vignetteStrength": vignette_strength,
            "adaptiveIntensity": round(adaptive_intensity, 3),
            "mlsBrightness": brightness_metrics,
            "cleanWhites": white_metrics,
            "windowBalance": window_metrics,
            "mlsFinish": finish_metrics,
        },
    }))


if __name__ == "__main__":
    main()
