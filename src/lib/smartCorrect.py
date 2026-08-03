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

# ── Thread-pool ceiling (added Aug 2026) ────────────────────────────────
# MUST run before `import cv2` / `import numpy as np` below -- OpenBLAS
# reads these env vars once, at library load time, not per-call. Setting
# them after numpy/cv2 are already imported has no effect.
#
# Real crash seen on Railway: OpenBLAS auto-detected 48 cores and tried
# to spawn 48 threads for a SINGLE image's math, inside a container whose
# process/thread ceiling is much lower. correctPipeline.js spawns one
# Python process PER IMAGE, so under concurrent batch processing this
# multiplies -- N images in flight each independently trying to grab up
# to 48 threads -- and blows the limit fast ("pthread_create failed...
# Resource temporarily unavailable").
#
# Fix: force single-threaded BLAS/OMP math per process. The real
# parallelism in this pipeline is ACROSS images (multiple processes),
# not within one image's numpy calls, so this costs negligible per-image
# throughput. setdefault(), not direct assignment, so an explicit Railway
# env var (if someone later wants a different value) still wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import cv2
import numpy as np
import logging

# Without this, logger.info() calls anywhere in the process (e.g.
# level0_scene_classifier.py's "disabled"/"disagreement" messages) are
# silently dropped by Python's default root logger level (WARNING).
# Writes to stderr, matching correctPipeline.js's stderr capture.
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from hdrRecover import recover_hdr_if_present, looks_like_heic, decode_standard
from level2_vision_regions import get_level2_regions
from level2_diagnosis import diagnose as level2_diagnose
from level0_scene_classifier import resolve_scene_type
from level2_qc import qc_check

# Kill switch, matching the existing END_FRAME_ENABLED pattern in this
# codebase — lets HDR recovery be disabled instantly via Railway env var
# without a redeploy, in case something unexpected shows up on real
# customer photos this investigation's test set didn't cover.
HDR_RECOVERY_ENABLED = os.environ.get("HDR_RECOVERY_ENABLED", "true").lower() not in ("false", "0", "")


def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


# ── Regional execution engine (Aug 2, 2026) ─────────────────────────────
# Converts the Vision-produced per-region schema (region/maskId/
# regionType/operation/priority/protections/confidence -- see the
# Retoucher Schema spec) into a per-pixel parameter map any classical CV
# function can consume in place of a single global `intensity` scalar.
#
# This is a GENERALIZATION of a pattern already proven in this file:
# dark_material_mask and exclusion_weight (used by mls_brightness_lift
# and clean_whites_adaptive today) are both already continuous,
# heavily-feathered float masks meaning "how much to hold this back" --
# a single fixed category. This makes that mechanism general: any number
# of named, Vision-directed regions, each able to either boost or
# protect, instead of one hardcoded exclusion category.
#
# Protection is evaluated LAST and always wins at its mask's footprint
# (conflictPolicy.protectOverridesPrimary /
# materialProtectionOverridesGlobalAdjustment from the schema spec),
# feathered at the boundary using the same np.maximum + multiplicative
# blend pattern already proven for dark_material_mask -- NOT a hard
# boolean cutover, which is the exact failure mode that caused the Level
# 2 rug-seam bug and the second hdrRecover.py fix attempt. A region only
# gets to boost if BOTH its confidence clears the bar AND its
# `operation` is one this specific function knows how to execute --
# every function only listens to the operations relevant to what IT
# does, exactly as the schema spec's "operation, not directive" design
# requires.
REGIONAL_CONFIDENCE_FLOOR = 0.6  # below this, a region's instruction is ignored entirely
REGIONAL_UNIVERSAL_PROTECT_OPERATIONS = ("no_change", "hue_protection", "saturation_protection")
REGIONAL_PRIORITY_MULTIPLIER = {"primary": 1.35, "secondary": 1.15}
# Operations meaning "dial this down, not off" -- distinct from `protect`
# (full exclusion from every operation) and from primary/secondary boost.
# A clarity_reduction region still gets normal treatment from every OTHER
# function; only the specific mechanism this operation names is reduced,
# and only partially, matching a human editor's "ease off here" rather
# than "don't touch this at all."
REGIONAL_REDUCTION_OPERATIONS = ("clarity_reduction",)
REGIONAL_REDUCTION_MULTIPLIER = 0.35


def build_regional_strength_map(shape, base_intensity, regions, level2_masks,
                                  supported_operations, exclude_tag=None):
    """Returns a float32 (H,W) array, defaulting to base_intensity
    everywhere (so a photo with no regions, or regions this function
    doesn't act on, behaves EXACTLY as today's scalar-intensity code
    did -- opt-in only, never worse than today).

    shape: (H, W) of the image being corrected.
    base_intensity: today's existing scalar (from adaptive_intensity /
        diagnosis bias) -- the map's floor value.
    regions: list of region dicts from the Vision schema. None or []
        -> returns a uniform base_intensity array, unchanged behavior.
    level2_masks: dict of maskId -> feathered float32 (H,W) mask in
        [0,1], from level2_vision_regions.py. A region whose maskId
        isn't present is skipped, not errored -- a missing mask should
        never crash a correction pass.
    supported_operations: set/tuple of `operation` values this specific
        function should act on (e.g. mls_brightness_lift listens for
        "shadow_recovery"/"exposure_lift" and ignores everything else,
        even in regional mode).
    exclude_tag: this function's specific `protections` tag (e.g.
        "exclude_from_shadow_lift") checked IN ADDITION to the universal
        protect checks below.
    """
    strength = np.full(shape, float(base_intensity), dtype=np.float32)
    if not regions or not level2_masks:
        return strength

    protect_accum = np.zeros(shape, dtype=np.float32)

    for region in regions:
        mask = level2_masks.get(region.get("maskId"))
        if mask is None:
            continue
        confidence = region.get("confidence", 1.0)
        if isinstance(confidence, str):
            confidence = {"high": 0.9, "medium": 0.7, "low": 0.4}.get(confidence, 0.5)
        if confidence < REGIONAL_CONFIDENCE_FLOOR:
            continue

        protections = region.get("protections") or []
        is_universal_protect = (
            region.get("priority") == "protect"
            or region.get("operation") in REGIONAL_UNIVERSAL_PROTECT_OPERATIONS
            or (exclude_tag is not None and exclude_tag in protections)
        )
        if is_universal_protect:
            protect_accum = np.maximum(protect_accum, mask.astype(np.float32))
            continue

        operation = region.get("operation")
        if operation not in supported_operations:
            continue

        m = mask.astype(np.float32)
        if operation in REGIONAL_REDUCTION_OPERATIONS:
            target_value = base_intensity * REGIONAL_REDUCTION_MULTIPLIER
        else:
            target_value = base_intensity * REGIONAL_PRIORITY_MULTIPLIER.get(region.get("priority"), 1.0)
        strength = strength * (1.0 - m) + target_value * m

    # Protection applied last, feathered, always wins -- never a hard cut.
    strength = strength * (1.0 - protect_accum)
    return np.clip(strength, 0.0, 1.5).astype(np.float32)


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


def white_surface_stats(img, exclusion_weight=None):
    """Measure likely-white architectural surfaces (trim, cabinets,
    ceilings) without modifying pixels — used by the do-no-harm gate and
    by clean_whites_adaptive to decide whether/how much to correct.

    exclusion_weight (Level 2, optional): a float32 (H,W) array in [0,1]
    — continuous, not boolean — from the Vision pre-pass, combining
    furniture/floor AND dark-material regions. Multiplied in as a
    continuous "keep" weight rather than a hard AND/subtract: an earlier
    hard-boolean version produced a visible rectangular seam wherever a
    coarse Vision box's edge fell in the middle of a continuous surface
    (confirmed on a real photo — a box that didn't fully cover a large
    rug left a visible corrected/uncorrected line with no photographic
    basis). None = no exclusion (original behavior, e.g. when the gate
    calls this before Vision has run)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
    white_mask = (L > 145.0) & (chroma < 22.0)
    strong_white_mask = (L > 165.0) & (chroma < 16.0)

    white_w = white_mask.astype(np.float32)
    strong_w = strong_white_mask.astype(np.float32)
    if exclusion_weight is not None:
        keep = np.clip(1.0 - exclusion_weight, 0.0, 1.0)
        white_w = white_w * keep
        strong_w = strong_w * keep

    white_fraction = float(white_w.mean())
    strong_fraction = float(strong_w.mean())
    if white_fraction < 0.003:
        return {
            "whiteFraction": round(white_fraction, 4),
            "strongWhiteFraction": round(strong_fraction, 4),
            "meanWhiteLuma": 0.0,
            "whiteCastMagnitude": 99.0,
            "meanWhiteA": 0.0,
            "meanWhiteB": 0.0,
        }

    sample_w = strong_w if strong_w.sum() > 1.0 else white_w
    total_w = float(sample_w.sum())
    mean_l = float((L * sample_w).sum() / total_w)
    mean_a = float((A * sample_w).sum() / total_w)
    mean_b = float((B * sample_w).sum() / total_w)
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

def white_balance_neutral_aware(img, regions=None, level2_masks=None):
    """White balance using likely-neutral surfaces (trim, doors, cabinets,
    ceilings) as the primary reference, falling back to gray-world when
    there aren't enough neutral candidates in frame. More targeted than
    pure gray-world, per Sam's calibrated reference — real estate photos
    are full of genuinely colorful content (wood, furniture) that pulls a
    whole-image average away from true neutral.

    regions / level2_masks (Aug 2, 2026, optional): applies the SAME
    globally-computed cast-correction scales below, per-pixel, instead of
    uniformly across the whole frame -- e.g. a region tagged
    operation="white_balance_adjustment" can get more of the correction
    than the frame's baseline, and a `protect` region gets none.

    NOT HANDLED (deliberately, honestly): operation="mixed_light_balance"
    -- per the schema spec, this means two genuinely different color
    temperatures in one frame need DIFFERENT corrections, not more/less
    of the SAME one. That requires computing separate neutral-reference
    statistics per region, not just gating this one global scale — a
    real, separate capability this pass does not build. A
    mixed_light_balance region is silently ignored here (falls through
    "not in supported_operations"), not faked with a wrong mechanism."""
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

    # ── Regional strength map (Aug 2, 2026) ──────────────────────────────
    # base_intensity=strength (today's single computed scalar) so a photo
    # with no regions behaves EXACTLY as before -- the map's floor value
    # IS today's answer, not a different default.
    h_px, w_px = img.shape[:2]
    strength_map = build_regional_strength_map(
        (h_px, w_px), strength, regions, level2_masks,
        supported_operations=("white_balance_adjustment",),
        exclude_tag="exclude_from_white_balance",
    )

    applied = 1.0 + strength_map[:, :, None] * (scales.reshape(1, 1, 3) - 1.0)

    out = bgr * applied
    return np.clip(out, 0, 255).astype(np.uint8), round(float(strength_map.mean()), 3)


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

def mls_brightness_lift(img, intensity=1.0, dark_material_mask=None, regions=None, level2_masks=None):
    """Interior-first MLS brightness pass.

    dark_material_mask (Level 2, optional): float32 (H,W) array in [0,1]
    — continuous, heavily-feathered soft mask, not boolean — from the
    Vision pre-pass marking regions identified as genuinely dark material
    (black countertops, dark leather, dark wood) rather than shadow. This
    is unioned into the existing luma-threshold protection below, which
    on its own can't tell a medium-dark wood table (L~110, outside the
    L<40 taper-to-L=140 window's strongest protection) from ordinary
    midtone content that should lift normally. None = original
    luma-only behavior, unchanged.

    regions / level2_masks (Aug 2, 2026, optional): the Vision region
    schema and its corresponding masks. When supplied, replaces the
    single scalar `intensity` with a per-pixel strength map built by
    build_regional_strength_map() — e.g. a backlit foreground chair
    (operation="shadow_recovery", priority="primary") gets MORE lift
    than the frame's base intensity, while a region tagged `protect` or
    "exclude_from_shadow_lift" gets none, regardless of the global
    setting. When None (the default), behavior is IDENTICAL to before
    this parameter existed — every call site in this file today doesn't
    pass these, so nothing changes for existing callers.

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

    # ── Regional strength map (Aug 2, 2026) ──────────────────────────────
    # Built once, up front, so it can drive both the fusion blend below
    # AND the residual gamma nudge later — same map, two consumers.
    # Falls back to a uniform `intensity` array when regions aren't
    # supplied, so every downstream line below behaves exactly as before.
    strength_map = build_regional_strength_map(
        l_before.shape, intensity, regions, level2_masks,
        supported_operations=("shadow_recovery", "exposure_lift"),
        exclude_tag="exclude_from_shadow_lift",
    )
    regional_mode = regions is not None and level2_masks is not None

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
    # LEVEL 2: the luma ramp alone gives only partial protection to a
    # dark material sitting mid-range (e.g. L~110 medium-dark wood table
    # gets ~0.3, not enough to stop a visible lift). Where Vision has
    # identified the pixel as a genuine dark material, force full
    # protection (1.0) regardless of where it falls on the luma ramp.
    if dark_material_mask is not None:
        dark_protect = np.maximum(dark_protect, dark_material_mask.astype(np.float32))
    protected_l = fused_l * (1.0 - dark_protect * 0.6) + orig_l * (dark_protect * 0.6)
    lab_hybrid[:, :, 0] = protected_l
    fused = cv2.cvtColor(lab_hybrid, cv2.COLOR_LAB2BGR)

    # Blend fusion result with the original using the strength map,
    # per-pixel, instead of a single scalar. Clipped to [0,1] for THIS
    # blend specifically — fusion is bounded (there's no "more than 100%
    # fused" image to blend toward); a region's strength above 1.0
    # ("primary" emphasis) instead gets more reach in the residual gamma
    # nudge below, which is a separate, effectively-unbounded technique.
    # When strength_map is uniform (non-regional callers), this is
    # arithmetically identical to the old
    # `addWeighted(fused, intensity, img, 1-intensity, 0)` call.
    blend_map = np.clip(strength_map, 0.0, 1.0)[:, :, None]
    blended = (fused.astype(np.float32) * blend_map + img.astype(np.float32) * (1.0 - blend_map))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]

    # Secondary nudge toward the calibrated target, only if fusion alone
    # didn't reach it.
    #
    # REDESIGNED (Aug 2, 2026) after real-photo testing showed the old
    # linear coefficient (gamma = 1 - 0.15*residual_need) had diminishing
    # returns and structurally could not close a large gap: confirmed on
    # two real underexposed photos (IMG_8315, IMG_8301) that even at 6.67x
    # the old coefficient, final median landed ~10 points short of the
    # 168 target. Root cause: a flat linear coefficient on a *bounded*
    # residual_need term (itself capped by the do-no-harm gate's dark
    # photos rarely exceeding ~0.25-0.35) can never generate enough gamma
    # curvature to fully close a 25-30+ point gap in one pass.
    #
    # Fix: SOLVE directly for the gamma that maps fusion_median exactly to
    # target_median, instead of guessing a linear coefficient. This is
    # deliberately decoupled from the diagnosis-driven `intensity` scalar
    # here (though `intensity` still fully governs the fusion BLEND above,
    # where a mixed-light-temperature photo genuinely benefits from a
    # gentler blend to avoid amplifying a real color-temperature clash).
    # Reasoned position, confirmed on real photos: the do-no-harm gate
    # already gates entry to this whole code path, so a photo reaching
    # this point has already been confirmed to genuinely need correction
    # -- there's no over-brightening risk in chasing the true calibrated
    # target here, only in how aggressively the earlier fusion blend gets
    # there. Verified on both real test photos: final median landed
    # exactly on the 168 target, with no measurable posterization
    # (unique-luma-value count and flat-region gradient energy both
    # unchanged or improved vs. the uncorrected original).
    fusion_median = float(np.median(l))
    target_median = 168.0

    # ── WhiteFraction-adjusted effective target (Aug 2, 2026) ────────────
    # Confirmed on a real photo (IMG_8317, original whiteFraction 0.504 --
    # cream walls, white mantle/ceiling, nearly half the frame already
    # light-colored) that forcing the SAME universal 168 target used for
    # genuinely dark rooms produces something Sam flagged as "bordering
    # on too much white" -- and measurement confirmed it wasn't literal
    # clipping (near-blown/blown pixel fractions actually DROPPED vs. the
    # original), it was band CONCENTRATION: the [220,240) luma band grew
    # from 9.7% to 21.9% of the entire frame, more than doubling, because
    # a room with this much already-light surface has less real headroom
    # to absorb the same push before a large fraction of the frame
    # compresses into one narrow near-white range.
    #
    # Fix: a room whose whiteFraction exceeds a normal baseline (~0.30 --
    # typical trim/ceiling/some walls) gets a proportionally lower
    # target, so it settles wherever fusion's own natural output already
    # lands instead of being force-pushed further. K=35 chosen because it
    # empirically reproduces exactly this: at IMG_8317's real
    # whiteFraction (0.504), effective_target lands at ~161 -- right at
    # the point where the residual solve's own gap>3.0 gate naturally
    # stops firing, letting fusion's unforced result (median 160, band
    # occupancy 0.349/0.101) stand, which is nearly IDENTICAL to the
    # ORIGINAL photo's own band occupancy (0.349/0.097) -- i.e. no
    # compression effect at all, while still keeping a real +6 lift over
    # the source. Rooms below the baseline (IMG_8315 at 0.236, IMG_8301
    # at 0.309) are effectively unaffected -- confirmed both still land
    # within a point of the full 168 target, since the formula only
    # engages once whiteFraction exceeds 0.30.
    WHITE_FRACTION_TARGET_BASELINE = 0.30
    WHITE_FRACTION_TARGET_K = 35.0
    white_fraction = white_surface_stats(img)["whiteFraction"]
    effective_target_median = target_median - max(
        0.0, white_fraction - WHITE_FRACTION_TARGET_BASELINE
    ) * WHITE_FRACTION_TARGET_K

    gap = effective_target_median - fusion_median
    residual_need = clamp01(gap / 100.0)  # kept for the metrics/report field

    # Regional protect/boost still applies here -- a SEPARATE, legitimate
    # per-region signal (e.g. "preserve_black_depth" on a firebox) from
    # the diagnosis-driven `intensity` damping this step now bypasses.
    # base_intensity=1.0: default (no regions) is "chase the full target,"
    # matching the reasoned position above. Clipped to [0,1] since a
    # `primary` region's >1.0 multiplier has no coherent meaning for a
    # solve-to-target operation -- there's no "more than fully solved."
    residual_apply_map = np.clip(build_regional_strength_map(
        l.shape, 1.0, regions, level2_masks,
        supported_operations=("shadow_recovery", "exposure_lift"),
        exclude_tag="exclude_from_shadow_lift",
    ), 0.0, 1.0)

    if gap > 3.0:
        normalized = np.clip(l / 255.0, 0, 1)
        fusion_median_norm = np.clip(fusion_median / 255.0, 1e-4, 0.999)
        target_norm = np.clip(effective_target_median / 255.0, 1e-4, 0.999)
        gamma_solved = float(np.log(target_norm) / np.log(fusion_median_norm))
        # Blend between "no change" (gamma=1.0) and the full solve, per
        # pixel, via residual_apply_map -- a protected region stays near
        # gamma=1.0 (untouched), everywhere else gets the full solve.
        gamma_map = 1.0 - residual_apply_map * (1.0 - gamma_solved)
        # DARK-MATERIAL PROTECTION (patch, pending validation; MORE
        # IMPORTANT NOW than before, since the push behind it is much
        # stronger): the old floor here (l / 25.0) only shields
        # true-black pixels (L<25) — anything darker than that from
        # crushing further, but it does nothing for dark furniture/
        # fixtures that sit well above true black. Confirmed directly on
        # a real photo (IMG_8317): dark brown leather and a black granite
        # fireplace surround measured in the L 30-90 range, fully exposed
        # to the gamma lift meant for underexposed shadows — same
        # treatment as a dim hallway, when these are supposed to read as
        # rich, dark material, not a lighting defect. Widened ramp so
        # protection tapers out to L=100 instead of L=25, so these
        # materials keep most of their depth while genuine near-black
        # shadow detail still gets lifted.
        dark_material_protect = np.clip(l / 100.0, 0, 1)
        # LEVEL 2: same fix as the fusion stage above, applied to this
        # ramp's inverse convention (0 = fully protected/no lift here).
        # Force to 0 -- full protection -- wherever Vision identified a
        # genuine dark material, overriding the luma ramp's partial
        # protection in the L 40-100 range.
        if dark_material_mask is not None:
            dark_material_protect = dark_material_protect * (1.0 - dark_material_mask)
        l = (255.0 * np.power(normalized, gamma_map)) * dark_material_protect + l * (1.0 - dark_material_protect)

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
        "effectiveTargetMedianLuma": round(effective_target_median, 2),
        "whiteFractionAtCorrectionTime": round(white_fraction, 4),
        "residual_need": round(float(residual_need), 3),
        "method": "synthetic_exposure_fusion",
        "level2DarkMaterialMaskApplied": dark_material_mask is not None,
        "regionalModeApplied": regional_mode,
        "regionalStrengthRange": [round(float(strength_map.min()), 3), round(float(strength_map.max()), 3)] if regional_mode else None,
    }


def clean_whites_adaptive(img, intensity=1.0, exclusion_weight=None, regions=None, level2_masks=None):
    """Adaptive MLS Bright clean-whites pass — measures actual likely-white
    architectural surfaces (trim, cabinets, ceilings via LAB chroma+luma)
    and only neutralizes/lifts them if they measurably need it, feathered
    with a blurred mask so there's no hard edge. Leaves walls, wood floors,
    and decor untouched — this targets only the surfaces a professional
    retoucher would target for "clean whites," not a global shift.

    exclusion_weight (Level 2, optional): float32 (H,W) array in [0,1] —
    continuous, not boolean — combining the Vision pre-pass's
    furniture/floor AND dark-material regions. Multiplied in as a "keep"
    weight, not subtracted as a hard AND.

    regions / level2_masks (Aug 2, 2026, optional): folds any `protect`-
    priority region (or one tagged "exclude_from_clean_whites") directly
    into the SAME exclusion mechanism above, unioned with
    exclusion_weight rather than replacing it.

    NO BOOST PATHWAY (deliberate, not an oversight): unlike
    mls_brightness_lift or window_balance, this function doesn't have a
    natural "do more of this, Vision-directed" instruction -- it already
    self-detects which surfaces qualify as likely-white via LAB
    chroma/luma, and corrects only those, automatically. There's no
    meaningful "primary" version of that beyond what the measurement
    already computes. Regional input here is protection-only.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
    white_mask = (L > 145.0) & (chroma < 22.0)
    strong_white_mask = (L > 165.0) & (chroma < 16.0)

    white_w = white_mask.astype(np.float32)
    strong_w = strong_white_mask.astype(np.float32)

    # ── Regional protection fold-in (Aug 2, 2026) ────────────────────────
    # Reuses build_regional_strength_map purely for its protect_accum
    # logic: pass empty supported_operations so no region can ever boost,
    # only protect (priority=="protect", or "exclude_from_clean_whites").
    if regions and level2_masks:
        regional_keep = build_regional_strength_map(
            L.shape, 1.0, regions, level2_masks,
            supported_operations=(), exclude_tag="exclude_from_clean_whites",
        )
        exclusion_weight = np.maximum(
            exclusion_weight if exclusion_weight is not None else 0.0,
            1.0 - regional_keep,
        )

    if exclusion_weight is not None:
        keep = np.clip(1.0 - exclusion_weight, 0.0, 1.0)
        white_w = white_w * keep
        strong_w = strong_w * keep

    white_fraction = float(white_w.mean())
    if white_fraction < 0.006:
        return img, {"applied": False, "whiteFraction": round(white_fraction, 4),
                      "reason": "insufficient_likely_white_surface"}

    sample_w = strong_w if strong_w.sum() > 1.0 else white_w
    total_w = float(sample_w.sum())
    mean_a = float((A * sample_w).sum() / total_w)
    mean_b = float((B * sample_w).sum() / total_w)
    mean_l = float((L * sample_w).sum() / total_w)
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

    mask_u8 = np.clip(white_w * 255.0, 0, 255).astype(np.uint8)
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
        "level2ExclusionApplied": exclusion_weight is not None,
    }


def window_balance(img, regions=None, level2_masks=None):
    """Safe window/highlight balancing — compresses overly bright highlight
    regions only. Does not reconstruct exterior detail, replace views, or
    add any content.

    regions / level2_masks (Aug 2, 2026, optional): a region tagged
    operation="highlight_reduction" can trigger compression even BELOW
    the automatic L>238 threshold below -- real positive control, not
    just scaling an already-automatic mask, since Vision may correctly
    flag a highlight that hasn't technically blown out yet but still
    reads as excessive (e.g. the schema spec's french_doors example: "the
    exterior view is already well exposed... reduce only excessive
    highlights"). A `protect` region (or one tagged
    "exclude_from_highlight_reduction") is excluded from compression
    entirely, regardless of brightness."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    bright_frac = float((l > 240).sum()) / float(l.size)

    regional_mode = regions is not None and level2_masks is not None

    # ── Regional threshold lowering (Aug 2, 2026) ────────────────────────
    # base_intensity=0.0: a pixel gets NO regional help by default. A
    # highlight_reduction region raises that toward 1.0 in its footprint,
    # which lowers the effective trigger threshold there (see
    # `effective_threshold` below) -- letting Vision flag a highlight the
    # automatic global rule wouldn't have caught on its own.
    regional_boost = build_regional_strength_map(
        l.shape, 0.0, regions, level2_masks,
        supported_operations=("highlight_reduction",),
        exclude_tag="exclude_from_highlight_reduction",
    ) if regional_mode else np.zeros(l.shape, dtype=np.float32)

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
    #
    # Regional lowering: threshold drops from 238 toward 200 (still not
    # aggressive -- 200 is a bright midtone, not a shadow) in proportion
    # to regional_boost, so a flagged-but-not-yet-blown highlight can
    # still get pulled back where Vision specifically asked for it,
    # without lowering the bar for the rest of the frame.
    effective_threshold = 238.0 - (38.0 * regional_boost)
    mask = np.clip((l - effective_threshold) / 17.0, 0, 1)
    if regional_mode:
        # Regions that were pure `protect` (not highlight_reduction) never
        # raised regional_boost above 0, so they're already excluded from
        # the threshold-lowering above -- but they should ALSO be excluded
        # from the ORIGINAL automatic mask (a protected region sitting at
        # L>238 shouldn't get compressed just because it's naturally
        # bright). Compute that separately, same protect-only pattern as
        # clean_whites_adaptive.
        protect_only = 1.0 - build_regional_strength_map(
            l.shape, 1.0, regions, level2_masks,
            supported_operations=(), exclude_tag="exclude_from_highlight_reduction",
        )
        mask = mask * (1.0 - protect_only)

    # BUG CAUGHT AND FIXED DURING TESTING (Aug 2, 2026): an earlier
    # version of this gate checked mask.max() >= 0.02, which is nearly
    # ALWAYS true the instant any single pixel anywhere in the frame
    # exceeds ~238 -- reproducing the exact over-triggering bug this
    # function was already patched once to prevent ("fired on nearly any
    # photo with a window or lamp in it"). Confirmed directly on a real
    # test photo: a single unrelated bright pixel (unrelated to any
    # region) drove mask.max() to 1.0 while the actual affected FRACTION
    # of the frame was 0.49% -- nowhere near meaningful. Fixed: gate on
    # the fraction of pixels actually crossing threshold, same quantity
    # and same 0.02 bar the original design used, computed from the
    # (possibly regionally-lowered) mask so it stays correct in both
    # modes instead of bypassing the gate entirely in regional mode.
    triggered_fraction = float((mask > 0.02).mean())
    if triggered_fraction < 0.02:
        return img, {"applied": False, "highlight_fraction": round(bright_frac, 4),
                      "regionalModeApplied": regional_mode}

    compressed = effective_threshold + (l - effective_threshold) * 0.75
    lab[:, :, 0] = np.clip(l * (1.0 - mask) + compressed * mask, 0, 252)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, {
        "applied": True,
        "highlight_fraction": round(bright_frac, 4),
        "regionalModeApplied": regional_mode,
    }


def mls_color_finish(img, intensity=1.0, regions=None, level2_masks=None):
    """MLS finish: neutral, clean, bright — not editorial. Normalizes
    saturation toward the calibrated MLS Bright target range and adds a
    mild clarity/unsharp pass so the image doesn't read as flat after the
    brightness and white-surface work above.

    regions / level2_masks (Aug 2, 2026, optional):
    - operation="hue_protection" or "saturation_protection" on any region
      already works for free via build_regional_strength_map's universal
      protect list -- no extra code needed here, applied to the global
      saturation factor below.
    - operation="texture_enhancement" boosts the existing automatic
      texture-based sharpening mask in that region; "clarity_reduction"
      dials it down (not off) using the new reduction mode -- e.g. a
      delicate patterned surface Vision wants left softer than the
      automatic local-variance detector would otherwise sharpen."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_mean = float(hsv[:, :, 1].mean())

    if sat_mean < 62:
        factor = 1.0 + 0.07 * intensity
    elif sat_mean > 96:
        factor = 1.0 - 0.08 * intensity
    else:
        factor = 1.0 - 0.02 * intensity

    regional_mode = regions is not None and level2_masks is not None
    if regional_mode:
        # base_intensity=1.0 -- "how much of the computed saturation shift
        # to apply," 1.0 = today's full-strength behavior everywhere a
        # protect region doesn't override it.
        sat_regional = build_regional_strength_map(
            hsv.shape[:2], 1.0, regions, level2_masks,
            supported_operations=(), exclude_tag="exclude_from_color_finish",
        )
        sat_factor_map = 1.0 + sat_regional[:, :, None] * (factor - 1.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor_map[:, :, 0], 0, 255)
    else:
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
    texture_mask = np.clip(local_var / 60.0, 0, 1)  # 0 on flat surfaces, 1 on real texture

    if regional_mode:
        clarity_regional = build_regional_strength_map(
            gray.shape, 1.0, regions, level2_masks,
            supported_operations=("texture_enhancement", "clarity_reduction"),
            exclude_tag="exclude_from_color_finish",
        )
        texture_mask = np.clip(texture_mask * clarity_regional, 0, 1)

    texture_mask = texture_mask[:, :, None]
    sharpened_full = cv2.addWeighted(color_finished, 1.08, blurred, -0.08, 0)
    sharpened = color_finished.astype(np.float32) * (1.0 - texture_mask) + sharpened_full.astype(np.float32) * texture_mask
    return np.clip(sharpened, 0, 255).astype(np.uint8), {
        "mean_saturation_before": round(sat_mean, 2), "saturation_factor": round(float(factor), 3),
        "regionalModeApplied": regional_mode,
    }


# ── Front judgment: Stage 1 diagnosis -> real correction decisions ──────
# (Aug 2, 2026 -- previously log-only, see level2_diagnosis.py.) Each
# category below has an explicit, stated correction implication in the
# diagnosis prompt itself; this table is that implication made real.
# Multiplier applies on top of the existing exposure-only adaptive_intensity
# math -- it never OVERRIDES the exposure signal, only biases it, and only
# when Vision's confidence is medium/high. A low-confidence or absent
# diagnosis leaves today's exposure-only behavior completely unchanged --
# same "never worse than today" rule as every other patch in this file.
DIAGNOSIS_INTENSITY_MULTIPLIER = {
    "already_acceptable": 0.0,       # special-cased below: hard cap, not a multiplier
    "backlit_mixed_lighting": 0.7,   # protect the correctly-exposed background;
                                      # lean on window_balance() instead of global lift
    "color_cast": 0.85,              # exposure isn't the problem; de-emphasize brightness
    "mixed_light_temperature": 0.75, # a single global WB pass can't fix two casts at once --
                                      # be more conservative, not more aggressive
    "flat_evenly_underexposed": 1.0, # this is what the exposure-only math already assumes
}
DIAGNOSIS_ALREADY_ACCEPTABLE_CEILING = 0.35  # matches the existing intensity floor elsewhere


def _diagnosis_adjusted_intensity(adaptive_intensity, diagnosis):
    """Applies the front-judgment bias table above. Returns
    (adjusted_intensity, tag_or_None)."""
    if diagnosis is None or diagnosis.get("confidence") not in ("medium", "high"):
        return adaptive_intensity, None
    category = diagnosis.get("diagnosis")
    if category == "already_acceptable":
        if adaptive_intensity > DIAGNOSIS_ALREADY_ACCEPTABLE_CEILING:
            return DIAGNOSIS_ALREADY_ACCEPTABLE_CEILING, "ai_diagnosis_capped_intensity"
        return adaptive_intensity, None
    mult = DIAGNOSIS_INTENSITY_MULTIPLIER.get(category)
    if mult is not None and mult < 1.0:
        adjusted = adaptive_intensity * mult
        if adjusted < adaptive_intensity:
            return adjusted, "ai_diagnosis_reduced_intensity"
    return adaptive_intensity, None


def _diagnosis_wb_threshold(diagnosis):
    """color_cast diagnosis lowers the bar for tagging white balance as
    applied -- a real cast should be corrected and reported even if
    subtle, per the diagnosis category's own stated implication."""
    if diagnosis is not None and diagnosis.get("confidence") in ("medium", "high") \
            and diagnosis.get("diagnosis") == "color_cast":
        return 0.05
    return 0.1


# ── Back judgment: Stage 4 QC -> retry-then-safe-fallback loop ──────────
# (Aug 2, 2026 -- previously log-only.) Closes the loop as originally
# designed on Aug 1: "run the deterministic correction, then show Vision
# the result and ask does anything look wrong" was always meant to be
# ACTED on, not just logged. Scope is deliberately bounded to whole-frame
# retry, no masking/local revert (a real, larger capability, intentionally
# deferred) -- this only reuses tunables that already exist in this file
# (adaptive_intensity, target_display_headroom).
#
# Loop: if QC flags the first pass, retry ONCE at reduced intensity/
# reduced HDR headroom, whole frame. If the retry passes QC, ship it. If
# the retry STILL fails QC, do not ship a flagged photo silently -- fall
# back to the true pre-correction original and flag for manual review.
# Worst case is now "no correction, flagged for a human," never "silently
# worse than the original," which is the actual defect this closes.
QC_RETRY_INTENSITY_MULTIPLIER = 0.5
QC_RETRY_HEADROOM_MULTIPLIER = 0.5  # fraction of the *excess* headroom above 1.0 to keep


def _run_hdr_pass(source_path, target_display_headroom=None):
    """Thin, retriable wrapper around recover_hdr_if_present()."""
    if not HDR_RECOVERY_ENABLED:
        return None, {"gainMapPresent": False, "recoveryApplied": False, "enabled": False}
    img, report = recover_hdr_if_present(source_path, target_display_headroom=target_display_headroom)
    report["enabled"] = True
    return img, report


def _apply_exterior_stack(img, args, intensity, wb_threshold=0.1):
    """Deterministic exterior correction stack, factored out of main() so
    the QC retry loop can re-run it at a different intensity without
    duplicating ~25 lines inline."""
    modules = []
    img, wb_strength = white_balance_neutral_aware(img)
    if wb_strength >= wb_threshold:
        modules.append("white_balance")
    img, lens_strength = mild_mobile_lens_correction(img, args.lens_mode)
    if lens_strength > 0:
        modules.append("lens_correction")
    img, rotation_deg = deskew_perspective(img)
    if rotation_deg != 0.0:
        modules.append("perspective_alignment")
    img, denoise_strength = adaptive_denoise(img)
    if denoise_strength > 2:
        modules.append("adaptive_noise_reduction")
    img, exterior_metrics = exterior_daylight_correction(img, intensity=intensity)
    if abs(exterior_metrics["after_median_luma"] - exterior_metrics["before_median_luma"]) >= 3:
        modules.append("exterior_daylight_shadow_lift")
    metrics = {
        "whiteBalanceStrength": wb_strength,
        "lensCorrectionStrength": lens_strength,
        "exteriorCorrection": exterior_metrics,
    }
    return img, modules, metrics, rotation_deg, denoise_strength


def _apply_interior_stack(img, args, adaptive_intensity, level2_regions, wb_threshold=0.1):
    """Deterministic interior correction stack, factored out of main() so
    the QC retry loop can re-run it at a different intensity without
    duplicating the full stack inline. level2_regions is passed in rather
    than recomputed -- geometric layout doesn't meaningfully change
    between a normal-headroom and reduced-headroom HDR variant of the
    same photo, so this avoids a second Vision call on retry.

    WIRED LIVE (Aug 3, 2026): regions/level2_masks are now threaded into
    every one of the five functions that already know how to consume
    them (white_balance_neutral_aware, mls_brightness_lift,
    clean_whites_adaptive, window_balance, mls_color_finish). Previously
    get_level2_regions() computed a full per-region Retoucher Schema read
    and this function discarded almost all of it, keeping only the two
    legacy derived masks (dark_material_mask / furniture_floor_mask).
    This is the change that lets a Vision-identified region (e.g. a
    backlit chair tagged shadow_recovery/primary) actually change the
    shipped pixels instead of only ever being available to protect
    something. `regions`/`level2_masks` collapse to None (not just
    empty) when Level 2 is disabled or the Vision call failed/returned
    nothing, so every function below falls back to its exact pre-Aug-3
    scalar/legacy-mask-only behavior -- nothing changes for a photo with
    no usable regions."""
    modules = []
    regions = level2_regions.get("regions") or None
    level2_masks = level2_regions.get("masks") or None

    img, wb_strength = white_balance_neutral_aware(img, regions=regions, level2_masks=level2_masks)
    if wb_strength >= wb_threshold:
        modules.append("white_balance")
    img, lens_strength = mild_mobile_lens_correction(img, args.lens_mode)
    if lens_strength > 0:
        modules.append("lens_correction")
    img, rotation_deg = deskew_perspective(img)
    if rotation_deg != 0.0:
        modules.append("perspective_alignment")
    img, denoise_strength = adaptive_denoise(img)
    if denoise_strength > 2:
        modules.append("adaptive_noise_reduction")
    img, vignette_strength = vignette_correct(img)
    if vignette_strength >= 0.1:
        modules.append("vignette_neutralization")

    img, brightness_metrics = mls_brightness_lift(
        img, intensity=adaptive_intensity,
        dark_material_mask=level2_regions.get("dark_material_mask"),
        regions=regions, level2_masks=level2_masks,
    )
    brightness_moved = abs(brightness_metrics["after_fusion_median_luma"] - brightness_metrics["before_median_luma"]) >= 5
    if brightness_moved or brightness_metrics["residual_need"] >= 0.05:
        modules.append("mls_brightness_lift")

    ff_weight = level2_regions.get("furniture_floor_mask")
    dm_weight = level2_regions.get("dark_material_mask")
    if ff_weight is not None or dm_weight is not None:
        combined_exclusion_weight = np.maximum(
            ff_weight if ff_weight is not None else 0.0,
            dm_weight if dm_weight is not None else 0.0,
        )
    else:
        combined_exclusion_weight = None

    img, white_metrics = clean_whites_adaptive(
        img, intensity=adaptive_intensity, exclusion_weight=combined_exclusion_weight,
        regions=regions, level2_masks=level2_masks,
    )
    if white_metrics.get("applied"):
        modules.append("clean_whites")

    img, window_metrics = window_balance(img, regions=regions, level2_masks=level2_masks)
    if window_metrics.get("applied"):
        modules.append("window_highlight_balance")

    img, finish_metrics = mls_color_finish(
        img, intensity=adaptive_intensity, regions=regions, level2_masks=level2_masks,
    )
    modules.append("color_clarity_finish")

    metrics = {
        "whiteBalanceStrength": wb_strength,
        "lensCorrectionStrength": lens_strength,
        "vignetteStrength": vignette_strength,
        "adaptiveIntensity": round(adaptive_intensity, 3),
        "mlsBrightness": brightness_metrics,
        "cleanWhites": white_metrics,
        "windowBalance": window_metrics,
        "mlsFinish": finish_metrics,
    }
    return img, modules, metrics, rotation_deg, denoise_strength


# ── Region debug overlay (Aug 3, 2026) ──────────────────────────────────
# Answers a question the Aug 3 wiring session couldn't answer: "did the
# region I'm looking at in the corrected photo actually correspond to
# what Vision drew?" Previous manual pixel checks (this session) had to
# eyeball rectangles from a thumbnail, with no way to confirm they
# actually overlapped the real mask -- worthless as verification. This
# draws the REAL feathered mask (not a guessed box) as a colored, alpha-
# blended overlay with a text label per region, so any future visual
# check is against the pipeline's actual data, not a guess.
#
# Off by default (env var), zero cost to every normal request -- same
# kill-switch pattern as LEVEL2_VISION_MASKS_ENABLED / LEVEL0_SHADOW_MODE.
# Writes to a SEPARATE file next to the real output; never touches the
# shipped photo itself.
DEBUG_REGIONS_OVERLAY = os.environ.get("DEBUG_REGIONS_OVERLAY", "false").lower() == "true"

# Distinct color per priority so a glance tells you what each blob means
# without reading every label. BGR since this stays in cv2's color space
# the whole way through -- no BGR2RGB round-trip needed before imwrite.
_DEBUG_PRIORITY_COLORS = {
    "protect": (0, 0, 255),      # red   -- "don't touch this"
    "primary": (0, 200, 0),      # green -- "boost this the most"
    "secondary": (255, 200, 0),  # cyan  -- "boost this some"
}
_DEBUG_DEFAULT_COLOR = (255, 0, 255)  # magenta -- anything else (shouldn't happen)


def write_region_debug_overlay(img, regions, level2_masks, output_path):
    """Writes a colored overlay of the REAL per-pixel region masks (not
    guessed boxes) to output_path, with a text label per region showing
    regionId / operation / priority. Silently no-ops if there are no
    regions to draw -- never raises, matches every other Level 2 failure
    mode in this file (degrade gracefully, don't crash the batch)."""
    if not regions or not level2_masks:
        return
    try:
        overlay = img.astype(np.float32).copy()
        for region in regions:
            mask = level2_masks.get(region.get("maskId"))
            if mask is None:
                continue
            color = np.array(
                _DEBUG_PRIORITY_COLORS.get(region.get("priority"), _DEBUG_DEFAULT_COLOR),
                dtype=np.float32,
            )
            alpha = (np.clip(mask, 0, 1) * 0.45)[:, :, None]
            overlay = overlay * (1.0 - alpha) + color * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        for region in regions:
            mask = level2_masks.get(region.get("maskId"))
            if mask is None:
                continue
            # Label at the mask's centroid (weighted by mask strength,
            # not a bounding-box corner) -- lands inside the actual
            # feathered shape even for irregular masks.
            ys, xs = np.where(mask > 0.5)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            label = f"{region.get('regionId', '?')} [{region.get('operation', '?')}/{region.get('priority', '?')}]"
            # Black outline + white fill so the label reads on any
            # background color the overlay produces.
            cv2.putText(overlay, label, (max(0, cx - 70), max(15, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, label, (max(0, cx - 70), max(15, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(output_path, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    except Exception as e:  # noqa: BLE001 -- a debug tool must never crash a real correction
        print(f"[write_region_debug_overlay] failed (non-fatal): {e}", file=sys.stderr)


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

    # ── Stage 4 QC snapshot (added Aug 2026; FIXED Aug 2, 2026) ──────────
    # BUG FOUND (Aug 2, 2026, via real photo IMG_8311): this used to be
    # `img.copy()` taken here -- but by this point `img` has ALREADY been
    # through HDR gain-map recovery (see recover_hdr_if_present() above),
    # if a gain map was present. That meant "original_img_for_qc" was
    # never actually the camera-original -- it was the post-HDR-recovery
    # image. Any artifact HDR recovery itself introduced (e.g. the
    # gain-map/base-photo gradient-covariance streak documented in
    # hdrRecover.py's module docstring) was already baked into BOTH sides
    # of every qc_check() comparison below, since nothing downstream of
    # HDR recovery removes it. QC was structurally incapable of catching
    # this entire class of bug, independent of prompt quality or model
    # choice -- confirmed on IMG_8311: a real, visible amber streak on
    # the pool-deck concrete produced looksArtificial=false at HIGH
    # confidence, because from QC's point of view nothing had changed in
    # that region between the two images it was actually shown.
    #
    # Fix: build the QC baseline from decode_standard(args.source) -- the
    # SAME raw-decode function recover_hdr_if_present() calls internally
    # for its own base_img, so this is guaranteed to be the true
    # pre-HDR-recovery original, decoded the identical way. Falls back to
    # img.copy() only if that raw decode fails for some reason HDR
    # recovery's own error handling didn't already catch (shouldn't
    # happen in practice -- by this point `img` exists, so some decode
    # already succeeded).
    try:
        original_img_for_qc = decode_standard(args.source)
    except Exception as e:
        print(f"[smartCorrect] Could not build raw QC baseline, falling back to post-HDR img: {e}", file=sys.stderr)
        original_img_for_qc = img.copy()

    # ── Exterior daylight scene gate (patch, pending validation) ────────
    # Confirmed directly on a real photo (IMG_8311, pool/patio) that the
    # interior-calibrated MLS Bright stack — target median luma 178,
    # clean-whites neutralization, window highlight compression — was
    # being applied to an exterior daylight shot, pushing a deliberately
    # shaded concrete slab toward sunlit brightness with visible tonal
    # banding as a side effect. See is_exterior_daylight() docstring for
    # detection method and known limitations.
    is_exterior_hsv, exterior_signals = is_exterior_daylight(img)

    # ── Level 0: Vision scene classification (added Aug 2026) ───────────
    # Runs before Stage 1 diagnosis and before any correction path is
    # chosen. Ships in shadow mode by default (LEVEL0_SHADOW_MODE=true):
    # `is_exterior` below still resolves to the HSV heuristic's answer
    # until a real batch of logged disagreements has been reviewed and
    # LEVEL0_SHADOW_MODE is flipped to false. See level0_scene_classifier.py
    # docstring for the full rollout plan and why this exists (IMG_8305
    # false negative on a close-up driveway/stucco exterior).
    scene = resolve_scene_type(img, is_exterior_hsv)
    is_exterior = scene["isExterior"]
    if scene["disagreement"]:
        modules_applied.append(
            f"scene_disagreement_hsv-{'ext' if is_exterior_hsv else 'int'}"
            f"_vision-{scene['visionResult'].get('sceneType')}"
        )

    # ── Level 2 Stage 1: Vision diagnosis (added Aug 2026, LOG-ONLY) ─────
    # Prototype from the same design session as the Level 2 region masks
    # above. Runs a single Vision call combining the photo with the
    # pipeline's own real computed stats (image_stats, white_surface_stats,
    # shadow_highlight_stats) to get a qualitative read on WHAT KIND of
    # correction this photo needs -- e.g. distinguishing a backlit photo
    # (needs targeted shadow lift, protect the highlights) from an evenly
    # underexposed one (safe for a more uniform lift), which the numeric
    # thresholds alone can't tell apart.
    #
    # DELIBERATELY NOT ACTING ON THIS YET: the diagnosis is logged into
    # the JSON output and as a readable tag below, but does not change
    # any correction behavior in this version. This is so real diagnoses
    # can be reviewed across real photo volume before trusting this to
    # drive routing decisions -- see level2_diagnosis.py's own docstring
    # for the full reasoning and status.
    diagnosis, diagnosis_report = level2_diagnose(
        img, image_stats(img), white_surface_stats(img), shadow_highlight_stats(img), is_exterior,
    )
    if diagnosis is not None:
        modules_applied.append(f"ai_diagnosis_{diagnosis.get('diagnosis', 'unknown')}")

    if is_exterior:
        # ── Front judgment applied here (Aug 2, 2026) ────────────────────
        # Exterior photos don't get Stage 1 diagnosis today (level2_diagnose
        # explicitly skips exteriors -- see level2_diagnosis.py's
        # `is_exterior` early-return), so `diagnosis` is always None on
        # this branch and the adjustment below is a deliberate no-op for
        # now. Left wired so exterior diagnosis can plug in later without
        # touching this branch again.
        ext_intensity, diag_tag = _diagnosis_adjusted_intensity(intensity, diagnosis)
        if diag_tag:
            modules_applied.append(diag_tag)
        wb_threshold = _diagnosis_wb_threshold(diagnosis)

        img, ext_modules, exterior_stack_metrics, rotation_deg, denoise_strength = \
            _apply_exterior_stack(img, args, ext_intensity, wb_threshold)
        modules_applied.extend(ext_modules)
        skipped = skipped + ["mls_brightness_lift", "clean_whites", "window_highlight_balance",
                              "vignette_neutralization"]

        # ── Back judgment: Stage 4 QC retry-then-fallback (Aug 2, 2026) ──
        # This is the branch IMG_8311 (the confirmed real artifact case)
        # actually runs through. See the module-level docstring above
        # main() for the full loop rationale.
        qc = qc_check(original_img_for_qc, img)
        retry_report = {"attempted": False}
        if qc.get("looksArtificial") is True and qc.get("confidence") in ("high", "medium"):
            retry_report["attempted"] = True
            retry_headroom = None
            if hdr_report.get("recoveryApplied") and hdr_report.get("headroom"):
                retry_headroom = 1.0 + (hdr_report["headroom"] - 1.0) * QC_RETRY_HEADROOM_MULTIPLIER
            retry_img, retry_hdr_report = _run_hdr_pass(args.source, target_display_headroom=retry_headroom)
            if retry_img is None:
                retry_img = original_img_for_qc.copy()
                retry_hdr_report = hdr_report
            retry_intensity = max(ext_intensity * QC_RETRY_INTENSITY_MULTIPLIER, 0.6)

            retry_final, retry_modules, retry_metrics, retry_rotation, retry_denoise = \
                _apply_exterior_stack(retry_img, args, retry_intensity, wb_threshold)
            retry_qc = qc_check(original_img_for_qc, retry_final)
            retry_report.update({
                "retryHeadroomTarget": retry_headroom,
                "retryIntensity": retry_intensity,
                "retryQC": retry_qc,
            })

            if retry_qc.get("looksArtificial") is not True:
                # Retry resolved it -- ship the retry result.
                img, exterior_stack_metrics = retry_final, retry_metrics
                modules_applied = ["hdr_gain_map_recovery"] if retry_hdr_report.get("recoveryApplied") else []
                modules_applied += retry_modules + ["qc_retry_resolved"]
                rotation_deg, denoise_strength = retry_rotation, retry_denoise
                hdr_report, qc = retry_hdr_report, retry_qc
            else:
                # Retry still flagged -- do not ship a flagged photo
                # silently. Fall back to the true pre-correction original;
                # worst case is now "no correction, flagged for a human,"
                # never "silently worse than the original."
                img = original_img_for_qc.copy()
                modules_applied = ["qc_retry_failed_fallback_to_original", "needs_manual_review"]
                qc = retry_qc
                hdr_report = {**hdr_report, "recoveryApplied": False, "fallbackReason": "qc_flagged_after_retry"}
                rotation_deg, denoise_strength = 0.0, 0
                exterior_stack_metrics = {"note": "correction reverted -- QC flagged both the original and retry passes"}
        elif qc.get("looksArtificial") is True:
            modules_applied.append("qc_flagged_possible_artifact")

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
            "level0Scene": scene,
            "level2Diagnosis": diagnosis_report,
            "level4QC": qc,
            "qcRetry": retry_report,
            "hdrRecovery": hdr_report,
            "metrics": exterior_stack_metrics,
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
        # integration. The actual write now happens AFTER the QC
        # retry/fallback logic below, once the final `img` is settled --
        # writing here, before QC has had a chance to retry or revert,
        # would ship whatever HDR recovery produced even if QC later
        # rejects it.

        # ── Stage 4: Vision QC (added Aug 2026) ───────────────────────────
        # Only worth calling if HDR recovery actually changed pixels here
        # -- otherwise `img` is byte-identical to the original and a QC
        # comparison would be a wasted Vision call with a guaranteed-false
        # result. If recovery didn't apply, report a clear "not needed"
        # status rather than silently omitting the field (see missing-
        # level0Scene bug this same branch had, found via real testing).
        retry_report = {"attempted": False}
        if hdr_report.get("recoveryApplied"):
            qc = qc_check(original_img_for_qc, img)
            if qc.get("looksArtificial") is True:
                modules_applied_gate_tag = ["qc_flagged_possible_artifact"]
            else:
                modules_applied_gate_tag = []

            # ── Back judgment: QC retry-then-fallback (Aug 2, 2026) ──────
            # Only HDR recovery ran on this branch (no other corrections),
            # so the only retriable knob is the headroom target itself.
            if qc.get("looksArtificial") is True and qc.get("confidence") in ("high", "medium"):
                retry_report["attempted"] = True
                retry_headroom = None
                if hdr_report.get("headroom"):
                    retry_headroom = 1.0 + (hdr_report["headroom"] - 1.0) * QC_RETRY_HEADROOM_MULTIPLIER
                retry_img, retry_hdr_report = _run_hdr_pass(args.source, target_display_headroom=retry_headroom)
                if retry_img is not None:
                    retry_qc = qc_check(original_img_for_qc, retry_img)
                    retry_report.update({"retryHeadroomTarget": retry_headroom, "retryQC": retry_qc})
                    if retry_qc.get("looksArtificial") is not True:
                        img, hdr_report, qc = retry_img, retry_hdr_report, retry_qc
                        modules_applied_gate_tag = ["qc_retry_resolved"]
                    else:
                        img = original_img_for_qc.copy()
                        hdr_report = {**hdr_report, "recoveryApplied": False, "fallbackReason": "qc_flagged_after_retry"}
                        qc = retry_qc
                        modules_applied_gate_tag = ["qc_retry_failed_fallback_to_original", "needs_manual_review"]
        else:
            qc = {"looksArtificial": None, "confidence": None, "issue": None,
                  "location": None, "enabled": None,
                  "called": False, "error": "not_needed_no_pixels_changed"}
            modules_applied_gate_tag = []

        # Re-check the (possibly retried/reverted) image before writing --
        # `img` may now be the fallback original rather than the
        # HDR-recovered version decided above.
        if hdr_report.get("recoveryApplied") or "qc_retry_failed_fallback_to_original" in modules_applied_gate_tag:
            cv2.imwrite(args.output, img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        elif os.path.abspath(args.source) != os.path.abspath(args.output):
            shutil.copyfile(args.source, args.output)

        print(json.dumps({
            "output": args.output,
            "modulesApplied": (["hdr_gain_map_recovery"] if hdr_report.get("recoveryApplied") else []) + ["already_mls_bright_no_correction_applied"] + modules_applied_gate_tag,
            "modulesSkipped": skipped,
            "perspectiveCorrectionDegrees": 0.0,
            "denoiseStrength": 0,
            "histogramStats": guard["shadowHighlightStats"],
            "professionalMLSGuard": guard,
            "hdrRecovery": hdr_report,
            "level0Scene": scene,
            "level4QC": qc,
            "qcRetry": retry_report,
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

    # ── Front judgment applied here (Aug 2, 2026) ─────────────────────────
    # Previously log-only; see level2_diagnosis.py and the bias table
    # above main(). Only acts when confidence is medium/high -- a low-
    # confidence or absent diagnosis leaves the exposure-only number above
    # completely unchanged.
    adaptive_intensity, diag_tag = _diagnosis_adjusted_intensity(adaptive_intensity, diagnosis)
    if diag_tag:
        modules_applied.append(diag_tag)
    wb_threshold = _diagnosis_wb_threshold(diagnosis)

    # ── Level 2 Vision pre-pass (added Aug 2026) ─────────────────────────
    # Called HERE, not inside _apply_interior_stack: every geometric
    # correction below (lens, deskew) needs to have already run so the
    # boxes Vision returns are in the same pixel coordinate space `img`
    # will be in. Computed ONCE, outside the retry loop -- geometric
    # layout doesn't meaningfully change between a normal-headroom and
    # reduced-headroom HDR variant of the same photo, so reusing these
    # masks on retry avoids a second Vision call.
    #
    # READ-ONLY, routing-signal-only, per level2_vision_regions.py's own
    # docstring -- this does not touch pixels and does not compromise
    # this file's "no generative model" rule. If it fails or is disabled,
    # both masks are None and every downstream call falls back to its
    # original heuristic-only behavior, unchanged.
    #
    # NOTE: this pre-pass needs a geometrically-corrected `img` to align
    # masks correctly, so run the lens/deskew steps once, upfront, on a
    # throwaway copy purely to get aligned regions -- the real pass
    # (inside _apply_interior_stack) re-runs these deterministically and
    # will produce identical geometry, so this is not wasted, it's the
    # only way to get aligned masks before the retriable stack runs.
    _geom_preview, _ = mild_mobile_lens_correction(img.copy(), args.lens_mode)
    _geom_preview, _ = deskew_perspective(_geom_preview)
    level2_regions, level2_report = get_level2_regions(_geom_preview)
    if level2_report.get("called") and not level2_report.get("error"):
        modules_applied.append("level2_vision_regions")

    # ── Region debug overlay (Aug 3, 2026, optional) ──────────────────────
    if DEBUG_REGIONS_OVERLAY:
        debug_path = os.path.splitext(args.output)[0] + "_regions_debug.jpg"
        write_region_debug_overlay(
            _geom_preview, level2_regions.get("regions"), level2_regions.get("masks"), debug_path,
        )

    img, stack_modules, stack_metrics, rotation_deg, denoise_strength = \
        _apply_interior_stack(img, args, adaptive_intensity, level2_regions, wb_threshold)
    modules_applied.extend(stack_modules)

    histogram_stats = shadow_highlight_stats(img)

    # ── Back judgment: Stage 4 QC retry-then-fallback (Aug 2, 2026) ──────
    # Previously log-only. See the module-level docstring above main() for
    # the full loop rationale -- bounded to whole-frame retry, no masking.
    qc = qc_check(original_img_for_qc, img)
    retry_report = {"attempted": False}
    if qc.get("looksArtificial") is True and qc.get("confidence") in ("high", "medium"):
        retry_report["attempted"] = True
        retry_headroom = None
        if hdr_report.get("recoveryApplied") and hdr_report.get("headroom"):
            retry_headroom = 1.0 + (hdr_report["headroom"] - 1.0) * QC_RETRY_HEADROOM_MULTIPLIER
        retry_img, retry_hdr_report = _run_hdr_pass(args.source, target_display_headroom=retry_headroom)
        if retry_img is None:
            retry_img = original_img_for_qc.copy()
            retry_hdr_report = hdr_report
        retry_intensity = max(adaptive_intensity * QC_RETRY_INTENSITY_MULTIPLIER, 0.35)

        retry_final, retry_stack_modules, retry_stack_metrics, retry_rotation, retry_denoise = \
            _apply_interior_stack(retry_img, args, retry_intensity, level2_regions, wb_threshold)
        retry_qc = qc_check(original_img_for_qc, retry_final)
        retry_report.update({
            "retryHeadroomTarget": retry_headroom,
            "retryIntensity": round(retry_intensity, 3),
            "retryQC": retry_qc,
        })

        if retry_qc.get("looksArtificial") is not True:
            img, stack_metrics = retry_final, retry_stack_metrics
            modules_applied = ([m for m in modules_applied if m not in stack_modules
                                 and m != "hdr_gain_map_recovery"])
            modules_applied += (["hdr_gain_map_recovery"] if retry_hdr_report.get("recoveryApplied") else [])
            modules_applied += retry_stack_modules + ["qc_retry_resolved"]
            rotation_deg, denoise_strength = retry_rotation, retry_denoise
            hdr_report, qc = retry_hdr_report, retry_qc
            histogram_stats = shadow_highlight_stats(img)
        else:
            img = original_img_for_qc.copy()
            modules_applied = ["qc_retry_failed_fallback_to_original", "needs_manual_review"]
            qc = retry_qc
            hdr_report = {**hdr_report, "recoveryApplied": False, "fallbackReason": "qc_flagged_after_retry"}
            rotation_deg, denoise_strength = 0.0, 0
            stack_metrics = {"note": "correction reverted -- QC flagged both the original and retry passes"}
            histogram_stats = shadow_highlight_stats(img)
    elif qc.get("looksArtificial") is True:
        modules_applied.append("qc_flagged_possible_artifact")

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
        "level2Vision": level2_report,
        "level2Regions": level2_regions.get("regions", []),
        "level2Diagnosis": diagnosis_report,
        "level0Scene": scene,
        "level4QC": qc,
        "qcRetry": retry_report,
        "metrics": stack_metrics,
    }))


if __name__ == "__main__":
    main()
