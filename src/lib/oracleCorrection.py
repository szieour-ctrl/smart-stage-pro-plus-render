"""
oracleCorrection.py

Oracle-driven correction pass for Smart Correct Level 2.

IMPORTANT — compliance note (see handoff, gotcha #3):
Any Oracle image used here is an internal, never-delivered analysis input.
It must never be exposed to or delivered to end users. This module produces
pixel corrections *guided by* Oracle deltas, gated by recoverability computed
from the ORIGINAL file alone. The architecture (generative render used only
as internal calibration/analysis input) has a defensible AB 723 reading, but
this is a legal call, not an engineering one -- needs John's sign-off before
any production/user-facing path. This module is prototype/test-harness only
until that review happens.

Two image classes handled differently (see handoff Step 2):
  - SAME-ROOM oracle (graded version of the identical photo): per-pixel
    alignment via homography, then dense delta maps.
  - DIFFERENT-ROOM oracle: no per-pixel correspondence exists. Do not attempt
    to align or diff spatially -- extract aggregate style targets instead
    (median luma, saturation, warmth) and apply as a global calibration, not
    a spatial map. This module currently implements the same-room path only;
    different-room path is a separate function (see NOTE at bottom).
"""

import cv2
import numpy as np


# ---- Recoverability classifier thresholds ----
# Validated on IMG_8310 last session (79.4% green / 9.5% yellow / 11.2% red).
# Do not tune without measuring against a new test photo -- see handoff gotcha #1.
WINDOW_SIZE = 25
CLIP_L_MAX = 3           # L <= 3 counts as clipped/blown for local clip fraction
RED_CLIP_FRAC = 0.20
RED_STD_MAX = 6
RED_MEAN_LOCAL_MAX = 35  # load-bearing guard: prevents bright-uniform-surface
                          # misclassification (ceiling) as "no data" (clipped shadow).
                          # Do NOT drop this AND condition.
GREEN_CLIP_FRAC = 0.05
GREEN_STD_MIN = 10
GREEN_MEAN_LOCAL_MIN = 60

GATE_GREEN = 1.0
GATE_YELLOW = 0.5
GATE_RED = 0.0

ILLUM_BLUR_SIGMA = 15
COLOR_BLUR_SIGMA = 15

# Gate feathering sigma. The recoverability gate is a hard 3-level classification
# (0.0/0.5/1.0) computed per-pixel-neighborhood from the Original. Used raw, its
# boundaries trace object silhouettes (e.g. a dark chair in shadow reads "red"
# while the lighter carpet/wall right next to it reads "green"), and multiplying
# that hard edge against a broad illumination lift produces a visible halo/ghost
# rim exactly at the object outline -- confirmed on IMG_8310 dining-chair and
# side-table-shelf test regions this session. Feathering at the same sigma as
# the delta-map blur removes the halo with negligible cost to the aggregate
# stats (chair region median L: 25 raw-gate vs 23 feathered; target was 35).
# This is the same fix class as `feathered float masks replacing hard-boolean
# box exclusion` elsewhere in the pipeline -- same failure mode, same fix.
GATE_FEATHER_SIGMA = 15


def align_oracle_to_original(orig_img, oracle_img, ransac_reproj_threshold=5.0):
    """
    Same-room alignment: resize Oracle to Original's resolution, then refine
    with a RANSAC homography estimated from CLAHE-normalized ORB features.

    CLAHE normalization is used purely to stabilize feature detection across
    the illumination shift between Original and Oracle -- it does not affect
    the pixel data that gets warped and returned.

    Returns:
        oracle_aligned: Oracle image warped into Original's pixel grid, same
            shape as orig_img.
        alignment_report: dict with inlier count, residual stats, and the
            homography matrix, for logging / sanity-checking every run.
    """
    target_h, target_w = orig_img.shape[:2]
    oracle_resized = cv2.resize(
        oracle_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
    )

    orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    oracle_gray = cv2.cvtColor(oracle_resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    orig_eq = clahe.apply(orig_gray)
    oracle_eq = clahe.apply(oracle_gray)

    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(orig_eq, None)
    kp2, des2 = orb.detectAndCompute(oracle_eq, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        raise ValueError(
            "Too few ORB features to align -- this pair may not be a valid "
            "same-room case. Check whether this should route to the "
            "different-room aggregate-style path instead."
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 10:
        raise ValueError(
            f"Only {len(good)} ratio-test matches found -- insufficient for "
            "reliable homography. Do not trust per-pixel deltas from this pair."
        )

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(
        pts1, pts2, cv2.RANSAC, ransacReprojThreshold=ransac_reproj_threshold
    )
    if H is None:
        raise ValueError("Homography estimation failed -- do not trust this pair.")

    inliers = mask.ravel().astype(bool)
    pts1_proj = cv2.perspectiveTransform(pts1[inliers].reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = np.linalg.norm(pts1_proj - pts2[inliers], axis=1)

    alignment_report = {
        "n_matches": len(good),
        "n_inliers": int(inliers.sum()),
        "mean_residual_px": float(resid.mean()) if len(resid) else None,
        "median_residual_px": float(np.median(resid)) if len(resid) else None,
        "max_residual_px": float(resid.max()) if len(resid) else None,
        "homography": H.tolist(),
    }

    # Warp Oracle (already at Original's resolution) into Original's frame.
    oracle_aligned = cv2.warpPerspective(
        oracle_resized, np.linalg.inv(H), (target_w, target_h),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
    )

    return oracle_aligned, alignment_report


def compute_recoverability_map(orig_img):
    """
    Classifies every pixel neighborhood in the ORIGINAL (no reference needed)
    as green/yellow/red for how much correction it can support.

    Returns:
        gate_map: float32 array, same H x W as orig_img, values in {0.0, 0.5, 1.0}
        classification: uint8 array, same H x W, values in {0, 1, 2} for red/yellow/green
            (kept separate from gate_map for logging/visualization use)
    """
    L = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    k = WINDOW_SIZE
    mean_local = cv2.boxFilter(L, ddepth=-1, ksize=(k, k), normalize=True)
    mean_sq_local = cv2.boxFilter(L * L, ddepth=-1, ksize=(k, k), normalize=True)
    var_local = np.clip(mean_sq_local - mean_local ** 2, 0, None)
    std_local = np.sqrt(var_local)

    clipped_mask = (L <= CLIP_L_MAX).astype(np.float32)
    clip_frac_local = cv2.boxFilter(clipped_mask, ddepth=-1, ksize=(k, k), normalize=True)

    red = (clip_frac_local > RED_CLIP_FRAC) | (
        (std_local < RED_STD_MAX) & (mean_local < RED_MEAN_LOCAL_MAX)
    )
    green = (~red) & (clip_frac_local < GREEN_CLIP_FRAC) & (
        (std_local > GREEN_STD_MIN) | (mean_local > GREEN_MEAN_LOCAL_MIN)
    )
    yellow = (~red) & (~green)

    classification = np.zeros(L.shape, dtype=np.uint8)
    classification[yellow] = 1
    classification[green] = 2
    # red stays 0

    gate_map = np.zeros(L.shape, dtype=np.float32)
    gate_map[red] = GATE_RED
    gate_map[yellow] = GATE_YELLOW
    gate_map[green] = GATE_GREEN

    return gate_map, classification


def compute_oracle_guided_deltas(orig_img, oracle_aligned, external_gate=None):
    """
    Computes illumination and color-temperature delta maps between an
    ALREADY-ALIGNED Oracle and the Original. Does not perform alignment --
    call align_oracle_to_original() first for the same-room case.

    external_gate: optional pre-computed float32 gate map (same H x W as
        orig_img, values 0..1), e.g. from
        level2_vision_recoverability.rasterize_vision_gate(). When given,
        this is used (still feathered -- Vision's boxes have hard edges
        too, see that module's docstring) INSTEAD OF the classical
        box-filter classifier. When None (default), falls back to the
        classical classifier unchanged -- see run_oracle_driven_pipeline
        for the shadow-mode logic that decides which to pass in
        production; this function itself has no opinion, it just uses
        what it's given.

    Returns:
        recoverability_map: float32 gate map, feathered, full-res, aligned to orig_img
        recoverability_classification: uint8 map (0/1/2 = red/yellow/green) from the
            CLASSICAL classifier specifically, always computed (cheap, no API call)
            regardless of which gate is actually used -- kept for logging/comparison
            even when external_gate drives the real correction, so shadow-mode
            agreement/disagreement is always loggable.
        illumination_delta: float32, L_oracle - L_original, Gaussian-blurred
        color_delta: float32, b_oracle - b_original (LAB b channel), Gaussian-blurred
    """
    orig_lab = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    oracle_lab = cv2.cvtColor(oracle_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_orig, _, b_orig = cv2.split(orig_lab)
    L_oracle, _, b_oracle = cv2.split(oracle_lab)

    illum_delta_raw = L_oracle - L_orig
    color_delta_raw = b_oracle - b_orig

    illumination_delta = cv2.GaussianBlur(
        illum_delta_raw, ksize=(0, 0), sigmaX=ILLUM_BLUR_SIGMA
    )
    color_delta = cv2.GaussianBlur(
        color_delta_raw, ksize=(0, 0), sigmaX=COLOR_BLUR_SIGMA
    )

    recoverability_map_raw, recoverability_classification = compute_recoverability_map(orig_img)

    if external_gate is not None:
        gate_source_raw = external_gate
    else:
        gate_source_raw = recoverability_map_raw

    # Feather whichever gate is actually being used -- see GATE_FEATHER_SIGMA
    # note above. recoverability_classification (the discrete classical 0/1/2
    # map) is returned unfeathered and un-substituted either way, since it's
    # for logging/comparison, not for multiplying into pixels.
    recoverability_map = cv2.GaussianBlur(
        gate_source_raw, ksize=(0, 0), sigmaX=GATE_FEATHER_SIGMA
    )

    return recoverability_map, recoverability_classification, illumination_delta, color_delta


def apply_oracle_guided_correction(orig_img, recoverability_map, illumination_delta, color_delta):
    """
    Applies Oracle-guided pixel correction to the ORIGINAL, gated by
    recoverability. Runs directly on the raw original -- deliberately bypasses
    the existing Vision-region/protect/do-no-harm pipeline (see handoff Step 3):
    this is an isolated read on what Oracle-guided correction alone produces.

    L_new = L_orig + illumination_delta * gate
    b_new = b_orig + color_delta * gate

    Near-white LAB-gamut caveat (see handoff Step 3 note): at very high target
    L values, achievable chroma compresses. We clip L to [0, 100] and b to LAB's
    valid range, then flag (not silently fix) any region that got clipped so it
    shows up as a candidate washed-out artifact in the render-and-look check.

    Returns:
        corrected_img: BGR uint8 image
        clip_report: dict with fraction of pixels clipped in L and b, for
            logging -- per gotcha #2, always check the actual render, this
            just tells you where to look.
    """
    orig_lab = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_orig, a_orig, b_orig = cv2.split(orig_lab)

    L_new_raw = L_orig + illumination_delta * recoverability_map
    b_new_raw = b_orig + color_delta * recoverability_map

    L_new = np.clip(L_new_raw, 0, 255)
    b_new = np.clip(b_new_raw, 0, 255)

    clip_report = {
        "L_clipped_fraction": float(np.mean(L_new_raw != L_new)),
        "b_clipped_fraction": float(np.mean(b_new_raw != b_new)),
    }

    corrected_lab = cv2.merge([L_new, a_orig, b_new]).astype(np.uint8)
    corrected_img = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    return corrected_img, clip_report


def score_region(img, box, label=""):
    """
    Stat-based scoring for a specific region (e.g. dining chairs, side-table
    shelf) -- median L, mean saturation, mean LAB b. Same methodology as the
    earlier saturation/warmth/brightness work, applied per-region rather than
    whole-frame, per the Step 4 reframe: track known-inconsistent regions
    run-to-run, since the whole-frame average already barely moves either way.

    box: (x1, y1, x2, y2) in pixel coords.
    Returns a dict; also returns the crop so callers can save/inspect it.
    """
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    return {
        "label": label,
        "median_L": float(np.median(L)),
        "mean_saturation": float(sat.mean()),
        "mean_b": float(b.mean()),
    }, crop


# Alignment-quality hard gate, not a soft warning. See handoff: today's
# Oracle images happened to align well by chance; a live per-photo Oracle
# render at production volume is not guaranteed to. These thresholds are
# the same numbers computed by hand during this session's validation
# (190/261 inliers, 2.34px mean residual on a real working pair) --
# codified here so a bad alignment fails LOUD (skip Oracle-guided
# correction, fall back to existing-pipeline-only for that photo, log it
# for review) instead of silently applying a broken delta map. Do not
# loosen these without a specific reason measured on real data, same
# discipline as every other threshold in this pipeline.
MIN_ALIGNMENT_INLIERS = 60
MAX_MEAN_RESIDUAL_PX = 8.0


def run_oracle_driven_pipeline(orig_img, oracle_img, log_prefix="[ORACLE-DRIVEN]",
                                vision_gate_regions=None, vision_report=None):
    """
    Single labeled entry point for the same-room Oracle-driven path:
    align -> (hard alignment-quality check) -> compute deltas -> apply correction.

    Every log line is explicitly prefixed so downstream comparison code and
    file naming can never blur this with the existing Vision-region/protect
    pipeline's output -- see handoff "specific mistake to not repeat".

    vision_gate_regions: optional output of
        level2_vision_recoverability.judge_recoverability()["regions"].
        When given, rasterized via rasterize_vision_gate and used as the
        gate SUBJECT TO SHADOW MODE (see level2_vision_recoverability.SHADOW_MODE):
        in shadow mode (default), the Vision gate is computed, rasterized,
        and its agreement with the classical gate is logged, but the
        CLASSICAL gate still drives the actual correction. Only once
        shadow mode is turned off does the Vision gate actually drive
        pixels. This mirrors level0_scene_classifier.py's own rollout
        exactly -- do not skip this phase just because oracleCorrection.py
        is newer/smaller than level0.
    vision_report: the report dict from judge_recoverability(), for logging
        alongside alignment_report/clip_report. Optional; only meaningful
        if vision_gate_regions is also given.

    Returns:
        corrected_img, report: dict merging alignment_report, clip_report,
            recoverability class percentages, and (if applicable) the
            Vision-gate shadow-mode comparison -- all clearly labeled.
        On a hard alignment failure, corrected_img is None and
        report["skipped"] is True -- callers MUST check this before using
        corrected_img, and should fall back to the existing pipeline's
        output for that photo rather than treat None as "no correction
        needed."
    """
    oracle_aligned, alignment_report = align_oracle_to_original(orig_img, oracle_img)

    alignment_ok = (
        alignment_report["n_inliers"] >= MIN_ALIGNMENT_INLIERS
        and alignment_report["mean_residual_px"] is not None
        and alignment_report["mean_residual_px"] <= MAX_MEAN_RESIDUAL_PX
    )
    if not alignment_ok:
        print(f"{log_prefix} ALIGNMENT FAILED HARD GATE: "
              f"{alignment_report['n_inliers']} inliers (min {MIN_ALIGNMENT_INLIERS}), "
              f"{alignment_report['mean_residual_px']} px mean residual (max {MAX_MEAN_RESIDUAL_PX}) "
              f"-- skipping Oracle-guided correction for this photo, existing pipeline output should be used instead.")
        return None, {
            "source": "oracle_driven_correction",
            "skipped": True,
            "skip_reason": "alignment_below_threshold",
            "alignment": alignment_report,
        }

    external_gate = None
    vision_gate_comparison = None
    if vision_gate_regions is not None:
        from level2_vision_recoverability import rasterize_vision_gate, SHADOW_MODE as VISION_SHADOW_MODE
        vision_gate_raw = rasterize_vision_gate(vision_gate_regions, orig_img.shape)

        if VISION_SHADOW_MODE:
            # Compute classical gate too, purely for comparison logging --
            # classical gate is what will actually be used below.
            classical_raw, _ = compute_recoverability_map(orig_img)
            diff = np.abs(vision_gate_raw - classical_raw)
            vision_gate_comparison = {
                "shadow_mode": True,
                "mean_abs_diff": float(diff.mean()),
                "used_for_correction": "classical",
            }
            print(f"{log_prefix} [VISION-GATE SHADOW MODE] mean |vision_gate - classical_gate| = "
                  f"{vision_gate_comparison['mean_abs_diff']:.3f} (classical still driving correction)")
        else:
            external_gate = vision_gate_raw
            vision_gate_comparison = {"shadow_mode": False, "used_for_correction": "vision"}
            print(f"{log_prefix} [VISION-GATE LIVE] Vision gate driving correction (shadow mode off)")

    recov_map, recov_class, illum_delta, color_delta = compute_oracle_guided_deltas(
        orig_img, oracle_aligned, external_gate=external_gate
    )
    corrected_img, clip_report = apply_oracle_guided_correction(
        orig_img, recov_map, illum_delta, color_delta
    )

    total_px = recov_class.size
    report = {
        "source": "oracle_driven_correction",  # NOT the existing pipeline -- explicit, always
        "skipped": False,
        "alignment": alignment_report,
        "clip": clip_report,
        "gate_source": "vision" if external_gate is not None else "classical",
        "vision_gate_comparison": vision_gate_comparison,
        "vision_report": vision_report,
        "recoverability_pct": {  # always the CLASSICAL classifier's breakdown, for
            # consistent logging regardless of which gate actually drove correction
            "red": 100.0 * (recov_class == 0).sum() / total_px,
            "yellow": 100.0 * (recov_class == 1).sum() / total_px,
            "green": 100.0 * (recov_class == 2).sum() / total_px,
        },
    }
    print(f"{log_prefix} alignment: {alignment_report['n_inliers']}/{alignment_report['n_matches']} "
          f"inliers, mean residual {alignment_report['mean_residual_px']:.2f}px")
    print(f"{log_prefix} gate source: {report['gate_source']}")
    print(f"{log_prefix} classical recoverability: green={report['recoverability_pct']['green']:.1f}% "
          f"yellow={report['recoverability_pct']['yellow']:.1f}% red={report['recoverability_pct']['red']:.1f}%")
    print(f"{log_prefix} clip: L={clip_report['L_clipped_fraction']*100:.2f}% "
          f"b={clip_report['b_clipped_fraction']*100:.2f}%")

    return corrected_img, report


# NOTE: different-room aggregate-style path is NOT implemented here.
# Per handoff Step 2, that case needs a separate mechanism entirely
# (global median-L / saturation / warmth calibration, no spatial diff).
# Build only once a different-room test photo is actually in hand --
# don't speculatively build against no data.
