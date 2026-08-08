"""
oracleCorrection.py

Oracle-driven correction pass for Smart Correct Level 2.

IMPORTANT — labeling requirement: any Oracle Scene Render (including any
GPT Image 2 output) is a digitally altered image and must be labeled as
such wherever it is shown or delivered. That labeling requirement is
what makes this compliant -- not a question still being evaluated.
Nothing in this module needs to change based on that; it's a delivery/UI
requirement on whatever surfaces Oracle-derived images, not on the
math here.

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
    Computes illumination and color delta maps between an ALREADY-ALIGNED
    Oracle and the Original. Does not perform alignment -- call
    align_oracle_to_original() first for the same-room case.

    ALL THREE LAB channels now computed (fixed this session -- see
    apply_oracle_guided_correction docstring for why leaving 'a' fixed at
    the Original's value was a real bug, not a simplification: confirmed
    on a real exterior photo, Oracle's own 'a' channel differed from the
    Original's by +7 in the sky region, and combining Original's 'a' with
    Oracle's L/b produced a LAB coordinate that belongs to NEITHER image,
    converting to an HSV saturation that overshot Oracle's own saturation
    (198 vs Oracle's 150 vs Original's 68). Moving all three channels
    together, then clamping (see apply_) is the fix.

    external_gate: optional pre-computed float32 gate map (same H x W as
        orig_img, values 0..1), e.g. from
        level2_vision_recoverability.rasterize_vision_gate(). When given,
        used (still feathered) INSTEAD OF the classical box-filter
        classifier. When None (default), falls back to the classical
        classifier unchanged.

    Returns:
        recoverability_map: float32 gate map, feathered, full-res, aligned to orig_img
        recoverability_classification: uint8 map (0/1/2 = red/yellow/green) from the
            CLASSICAL classifier specifically, always computed regardless of gate source.
        illumination_delta: float32, L_oracle - L_original, Gaussian-blurred
        color_delta_a: float32, a_oracle - a_original, Gaussian-blurred (NEW)
        color_delta_b: float32, b_oracle - b_original, Gaussian-blurred
    """
    orig_lab = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    oracle_lab = cv2.cvtColor(oracle_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_orig, a_orig, b_orig = cv2.split(orig_lab)
    L_oracle, a_oracle, b_oracle = cv2.split(oracle_lab)

    illum_delta_raw = L_oracle - L_orig
    color_delta_a_raw = a_oracle - a_orig
    color_delta_b_raw = b_oracle - b_orig

    illumination_delta = cv2.GaussianBlur(illum_delta_raw, ksize=(0, 0), sigmaX=ILLUM_BLUR_SIGMA)
    color_delta_a = cv2.GaussianBlur(color_delta_a_raw, ksize=(0, 0), sigmaX=COLOR_BLUR_SIGMA)
    color_delta_b = cv2.GaussianBlur(color_delta_b_raw, ksize=(0, 0), sigmaX=COLOR_BLUR_SIGMA)

    recoverability_map_raw, recoverability_classification = compute_recoverability_map(orig_img)
    gate_source_raw = external_gate if external_gate is not None else recoverability_map_raw
    recoverability_map = cv2.GaussianBlur(gate_source_raw, ksize=(0, 0), sigmaX=GATE_FEATHER_SIGMA)

    return recoverability_map, recoverability_classification, illumination_delta, color_delta_a, color_delta_b


def apply_oracle_guided_correction(orig_img, oracle_aligned, recoverability_map,
                                    illumination_delta, color_delta_a, color_delta_b):
    """
    Applies Oracle-guided pixel correction to the ORIGINAL, gated by
    recoverability. Runs directly on the raw original -- deliberately bypasses
    the existing Vision-region/protect pipeline entirely for this test.

    FIXED THIS SESSION -- two real bugs, found on a real exterior photo test
    (not theoretical): the corrected sky ended up MORE saturated (HSV S=198)
    than even Oracle itself (S=150), against an Original of S=68.

    Bug 1: only L and b were being moved; 'a' stayed fixed at the Original's
    value. Confirmed Oracle's own 'a' differs from Original's by a real
    amount (+7 in the sky test case) -- combining Original's 'a' with
    Oracle's L/b produces a LAB coordinate that belongs to NEITHER image.
    Fixed by moving all three channels together, same gate.

    Bug 2 (the actual overshoot-prevention fix, kept even after fixing bug 1,
    since gate<1.0 blending plus feathering can still occasionally produce a
    per-pixel value outside either endpoint): every channel is now hard-
    clamped, per-pixel, to [min(orig, oracle), max(orig, oracle)]. This is
    the direct implementation of "the corrected image should never look like
    a DIFFERENT captured moment than either the Original or Oracle" -- it
    makes an overshoot like the one found this session structurally
    impossible, not just less likely.

    oracle_aligned: needed now (wasn't before) so the clamp bounds can be
    computed per-pixel against Oracle's actual LAB values, not just its delta.

    Returns:
        corrected_img: BGR uint8 image
        clip_report: dict with fraction of pixels clamped in each channel,
            for logging -- per gotcha #2, always check the actual render,
            this just tells you where to look. NOTE: "clipped" here now
            means "hit the [min(orig,oracle),max(orig,oracle)] clamp",
            not the old 0-255 clamp -- a meaningfully more informative
            signal (it tells you the raw math wanted to go somewhere
            neither image supports, not just that it hit format limits).
    """
    orig_lab = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    oracle_lab = cv2.cvtColor(oracle_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_orig, a_orig, b_orig = cv2.split(orig_lab)
    L_oracle, a_oracle, b_oracle = cv2.split(oracle_lab)

    L_new_raw = L_orig + illumination_delta * recoverability_map
    a_new_raw = a_orig + color_delta_a * recoverability_map
    b_new_raw = b_orig + color_delta_b * recoverability_map

    def clamp_to_span(new_raw, orig_ch, oracle_ch):
        lo = np.minimum(orig_ch, oracle_ch)
        hi = np.maximum(orig_ch, oracle_ch)
        clamped = np.clip(new_raw, lo, hi)
        clamped = np.clip(clamped, 0, 255)  # also enforce valid image range
        return clamped, float(np.mean(new_raw != clamped))

    L_new, L_clipped_frac = clamp_to_span(L_new_raw, L_orig, L_oracle)
    a_new, a_clipped_frac = clamp_to_span(a_new_raw, a_orig, a_oracle)
    b_new, b_clipped_frac = clamp_to_span(b_new_raw, b_orig, b_oracle)

    clip_report = {
        "L_clipped_fraction": L_clipped_frac,
        "a_clipped_fraction": a_clipped_frac,
        "b_clipped_fraction": b_clipped_frac,
    }

    corrected_lab = cv2.merge([L_new, a_new, b_new]).astype(np.uint8)
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
                                vision_gate_regions=None, vision_report=None,
                                wall_trim_grid=None, ceiling_grid=None):
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

    wall_trim_grid: optional ["grid"] output of
        level2_wall_trim_mask.identify_wall_trim(). ceiling_grid: optional
        ["grid"] output of level2_ceiling_mask.identify_ceiling(). When
        either is given, rasterized and UNIONed into a single
        architectural-surface mask (see apply_hue_fidelity_gate and
        level2_wall_trim_mask's module docstring for why ceiling is a
        separate Vision call unioned in here, not part of wall_trim's own
        category). SHADOW MODE, same posture as vision_gate_regions and
        every other Vision gate in this pipeline: this function computes
        the mask and logs its coverage/would-be effect
        (a_reverted_fraction / b_reverted_fraction from
        apply_hue_fidelity_gate, run but NOT applied to corrected_img) --
        it does NOT call apply_hue_fidelity_gate on the returned
        corrected_img yet. Callers wanting the gate LIVE must call
        apply_hue_fidelity_gate explicitly themselves on this function's
        output, same as apply_saturation_cap and smooth_deltas_in_mask
        are already caller-invoked rather than auto-applied here. This
        keeps the rollout discipline identical across every gate in this
        module: compute, log, review a real batch, THEN wire live -- do
        not skip straight to auto-applying just because the shadow-mode
        plumbing for vision_gate_regions already exists above.

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

    recov_map, recov_class, illum_delta, color_delta_a, color_delta_b = compute_oracle_guided_deltas(
        orig_img, oracle_aligned, external_gate=external_gate
    )
    corrected_img, clip_report = apply_oracle_guided_correction(
        orig_img, oracle_aligned, recov_map, illum_delta, color_delta_a, color_delta_b
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

    # --- Hue-fidelity gate: SHADOW MODE ONLY, same posture as the Vision
    # recoverability gate above. Computes the architectural mask and runs
    # apply_hue_fidelity_gate to find out what IT WOULD DO, logs that,
    # and discards the result -- corrected_img returned below is always
    # the pre-gate version until this is explicitly wired live by a
    # caller. See run_oracle_driven_pipeline's docstring for why this
    # stays shadow-only here even though the plumbing exists.
    hue_gate_shadow = None
    if wall_trim_grid is not None or ceiling_grid is not None:
        from level2_wall_trim_mask import rasterize_wall_trim_mask, GRID_COLS as WALL_TRIM_GRID_COLS
        wall_mask = (
            rasterize_wall_trim_mask(wall_trim_grid, orig_img.shape)
            if wall_trim_grid is not None
            else np.zeros(orig_img.shape[:2], dtype=np.float32)
        )
        ceiling_mask_for_hue = np.zeros(orig_img.shape[:2], dtype=np.float32)
        if ceiling_grid is not None:
            try:
                from level2_ceiling_mask import rasterize_ceiling_mask
                ceiling_mask_for_hue = rasterize_ceiling_mask(ceiling_grid, orig_img.shape)
            except ImportError:
                print(f"{log_prefix} level2_ceiling_mask not importable, "
                      f"hue-gate shadow test proceeding with wall/trim only")

        architectural_mask = np.clip(wall_mask + ceiling_mask_for_hue, 0.0, 1.0)

        _shadow_result, hue_gate_shadow = apply_hue_fidelity_gate(
            orig_img, corrected_img, architectural_mask, mask_grid_cols=WALL_TRIM_GRID_COLS
        )
        # _shadow_result deliberately discarded -- shadow mode never
        # changes what this function returns.
        print(f"{log_prefix} [HUE-GATE SHADOW MODE] architectural coverage="
              f"{hue_gate_shadow.get('masked_area_fraction', 0.0)*100:.1f}%  "
              f"would-revert a={hue_gate_shadow.get('a_reverted_fraction', 0.0)*100:.1f}% "
              f"b={hue_gate_shadow.get('b_reverted_fraction', 0.0)*100:.1f}% "
              f"(NOT applied -- shadow mode, corrected_img unchanged)")

    report["hue_gate_shadow"] = hue_gate_shadow

    print(f"{log_prefix} alignment: {alignment_report['n_inliers']}/{alignment_report['n_matches']} "
          f"inliers, mean residual {alignment_report['mean_residual_px']:.2f}px")
    print(f"{log_prefix} gate source: {report['gate_source']}")
    print(f"{log_prefix} classical recoverability: green={report['recoverability_pct']['green']:.1f}% "
          f"yellow={report['recoverability_pct']['yellow']:.1f}% red={report['recoverability_pct']['red']:.1f}%")
    print(f"{log_prefix} clip: L={clip_report['L_clipped_fraction']*100:.2f}% "
          f"b={clip_report['b_clipped_fraction']*100:.2f}%")

    return corrected_img, report


def apply_saturation_cap(orig_img, corrected_img, restricted_mask, max_increase_ratio=1.15,
                          feather_sigma=None, mask_grid_cols=None):
    """
    Scoped saturation ceiling -- applies ONLY inside restricted_mask
    (e.g. from level2_sky_grass_mask.rasterize_sky_veg_mask), not
    globally. See conversation: a global cap would recreate the exact
    over-defensive "Protect" failure mode found earlier this session (the
    wall-seam/column-wedge artifacts) by restricting correction on content
    -- wood tone, fabric, a genuinely color-cast wall -- that was never the
    actual risk and that the governing Oracle prompts explicitly permit
    correcting. The risk this addresses is specifically named in the
    exterior prompt (Weather/Sky character/Cloud formations under Identity
    Preservation; the standalone Vegetation section) -- nowhere else.

    max_increase_ratio: corrected HSV saturation, inside the masked region,
        is capped at this multiple of the ORIGINAL's saturation at that
        pixel. Default 1.15 (+15%) is a starting point, not a measured
        constant -- there is no real-batch data behind this number yet.
        Revisit once a batch of real exterior photos has been reviewed.

    This is independent of Oracle's own quality -- unlike the
    [min(orig,oracle),max(orig,oracle)] clamp added earlier this session
    (which stops the correction from exceeding Oracle but does nothing if
    Oracle itself is already too saturated), this cap is measured against
    the ORIGINAL only, so it still holds even when a future Oracle render
    is flawed in a way nobody caught ahead of time.

    feather_sigma: if None, DERIVED from mask_grid_cols when given (see
        below), else falls back to GATE_FEATHER_SIGMA. NOT safe to leave
        at a small fixed constant when the mask came from a coarse grid --
        found live on a real test: a single wrong grid cell (24-wide grid,
        ~238px cells on a real photo) left a hard, clearly visible blocky
        patch, because a 15px feather cannot meaningfully soften a 238px
        cell against its neighbors. A bbox-sourced mask and a grid-sourced
        mask need different feather scales; this isn't a one-time tuning
        fix, it's a structural fact about the mask's own resolution.

    mask_grid_cols: if restricted_mask came from a GRID_COLS x GRID_ROWS
        classification (level2_sky_grass_mask), pass GRID_COLS here
        so feather_sigma can be derived proportionally to actual cell size
        (half a cell width) rather than guessed. This makes an isolated
        misclassified cell blend into its neighbors instead of showing as
        a hard block -- the correct response to an occasional wrong cell
        is for it to fade in gracefully, not to trust every cell as
        precisely correct.

    Returns: corrected BGR image with the cap applied, and a report dict.
    """
    if feather_sigma is None:
        if mask_grid_cols:
            cell_width_px = restricted_mask.shape[1] / float(mask_grid_cols)
            feather_sigma = cell_width_px / 2.0
        else:
            feather_sigma = GATE_FEATHER_SIGMA

    mask = cv2.GaussianBlur(restricted_mask.astype(np.float32), ksize=(0, 0), sigmaX=feather_sigma)
    mask = np.clip(mask, 0.0, 1.0)

    if mask.max() < 1e-6:
        # No sky/vegetation regions given (module disabled, call failed, or
        # genuinely none in this photo) -- apply no cap at all, per this
        # module's "empty regions = apply no cap" contract. Do NOT
        # interpret an empty mask as "cap everything" or "cap nothing
        # differently than before" -- it must be a true no-op.
        return corrected_img, {"applied": False, "reason": "empty_mask", "pixels_capped_fraction": 0.0}

    orig_hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    corr_hsv = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2HSV).astype(np.float32)

    orig_s = orig_hsv[:, :, 1]
    corr_s = corr_hsv[:, :, 1]
    ceiling = orig_s * max_increase_ratio

    over_ceiling = corr_s > ceiling
    # Blend toward the ceiling by mask strength rather than hard-clip, so a
    # partially-confident mask edge doesn't produce a visible saturation
    # step -- consistent with every other gate in this pipeline being a
    # continuous blend, not a hard switch.
    capped_s = corr_s * (1 - mask) + np.minimum(corr_s, ceiling) * mask

    pixels_capped_fraction = float((over_ceiling & (mask > 0.5)).mean())

    new_hsv = corr_hsv.copy()
    new_hsv[:, :, 1] = capped_s
    new_hsv = np.clip(new_hsv, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)

    return result, {
        "applied": True,
        "max_increase_ratio": max_increase_ratio,
        "pixels_capped_fraction": pixels_capped_fraction,
    }


def apply_hue_fidelity_gate(orig_img, corrected_img, architectural_mask,
                             feather_sigma=None, mask_grid_cols=None):
    """
    Scoped hue-fidelity constraint -- applies ONLY inside architectural_mask
    (e.g. Vision-identified walls/ceiling/trim, same rasterization pattern
    as level2_sky_grass_mask / level2_ceiling_mask), not globally.

    PROBLEM THIS FIXES: apply_oracle_guided_correction() trusts Oracle's a/b
    (color) channels exactly as much as its L (brightness) channel, clamped
    only to [min(orig,oracle), max(orig,oracle)] -- a magnitude bound, not
    a direction bound. Confirmed on a real photo (IMG_8310, dining/living
    room): a neutral gray wall's Oracle render carried a real hue-family
    shift, not just a brightness lift -- LAB a went from +1.9 (orig) to
    -1.0 (Oracle) on one wall region and -2.3 to -3.9 on another; b went
    from +9.8 to +16.2 and from -4.6 to +1.4 -- crossing from cool to warm
    outright on the second region. The existing clamp does nothing here:
    it only bounds how far a pixel can move, not whether Oracle's own
    endpoint value is itself correct, and a globally-biased Oracle render
    stays fully "in bounds" of its own bad value.

    ROOT CAUSE, why this is scoped and not global: this is a known
    generative-rendering failure mode specifically on underexposed rooms --
    weak real signal in a dark room gives Flux more room to default toward
    a statistically "plausible" neutral color rather than the actual
    captured one (a same-session photo of a well-lit room, same style of
    region, showed no measurable a/b drift -- the failure tracks how dark
    the original region was, not a universal Oracle bias). A global hue
    lock would recreate the exact over-defensive "Protect" failure this
    pipeline already moved away from once -- wood tone, fabric, and
    genuinely color-cast walls are explicitly ALLOWED to shift hue as
    part of legitimate cast removal (see oracleGeneration.py's Color
    Temperature -- Independent of Brightness section). This gate is
    deliberately scoped to Vision-identified architectural/neutral
    surfaces only (walls, ceiling, trim) -- the same category of surface
    the governing prompt already treats as "must hold true captured
    color," never furniture, fabric, or wood.

    WHAT "toward neutral" MEANS, per-pixel per-channel (a and b handled
    independently): Oracle's move is accepted only if it reduces the
    channel's distance from true zero (LAB neutral) without crossing zero
    -- i.e. genuine color-cast neutralization. Any move that INCREASES
    magnitude or FLIPS SIGN is rejected outright for that pixel/channel,
    and the already-corrected image's Original value is kept instead.
    This deliberately does NOT touch L -- brightness recovery is
    untouched by this gate, still fully driven by Oracle via
    apply_oracle_guided_correction. Only the already-corrected image's
    a/b channels, inside the mask, are re-evaluated against this rule.

    Prototyped against the real IMG_8310 wall crops before being written
    here: with this rule applied, both wall regions kept L at Oracle's
    full target (71/75, unchanged) while a/b returned to within ~1 LAB
    unit of the Original's true near-neutral values, instead of the
    uncorrected pipeline's +16.2b / sign-flipped-to-warm result. NOT yet
    re-run through the live pipeline on a real batch -- a hand-computed
    prototype on two crops is not the same as confirming this end-to-end.

    architectural_mask: raw (unfeathered) float32/bool mask, same H x W
        as orig_img, from a Vision-identified walls/ceiling/trim
        classifier -- NOT YET BUILT as of this function. Follow the same
        grid-based pattern as level2_sky_grass_mask.py /
        level2_ceiling_mask.py rather than a bbox approach, per this
        module's established reasoning for why bbox failed twice on real
        photos. Until that classifier exists, this function is dead code
        with no caller -- see run_oracle_driven_pipeline, which does NOT
        call this yet, same as apply_saturation_cap and
        smooth_deltas_in_mask are also caller-invoked, not auto-wired.
        An empty/all-zero mask (module not yet wired in, disabled, or
        genuinely no architectural surface detected) is a true no-op --
        returns corrected_img unchanged, same "empty mask = apply
        nothing" contract as apply_saturation_cap and
        smooth_deltas_in_mask.

    feather_sigma / mask_grid_cols: same meaning and same derivation
        logic as apply_saturation_cap -- see that docstring, not
        duplicated here. Same failure mode (hard mask edge -> visible
        seam), same fix (feather proportional to source grid resolution).

    Returns: corrected BGR image with the gate applied, and a report dict
        with the fraction of masked pixels where Oracle's a/b move was
        rejected and reverted toward Original -- always check the actual
        render, this just tells you where to look, same as every other
        report dict in this module.
    """
    if feather_sigma is None:
        if mask_grid_cols:
            cell_width_px = architectural_mask.shape[1] / float(mask_grid_cols)
            feather_sigma = cell_width_px / 2.0
        else:
            feather_sigma = GATE_FEATHER_SIGMA

    mask = cv2.GaussianBlur(architectural_mask.astype(np.float32), ksize=(0, 0), sigmaX=feather_sigma)
    mask = np.clip(mask, 0.0, 1.0)

    if mask.max() < 1e-6:
        return corrected_img, {
            "applied": False, "reason": "empty_mask",
            "a_reverted_fraction": 0.0, "b_reverted_fraction": 0.0,
        }

    orig_lab = cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    corr_lab = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_corr, a_corr, b_corr = cv2.split(corr_lab)
    _, a_orig, b_orig = cv2.split(orig_lab)

    # OpenCV stores LAB a/b as 0-255 with 128 as the neutral (zero) point.
    # Recenter to signed values so "toward zero" and "sign flip" mean the
    # actual neutral-axis crossing, not an artifact of the 0-255 encoding.
    a_orig_s = a_orig - 128.0
    b_orig_s = b_orig - 128.0
    a_corr_s = a_corr - 128.0
    b_corr_s = b_corr - 128.0

    def constrain_toward_neutral(orig_s, corr_s):
        same_sign = np.sign(orig_s) == np.sign(corr_s)
        shrank = np.abs(corr_s) <= np.abs(orig_s)
        allowed = same_sign & shrank
        constrained_s = np.where(allowed, corr_s, orig_s)
        reverted_fraction = float((~allowed).mean())
        return constrained_s, reverted_fraction

    a_constrained_s, a_reverted_frac = constrain_toward_neutral(a_orig_s, a_corr_s)
    b_constrained_s, b_reverted_frac = constrain_toward_neutral(b_orig_s, b_corr_s)

    # Blend by mask strength, same reasoning as apply_saturation_cap -- a
    # partially-confident mask edge should not produce a visible seam.
    a_new_s = a_corr_s * (1 - mask) + a_constrained_s * mask
    b_new_s = b_corr_s * (1 - mask) + b_constrained_s * mask

    a_new = np.clip(a_new_s + 128.0, 0, 255)
    b_new = np.clip(b_new_s + 128.0, 0, 255)

    new_lab = cv2.merge([L_corr, a_new, b_new]).astype(np.uint8)
    result = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

    # Reverted fractions are computed over the WHOLE frame for simplicity;
    # they understate the effective in-mask impact where mask < 1. Cross-
    # reference against masked_area_fraction if an in-mask-only rate is
    # needed for logging.
    return result, {
        "applied": True,
        "a_reverted_fraction": a_reverted_frac,
        "b_reverted_fraction": b_reverted_frac,
        "masked_area_fraction": float(mask.mean()),
    }


def smooth_deltas_in_mask(illum_delta, color_delta_a, color_delta_b, ceiling_mask,
                           extra_blur_sigma=None, min_blur_sigma=40.0):
    """
    Re-blurs the three Oracle delta maps at a MUCH larger sigma than the
    pipeline's default (ILLUM_BLUR_SIGMA/COLOR_BLUR_SIGMA=15), scoped to
    ceiling_mask only, then blended back in proportional to mask
    confidence. Built for a real, confirmed-repeatable artifact: Flux
    Kontext's own interior prompt explicitly asks it to preserve
    "believable luminance gradients" on the ceiling (deliberately, so it
    doesn't render a flat, sterile ceiling) -- but this pipeline uses
    Oracle purely as a source of PER-PIXEL DELTAS against the Original,
    not as a standalone image. Flux's own invented small-scale ceiling
    variation doesn't spatially match the Original's real lighting
    pattern, so subtracting the two directly produces visible blotchy,
    uneven correction on what should read as one continuous painted
    plane -- confirmed on two separate real photos across two separate
    sessions, not a one-off render artifact.

    NOT fixed by touching the Oracle prompt: that would reintroduce the
    exact "Protect"-style over-restraint this pipeline already moved
    away from once (a deliberately flat ceiling looks cheap/fake as a
    standalone render). The fix belongs at the delta-APPLICATION stage,
    where the pipeline decides what to trust from Oracle's render, not
    at generation -- Flux stays free to render however it wants.

    WHY THIS IS A DIFFERENT SIGMA DERIVATION THAN apply_saturation_cap's
    mask_grid_cols-based feather_sigma: that function softens the EDGE
    of a mask (roughly cell-width scale) so a grid boundary doesn't show
    as a hard step. This function needs the opposite problem solved --
    heavy smoothing across the WHOLE INTERIOR of a large surface, to
    average out Oracle's own invented small-scale noise while preserving
    the surface's real, coarse gradient (brighter near a fixture, dimmer
    at a far corner). The right scale for that is proportional to the
    surface's own measured size, not the grid resolution used to detect
    it.

    ceiling_mask: raw (unfeathered) float32/bool mask, same H x W as the
        delta maps, from level2_ceiling_mask.rasterize_ceiling_mask(). An
        empty/all-zero mask (module disabled, call failed, or genuinely
        no ceiling in this photo) is a true no-op -- returns the input
        deltas unchanged, per this module's "empty mask = apply nothing"
        contract, same as apply_saturation_cap.

    extra_blur_sigma: if None, derived from the mask's own measured
        extent (sqrt of masked pixel area) rather than a fixed constant
        -- a ceiling spanning most of a wide-angle frame needs much
        heavier smoothing than a small ceiling sliver in a tight shot.
        This is a simple, defensible proxy for "how big is this
        surface," not a measured constant -- revisit once a real batch
        shows it over- or under-smoothing, same discipline as every
        other unmeasured constant in this pipeline (see
        apply_saturation_cap's max_increase_ratio).

    min_blur_sigma: floor so a small/fragmented ceiling detection doesn't
        get a near-zero derived sigma and effectively skip smoothing --
        the entire premise here is "ceilings need MORE smoothing than
        the pipeline's default," not "scale it down to nothing for small
        ones." 40.0 (roughly 2.5x the pipeline's own ILLUM_BLUR_SIGMA) is
        a starting floor, not a measured constant.

    Returns (illum_delta_out, color_delta_a_out, color_delta_b_out, report).
    On an empty mask, the three deltas are returned UNCHANGED (same
    object references) and report["applied"] is False.
    """
    mask = np.clip(ceiling_mask.astype(np.float32), 0.0, 1.0)

    if mask.max() < 1e-6:
        return illum_delta, color_delta_a, color_delta_b, {
            "applied": False, "reason": "empty_mask", "extra_blur_sigma": None,
        }

    if extra_blur_sigma is None:
        masked_area_px = float(mask.sum())
        extra_blur_sigma = max(min_blur_sigma, np.sqrt(masked_area_px) * 0.5)

    # Feather the mask itself too, same reasoning as every other gate in
    # this pipeline (apply_saturation_cap, GATE_FEATHER_SIGMA) -- a hard
    # mask boundary would otherwise produce a visible seam where heavily-
    # smoothed ceiling delta meets normally-smoothed wall delta right at
    # the edge.
    feathered_mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=GATE_FEATHER_SIGMA)
    feathered_mask = np.clip(feathered_mask, 0.0, 1.0)

    def _blend(delta):
        heavily_smoothed = cv2.GaussianBlur(delta, ksize=(0, 0), sigmaX=extra_blur_sigma)
        return delta * (1 - feathered_mask) + heavily_smoothed * feathered_mask

    illum_out = _blend(illum_delta)
    a_out = _blend(color_delta_a)
    b_out = _blend(color_delta_b)

    return illum_out, a_out, b_out, {
        "applied": True,
        "extra_blur_sigma": float(extra_blur_sigma),
        "masked_area_fraction": float(mask.mean()),
    }


# NOTE: different-room aggregate-style path is NOT implemented here.
# Per handoff Step 2, that case needs a separate mechanism entirely
# (global median-L / saturation / warmth calibration, no spatial diff).
# Build only once a different-room test photo is actually in hand --
# don't speculatively build against no data.
