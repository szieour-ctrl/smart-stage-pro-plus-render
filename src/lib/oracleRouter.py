#!/usr/bin/env python3
"""
oracleRouter.py

Top-level entry point for Smart Correct with Oracle-driven correction --
same CLI contract as smartCorrect.py (--source, --output, JSON on stdout,
error JSON on stderr + exit 1 on failure), so correctPipeline.js's existing
spawn("python3", [CORRECT_PY, "--source", ..., "--output", ...]) needs to
point at this file instead of smartCorrect.py directly -- no other change
needed on the Node side for the CLI contract itself. (Whether the async/
sync shape around that spawn call still fits is a separate question --
see oracleGeneration.py's docstring on expected per-image latency now that
Flux Kontext is the generation backend, and re-confirm against real
measured latency before assuming the existing synchronous route handles
this without change.)

ROUTING DECISION (this session, confirmed against real data, not just
architecture reasoning): Oracle-driven correction is INTERIOR ONLY.
Exterior photos route directly to smartCorrect.py's own classical CV path
-- no Level 0 Vision call spent on this file's own account, no Flux
generation call, no Oracle pipeline at all for exteriors, ever.

Why, concretely: tested head-to-head on a real exterior photo this
session -- smartCorrect.py's own classical correction produced a larger,
more useful lift (house facade +7.2 L, lawn +6.3 L) than the full
Oracle-driven pipeline did on the SAME photo (+0.5 L, -0.1 L --
functionally a no-op past a modest, capped sky adjustment), while costing
zero extra API calls and carrying none of the foliage-alignment risk
documented earlier this session (vegetation genuinely doesn't align
rigidly between two separate images the way furniture does).

Structural reason this generalizes, not just true of one test photo:
Oracle-driven correction exists to solve severe DYNAMIC RANGE COMPRESSION
from multiple competing light sources in one frame -- a blown-out window
next to a shadowed room, warm lamp light fighting cool daylight through
glass. That is an interior-specific failure mode. A daylight exterior is
usually lit by one dominant source (the sun), so the specific problem
this whole pipeline exists to solve rarely occurs there. Real exceptions
exist (deep shade, strong backlighting) -- rare, not impossible, which is
why this is a routing DEFAULT, not a hard restriction; if a real batch
later shows exteriors that would clearly benefit, that's a reason to
revisit this file's routing logic specifically, not the whole pipeline.

SCENE CLASSIFICATION: uses level0_scene_classifier.classify_scene()
directly (the raw Vision call), not the full resolve_scene_type() (which
additionally blends in an HSV heuristic that lives inside smartCorrect.py
itself and isn't duplicated here). When Vision is unavailable or
unconfident (sceneType is None), this router defaults to the classical
CV path -- safe by construction, since smartCorrect.py re-derives scene
type internally anyway (HSV heuristic + its own Level 0 call) regardless
of what happens here, so falling back to it is never a dead end, just a
missed opportunity to route an interior photo to the richer pipeline for
this one image.

FAILURE HANDLING: any failure at any Oracle-pipeline step (generation,
alignment-quality gate, gate computation, correction) falls back to
smartCorrect.py's own output for that image, never a hard error and
never a half-finished Oracle result shipped as final. The Oracle path is
strictly additive -- an image this file mishandles should never come out
worse than the classical-only path already in production today.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from level0_scene_classifier import classify_scene  # noqa: E402
from oracleGeneration import generate_oracle_scene  # noqa: E402
from oracleCorrection import (  # noqa: E402
    align_oracle_to_original,
    compute_oracle_guided_deltas,
    apply_oracle_guided_correction,
    apply_hue_fidelity_gate,
    apply_wall_color_anchor,
    apply_illumination_floor,
    apply_global_hue_angle_gate,
    compute_recoverability_map,
    smooth_deltas_in_mask,
    MIN_ALIGNMENT_INLIERS,
    MAX_MEAN_RESIDUAL_PX,
)
from level2_vision_recoverability import (  # noqa: E402
    judge_recoverability,
    rasterize_vision_gate,
    SHADOW_MODE as RECOVERABILITY_SHADOW_MODE,
)
from level2_ceiling_mask import (  # noqa: E402
    identify_ceiling,
    rasterize_ceiling_mask,
    SHADOW_MODE as CEILING_MASK_SHADOW_MODE,
)
from level2_wall_trim_mask import (  # noqa: E402
    identify_wall_trim,
    rasterize_wall_trim_mask,
    GRID_COLS as WALL_TRIM_GRID_COLS,
    SHADOW_MODE as WALL_TRIM_MASK_SHADOW_MODE,
)

SMARTCORRECT_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smartCorrect.py")

# ILLUMINATION FLOOR -- oracleCorrection.apply_illumination_floor is classical
# CV (no Vision call), so unlike the wall/trim, ceiling, and recoverability
# gates it has no level2_ module of its own to own this constant -- defined
# here instead. Was previously computed shadow-only and discarded inside
# oracleCorrection.run_oracle_driven_pipeline() itself (that function is
# deliberately never called by this router -- see run_oracle_interior's own
# docstring) and was explicitly NOT wired into this router when the
# hue-fidelity gate was added, per that patch's own comment: "a separate,
# not-yet-revalidated piece." Wired in live (default apply, not shadow-first)
# now that ceiling_smoothing and the hue-fidelity gate are both confirmed
# firing correctly on real photos (IMG_8291, IMG_8310) but walls/ceiling
# still weren't reaching the MLS Bright brightness target -- this is the
# missing piece, not a replacement for either of those. Kill switch:
# ORACLE_ILLUM_FLOOR_SHADOW_MODE=true reverts to compute-and-log-only
# without a code change, same discipline as every other gate here.
ILLUM_FLOOR_SHADOW_MODE = os.environ.get(
    "ORACLE_ILLUM_FLOOR_SHADOW_MODE", "false"
).lower() not in ("false", "0", "")

# GLOBAL HUE ANGLE GATE -- oracleCorrection.apply_global_hue_angle_gate was
# written, and its docstring cites a real measured hue-family drift on
# IMG_8310 as the reason it exists, but it was never imported or called
# anywhere in this codebase (confirmed by grep across oracleCorrection.py
# and this file -- Aug 2026). Unlike hue_fidelity_gate/illumination_floor
# above, this gate has NO Vision dependency at all -- no ceiling_mask, no
# wall_trim_mask, whole-frame, classical LAB math only -- so it is not
# subject to ceiling_mask's 0%-real-batch-pass-rate problem those two
# gates are currently starved by. Wiring in now, live by default, as the
# final step of correction: it runs after every other step (Vision-gated
# or not) so it can catch whatever hue drift survives everything above it,
# same as its own docstring's intended role as a whole-frame backstop.
# Kill switch for consistency with every other gate in this router.
GLOBAL_HUE_GATE_SHADOW_MODE = os.environ.get(
    "ORACLE_GLOBAL_HUE_GATE_SHADOW_MODE", "false"
).lower() not in ("false", "0", "")


def run_classical_only(source_path: str, output_path: str, lens_mode: str, intensity: float) -> dict:
    """
    Shells out to the existing, unmodified smartCorrect.py -- same
    subprocess pattern correctPipeline.js's own correctOneImage() already
    uses, ported to Python since this router is itself a CLI entry point
    rather than a Node caller. This is the exterior default AND the
    fallback for any interior photo where the Oracle pipeline fails at
    any step.

    Returns the parsed JSON dict smartCorrect.py prints to stdout, with
    "oracleRouting" added so callers can always tell which path actually
    produced a given result -- never leave that ambiguous, same reasoning
    as oracleCorrection.py's "source" field on every report it returns.
    """
    proc = subprocess.run(
        ["python3", SMARTCORRECT_PY, "--source", source_path, "--output", output_path,
         "--lens-mode", lens_mode, "--intensity", str(intensity)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"smartCorrect.py failed (exit {proc.returncode}): {proc.stderr[:500]}")
    result = json.loads(proc.stdout.strip())
    result["oracleRouting"] = {"path": "classical_only", "reason": None}
    return result


def run_oracle_interior(source_path: str, output_path: str, orig_img) -> dict:
    """
    The full interior Oracle-driven path: generate -> align (hard gate) ->
    Vision recoverability gate (shadow-mode aware) -> correct. Raises on
    any failure -- caller (main()) is responsible for catching this and
    falling back to run_classical_only(), per this module's stated
    "never ship a half-finished Oracle result" contract.
    """
    routing_report = {"path": "oracle_interior", "steps": {}}

    ok, buf = cv2.imencode(".png", orig_img)
    if not ok:
        raise RuntimeError("failed to encode original image for Oracle generation")

    oracle_bytes, gen_report = generate_oracle_scene(buf.tobytes(), is_exterior=False)
    routing_report["steps"]["generation"] = gen_report
    if oracle_bytes is None:
        raise RuntimeError(f"Oracle generation unavailable: {gen_report.get('error')}")

    oracle_img = cv2.imdecode(np.frombuffer(oracle_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if oracle_img is None:
        raise RuntimeError("failed to decode Oracle-generated image")

    oracle_aligned, alignment_report = align_oracle_to_original(orig_img, oracle_img)
    routing_report["steps"]["alignment"] = alignment_report

    # DIAGNOSTIC ONLY -- added to directly answer one question: does the
    # rectangular-artifact issue seen on IMG_8291 originate in Flux
    # Kontext's own render, or somewhere in this pipeline's own code?
    # Every file between generation and the final corrected_img has now
    # been checked (oracleRouter.py, oracleCorrection.py,
    # level2_ceiling_mask.py, level2_wall_trim_mask.py,
    # correctPipeline.js, oracleGeneration.py) and none of them draw
    # anything onto an image -- the one thing never actually inspected is
    # what Flux itself returned. Same opt-in discipline as
    # DEBUG_REGIONS_OVERLAY in correctPipeline.js: writes a SEPARATE
    # file, never touches corrected_img, off by default so this costs
    # nothing in normal operation. Same file-naming convention (suffix
    # next to output_path) so it's easy to find.
    if os.environ.get("ORACLE_DEBUG_SAVE_RENDER", "false").lower() not in ("false", "0", ""):
        debug_render_path = os.path.splitext(output_path)[0] + "_oracle_render_debug.jpg"
        cv2.imwrite(debug_render_path, oracle_aligned)
        print(f"[ORACLE-ROUTER] [DEBUG] saved raw Oracle render to {debug_render_path}",
              file=sys.stderr)

    alignment_ok = (
        alignment_report["n_inliers"] >= MIN_ALIGNMENT_INLIERS
        and alignment_report["mean_residual_px"] is not None
        and alignment_report["mean_residual_px"] <= MAX_MEAN_RESIDUAL_PX
    )
    if not alignment_ok:
        raise RuntimeError(
            f"alignment below threshold: {alignment_report['n_inliers']} inliers "
            f"(min {MIN_ALIGNMENT_INLIERS}), {alignment_report['mean_residual_px']}px "
            f"mean residual (max {MAX_MEAN_RESIDUAL_PX})"
        )

    # Vision recoverability gate -- shadow mode respected exactly as
    # level2_vision_recoverability.py itself defines it. In shadow mode
    # (the current default), this is computed and logged but the classical
    # box-filter gate still drives the actual correction -- external_gate
    # stays None below in that case, matching run_oracle_driven_pipeline's
    # own shadow-mode logic in oracleCorrection.py.
    #
    # SHADOW-MODE COMPARISON LOGGING (added -- see handoff open item #3):
    # oracleCorrection.run_oracle_driven_pipeline() computes and logs
    # mean |vision_gate - classical_gate| whenever it's given vision_gate_
    # regions, specifically so a real batch has something to review before
    # deciding whether RECOVERABILITY_SHADOW_MODE should ever flip to
    # false. This router does NOT call that function directly (see below)
    # -- it reimplements the same steps inline instead, so this logging
    # has to be reimplemented here too, or a real batch run through this
    # router produces zero comparison data despite shadow mode being on.
    # Deliberately NOT calling run_oracle_driven_pipeline() itself here:
    # that function does its own alignment internally via
    # align_oracle_to_original(), and this router already has
    # oracle_aligned from the alignment step above -- calling it would
    # mean a SECOND, independently-seeded RANSAC pass on the same image
    # pair, which is not guaranteed to reproduce identical inliers/
    # residuals as the first (cv2.RANSAC has its own internal randomness).
    # Reusing the alignment this router already computed, once, is the
    # more correct choice here, not just a shortcut.
    vision_result, vision_report = judge_recoverability(orig_img, oracle_aligned)
    routing_report["steps"]["vision_recoverability"] = vision_report

    external_gate = None
    gate_source = "classical"
    vision_gate_comparison = None

    # Only log/act on a gate if Vision actually returned something usable
    # this run -- an empty regions list from a genuine failure (disabled,
    # missing key, timeout, bad JSON -- see judge_recoverability's own
    # contract) would just rasterize to an uninformative all-1.0 gate and
    # compare that against the classical gate, adding noise rather than
    # signal to the review data.
    if vision_result.get("error") is None:
        vision_gate_raw = rasterize_vision_gate(vision_result["regions"], orig_img.shape)

        if RECOVERABILITY_SHADOW_MODE:
            # Compute classical gate too, purely for comparison logging --
            # classical gate is what will actually be used below.
            classical_raw, _ = compute_recoverability_map(orig_img)
            diff = np.abs(vision_gate_raw - classical_raw)
            vision_gate_comparison = {
                "shadow_mode": True,
                "mean_abs_diff": float(diff.mean()),
                "used_for_correction": "classical",
            }
            # file=sys.stderr -- NOT plain print(). This module's own CLI
            # contract (see docstring) requires stdout to contain ONLY the
            # final json.dumps(result) call; a stray stdout line here
            # breaks correctPipeline.js's JSON.parse(stdout.trim()) with
            # exactly the failure mode this line previously caused
            # ("Unexpected token O in JSON at position 1" -- '[' parsed as
            # a valid array-open token, then this line's literal text hit
            # as the next token). Confirmed on a real batch run, not
            # theoretical -- fixed here, not just diagnosed.
            print(f"[ORACLE-ROUTER] [VISION-GATE SHADOW MODE] mean |vision_gate - classical_gate| = "
                  f"{vision_gate_comparison['mean_abs_diff']:.3f} (classical still driving correction)",
                  file=sys.stderr)
        else:
            external_gate = vision_gate_raw
            gate_source = "vision"
            vision_gate_comparison = {"shadow_mode": False, "used_for_correction": "vision"}
            print(f"[ORACLE-ROUTER] [VISION-GATE LIVE] Vision gate driving correction (shadow mode off)",
                  file=sys.stderr)

    routing_report["steps"]["vision_gate_comparison"] = vision_gate_comparison

    recov_map, recov_class, illum_delta, color_delta_a, color_delta_b = compute_oracle_guided_deltas(
        orig_img, oracle_aligned, external_gate=external_gate
    )

    # CEILING-SCOPED DELTA SMOOTHING -- fixes a real, confirmed-repeatable
    # artifact: Flux Kontext's own invented ceiling texture doesn't
    # spatially match the Original's real lighting, producing blotchy,
    # uneven correction on what should read as one continuous surface.
    # Confirmed on two separate real photos across two separate sessions.
    # See oracleCorrection.smooth_deltas_in_mask's own docstring for the
    # full mechanism. Shadow-mode gated like every other Vision addition
    # here -- identify_ceiling() runs and its result is always logged in
    # routing_report, but only allowed to actually change illum_delta/
    # color_delta_a/color_delta_b once CEILING_MASK_SHADOW_MODE is
    # explicitly turned off after a real batch review. Deliberately NOT
    # using print() anywhere in this block -- see this router's own
    # history (VISION-GATE logging previously broke correctPipeline.js's
    # JSON.parse by writing to stdout) -- all ceiling_mask visibility
    # goes through routing_report only, which is safe by construction
    # since it's serialized into the final json.dumps(result) call, not
    # printed independently.
    ceiling_result, ceiling_report = identify_ceiling(orig_img)
    ceiling_smoothing_report = {"applied": False, "reason": "shadow_mode_or_no_grid"}
    if ceiling_result.get("grid") and not CEILING_MASK_SHADOW_MODE:
        ceiling_mask = rasterize_ceiling_mask(ceiling_result["grid"], orig_img.shape)
        illum_delta, color_delta_a, color_delta_b, ceiling_smoothing_report = smooth_deltas_in_mask(
            illum_delta, color_delta_a, color_delta_b, ceiling_mask
        )
    routing_report["steps"]["ceiling_mask"] = ceiling_report
    routing_report["steps"]["ceiling_smoothing"] = ceiling_smoothing_report

    corrected_img, clip_report = apply_oracle_guided_correction(
        orig_img, oracle_aligned, recov_map, illum_delta, color_delta_a, color_delta_b
    )
    routing_report["steps"]["clip"] = clip_report

    # WALL/TRIM + CEILING ARCHITECTURAL MASK -- feeds the hue-fidelity
    # gate below. Ceiling identification already ran above for the
    # smoothing step; reuse that result rather than calling
    # identify_ceiling() a second time for the same photo.
    wall_trim_result, wall_trim_report = identify_wall_trim(orig_img)
    routing_report["steps"]["wall_trim_mask"] = wall_trim_report

    wall_mask = (
        rasterize_wall_trim_mask(wall_trim_result["grid"], orig_img.shape)
        if wall_trim_result.get("grid")
        else np.zeros(orig_img.shape[:2], dtype=np.float32)
    )
    ceiling_mask_for_arch = (
        rasterize_ceiling_mask(ceiling_result["grid"], orig_img.shape)
        if ceiling_result.get("grid")
        else np.zeros(orig_img.shape[:2], dtype=np.float32)
    )
    architectural_mask = np.clip(wall_mask + ceiling_mask_for_arch, 0.0, 1.0)

    # HUE-FIDELITY GATE -- validated against two separate real photos this
    # session (IMG_8310, two different Oracle generations of the same
    # room): a neutral wall's Oracle render can carry a real hue-family
    # shift (measured as a visible warm/khaki cast both times, confirmed
    # by direct RGB swatch comparison against the correct target, not
    # just LAB deltas). The existing [min,max] clamp does nothing to
    # catch this since it only bounds magnitude, not whether Oracle's own
    # color endpoint is itself valid. See
    # oracleCorrection.apply_hue_fidelity_gate's docstring for the full
    # evidence and mechanism.
    #
    # SCOPING CHANGE (Aug 2026, second IMG_8310 investigation): this gate
    # now runs on ceiling_mask_for_arch ALONE, not the combined wall+
    # ceiling architectural_mask it used before. Wall gets its own,
    # different treatment immediately below (apply_wall_color_anchor) --
    # see that function's docstring for why wall and ceiling need
    # opposite rules, not the same one. Confirmed on a real production
    # run that the combined mask was the actual mechanism behind walls
    # coming back visibly cream/tan: the gate was working exactly as
    # designed, on a rule that's correct for ceiling and wrong for wall.
    #
    # Shadow-mode gated via WALL_TRIM_MASK_SHADOW_MODE -- computed and
    # logged every run, only allowed to change corrected_img once that
    # flag is explicitly turned off after review.
    if WALL_TRIM_MASK_SHADOW_MODE:
        _hue_shadow_result, hue_gate_shadow = apply_hue_fidelity_gate(
            orig_img, corrected_img, ceiling_mask_for_arch, mask_grid_cols=WALL_TRIM_GRID_COLS
        )
        print(f"[ORACLE-ROUTER] [HUE-GATE SHADOW MODE] ceiling coverage="
              f"{hue_gate_shadow.get('masked_area_fraction', 0.0)*100:.1f}%  "
              f"would-revert a={hue_gate_shadow.get('a_reverted_fraction', 0.0)*100:.1f}% "
              f"b={hue_gate_shadow.get('b_reverted_fraction', 0.0)*100:.1f}% "
              f"(NOT applied -- shadow mode, corrected_img unchanged)", file=sys.stderr)
    else:
        corrected_img, hue_gate_shadow = apply_hue_fidelity_gate(
            orig_img, corrected_img, ceiling_mask_for_arch, mask_grid_cols=WALL_TRIM_GRID_COLS
        )
        print(f"[ORACLE-ROUTER] [HUE-GATE LIVE] applied to ceiling region "
              f"(shadow mode off)", file=sys.stderr)
    routing_report["steps"]["hue_fidelity_gate"] = hue_gate_shadow

    # WALL COLOR ANCHOR -- see oracleCorrection.apply_wall_color_anchor's
    # docstring for the full mechanism and evidence. Wall's true paint
    # color is not neutral and isn't supposed to become neutral, unlike
    # ceiling above -- this locks wall a/b to the Original's own captured
    # values rather than gating Oracle's move toward zero. Same shadow-
    # mode flag as the hue-fidelity gate, since it depends on the same
    # wall_trim_mask Vision call -- NOT yet validated against a real
    # batch, ship shadow-first same as everything else here.
    if WALL_TRIM_MASK_SHADOW_MODE:
        _wall_anchor_shadow_result, wall_color_anchor_report = apply_wall_color_anchor(
            orig_img, corrected_img, wall_mask, mask_grid_cols=WALL_TRIM_GRID_COLS
        )
        print(f"[ORACLE-ROUTER] [WALL-ANCHOR SHADOW MODE] wall coverage="
              f"{wall_color_anchor_report.get('masked_area_fraction', 0.0)*100:.1f}% "
              f"(NOT applied -- shadow mode, corrected_img unchanged)", file=sys.stderr)
    else:
        corrected_img, wall_color_anchor_report = apply_wall_color_anchor(
            orig_img, corrected_img, wall_mask, mask_grid_cols=WALL_TRIM_GRID_COLS
        )
        print(f"[ORACLE-ROUTER] [WALL-ANCHOR LIVE] applied to wall region "
              f"(shadow mode off)", file=sys.stderr)
    routing_report["steps"]["wall_color_anchor"] = wall_color_anchor_report

    # ILLUMINATION FLOOR -- see oracleCorrection.apply_illumination_floor's
    # own docstring for the full mechanism and evidence: Oracle's own
    # render can undershoot the MLS Bright brightness target on walls/
    # ceiling/trim (confirmed on IMG_8253/IMG_8317 as a mean-L lift of only
    # +1.6 to +2.2 out of 255), and apply_oracle_guided_correction's clamp
    # silently inherits that conservatism as a hard ceiling on the final
    # correction. Still uses the COMBINED architectural_mask (wall +
    # ceiling) deliberately -- unlike the two hue functions above,
    # brightness scoping is not part of this session's separation fix,
    # per Sam's explicit instruction not to push ceiling brightness any
    # further; this only touches L, never a/b, so it doesn't reintroduce
    # the wall/ceiling hue-conflation problem those two functions above
    # just fixed. Same recov_map already computed earlier in this
    # function -- no new Vision call, no new mask.
    #
    # FEATHERING FIX (Aug 2026, IMG_8310 investigation): mask_grid_cols
    # is now passed here -- it previously was not, even though the
    # hue-fidelity gate call right above it always has. apply_
    # illumination_floor's own docstring now documents why that omission
    # produced hard rectangular grid-cell edges the first time this
    # floor actually went live on a real photo (wall_trim_mask
    # succeeding for the first time this session exposed a bug that had
    # been sitting here, unreachable, all along).
    if ILLUM_FLOOR_SHADOW_MODE:
        _floor_shadow_result, illum_floor_report = apply_illumination_floor(
            orig_img, corrected_img, recov_map, architectural_mask, mask_grid_cols=WALL_TRIM_GRID_COLS
        )
        print(f"[ORACLE-ROUTER] [ILLUM-FLOOR SHADOW MODE] mean_l_before="
              f"{illum_floor_report.get('mean_l_before', 0.0):.1f}  "
              f"lift_strength={illum_floor_report.get('lift_strength', 0.0):.2f}  "
              f"would-raise={illum_floor_report.get('floor_raised_fraction', 0.0)*100:.1f}% "
              f"of masked pixels (NOT applied -- shadow mode, corrected_img unchanged)",
              file=sys.stderr)
    else:
        corrected_img, illum_floor_report = apply_illumination_floor(
            orig_img, corrected_img, recov_map, architectural_mask, mask_grid_cols=WALL_TRIM_GRID_COLS
        )
        print(f"[ORACLE-ROUTER] [ILLUM-FLOOR LIVE] applied to corrected_img "
              f"(shadow mode off) mean_l_before={illum_floor_report.get('mean_l_before', 0.0):.1f} "
              f"raised={illum_floor_report.get('floor_raised_fraction', 0.0)*100:.1f}% of masked pixels",
              file=sys.stderr)
    routing_report["steps"]["illumination_floor"] = illum_floor_report

    # GLOBAL HUE ANGLE GATE -- runs last, whole-frame, no Vision dependency.
    # See GLOBAL_HUE_GATE_SHADOW_MODE comment above for why this is wired
    # in now and why it's independent of everything above it in this
    # function. Deliberately placed after illumination_floor so it can
    # catch drift introduced anywhere upstream -- Oracle's own generation,
    # or any Vision-gated step above -- not just Oracle's raw output.
    if GLOBAL_HUE_GATE_SHADOW_MODE:
        _hue_angle_shadow_result, hue_angle_report = apply_global_hue_angle_gate(
            orig_img, corrected_img
        )
        print(f"[ORACLE-ROUTER] [GLOBAL-HUE-GATE SHADOW MODE] "
              f"violated_fraction={hue_angle_report.get('violated_fraction', 0.0)*100:.1f}% "
              f"mean_drift={hue_angle_report.get('mean_hue_drift_all_pixels', 0.0):.1f} deg "
              f"(NOT applied -- shadow mode, corrected_img unchanged)", file=sys.stderr)
    else:
        corrected_img, hue_angle_report = apply_global_hue_angle_gate(
            orig_img, corrected_img
        )
        print(f"[ORACLE-ROUTER] [GLOBAL-HUE-GATE LIVE] "
              f"violated_fraction={hue_angle_report.get('violated_fraction', 0.0)*100:.1f}% "
              f"mean_drift={hue_angle_report.get('mean_hue_drift_all_pixels', 0.0):.1f} deg "
              f"(applied -- shadow mode off)", file=sys.stderr)
    routing_report["steps"]["global_hue_angle_gate"] = hue_angle_report

    ok = cv2.imwrite(output_path, corrected_img)
    if not ok:
        raise RuntimeError(f"failed to write output image to {output_path}")

    total_px = recov_class.size
    return {
        "oracleRouting": routing_report,
        "level0Scene": "interior",
        "gateSource": gate_source,
        "recoverabilityPct": {
            "red": 100.0 * (recov_class == 0).sum() / total_px,
            "yellow": 100.0 * (recov_class == 1).sum() / total_px,
            "green": 100.0 * (recov_class == 2).sum() / total_px,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lens-mode", choices=["auto", "mild", "off"], default="auto")
    parser.add_argument("--intensity", type=float, default=1.0)
    args = parser.parse_args()

    orig_img = cv2.imread(args.source)
    if orig_img is None:
        print(json.dumps({"error": f"Could not read image: {args.source}"}), file=sys.stderr)
        sys.exit(1)

    scene = classify_scene(orig_img)
    is_exterior = scene.get("sceneType") == "exterior"
    scene_confident = scene.get("sceneType") is not None

    # Exterior, or Vision unavailable/unconfident -> classical only, always.
    # This is the routing decision itself, stated once, in code, not spread
    # across conditionals -- exterior is a hard route, not a preference.
    if is_exterior or not scene_confident:
        try:
            result = run_classical_only(args.source, args.output, args.lens_mode, args.intensity)
            result["oracleRouting"]["reason"] = (
                "exterior" if is_exterior else "scene_classification_unavailable"
            )
            print(json.dumps(result))
            return
        except Exception as e:  # noqa: BLE001 -- this IS the fallback path, must not itself fail silently
            print(json.dumps({"error": f"classical correction failed: {e}"}), file=sys.stderr)
            sys.exit(1)

    # Interior with confident classification -> try Oracle, fall back to
    # classical on ANY failure at ANY step. Never ships a half-finished
    # Oracle result, per this module's own stated contract.
    try:
        result = run_oracle_interior(args.source, args.output, orig_img)
        print(json.dumps(result))
    except Exception as e:  # noqa: BLE001
        try:
            result = run_classical_only(args.source, args.output, args.lens_mode, args.intensity)
            result["oracleRouting"]["reason"] = f"oracle_pipeline_failed: {e}"
            print(json.dumps(result))
        except Exception as fallback_err:
            print(json.dumps({"error": f"both Oracle and classical correction failed: "
                                        f"oracle={e}, classical={fallback_err}"}), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
