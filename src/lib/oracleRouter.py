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
    compute_recoverability_map,
    MIN_ALIGNMENT_INLIERS,
    MAX_MEAN_RESIDUAL_PX,
)
from level2_vision_recoverability import (  # noqa: E402
    judge_recoverability,
    rasterize_vision_gate,
    SHADOW_MODE as RECOVERABILITY_SHADOW_MODE,
)

SMARTCORRECT_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smartCorrect.py")


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
    corrected_img, clip_report = apply_oracle_guided_correction(
        orig_img, oracle_aligned, recov_map, illum_delta, color_delta_a, color_delta_b
    )
    routing_report["steps"]["clip"] = clip_report

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
