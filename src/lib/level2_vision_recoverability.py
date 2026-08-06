"""
level2_vision_recoverability.py

Vision-based recoverability gate for the Oracle-driven correction pass
(oracleCorrection.py). Replaces compute_recoverability_map()'s 25px
box-filter classifier as the thing that decides WHERE the measured
Oracle-vs-Original delta is trusted -- not the delta itself, which stays
a real measured quantity from two real images (see oracleCorrection.py).

WHY THIS EXISTS: the box-filter classifier judges recoverability at a
fixed small spatial scale (a 25px sliding window) and got confirmed wrong
on real data -- a dark dining chair in shadow reads identically to a
genuinely clipped shadow at that scale (both: low local std, low local
mean), so the classifier can't tell "object in shadow, fully recoverable
from context two feet away" from "truly no data here." That produced a
real halo artifact on IMG_8310 (fixed there by feathering, which softens
the SYMPTOM) and is the same root cause implicated in the existing
pipeline's wall-seam and column-shadow-wedge artifacts on the same file
(different mechanism, same failure class: hard-edged, small-context
region judgment). Feathering a wrong local judgment still produces a
smoothed version of the wrong judgment.

Confirmed directly on IMG_8301 vs Oracle2 (Aug 2026 session): the
box-filter classifier found only 1.5% of the frame genuinely "red" (no
data) -- and every one of those small red pockets, checked by hand, had
abundant real structure in the wider region around it (std 30-66 in a
120px window around a 25px patch that read as flat). A shoe rack read
"red" at its exact geometric center while individual shoes were legible
one shelf-width away. This module asks Vision the question a human
retoucher actually asks -- "does the OBJECT/SCENE this belongs to have
data" -- instead of "does this exact small patch have data."

WHAT THIS DOES NOT DO: it does not compute the illumination/color delta
-- that stays oracleCorrection.compute_oracle_guided_deltas(), a real
measured quantity from two real images. This module ONLY judges where
that measured delta should be trusted. Vision is never asked to invent
or estimate a correction magnitude here, the same discipline
level2_diagnosis.py already applies (diagnose the TYPE of problem, never
estimate the correction VALUE) -- Vision names regions and a trust
level with evidence; the actual math is still deterministic.

ROLLOUT (same phased discipline as level0_scene_classifier.py -- do not
skip steps): ships in SHADOW MODE by default. Vision's gate is computed,
returned, and logged (including agreement/disagreement against the
classical gate) on every run, but oracleCorrection.run_oracle_driven_pipeline
still uses the classical box-filter gate to actually drive correction
until a real batch has been reviewed. Flip via
RECOVERABILITY_SHADOW_MODE=false once that review is done. The classical
gate is never removed -- it remains the fallback for when Vision is
disabled, unavailable, or answers with low confidence, exactly as the
HSV heuristic remains level0's fallback.
"""

import os
import json
import logging
import base64
import urllib.error
import urllib.request
from typing import Optional, TypedDict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

RECOVERABILITY_ENABLED = os.environ.get("LEVEL2_RECOVERABILITY_ENABLED", "true").lower() not in ("false", "0", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RECOVERABILITY_MODEL = os.environ.get("LEVEL2_RECOVERABILITY_MODEL", "claude-haiku-4-5-20251001")
TIMEOUT_SECONDS = int(os.environ.get("LEVEL2_RECOVERABILITY_TIMEOUT_SECONDS", "30"))  # 30s
# from the start, not a Haiku-era 12s -- this call asks for several regions
# each with a reasoning sentence (see PROMPT below), the same output shape
# that forced level2_diagnosis.py and level2_qc.py to each raise their
# timeout after hitting it live. Sized correctly up front instead of
# waiting to reproduce that same incident a third time.
MAX_VISION_EDGE = 1568  # matches level0/level2_diagnosis/level2_qc

# Shadow mode default TRUE (opposite of level0's current state, matching
# level0's OWN starting state before its Aug 3 promotion) -- this module
# has zero real-batch review yet. Do not flip without that review; see
# module docstring.
SHADOW_MODE = os.environ.get("RECOVERABILITY_SHADOW_MODE", "true").lower() not in ("false", "0", "")

MAX_REGIONS = int(os.environ.get("LEVEL2_RECOVERABILITY_MAX_REGIONS", "12"))
GATE_FEATHER_SIGMA = 15  # matches oracleCorrection.GATE_FEATHER_SIGMA -- kept as a
# separate constant (not an import) so this module has no hard dependency
# on oracleCorrection's internals, only on the array shape it hands back.


class RecoverabilityRegion(TypedDict, total=False):
    description: str
    bbox: List[int]  # [x1, y1, x2, y2] in pixel coords of the ORIGINAL image
    recoverable: bool
    confidence: float  # 0.0-1.0
    evidence: str


class RecoverabilityResult(TypedDict, total=False):
    regions: List[RecoverabilityRegion]
    error: Optional[str]


SYSTEM_PROMPT = """You are a professional real estate photo retoucher judging WHERE a lighting/color correction can be safely applied to a real photo -- not what the correction should be, just where it's trustworthy.

You are shown two images of the same room: the ORIGINAL (as-shot, unedited) and an ORACLE (a target reference showing the same room after expert correction -- same architecture, same furniture, same everything, just properly exposed and color-balanced). A separate, purely mathematical step has already measured the exact per-pixel difference between them. Your job is NOT to describe that difference -- it's to judge, region by region, whether the ORIGINAL actually contains enough real visual data to justify trusting that measured difference, or whether a region is so fully lacking in detail that applying it would be invention rather than correction.

The question for each region is: "if I only had the ORIGINAL and a skilled retoucher's eye -- no Oracle, no measurement -- could I confidently lift/correct this area using data visible ELSEWHERE in the same photo (the rest of the same object, the same continuous surface, the same material under better light nearby), the way a human dodges and burns using context?" If yes, mark it recoverable, even if the exact region itself looks dark or flat up close -- a shadowed object sitting next to its own well-lit surroundings is the normal case, not the exception. If a region is truly, fully occluded or void of any surrounding context anywhere in the frame (a cavity with nothing visible near it, a window blown to pure white with zero cast-shadow or reflection cues anywhere), mark it not recoverable.

Do NOT judge recoverability by how dark or flat a small patch looks in isolation -- judge it by whether the OBJECT or SURFACE that patch belongs to has legible structure anywhere in the frame. A dark gap between two chair-back spindles is recoverable if the rest of the chair is visible. A wall in deep shadow is recoverable if the same wall is visible in better light elsewhere in the frame, or if it's a single continuous painted surface with legible texture. Do NOT propose named-category regions ("the wall", "the carpet") that would only be blended by something downstream -- instead give bounding boxes small enough that each one is internally consistent (same material, same rough lighting condition), so no single box straddles a real material or lighting boundary.

Return at most {max_regions} regions -- prioritize the darkest/most ambiguous areas of the ORIGINAL; do not bother with areas that are obviously already well-lit.

Return ONLY strict JSON, no markdown fences, no prose outside the JSON:
{{"regions": [{{"description": "<a few words>", "bbox": [x1, y1, x2, y2], "recoverable": true|false, "confidence": 0.0-1.0, "evidence": "<one sentence: what elsewhere in the frame supports this judgment>"}}]}}"""


def _encode_for_vision(img) -> Optional[tuple]:
    """Same downscale/encode as level0/level2_diagnosis/level2_qc's helper,
    but ALSO returns the (h, w) actually sent to Vision -- unlike those
    modules, this one needs to rescale bbox coordinates Vision returns
    back up to the Original's full-resolution pixel grid (see
    judge_recoverability), so the downscaled shape has to survive the
    call. Returns None on encode failure, (b64_str, (h, w)) on success."""
    h, w = img.shape[:2]
    scale = min(1.0, MAX_VISION_EDGE / float(max(h, w)))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii"), small.shape[:2]


def _call_vision_api(original_b64: str, oracle_b64: str, media_type: str) -> dict:
    """Raw urllib.request call, no SDK -- same pattern as level2_diagnosis.py
    and level2_qc.py's _call_vision_api, proven in production."""
    prompt_text = SYSTEM_PROMPT.format(max_regions=MAX_REGIONS)
    body = json.dumps({
        "model": RECOVERABILITY_MODEL,
        "max_tokens": 1536,  # sized for MAX_REGIONS regions each with a reasoning
        # sentence -- same rationale as level2_diagnosis.py's 512->1024 raise,
        # scaled up further since this response can contain up to MAX_REGIONS
        # structured objects rather than one.
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "ORIGINAL (as-shot):"},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": original_b64}},
                {"type": "text", "text": "ORACLE (target reference):"},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": oracle_b64}},
                {"type": "text", "text": prompt_text},
            ],
        }],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    text = "".join(
        block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
    ).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()

    # Same hardened parsing as level2_diagnosis.py / level2_qc.py, applied
    # up front rather than added after a live failure this time: empty-
    # response check, find first '{' rather than assume position 0 (leading
    # prose), raw_decode rather than plain loads (trailing chatter).
    if not text:
        raise ValueError(f"empty_response_text (stop_reason={payload.get('stop_reason')!r})")
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"no_json_object_found -- raw_text={text[:300]!r}")
    try:
        return json.JSONDecoder().raw_decode(text, brace_start)[0]
    except json.JSONDecodeError as e:
        raise ValueError(f"{e} -- raw_text={text[:300]!r}") from e


def judge_recoverability(original_img, oracle_aligned_img) -> tuple:
    """
    Vision judgment of WHERE the Oracle-vs-Original delta should be trusted.
    Takes decoded cv2/numpy BGR arrays, same calling convention as
    level2_diagnosis.diagnose() and level2_qc.qc_check() -- no re-reading
    from disk.

    original_img: the raw Original, full resolution.
    oracle_aligned_img: Oracle already warped into the Original's frame
        (see oracleCorrection.align_oracle_to_original) -- passing the
        aligned version, not the raw Oracle, so bbox coordinates Vision
        returns land correctly on the Original's pixel grid.

    Returns (result, report). result["regions"] is a list of
    RecoverabilityRegion dicts in ORIGINAL-image pixel coordinates
    (rescaled up from whatever size was actually sent to Vision -- see
    MAX_VISION_EDGE downscaling). Never raises. On any failure, returns
    ({"regions": [], "error": ...}, report) -- callers must treat an
    empty regions list as "Vision gate unavailable," not "nothing is
    recoverable," and fall back to the classical gate.
    """
    report = {"enabled": RECOVERABILITY_ENABLED, "called": False, "model": RECOVERABILITY_MODEL,
              "shadowMode": SHADOW_MODE, "error": None}

    if not RECOVERABILITY_ENABLED:
        report["error"] = "disabled_via_env"
        return {"regions": [], "error": "disabled_via_env"}, report
    if not ANTHROPIC_API_KEY:
        logger.warning("level2_vision_recoverability: missing ANTHROPIC_API_KEY, degrading to empty")
        report["error"] = "missing_ANTHROPIC_API_KEY"
        return {"regions": [], "error": "missing_ANTHROPIC_API_KEY"}, report

    try:
        orig_h, orig_w = original_img.shape[:2]

        enc_orig = _encode_for_vision(original_img)
        enc_oracle = _encode_for_vision(oracle_aligned_img)
        if enc_orig is None or enc_oracle is None:
            report["error"] = "jpeg_encode_failed"
            return {"regions": [], "error": "jpeg_encode_failed"}, report

        original_b64, sent_shape = enc_orig
        oracle_b64, _ = enc_oracle

        report["called"] = True
        parsed = _call_vision_api(original_b64, oracle_b64, "image/jpeg")

        raw_regions = parsed.get("regions")
        if not isinstance(raw_regions, list):
            report["error"] = f"unparseable_regions:{type(raw_regions).__name__}"
            return {"regions": [], "error": report["error"]}, report

        # Rescale bboxes from the (possibly downscaled) image Vision actually
        # saw back up to the Original's full-resolution pixel grid -- Vision
        # doesn't know MAX_VISION_EDGE was applied, its coordinates are in
        # the image it was shown.
        sent_h, sent_w = sent_shape
        scale_x = orig_w / float(sent_w)
        scale_y = orig_h / float(sent_h)

        regions: List[RecoverabilityRegion] = []
        for r in raw_regions[:MAX_REGIONS]:
            bbox = r.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = bbox
            regions.append({
                "description": str(r.get("description", ""))[:200],
                "bbox": [
                    int(max(0, min(orig_w, x1 * scale_x))),
                    int(max(0, min(orig_h, y1 * scale_y))),
                    int(max(0, min(orig_w, x2 * scale_x))),
                    int(max(0, min(orig_h, y2 * scale_y))),
                ],
                "recoverable": bool(r.get("recoverable", False)),
                "confidence": float(r.get("confidence", 0.0)),
                "evidence": str(r.get("evidence", ""))[:400],
            })

        return {"regions": regions, "error": None}, report

    except Exception as e:  # noqa: BLE001 -- any failure must fall back, never crash the pipeline
        logger.warning(f"level2_vision_recoverability: failed ({type(e).__name__}: {e}), degrading to empty")
        report["error"] = f"{type(e).__name__}: {e}"
        return {"regions": [], "error": report["error"]}, report


def rasterize_vision_gate(regions: List[RecoverabilityRegion], img_shape,
                           default_gate: float = 1.0) -> np.ndarray:
    """
    Turns Vision's bbox+confidence regions into a continuous float32 gate
    map, same shape/contract as oracleCorrection.compute_recoverability_map's
    RAW (unfeathered) output -- drop-in replacement for the classical gate's
    raw output, meant to be passed as external_gate= into
    oracleCorrection.compute_oracle_guided_deltas(), which does the
    feathering itself. Deliberately NOT feathered here -- feathering twice
    (once here, once there) would over-blur relative to the classical
    path with no benefit; keeping exactly one feather step, in one place,
    for both gate sources, keeps them comparable in shadow-mode logging.

    default_gate: value for any pixel NOT covered by any region Vision
        returned. Defaults to 1.0 (fully trust the measured delta) rather
        than 0.0 -- Vision was told to only bother naming ambiguous/dark
        areas ("do not bother with areas that are obviously already
        well-lit"), so silence on a region means "not ambiguous," not
        "not recoverable." Starting everywhere-trusted and only pulling
        DOWN the specific areas Vision flagged as not-recoverable matches
        that instruction; starting everywhere-untrusted would silently
        zero out the entire frame outside Vision's named regions.

    Regions are painted in the order given, at their stated confidence
    (recoverable=True -> gate approaches confidence; recoverable=False ->
    gate approaches (1-confidence), i.e. a high-confidence NOT-recoverable
    call pulls the gate toward 0). Later regions overwrite earlier ones on
    overlap.
    """
    h, w = img_shape[:2]
    gate = np.full((h, w), default_gate, dtype=np.float32)

    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        conf = max(0.0, min(1.0, r.get("confidence", 0.0)))
        value = conf if r.get("recoverable") else (1.0 - conf)
        gate[y1:y2, x1:x2] = value

    return gate
