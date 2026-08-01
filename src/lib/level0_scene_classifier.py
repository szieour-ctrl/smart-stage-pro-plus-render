"""
level0_scene_classifier.py

Stage 0 (Level 0) — Vision-based interior/exterior scene classification.

Runs as the FIRST step in the pipeline, before any geometric correction,
before Stage 1 diagnosis, and (during rollout) alongside the existing
`is_exterior_daylight()` HSV heuristic rather than replacing it outright.

Why this exists: the HSV heuristic (green-pixel fraction / sky-pixel
fraction) misclassified IMG_8305 — an unambiguous house-exterior photo
(driveway, garage, stucco) — as interior, because it has low visible
vegetation and low visible sky. The heuristic's own docstring already
flagged this exact blind spot ("a close-up patio/pool shot with little
visible plant life... would not be caught"). This module gives Vision
the interior/exterior judgment call directly, the way a human editor
would just look at the photo and know.

IMPLEMENTATION NOTE (fixed Aug 1, 2026): originally written against the
`anthropic` Python SDK, which is NOT installed on this service (only
opencv-python-headless/numpy are, per requirements.txt/Dockerfile) and
has no reason to be — level2_diagnosis.py already proved the pattern
this codebase actually uses: a raw `urllib.request` POST to
api.anthropic.com, stdlib only, zero extra dependencies, zero rebuild
risk. This module was rewritten to match that exact pattern rather than
add a new dependency footprint. First deploy attempt failed silently
(degraded to sceneType=None on every photo, "No module named 'anthropic'")
until level0Scene was added to the JSON output and made this visible —
see smartCorrect.py's output block for where that surfaces.

Rollout plan (mirrors the Level 0-5 testing discipline used for Stage 1
diagnosis — do not skip steps just because this module feels simpler):
  1. SHADOW_MODE=true (default): Vision runs, result is logged/returned,
     but the HSV heuristic still drives actual routing. Nothing in
     production behavior changes yet.
  2. Review a real batch via the `level0Scene` field now present in
     every JSON result (see resolve_scene_type below). Confirm Vision
     gets IMG_8305 right, confirm it doesn't introduce new false
     positives (e.g. a bright sunroom/conservatory called "exterior").
  3. Only after that review, flip LEVEL0_SHADOW_MODE=false so Vision's
     answer becomes the actual gate. HSV heuristic remains as a
     fallback for when Vision is disabled/unavailable/fails.
"""

import os
import json
import logging
import base64
import urllib.error
import urllib.request
from typing import Optional, TypedDict

import cv2

logger = logging.getLogger(__name__)

LEVEL0_ENABLED = os.environ.get("LEVEL0_VISION_CLASSIFIER_ENABLED", "true").lower() not in ("false", "0", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SCENE_MODEL = os.environ.get("LEVEL0_SCENE_MODEL", "claude-haiku-4-5-20251001")
TIMEOUT_SECONDS = int(os.environ.get("LEVEL0_VISION_TIMEOUT_SECONDS", "12"))
MAX_VISION_EDGE = 1568  # matches level2_diagnosis.py / the existing cap used elsewhere in this codebase

# SHADOW_MODE=true: Vision result is computed and returned/logged only;
# the HSV heuristic still decides routing. Flip to "false" only after
# reviewing a real batch of logged disagreements via level0Scene in the
# JSON output.
SHADOW_MODE = os.environ.get("LEVEL0_SHADOW_MODE", "true").lower() not in ("false", "0", "")


class SceneClassification(TypedDict, total=False):
    sceneType: Optional[str]      # "interior" | "exterior" | None
    confidence: Optional[str]     # "high" | "medium" | "low"
    reasoning: Optional[str]
    error: Optional[str]


SYSTEM_PROMPT = """You are assisting a real estate photo correction pipeline. \
Your ONLY job is to classify whether the attached photo is an INTERIOR or \
EXTERIOR shot of a residential property.

Judge based on what's actually visible in the frame, not what's implied by \
a small detail. A photo taken from indoors looking out a window or patio \
door is INTERIOR, even if a yard or sky is visible through the glass — \
indoor surfaces (walls, ceiling, floor, furniture) dominate the frame. A \
close-up of a driveway, garage door, stucco wall, patio, pool deck, or \
building facade is EXTERIOR, even with little sky or grass visible.

WATCH FOR THIS SPECIFIC TRAP: covered outdoor spaces — patios, breezeways, \
covered balconies, verandas, covered walkways — often have a finished \
ceiling, recessed can lighting, and tile or concrete flooring that can look \
"indoor" at a glance. Do NOT classify based on ceiling finish, lighting \
fixtures, or floor material alone. Instead, look for OPEN AIR: a railing, \
column, or open edge with unobstructed depth beyond it (distant rooftops, \
trees, sky, or yard visible past that edge, even out of focus or in a thin \
sliver). If that open-air depth is present, this is a COVERED EXTERIOR \
space, not an interior room — classify it as exterior. True interiors have \
walls (not railings) closing off the space, and anything visible through a \
window is a distinct, framed view rather than open depth running past an \
open edge.

Respond with ONLY strict JSON, no markdown fences, no preamble, no \
explanation outside the JSON:
{"sceneType": "interior" | "exterior", "confidence": "high" | "medium" | "low", "reasoning": "<one sentence>"}"""


def _call_vision_api(image_b64: str, media_type: str) -> dict:
    """Raw urllib.request call, no SDK — same pattern as level2_diagnosis.py's
    _call_vision_api, proven working in production for Stage 1 diagnosis."""
    body = json.dumps({
        "model": SCENE_MODEL,
        "max_tokens": 200,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": "Classify this photo."},
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

    # Use raw_decode rather than a plain json.loads: models occasionally
    # append trailing text after a complete, valid JSON object despite
    # being told not to (confirmed failure mode in level2_qc.py's
    # identical parsing logic -- "Extra data: line 2 column 1"). This
    # parses just the first valid JSON value and ignores anything after.
    return json.JSONDecoder().raw_decode(text)[0]


def classify_scene(img) -> SceneClassification:
    """
    Classify a decoded image (cv2/numpy array, BGR) as interior or exterior
    using Vision. Takes the same `img` object the rest of the pipeline
    already has in memory -- no re-reading from disk, matches
    level2_diagnosis.diagnose()'s calling convention.

    Never raises. On any failure (disabled, missing key, timeout, bad
    JSON, unexpected value), returns {"sceneType": None, ...} so callers
    fall back to the HSV heuristic.
    """
    if not LEVEL0_ENABLED:
        logger.info("level0_scene_classifier: disabled via env, skipping")
        return {"sceneType": None, "confidence": None, "reasoning": None, "error": "disabled_via_env"}
    if not ANTHROPIC_API_KEY:
        logger.warning("level0_scene_classifier: missing ANTHROPIC_API_KEY, degrading to None")
        return {"sceneType": None, "confidence": None, "reasoning": None, "error": "missing_ANTHROPIC_API_KEY"}

    try:
        h, w = img.shape[:2]
        scale = min(1.0, MAX_VISION_EDGE / float(max(h, w)))
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img

        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            logger.warning("level0_scene_classifier: jpeg encode failed, degrading to None")
            return {"sceneType": None, "confidence": None, "reasoning": None, "error": "jpeg_encode_failed"}
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        parsed = _call_vision_api(image_b64, "image/jpeg")

        scene_type = parsed.get("sceneType")
        if scene_type not in ("interior", "exterior"):
            logger.warning(f"level0_scene_classifier: unexpected sceneType '{scene_type}'")
            return {"sceneType": None, "confidence": None, "reasoning": None, "error": f"unparseable_scene_type:{scene_type}"}

        return {
            "sceneType": scene_type,
            "confidence": parsed.get("confidence"),
            "reasoning": parsed.get("reasoning"),
            "error": None,
        }

    except Exception as e:  # noqa: BLE001 -- any failure must fall back, never crash the pipeline
        logger.warning(f"level0_scene_classifier: failed ({type(e).__name__}: {e}), degrading to None")
        return {"sceneType": None, "confidence": None, "reasoning": None, "error": f"{type(e).__name__}: {e}"}


def resolve_scene_type(img, hsv_heuristic_result: bool) -> dict:
    """
    Combines Vision classification with the existing HSV heuristic and
    returns the FINAL routing decision, plus both raw signals. Always
    included in smartCorrect.py's JSON output as "level0Scene" -- do not
    rely on logs alone to see this (see the Aug 1 incident note above).

    hsv_heuristic_result: output of the existing is_exterior_daylight()
    (True = heuristic says exterior).
    """
    vision_result = classify_scene(img)
    vision_available = vision_result.get("sceneType") is not None
    vision_says_exterior = vision_result.get("sceneType") == "exterior"

    if SHADOW_MODE or not vision_available:
        final_is_exterior = hsv_heuristic_result
        source = "hsv_heuristic (shadow mode)" if SHADOW_MODE else "hsv_heuristic (vision unavailable)"
    else:
        final_is_exterior = vision_says_exterior
        source = "vision"

    disagreement = vision_available and (vision_says_exterior != hsv_heuristic_result)
    if disagreement:
        logger.info(
            f"level0_scene_classifier: DISAGREEMENT — hsv_exterior={hsv_heuristic_result} "
            f"vision_exterior={vision_says_exterior} "
            f"(confidence={vision_result.get('confidence')}) "
            f"reasoning='{vision_result.get('reasoning')}'"
        )

    return {
        "isExterior": final_is_exterior,
        "source": source,
        "hsvResult": hsv_heuristic_result,
        "visionResult": vision_result,
        "disagreement": disagreement,
    }
