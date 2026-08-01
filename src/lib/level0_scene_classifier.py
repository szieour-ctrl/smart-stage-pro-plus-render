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

Rollout plan (mirrors the Level 0-5 testing discipline used for Stage 1
diagnosis — do not skip steps just because this module feels simpler):
  1. SHADOW_MODE=true (default): Vision runs, result is logged, but the
     HSV heuristic still drives actual routing. Nothing in production
     behavior changes yet.
  2. Review a real batch of disagreements (see resolve_scene_type's
     `disagreement` flag in logs). Confirm Vision gets IMG_8305 right,
     confirm it doesn't introduce new false positives (e.g. a bright
     sunroom or conservatory getting called "exterior").
  3. Only after that review, flip LEVEL0_SHADOW_MODE=false so Vision's
     answer becomes the actual gate. HSV heuristic remains as a
     fallback for when Vision is disabled/unavailable/fails.
"""

import os
import json
import base64
import logging
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MODEL = os.environ.get("LEVEL0_SCENE_MODEL", DEFAULT_MODEL)
ENABLED = os.environ.get("LEVEL0_VISION_CLASSIFIER_ENABLED", "true").lower() == "true"
TIMEOUT_SECONDS = int(os.environ.get("LEVEL0_VISION_TIMEOUT_SECONDS", "12"))

# SHADOW_MODE=true: Vision result is computed and logged only; the HSV
# heuristic still decides routing. Flip to "false" only after reviewing
# a real batch of logged disagreements.
SHADOW_MODE = os.environ.get("LEVEL0_SHADOW_MODE", "true").lower() == "true"


class SceneClassification(TypedDict, total=False):
    sceneType: Optional[str]      # "interior" | "exterior" | None
    confidence: Optional[str]     # "high" | "medium" | "low"
    reasoning: Optional[str]


SYSTEM_PROMPT = """You are assisting a real estate photo correction pipeline. \
Your ONLY job is to classify whether the attached photo is an INTERIOR or \
EXTERIOR shot of a residential property.

Judge based on what's actually visible in the frame, not what's implied by \
a small detail. A photo taken from indoors looking out a window or patio \
door is INTERIOR, even if a yard or sky is visible through the glass — \
indoor surfaces (walls, ceiling, floor, furniture) dominate the frame. A \
close-up of a driveway, garage door, stucco wall, patio, pool deck, or \
building facade is EXTERIOR, even with little sky or grass visible.

Respond with ONLY strict JSON, no markdown fences, no preamble, no \
explanation outside the JSON:
{"sceneType": "interior" | "exterior", "confidence": "high" | "medium" | "low", "reasoning": "<one sentence>"}
"""


def classify_scene(image_bytes: bytes, media_type: str = "image/jpeg") -> SceneClassification:
    """
    Classify a photo as interior or exterior using Vision.

    Never raises. On any failure (disabled, missing key, timeout, bad
    JSON, unexpected value), returns {"sceneType": None, ...} so callers
    can fall back to the HSV heuristic — same degrade-gracefully pattern
    used throughout Level 2 (see level2_diagnosis.py).
    """
    if not ENABLED:
        logger.info("level0_scene_classifier: disabled via env, skipping")
        return {"sceneType": None, "confidence": None, "reasoning": "disabled"}

    try:
        import anthropic  # local import, same pattern as level2_diagnosis.py

        client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS)
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": "Classify this photo."},
                ],
            }],
        )

        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        scene_type = parsed.get("sceneType")
        if scene_type not in ("interior", "exterior"):
            logger.warning(
                f"level0_scene_classifier: unexpected sceneType '{scene_type}', "
                f"treating as unknown"
            )
            return {"sceneType": None, "confidence": None, "reasoning": f"unparseable: {raw[:200]}"}

        return {
            "sceneType": scene_type,
            "confidence": parsed.get("confidence"),
            "reasoning": parsed.get("reasoning"),
        }

    except Exception as e:
        logger.warning(f"level0_scene_classifier: failed ({e}), degrading to None")
        return {"sceneType": None, "confidence": None, "reasoning": None}


def resolve_scene_type(
    image_bytes: bytes,
    hsv_heuristic_result: bool,
    media_type: str = "image/jpeg",
) -> dict:
    """
    Combines Vision classification with the existing HSV heuristic and
    returns the FINAL routing decision, plus both raw signals so main()
    can log disagreements during the shadow-mode rollout.

    hsv_heuristic_result: output of the existing is_exterior_daylight()
    (True = heuristic says exterior).
    """
    vision_result = classify_scene(image_bytes, media_type)
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
