"""
level2_qc.py

Stage 4 of the four-stage architecture (design session, Aug 2026):
Vision diagnoses -> Vision routes regions -> classical CV executes ->
**Vision QCs the result**. This is that last, previously-unbuilt piece.

WHAT THIS IS: a single Vision call made AFTER correction runs, shown
BOTH the original and corrected photo, asked a holistic "does anything
here look artificially altered, streaked, or unnatural" question. This
plays to what Vision models are actually reliable at (a holistic "does
this look wrong" read) rather than asking them to originate calibrated
correction values (already rejected for Stage 3) or self-diagnose a
subtle pixel-level artifact from a single image with no reference.

WHY THIS EXISTS NOW: two independent HDR gain-map fix attempts (Aug 1,
2026) both failed real-photo testing -- and both failures were only
caught because Sam manually measured local-contrast deltas on the
output. A human retoucher wouldn't have needed that measurement; they'd
have looked at the result and said "no, that streak isn't right." This
module is that same look, automated. It does NOT fix the HDR streak bug
-- hdrRecover.py is unchanged, root cause (gain-map/base-photo gradient
covariance) remains unresolved -- but it gives a real, running signal on
whether that bug (and anything like it) is actually rare at real volume,
and a backstop that can flag a bad photo for manual review even with no
correction-side fix in place.

Sends BOTH original and corrected images in one call (not corrected
alone) specifically so Vision can distinguish an artifact INTRODUCED by
correction from a pre-existing quality issue in the source photo itself
-- a single-image QC call has no way to make that distinction and would
false-flag on ordinary difficult source photos.

Rollout plan (same discipline as Level 0 and Stage 1 -- do not skip
steps): ships in LOG-ONLY mode. The QC verdict is computed and included
in the JSON output under "level4QC" on every corrected photo, but never
blocks, rejects, or changes what's returned to the browser. Review a
real batch for: (a) does it actually flag a real HDR-streak-type photo
if one comes through, (b) false-positive rate on normal good
corrections -- before this is wired to anything user-facing (e.g. a
results-screen warning badge, or an auto-hold-for-review queue).
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

LEVEL4_QC_ENABLED = os.environ.get("LEVEL4_QC_ENABLED", "true").lower() not in ("false", "0", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
QC_MODEL = os.environ.get("LEVEL4_QC_MODEL", "claude-haiku-4-5-20251001")
TIMEOUT_SECONDS = int(os.environ.get("LEVEL4_QC_TIMEOUT_SECONDS", "12"))
MAX_VISION_EDGE = 1568  # matches level0_scene_classifier.py / level2_diagnosis.py

SYSTEM_PROMPT = """You are a professional real estate photo retoucher doing \
a final quality check. You will see two photos of the same room or scene: \
the ORIGINAL (before correction) and the CORRECTED version (after automatic \
brightness/color correction was applied).

Your ONLY job is to spot anything the correction INTRODUCED that looks \
artificial, unnatural, or wrong -- NOT pre-existing issues in the original \
photo (bad furniture arrangement, clutter, a plain original exposure, etc. \
are not your concern). Look specifically for:
- Streaks, bands, or gradients that don't follow real light/surface physics
- A visible seam or edge where correction was applied unevenly
- Blown-out highlights or crushed shadows that lost real detail
- Color shifts that look artificial rather than like natural light
- Any area that looks "processed" or synthetic rather than photographic

If the corrected version looks like a clean, natural, professionally-lit \
photo (even if quite different in brightness/color from the original --  \
that's the correction working as intended), say so plainly.

Respond with ONLY strict JSON, no markdown fences, no preamble:
{"looksArtificial": true | false, "confidence": "high" | "medium" | "low", "issue": "<one sentence describing what's wrong, or null if none>", "location": "<brief description of where in the frame, or null if none>"}"""


def _call_vision_api(original_b64: str, corrected_b64: str, media_type: str) -> dict:
    """Raw urllib.request call, no SDK -- same proven pattern as
    level2_diagnosis.py and level0_scene_classifier.py."""
    body = json.dumps({
        "model": QC_MODEL,
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "ORIGINAL:"},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": original_b64}},
                {"type": "text", "text": "CORRECTED:"},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": corrected_b64}},
                {"type": "text", "text": "Does the corrected version show anything artificially altered, streaked, or unnatural that isn't in the original?"},
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

    # Use raw_decode rather than a plain json.loads: despite the prompt
    # saying "ONLY JSON, no prose", models occasionally append a trailing
    # sentence after a complete, valid JSON object (confirmed on a real
    # response here -- "Extra data: line 2 column 1"). raw_decode parses
    # just the first valid JSON value and ignores anything after it,
    # rather than failing the whole call over trailing chatter.
    return json.JSONDecoder().raw_decode(text)[0]


def _encode_for_vision(img) -> Optional[str]:
    h, w = img.shape[:2]
    scale = min(1.0, MAX_VISION_EDGE / float(max(h, w)))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def qc_check(original_img, corrected_img) -> dict:
    """
    Compares original vs. corrected image via Vision, looking for
    artifacts introduced by correction. LOG-ONLY by design -- this never
    blocks or changes the pipeline's output; callers should surface the
    result (e.g. under "level4QC" in the JSON output) for review, not
    act on it yet.

    Never raises. On any failure (disabled, missing key, timeout, bad
    JSON), returns {"looksArtificial": None, ...} -- treat that as "QC
    unavailable for this photo," not "photo is clean."
    """
    report = {"enabled": LEVEL4_QC_ENABLED, "called": False, "error": None}

    if not LEVEL4_QC_ENABLED:
        report["error"] = "disabled_via_env"
        return {"looksArtificial": None, "confidence": None, "issue": None, "location": None, **report}
    if not ANTHROPIC_API_KEY:
        logger.warning("level2_qc: missing ANTHROPIC_API_KEY, degrading to None")
        report["error"] = "missing_ANTHROPIC_API_KEY"
        return {"looksArtificial": None, "confidence": None, "issue": None, "location": None, **report}

    try:
        original_b64 = _encode_for_vision(original_img)
        corrected_b64 = _encode_for_vision(corrected_img)
        if original_b64 is None or corrected_b64 is None:
            report["error"] = "jpeg_encode_failed"
            return {"looksArtificial": None, "confidence": None, "issue": None, "location": None, **report}

        report["called"] = True
        result = _call_vision_api(original_b64, corrected_b64, "image/jpeg")

        looks_artificial = result.get("looksArtificial")
        if not isinstance(looks_artificial, bool):
            logger.warning(f"level2_qc: unexpected looksArtificial value '{looks_artificial}'")
            report["error"] = f"unparseable_looksArtificial:{looks_artificial}"
            return {"looksArtificial": None, "confidence": None, "issue": None, "location": None, **report}

        if looks_artificial:
            logger.info(
                f"level2_qc: FLAGGED — confidence={result.get('confidence')} "
                f"issue='{result.get('issue')}' location='{result.get('location')}'"
            )

        return {
            "looksArtificial": looks_artificial,
            "confidence": result.get("confidence"),
            "issue": result.get("issue"),
            "location": result.get("location"),
            **report,
        }

    except Exception as e:  # noqa: BLE001 -- any failure must fall back, never crash the pipeline
        logger.warning(f"level2_qc: failed ({type(e).__name__}: {e}), degrading to None")
        report["error"] = f"{type(e).__name__}: {e}"
        return {"looksArtificial": None, "confidence": None, "issue": None, "location": None, **report}
