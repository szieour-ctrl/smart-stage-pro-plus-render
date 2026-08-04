A"""
level2_qc.py

Stage 4 of the four-stage architecture (design session, Aug 2026):
Vision diagnoses -> Vision routes regions -> classical CV executes ->
**Vision QCs the result**. This is that last, previously-unbuilt piece.

WHAT THIS IS: a single Vision call made AFTER correction runs, shown
BOTH the original and corrected photo, asked a holistic "does anything
here look artificially altered, streaked, or unnatural" question. This
plays to what Vision models aAre actually reliable at (a holistic "does
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

SYSTEM_PROMPT = """You are a senior professional photo retoucher with 20+ years \
in real estate and architectural photography, performing a final forensic \
quality-control pass before a corrected photo ships to a listing. You are \
being shown the ORIGINAL (pre-correction) and CORRECTED (post-correction) \
versions of the same photo.

Your job is NOT to judge whether the corrected photo looks nice. Your job is \
to determine whether the correction process introduced any artifact that a \
trained editor's eye would catch and a client would eventually notice. \
Assume the correction changed exposure and color balance intentionally -- \
a uniform, frame-wide shift in brightness or color temperature is expected \
and NOT something to flag, even if it's substantial (this includes real HDR \
highlight recovery, which can legitimately brighten and add detail to a \
large portion of the frame all at once -- that alone is not a defect).

METHOD -- work through this explicitly before answering, don't skip straight \
to a gut impression:
1. Mentally divide the frame into its major surfaces: sky, foliage/trees, \
   walls, roofline, concrete/pavement, water, glass/reflective surfaces, \
   flooring, furniture, and any other large continuous material.
2. For EACH surface, compare its appearance in the original against the \
   same surface in the corrected version. Ask: did this surface change \
   UNIFORMLY (the same kind of shift in tone/color across its whole visible \
   area), or did only PART of it change while the rest of the same surface \
   did not?
3. A real, correctly-applied correction changes a given surface consistently \
   across its whole visible extent. A processing artifact typically does \
   NOT respect the surface's real boundaries -- it appears as a gradient, \
   band, streak, or patch that starts and stops in the middle of a single \
   continuous material, unrelated to any real seam, joint, shadow line, or \
   texture change present in the original photo.
4. Concrete, pavement, glass, water, and sky are the highest-risk surfaces \
   for this failure mode -- they are large, visually uniform, and any \
   unnatural gradient on them is both easiest to introduce and easiest to \
   miss on a quick look. Give these surfaces particular scrutiny before \
   concluding there's nothing wrong.
5. Also check: blown-out highlights or crushed shadows that lost real detail \
   present in the original; a visible seam or edge where correction was \
   applied unevenly; color shifts that look chemical or synthetic rather \
   than like a real light source.

Do NOT flag: pre-existing issues in the original (clutter, plain exposure, \
composition) -- those are not your concern. Do NOT flag a large uniform \
brightness or color shift by itself -- that's the correction (including HDR \
recovery) working as intended, PROVIDED it's applied evenly across each \
surface.

If, after this surface-by-surface check, nothing shows a localized, \
boundary-violating change, say so plainly with high confidence. If anything \
does, describe exactly which surface and where in the frame, even if it's \
subtle -- subtle is exactly the kind of miss a careless review would make \
and a careful one wouldn't.

CALIBRATION NOTE from a real miss: this exact check was run on a photo with \
a genuine streak -- a lightened, smooth vertical band cutting through an \
otherwise uniformly stippled/speckled concrete slab -- and returned \
looksArtificial: false at HIGH confidence. The artifact was unambiguous \
once someone looked closely at that one surface; it was missed on a first \
holistic pass across the whole frame. A confident "false" must mean you \
actually traced each high-risk surface's texture/grain from one edge to \
the other and confirmed it stays continuous -- not that the frame's overall \
brightness and color looked plausible at a glance. If you have not \
mentally traced the concrete, pavement, or any other large uniform surface \
edge-to-edge, do not report high confidence on it.

Respond with ONLY strict JSON, no markdown fences, no preamble:
{"looksArtificial": true | false, "confidence": "high" | "medium" | "low", "issue": "<one sentence describing what's wrong, or null if none>", "location": "<brief description of where in the frame, or null if none>"}"""


def _call_vision_api(original_b64: str, corrected_b64: str, media_type: str) -> dict:
    """Raw urllib.request call, no SDK -- same proven pattern as
    level2_diagnosis.py and level0_scene_classifier.py."""
    body = json.dumps({
        "model": QC_MODEL,
        # 300 was sized for Haiku's typical terse output. Confirmed real
        # failure (Aug 3, 2026): Sonnet, given the same prompt, actually
        # narrates the full surface-by-surface METHOD in visible prose
        # ("I'll work through this surface by surface carefully...
        # **Sky:**... **Foliage/Trees:**...") before ever reaching the
        # closing JSON -- and was very likely getting cut off by the 300
        # token ceiling before it got there at all. Raised generously;
        # at Sonnet's $15/MTok output rate even 1500 tokens is
        # ~$0.0225/call, trivial next to the cost of a QC call that
        # never produces a usable verdict.
        "max_tokens": 1500,
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

    # Confirmed real failure (Aug 3, 2026): a QC call returned
    # "JSONDecodeError: Expecting value: line 1 column 1" with no way to
    # tell whether Vision sent back an empty response, a refusal, or
    # something else -- the raw text was never preserved anywhere, so
    # the failure was a dead end to diagnose. Two fixes: flag the empty-
    # response case explicitly (the most likely cause of a column-1
    # failure), and on any other parse failure, attach a truncated
    # snippet of the actual raw text to the exception so it survives
    # into qc_check's report["error"] instead of vanishing.
    if not text:
        raise ValueError(f"empty_response_text (stop_reason={payload.get('stop_reason')!r})")

    # Confirmed real failure (Aug 3, 2026): raw_decode() starting at
    # position 0 assumes JSON is the FIRST thing in the text. That broke
    # the moment a model (Sonnet, narrating its reasoning per the
    # prompt's own METHOD instructions) put paragraphs of prose before
    # the JSON instead of after it -- the original raw_decode fallback
    # only ever handled TRAILING junk, not leading. Find the first '{'
    # and start there instead of assuming position 0; falls through to
    # the original behavior unchanged when the response is pure JSON
    # with nothing before it (brace_start == 0).
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"no_json_object_found -- raw_text={text[:300]!r}")

    # Use raw_decode rather than a plain json.loads: despite the prompt
    # saying "ONLY JSON, no prose", models occasionally append a trailing
    # sentence after a complete, valid JSON object (confirmed on a real
    # response here -- "Extra data: line 2 column 1"). raw_decode parses
    # just the first valid JSON value and ignores anything after it,
    # rather than failing the whole call over trailing chatter.
    try:
        return json.JSONDecoder().raw_decode(text, brace_start)[0]
    except json.JSONDecodeError as e:
        raise ValueError(f"{e} -- raw_text={text[:300]!r}") from e


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
    # "model" included from the start, not just on success (Aug 3, 2026,
    # same fix already made in level2_vision_regions.py after the
    # identical blind spot cost real time there): three QC runs came
    # back with no way to confirm whether LEVEL4_QC_MODEL had actually
    # taken effect on Railway. Reflects QC_MODEL's resolved value (env
    # override or the hardcoded default) on every path, success or not.
    report = {"enabled": LEVEL4_QC_ENABLED, "called": False, "model": QC_MODEL, "error": None}

    if not LEVEL4_QC_ENABLED:
        logger.info("level2_qc: disabled via env, skipping")
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
