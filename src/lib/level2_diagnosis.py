#!/usr/bin/env python3
"""
level2_diagnosis.py — Stage 1 of the four-stage architecture (Aug 2026
design session): Vision diagnoses -> Vision routes regions (existing
Level 2) -> classical CV executes and measures (unchanged) -> Vision QCs
the result (not yet built).

WHAT THIS IS: a single Vision call made BEFORE any correction runs,
combining the photo itself with the pipeline's own real computed stats
(image_stats, white_surface_stats, shadow_highlight_stats — the same
functions the do-no-harm gate already uses). Returns a qualitative
diagnosis of WHAT KIND of correction this photo needs, not calibrated
correction values.

WHY BOTH THE PHOTO AND THE STATS, NOT JUST ONE:
  - Stats alone can't distinguish "backlit, protect the highlights" from
    "flat and evenly underexposed, safe to lift everything" -- both can
    produce a similar mean_luma/shadowFraction reading, but they need
    opposite correction strategies (a uniform lift strong enough to fix
    a backlit foreground would blow out highlights that are currently
    fine). Confirmed directly on a real photo (IMG_8310): mean_luma 88.2,
    shadowFraction 0.264 reads as "needs a strong global lift" from
    numbers alone, but the photo shows correctly-exposed windows behind
    a silhouetted foreground chair -- a global lift strong enough for the
    foreground would blow those windows out. Only the visual read
    catches this.
  - The photo alone, without real numbers, means Vision is ESTIMATING
    exposure/color-cast magnitude from appearance -- the same
    non-deterministic guessing problem already rejected for stage 3
    (extracting actual correction values). A model asked "how underexposed
    is this" from the image alone will give a different answer on repeat
    calls. Grounding it in real computed numbers removes that guessing
    for the parts classical CV already measures accurately, and confines
    Vision's job to the qualitative judgment call it's actually suited
    for: what KIND of problem is this.

WHAT THIS DOES NOT DO: it does not compute correction strength, gamma
values, or masks. That's still classical CV (stage 3, unchanged) and the
existing Level 2 region routing (stage 2, unchanged). This module's
output is meant to inform which correction path/emphasis to use, not to
replace any of the deterministic math.

STATUS UPDATE (Aug 4, 2026): the "prototype, not yet wired in" note above
is stale -- this has been live and authoritative in smartCorrect.py's
main() since at least Aug 3. Every real batch run that day
(ai_diagnosis_mixed_light_temperature, ai_diagnosis_already_acceptable,
ai_diagnosis_reduced_intensity, etc. in modulesApplied) shows this
diagnosis actively driving _diagnosis_adjusted_intensity(). Left the
original note above for history rather than deleting it, but do not
treat this module as unproven or optional -- it is a live, production
routing signal today.
"""

import base64
import json
import os
import urllib.error
import urllib.request

import cv2
import numpy as np

LEVEL2_DIAGNOSIS_ENABLED = os.environ.get("LEVEL2_DIAGNOSIS_ENABLED", "true").lower() not in ("false", "0", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DIAGNOSIS_MODEL = os.environ.get("LEVEL2_DIAGNOSIS_MODEL", "claude-haiku-4-5-20251001")
DIAGNOSIS_TIMEOUT_SECONDS = int(os.environ.get("LEVEL2_DIAGNOSIS_TIMEOUT_SECONDS", "30"))  # raised
# from a hardcoded 12s with NO env override at all (Aug 4, 2026): this exact
# gap -- a Haiku-era timeout with no way to raise it -- was already found
# and fixed in level2_qc.py and level2_vision_regions.py after each hit a
# real live timeout under Sonnet. This file never got the same treatment.
# Fixed proactively here, informed by those two real incidents, rather
# than waiting to hit it live a third time.
MAX_VISION_EDGE = 1568  # matches the existing cap used elsewhere in this codebase

PROMPT_TEMPLATE = """You are a professional real estate photo retoucher doing the FIRST look at a photo before any correction is applied — the same glance a human retoucher gives before touching anything.

You have two things: the photo itself, and the pipeline's own real computed measurements from it (not your estimate — actual numbers):

{stats_json}

Diagnose what KIND of correction this photo needs. Do not estimate correction values (no gamma numbers, no percentages) — that part is handled separately by deterministic math using the real measurements above. Your job is the qualitative call a human makes by looking: what TYPE of lighting problem is this, and what should the correction strategy protect against.

Common categories (use your own words if none fit well):
- "backlit_mixed_lighting" -- a subject or foreground is in shadow/silhouette while a background area (often a window) is correctly exposed. Needs targeted shadow/fill lift, NOT a uniform global lift, which would blow out the already-correct background.
- "flat_evenly_underexposed" -- the whole frame is uniformly dark with no strong bright/dark split. Safe for a more uniform lift.
- "color_cast" -- exposure is roughly fine but there's a visible color tint (warm/cool/green) that needs correcting, independent of brightness.
- "already_acceptable" -- looks like a professionally-lit, well-exposed photo already; minimal correction needed regardless of what any single threshold says.
- "mixed_light_temperature" -- warm interior lighting (lamps, fixtures) mixed with cool daylight from windows in the same frame, creating competing color casts in different regions.

Return ONLY strict JSON, no prose, no markdown fences:
{{"diagnosis": "<category>", "confidence": "<low|medium|high>", "reasoning": "<one sentence, what you SEE that the numbers alone wouldn't tell you>", "correctionEmphasis": "<one sentence, what the correction should prioritize or protect>"}}"""


def _call_vision_api(image_b64: str, media_type: str, stats_json: str) -> dict:
    body = json.dumps({
        "model": DIAGNOSIS_MODEL,
        # Raised from 512 (Aug 4, 2026): not yet a confirmed live failure
        # here the way level2_qc.py's 300 was, but the same shape of risk
        # -- a model given room to reason fully (this prompt explicitly
        # asks for a one-sentence reasoning AND a one-sentence correction
        # emphasis) can run longer than a token budget sized for a terser
        # model's typical output. Fixed proactively rather than waiting
        # for a repeat of that exact incident.
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": PROMPT_TEMPLATE.format(stats_json=stats_json)},
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
    with urllib.request.urlopen(req, timeout=DIAGNOSIS_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    text = "".join(
        block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
    ).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()

    # Hardened parsing (Aug 4, 2026): this was a plain json.loads(), not
    # protected against either failure mode already confirmed real
    # elsewhere in this same pipeline -- trailing chatter after valid
    # JSON (level2_qc.py: "Extra data: line 2 column 1") or, worse,
    # leading prose BEFORE the JSON (level2_qc.py again, once Sonnet was
    # given room to actually narrate its reasoning per this prompt's own
    # "one sentence" asks). Same fix as that file: find the first '{' and
    # parse from there, rather than assume the JSON is the very first
    # character of the response.
    if not text:
        raise ValueError("empty_response_text")
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"no_json_object_found -- raw_text={text[:300]!r}")
    try:
        return json.JSONDecoder().raw_decode(text, brace_start)[0]
    except json.JSONDecodeError as e:
        raise ValueError(f"{e} -- raw_text={text[:300]!r}") from e


def diagnose(img, image_stats_dict, white_surface_stats_dict, shadow_highlight_stats_dict, is_exterior):
    """Returns (diagnosis, report). diagnosis is None on disable/failure —
    callers must treat that as 'no diagnosis available, proceed with
    existing threshold-based logic unchanged.' Never raises."""
    # "model" included from the start, not just on success (Aug 4, 2026,
    # same fix already applied to level2_vision_regions.py and
    # level2_qc.py after each cost real diagnostic time from not having
    # it): reflects DIAGNOSIS_MODEL's resolved value on every path,
    # success or not, so which model served a given diagnosis is never
    # a guess.
    report = {"enabled": LEVEL2_DIAGNOSIS_ENABLED, "called": False, "model": DIAGNOSIS_MODEL, "error": None}

    if not LEVEL2_DIAGNOSIS_ENABLED:
        report["error"] = "disabled_via_env"
        return None, report
    if not ANTHROPIC_API_KEY:
        report["error"] = "missing_ANTHROPIC_API_KEY"
        return None, report
    if is_exterior:
        report["error"] = "skipped_exterior_photo"
        return None, report

    h, w = img.shape[:2]
    scale = min(1.0, MAX_VISION_EDGE / float(max(h, w)))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img

    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        report["error"] = "jpeg_encode_failed"
        return None, report
    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    stats_json = json.dumps({
        "imageStats": image_stats_dict,
        "whiteSurfaceStats": white_surface_stats_dict,
        "shadowHighlightStats": shadow_highlight_stats_dict,
    }, indent=2)

    try:
        report["called"] = True
        result = _call_vision_api(image_b64, "image/jpeg", stats_json)
    except Exception as e:  # noqa: BLE001 -- any failure must fall back, never crash the pipeline
        report["error"] = f"{type(e).__name__}: {e}"
        return None, report

    report["diagnosis"] = result.get("diagnosis")
    report["confidence"] = result.get("confidence")
    return result, report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python level2_diagnosis.py <image_path>")
        sys.exit(1)

    # Reuse the real production stats functions rather than re-implementing them
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import smartCorrect as sc

    img = cv2.imread(sys.argv[1])
    stats = sc.image_stats(img)
    whites = sc.white_surface_stats(img)
    shadows = sc.shadow_highlight_stats(img)
    is_ext, _ = sc.is_exterior_daylight(img)

    diagnosis, report = diagnose(img, stats, whites, shadows, is_ext)
    print(json.dumps({"diagnosis": diagnosis, "report": report}, indent=2))
