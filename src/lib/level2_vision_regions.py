#!/usr/bin/env python3
"""
level2_vision_regions.py — Level 2 Vision pre-pass for smartCorrect.py.

REWRITTEN (Aug 3, 2026) to emit the Retoucher Schema: an open-ended list
of named regions, each carrying a controlled-vocabulary regionType,
operation, priority, protections, and confidence -- instead of the
previous two fixed categories (furniture_floor / dark_material). This is
what makes Level 2 an actual per-photo judgment call rather than a
one-size-fits-all label: a photo can now come back with, say, "backlit
foreground chair -> shadow_recovery, primary" and "French doors ->
highlight_reduction, protect" in the SAME frame, each tied to its own
mask -- not one exclusion mask applied uniformly regardless of what's
actually wrong (or already fine) in that specific region.

Generates READ-ONLY routing masks using Claude Haiku 4.5 Vision. This
module NEVER touches pixel values -- it only classifies WHAT and WHERE,
so the existing classical CV corrections in smartCorrect.py know how to
apply their own deterministic math per region. All actual pixel
modification remains classical CV in smartCorrect.py, unchanged.

WHY THIS DOESN'T VIOLATE smartCorrect.py's "no generative model" rule
(see that file's own top docstring): Vision here plays the same role a
human retoucher's eye plays before reaching for the dodge/burn tool --
deciding what needs attention, what kind, and how urgently. It judges
and classifies, it doesn't generate pixels. If this module fails, times
out, or is disabled, smartCorrect.py's corrections fall back to their
existing heuristic-only behavior unchanged -- this is a strictly
additive routing signal, never a dependency the pipeline can't run
without.

DESIGN CARRIED FORWARD from the original (July 31, 2026 test session,
Notion: "Session handoff -- Level 2 mask test -- Haiku vs Sonnet, routing
vs hard-mask"):
  - Haiku 4.5 remains the production default -- that test found Haiku
    and Sonnet produce near-identical region calls on real hard-case
    photos; cost/quality gap is immaterial at ~$0.004-0.005/image. No
    evidence yet that the richer schema below needs the larger model,
    but this is worth re-validating once real photos have gone through
    the new prompt, not assumed to carry over automatically.
  - Coarse, GENEROUSLY PADDED boxes remain the deliverable, not tight
    polygons or pixel-accurate segmentation -- box precision was never
    validated to pixel-tight accuracy and shouldn't be trusted to that
    level regardless of how much richer the per-region metadata is now.
  - No hallucinated regions on the control (no-wall/window) exterior
    photo in the original test -- reasonable confidence against
    false-positive region invention, but that was the ORIGINAL prompt,
    not this rewrite's schema. Needs its own re-validation.

BACKWARD COMPATIBILITY: every existing call site in smartCorrect.py
(main()'s _apply_interior_stack, the exterior branch) was written
against the OLD two-key return shape and has NOT yet been updated to
consume the new `regions`/`masks` this rewrite adds -- that's real,
separate wiring work, intentionally not done in this same pass (see
Notion handoff for what's next). `get_level2_regions()` still returns
"furniture_floor_mask" and "dark_material_mask" in its return dict,
derived by unioning every new region whose regionType matches the old
category's intent, so nothing existing breaks today. New callers should
read the same dict's "regions" and "masks" keys instead.
"""

import base64
import json
import os
import urllib.error
import urllib.request

import cv2
import numpy as np

LEVEL2_VISION_MASKS_ENABLED = os.environ.get("LEVEL2_VISION_MASKS_ENABLED", "true").lower() not in ("false", "0", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VISION_MODEL = os.environ.get("LEVEL2_VISION_MODEL", "claude-haiku-4-5-20251001")
VISION_TIMEOUT_SECONDS = 12
MAX_VISION_EDGE = 1568  # matches the existing cap in autoSelect.js (Vision API 8000px ceiling)
DEFAULT_PAD_FRAC = 0.08  # pad each box by 8% of its own size on every side

# ── Controlled vocabulary (Retoucher Schema) ─────────────────────────────
# Anything Vision returns outside these sets is dropped, not guessed at --
# same "never trust an unvalidated Vision string as a code branch" rule
# used everywhere else tonight (level2_diagnosis.py's category set,
# smartCorrect.py's DIAGNOSIS_INTENSITY_MULTIPLIER table).
REGION_TYPES = {
    "dark_furniture", "dark_stone", "dark_fixture",
    "furniture", "flooring", "rug_textile",
    "window_glass", "mirror", "screen_display",
    "light_wall_trim", "sky_exterior", "other",
}
OPERATIONS = {
    "exposure_lift", "shadow_recovery", "highlight_reduction",
    "white_balance_adjustment", "mixed_light_balance",
    "texture_enhancement", "clarity_reduction",
    "saturation_protection", "hue_protection", "no_change",
}
PRIORITIES = {"primary", "secondary", "protect"}

# regionTypes that feed the BACKWARD-COMPATIBLE legacy masks below.
# "dark_furniture" deliberately contributes to BOTH -- a dark couch is
# both furniture (exclude from clean_whites) and a dark material
# (protect from brightness lift), matching the old scheme's behavior
# for that same case.
LEGACY_FURNITURE_FLOOR_TYPES = {"dark_furniture", "furniture", "flooring", "rug_textile"}
LEGACY_DARK_MATERIAL_TYPES = {"dark_furniture", "dark_stone", "dark_fixture"}

PROMPT_TEMPLATE = """You are a senior professional photo retoucher reviewing a real estate \
listing photo before correction, the way you would before reaching for \
dodge/burn or a graduated filter. Return ONLY strict JSON, no prose, no \
markdown fences.

Identify any regions in this photo that need SPECIFIC attention -- either \
because they need a targeted correction different from what the rest of \
the frame needs, or because they must be PROTECTED from whatever \
correction the rest of the frame gets. Most photos will have a FEW such \
regions, not many -- do not invent regions just to fill out the list, and \
return an empty list if nothing in the frame needs individual treatment \
beyond a normal, uniform correction.

For EACH region, return:
- "regionId": a short human-readable label, e.g. "foreground_chair"
- "box": [x1, y1, x2, y2] in pixel coordinates for THIS image \
(width={w}, height={h}). Use a generously PADDED box -- err larger, not \
tighter. This is a coarse routing signal, not a precise boundary.
- "regionType": exactly one of: dark_furniture, dark_stone, dark_fixture, \
furniture, flooring, rug_textile, window_glass, mirror, screen_display, \
light_wall_trim, sky_exterior, other
- "operation": exactly one of: exposure_lift, shadow_recovery, \
highlight_reduction, white_balance_adjustment, mixed_light_balance, \
texture_enhancement, clarity_reduction, saturation_protection, \
hue_protection, no_change
- "priority": exactly one of: primary (this needs real correction), \
secondary (mild correction, lower priority than primary regions), \
protect (this must NOT receive whatever correction the rest of the \
frame gets)
- "protections": a list of any of: preserve_hue, preserve_black_depth, \
preserve_texture, preserve_exterior_view, preserve_source_detail, \
exclude_from_shadow_lift, exclude_from_white_balance, \
exclude_from_color_finish, exclude_from_highlight_reduction -- only the \
ones that genuinely apply, can be an empty list
- "confidence": a number from 0 to 1 for how sure you are this region \
genuinely needs distinct treatment
- "reasoning": one short sentence

Examples of what belongs on this list: a foreground subject sitting in \
shadow against a bright window behind it (shadow_recovery, primary); a \
genuinely black countertop or dark leather that should stay dark, not \
get brightened like the rest of the room (protect, preserve_black_depth); \
an already well-exposed window or doorway that shouldn't be pushed \
brighter along with a dim room (highlight_reduction, protect); a TV or \
monitor screen showing real content, which should never be treated like \
a wall or trim surface (screen_display, no_change, protect).

Do NOT include ordinary, evenly-lit content that just needs whatever \
uniform correction the rest of the frame gets -- that's the default \
behavior and doesn't need a region entry.

Return exactly this shape:
{{"regions": [{{"regionId": "...", "box": [x1,y1,x2,y2], "regionType": "...", \
"operation": "...", "priority": "...", "protections": [...], \
"confidence": 0.0, "reasoning": "..."}}, ...]}}

If nothing in this photo needs individual treatment, return {{"regions": []}}."""


def _call_vision_api(image_b64: str, media_type: str, w: int, h: int) -> dict:
    body = json.dumps({
        "model": VISION_MODEL,
        "max_tokens": 2048,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": PROMPT_TEMPLATE.format(w=w, h=h)},
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
    with urllib.request.urlopen(req, timeout=VISION_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    text = "".join(
        block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
    ).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _boxes_to_mask(boxes, shape, pad_frac=DEFAULT_PAD_FRAC, feather_frac=0.035, feather_min_px=18):
    """Rasterize [x1,y1,x2,y2] boxes into a SOFT (H,W) float32 mask in
    [0,1], not a hard boolean. Unchanged from the original -- this
    technique is generic to any box-shaped region regardless of what
    category it represents, and the feathering fix behind it (see
    below) applies exactly the same way to every region this file now
    produces, not just the original two categories.

    FEATHERING (added after first real-photo test, Aug 2026): the first
    version returned a hard binary mask. Confirmed directly on real
    photos that this creates a visible artificial seam wherever a box
    edge falls in the middle of a large continuous surface it doesn't
    fully cover -- e.g. Vision's furniture_floor box for a large area
    rug didn't reach the rug's actual edges, so the rug got corrected
    (neutralized/lifted) right up to the box boundary and untouched
    inside it, producing a visible rectangular seam with no photographic
    basis (nothing in the room actually changes at that line). Same
    mechanism produced a visible streak where a box edge crossed window
    glass. A box is a coarse routing signal, not a tight boundary --
    treating its edge as a hard cutoff was the bug, not the box itself.

    Fix: rasterize the (already padded) hard box, then apply a Gaussian
    blur wide enough that the box's own edges become a gradual falloff
    rather than a step. feather_frac scales the blur sigma to image size
    so this behaves consistently across different photo resolutions."""
    h, w = shape[:2]
    hard = np.zeros((h, w), dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * pad_frac, bh * pad_frac
        x1c = max(0, int(round(x1 - pad_x)))
        y1c = max(0, int(round(y1 - pad_y)))
        x2c = min(w, int(round(x2 + pad_x)))
        y2c = min(h, int(round(y2 + pad_y)))
        if x2c > x1c and y2c > y1c:
            hard[y1c:y2c, x1c:x2c] = 1.0

    if not np.any(hard):
        return hard

    sigma = max(feather_min_px, feather_frac * max(h, w))
    soft = cv2.GaussianBlur(hard, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(soft, 0.0, 1.0)


def _sanitize_region(raw, index):
    """Validates one raw region dict from Vision against the controlled
    vocabulary. Returns None (drop it) rather than guess at anything
    outside the enum -- an unrecognized regionType/operation/priority is
    treated as Vision not being usable for this specific region, not as
    a new category to invent handling for on the fly."""
    box = raw.get("box")
    if not (isinstance(box, list) and len(box) == 4):
        return None
    region_type = raw.get("regionType")
    operation = raw.get("operation")
    priority = raw.get("priority")
    if region_type not in REGION_TYPES or operation not in OPERATIONS or priority not in PRIORITIES:
        return None
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    protections = raw.get("protections")
    protections = [p for p in protections if isinstance(p, str)] if isinstance(protections, list) else []

    mask_id = f"mask_{index:02d}"
    region_id = raw.get("regionId")
    region_id = region_id if isinstance(region_id, str) and region_id.strip() else mask_id

    return {
        "regionId": region_id,
        "maskId": mask_id,
        "regionType": region_type,
        "operation": operation,
        "priority": priority,
        "protections": protections,
        "confidence": confidence,
        "reasoning": raw.get("reasoning") if isinstance(raw.get("reasoning"), str) else None,
    }, box


def get_level2_regions(img):
    """Returns (level2_regions, report).

    level2_regions = {
        # NEW (Aug 3, 2026):
        "regions": [ {regionId, maskId, regionType, operation, priority,
                       protections, confidence, reasoning, box}, ... ],
        "masks": { maskId: float32 (H,W) ndarray in [0,1], ... },
        # BACKWARD-COMPATIBLE (unchanged shape/meaning from before):
        "furniture_floor_mask": float32 (H,W) ndarray in [0,1], or None,
        "dark_material_mask": float32 (H,W) ndarray in [0,1], or None,
    }
    All masks are SOFT (heavily feathered, see _boxes_to_mask) -- treat
    as a continuous weight, not a boolean. Multiply, don't index/AND.

    On any failure/disable: "regions" is [], "masks" is {}, and the two
    legacy keys are None -- callers must treat that as "no exclusion/no
    extra protection, no regional signal," i.e. fall back to whatever
    heuristic-only behavior already exists. This function never raises;
    every failure mode is caught and reported instead.
    """
    empty = {"regions": [], "masks": {}, "furniture_floor_mask": None, "dark_material_mask": None}
    report = {"enabled": LEVEL2_VISION_MASKS_ENABLED, "called": False, "error": None}

    if not LEVEL2_VISION_MASKS_ENABLED:
        report["error"] = "disabled_via_env"
        return empty, report

    if not ANTHROPIC_API_KEY:
        report["error"] = "missing_ANTHROPIC_API_KEY"
        return empty, report

    h, w = img.shape[:2]
    scale = min(1.0, MAX_VISION_EDGE / float(max(h, w)))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    sh, sw = small.shape[:2]

    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        report["error"] = "jpeg_encode_failed"
        return empty, report

    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    try:
        report["called"] = True
        result = _call_vision_api(image_b64, "image/jpeg", sw, sh)
    except Exception as e:  # noqa: BLE001 -- any Vision failure must fall back, never crash the pipeline
        report["error"] = f"{type(e).__name__}: {e}"
        return empty, report

    inv_scale = (1.0 / scale) if scale < 1.0 else 1.0
    raw_regions = result.get("regions", [])
    if not isinstance(raw_regions, list):
        report["error"] = "malformed_response: 'regions' is not a list"
        return empty, report

    regions = []
    masks = {}
    dropped_count = 0
    ff_boxes, dm_boxes = [], []

    for i, raw in enumerate(raw_regions):
        try:
            sanitized = _sanitize_region(raw, i)
        except (TypeError, ValueError, AttributeError):
            sanitized = None
        if sanitized is None:
            dropped_count += 1
            continue
        region, box = sanitized
        try:
            scaled_box = [c * inv_scale for c in box]
        except (TypeError, ValueError):
            dropped_count += 1
            continue

        mask = _boxes_to_mask([scaled_box], img.shape)
        masks[region["maskId"]] = mask
        # Exposed on the region dict itself (Aug 3, 2026, diagnostic
        # patch): previously only the final feathered mask was kept --
        # once a region looked wrong on a real photo (oversized protect
        # coverage on IMG_8315/IMG_8317, confirmed via the new debug
        # overlay), there was no way to tell whether Vision itself
        # returned a loose box or whether _boxes_to_mask's padding/
        # feathering was responsible, without instrumenting this file
        # directly. Full-image pixel coordinates, already inv_scale-
        # corrected -- same coordinate space smartCorrect.py's `img`
        # is in, so this can be drawn directly without further math.
        region["box"] = [round(c, 1) for c in scaled_box]
        regions.append(region)

        if region["regionType"] in LEGACY_FURNITURE_FLOOR_TYPES:
            ff_boxes.append(scaled_box)
        # BUG FOUND AND FIXED (Aug 3, 2026, real photo IMG_8317): a
        # fireplace's dark glass front got tagged regionType=
        # "screen_display" (Vision's reasonable-but-wrong read of a dark
        # reflective surface) instead of "dark_fixture" -- and since
        # main() isn't wired to the new regions/masks yet, ONLY this
        # legacy-derived mask actually protects anything today. A
        # regionType outside LEGACY_DARK_MATERIAL_TYPES meant zero
        # protection reached this genuinely protect-priority region,
        # confirmed by real pixel measurement: this exact photo's firebox
        # moved +20 luma this run vs. +16 under the OLD dedicated
        # dark-material prompt -- worse, not better.
        #
        # Fix: ANY region tagged priority=="protect" feeds this legacy
        # fallback, regardless of regionType. The whole meaning of
        # `protect` is "don't brighten this like the rest of the frame,"
        # which is exactly what dark_material_mask does for
        # mls_brightness_lift today -- this doesn't depend on getting
        # the regionType label exactly right, and closes the gap
        # immediately rather than waiting on a prompt fix (which may
        # also be worth doing, but shouldn't be the ONLY fix for
        # something this consequential).
        if region["regionType"] in LEGACY_DARK_MATERIAL_TYPES or region["priority"] == "protect":
            dm_boxes.append(scaled_box)

    ff_mask = _boxes_to_mask(ff_boxes, img.shape) if ff_boxes else np.zeros((h, w), dtype=np.float32)
    dm_mask = _boxes_to_mask(dm_boxes, img.shape) if dm_boxes else np.zeros((h, w), dtype=np.float32)

    report["regionCount"] = len(regions)
    report["droppedCount"] = dropped_count
    # Kept for anything still reading these exact report fields.
    report["furnitureFloorBoxCount"] = len(ff_boxes)
    report["darkMaterialBoxCount"] = len(dm_boxes)

    return {
        "regions": regions,
        "masks": masks,
        "furniture_floor_mask": ff_mask,
        "dark_material_mask": dm_mask,
    }, report
