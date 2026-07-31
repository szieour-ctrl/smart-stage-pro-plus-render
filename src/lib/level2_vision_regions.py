#!/usr/bin/env python3
"""
level2_vision_regions.py — Level 2 Vision pre-pass for smartCorrect.py.

Generates READ-ONLY routing masks (furniture/floor exclusion, dark-material
protection) using Claude Haiku 4.5 Vision. This module NEVER touches pixel
values — it only classifies WHERE regions are, so the existing classical CV
corrections in smartCorrect.py know where to apply or skip their own
deterministic math. All actual pixel modification remains classical CV in
smartCorrect.py, unchanged.

WHY THIS DOESN'T VIOLATE smartCorrect.py's "no generative model" rule
(see that file's own top docstring): Vision here plays the same role a
human retoucher's eye plays before reaching for the dodge/burn tool —
deciding what's furniture vs. trim, what's a real dark material vs. a
shadow. It classifies, it doesn't generate pixels. If this module fails,
times out, or is disabled, smartCorrect.py's corrections fall back to
their existing heuristic-only behavior unchanged — this is a strictly
additive routing signal, never a dependency the pipeline can't run without.

DESIGN, per the July 31, 2026 test session (Notion: "Session handoff —
Level 2 mask test — Haiku vs Sonnet, routing vs hard-mask"):
  - Haiku 4.5 and Sonnet 5 produced near-identical region calls on real
    hard-case photos (dark granite, backlit windows, dark leather +
    hardwood). Haiku 4.5 is the production default; cost/quality gap is
    immaterial at ~$0.004-0.005/image.
  - Coarse, GENEROUSLY PADDED boxes are the deliverable, not tight
    polygons or pixel-accurate segmentation — that test found the model
    does real material reasoning (specular highlights, grain, gradient
    falloff vs. hard-edged uniform darkness) well enough for a routing
    signal, but box precision was never validated to pixel-tight
    accuracy and shouldn't be trusted to that level.
  - No hallucinated regions on the control (no-wall/window) exterior
    photo in that test — reasonable confidence against false-positive
    region invention, but this hasn't been tested at production volume.

TWO INTEGRATION POINTS this feeds (see smartCorrect.py):
  1. furniture_floor_mask -> subtracted from white_surface_stats()'s
     white_mask before clean_whites_adaptive() runs. This is the
     boundary-critical case: white_mask is a proxy for "wall/trim/
     ceiling/cabinet," and when it's wrong it repaints real color
     (A/B channels) on whatever it caught — a light wood floor or
     cream sofa fabric can get miscategorized as white trim and
     desaturated. Because the risk here is misclassifying material IN,
     a padded EXCLUSION box is inherently safer than a tight inclusion
     boundary: over-excluding only costs a little legitimate trim near
     furniture; under-excluding is the actual compliance risk.
  2. dark_material_mask -> unioned into mls_brightness_lift()'s existing
     luma-threshold dark-material protection. Today that protection is
     a blunt L<40 taper-to-L=140 rule that can't distinguish a black
     granite countertop from a shadow at the same brightness. This mask
     lets genuinely dark materials outside that luma window (e.g. a
     medium-dark wood table around L~110) also get protected.
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

PROMPT_TEMPLATE = """You are analyzing a real estate listing photo for a photo-correction pipeline. Return ONLY strict JSON, no prose, no markdown fences.

Identify bounding boxes for two region categories. Use generously PADDED boxes -- err on the side of larger boxes, not tight ones. Coordinates are pixel values in this exact image (width={w}, height={h}), format [x1, y1, x2, y2].

1. "furniture_floor": movable furniture (sofas, chairs, tables, beds, rugs, decor) AND flooring (carpet, hardwood, tile) -- anything that is NOT a wall, ceiling, or built-in trim/cabinet surface.
2. "dark_material": any region where the material itself is genuinely dark by design -- black/dark countertops, dark leather or fabric upholstery, dark wood furniture, black fixture surrounds, dark stone. Do NOT include areas that are dark only because of shadow/lighting with no distinct dark material -- those should stay eligible for brightening.

Return exactly this shape:
{{"furniture_floor": [[x1,y1,x2,y2], ...], "dark_material": [[x1,y1,x2,y2], ...]}}

If a category has no regions in this photo, return an empty list for it."""


def _call_vision_api(image_b64: str, media_type: str, w: int, h: int) -> dict:
    body = json.dumps({
        "model": VISION_MODEL,
        "max_tokens": 1024,
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
    [0,1], not a hard boolean.

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
    glass. A box is a coarse routing signal, not a tight boundary (see
    level2_vision_regions.py's own design notes) -- treating its edge as
    a hard cutoff was the bug, not the box itself.

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


def get_level2_regions(img):
    """Returns (regions, report).

    regions = {
        "furniture_floor_mask": float32 (H,W) ndarray in [0,1], or None,
        "dark_material_mask": float32 (H,W) ndarray in [0,1], or None,
    }
    These are SOFT masks (heavily feathered, see _boxes_to_mask) -- treat
    as a continuous weight, not a boolean. Multiply, don't index/AND.
    None means disabled/unconfigured/failed -- callers must treat None as
    "no exclusion/no extra protection," i.e. fall back to current
    heuristic-only behavior. This function never raises; every failure
    mode is caught and reported instead.
    """
    report = {"enabled": LEVEL2_VISION_MASKS_ENABLED, "called": False, "error": None}

    if not LEVEL2_VISION_MASKS_ENABLED:
        report["error"] = "disabled_via_env"
        return {"furniture_floor_mask": None, "dark_material_mask": None}, report

    if not ANTHROPIC_API_KEY:
        report["error"] = "missing_ANTHROPIC_API_KEY"
        return {"furniture_floor_mask": None, "dark_material_mask": None}, report

    h, w = img.shape[:2]
    scale = min(1.0, MAX_VISION_EDGE / float(max(h, w)))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    sh, sw = small.shape[:2]

    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        report["error"] = "jpeg_encode_failed"
        return {"furniture_floor_mask": None, "dark_material_mask": None}, report

    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    try:
        report["called"] = True
        result = _call_vision_api(image_b64, "image/jpeg", sw, sh)
    except Exception as e:  # noqa: BLE001 -- any Vision failure must fall back, never crash the pipeline
        report["error"] = f"{type(e).__name__}: {e}"
        return {"furniture_floor_mask": None, "dark_material_mask": None}, report

    inv_scale = (1.0 / scale) if scale < 1.0 else 1.0
    try:
        ff_boxes = [[c * inv_scale for c in box] for box in result.get("furniture_floor", [])]
        dm_boxes = [[c * inv_scale for c in box] for box in result.get("dark_material", [])]
    except (TypeError, ValueError) as e:
        report["error"] = f"malformed_box_data: {type(e).__name__}: {e}"
        return {"furniture_floor_mask": None, "dark_material_mask": None}, report

    ff_mask = _boxes_to_mask(ff_boxes, img.shape) if ff_boxes else np.zeros((h, w), dtype=np.float32)
    dm_mask = _boxes_to_mask(dm_boxes, img.shape) if dm_boxes else np.zeros((h, w), dtype=np.float32)

    report["furnitureFloorBoxCount"] = len(ff_boxes)
    report["darkMaterialBoxCount"] = len(dm_boxes)

    return {"furniture_floor_mask": ff_mask, "dark_material_mask": dm_mask}, report
