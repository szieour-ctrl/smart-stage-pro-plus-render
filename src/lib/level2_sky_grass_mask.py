"""
level2_sky_grass_mask.py

Vision-based identification of SKY and GRASS (mowed lawn/turf ONLY)
regions in exterior photos -- exists for exactly one purpose: scoping
the saturation cap in oracleCorrection.py to large, homogeneous areas
where AB 723 specifically names the risk of a materially misleading
change (Identity Preservation: "Weather," "Sky character," "Cloud
formations"; a large lawn area misrepresented as healthier/greener than
captured), without touching correction strength anywhere else in the
photo.

NARROWED FROM "vegetation" TO "grass" THIS SESSION -- not a naming
tweak, a real scope correction found on a real photo. The original
version capped saturation across ALL vegetation (shrubs, ornamental
grasses, flowers, trees) using the same per-pixel ratio as sky. Tested
on the actual homeowner's real photo and found wrong: a golden ornamental
grass -- one of a dozen individually-colored plants in a mixed bed, not
a large uniform area -- lost real color character it should have kept.
The AB 723 concern this module exists to address is specifically about
LARGE, HOMOGENEOUS areas reading as a different day/condition than
captured (an entire sky, an entire lawn) -- not individually distinct
ornamental plants, where recovering their actual captured color is
exactly what correction is supposed to do, not a risk to guard against.
Applying the same cap to both was repeating the original "Protect"
mistake this whole project has been working to move away from: one
mechanism, built for one real risk, applied indiscriminately to content
that was never that risk. Sky keeps the cap. Individual ornamental
plants, shrubs, flowers, and accent grasses do not -- they rely on the
recoverability gate and the Oracle-range clamp (oracleCorrection.py)
alone, same as every other correctable surface in the photo.

WHY A SEPARATE CALL FROM level2_vision_recoverability.py: recoverability
answers "is there enough data here to trust a correction at all." This
answers a different question -- "what KIND of thing is this pixel" --
the same category distinction level0_scene_classifier.py already
established between interior/exterior judgment and this pipeline's other
Vision calls.

WHY NOT A CLASSICAL COLOR/POSITION HEURISTIC: level0_scene_classifier.py
already documents a classical heuristic failing at a structurally similar
judgment (interior/exterior from HSV color fractions) and being replaced
by Vision after real failures. A blue/green color heuristic for sky/grass
is exactly as brittle for the same reason -- a blue accent wall, an
ornamental grass that happens to be green, would fool a pure-color rule.
This is also exactly why "grass" needs Vision rather than a simple
green-hue threshold: a color rule cannot distinguish mowed turf from an
ornamental grass planting the way a scene-level judgment can.

GRID CLASSIFICATION, NOT BOUNDING BOXES (changed earlier this session --
not a stylistic choice). First version used bboxes, same as
level2_vision_recoverability.py. Tested on a real exterior photo and
found broken twice on the same image: sky visible around a roofline is a
genuinely irregular, often disconnected shape (interrupted by two roof
peaks and a tree in the test case) -- a rectangle union missed real sky
between/around those obstructions no matter how many boxes were added.
Bounding boxes are the right tool for level2_vision_recoverability.py's
regions (roughly convex, contiguous problem areas) and the wrong tool
here. A coarse grid, classified cell by cell, handles arbitrary and
disconnected shapes without this failure mode by construction.

SHADOW MODE: deliberately NOT implemented the way level0/level2_vision_
recoverability use it. Both failure directions here are low-severity by
construction: a false negative (grass cell missed) just means the
saturation cap doesn't get scoped there and behaves exactly as it did
before this module existed (not worse); a false positive just means
that cell gets a saturation ceiling it didn't strictly need -- though
per the narrowing above, a false positive that mistakes an ornamental
plant for "grass" is now the failure mode to actually watch for in real
review, since that's precisely the mistake found this session. ENABLED
is still an env-gated kill switch, and any failure degrades to "no
mask" -- the caller must treat empty/all-zero output as "apply no cap,"
never as "the whole photo is sky/grass."
"""

import os
import json
import logging
import base64
import urllib.error
import urllib.request
from typing import Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SKY_GRASS_MASK_ENABLED = os.environ.get("LEVEL2_SKY_GRASS_MASK_ENABLED", "true").lower() not in ("false", "0", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SKY_GRASS_MASK_MODEL = os.environ.get("LEVEL2_SKY_GRASS_MASK_MODEL", "claude-haiku-4-5-20251001")
TIMEOUT_SECONDS = int(os.environ.get("LEVEL2_SKY_GRASS_MASK_TIMEOUT_SECONDS", "20"))
MAX_VISION_EDGE = 1568  # matches level0/level2_diagnosis/level2_qc/level2_vision_recoverability

# Grid resolution. Coarse on purpose -- this scopes a conservative safety
# cap, not a pixel-accurate mask (same non-goal as the old bbox version's
# docstring said explicitly). 24x18 is small enough for a model to
# classify reliably cell-by-cell in one pass, fine enough to follow a
# roofline/treeline reasonably closely once feathered.
GRID_COLS = int(os.environ.get("LEVEL2_SKY_GRASS_GRID_COLS", "24"))
GRID_ROWS = int(os.environ.get("LEVEL2_SKY_GRASS_GRID_ROWS", "18"))
MASK_FEATHER_SIGMA = 15  # same value as GATE_FEATHER_SIGMA elsewhere


SYSTEM_PROMPT = """You are looking at a real estate exterior photograph, overlaid with a {cols}x{rows} grid (columns lettered A-{last_col}, rows numbered 1-{rows}, top-left cell is A1).

For EVERY cell, classify what's predominantly in it: "sky", "grass" (mowed lawn / turf ONLY -- see exclusions below), or "other" (everything else: architecture, hardscape, sidewalk, vehicles, ornamental plants, shrubs, ornamental/accent grasses, trees, flowers, mulch, gravel, potted plants -- anything that is not sky or mowed turf). A cell that's a mix should get whichever category covers more of that cell's area.

IMPORTANT -- "grass" here means mowed lawn/turf specifically, NOT vegetation broadly. Do NOT classify as "grass": ornamental grasses (fountain grass, pampas grass, or any individually-planted accent grass with a distinct clump/spray form rather than a mowed carpet), shrubs, flowering plants, groundcover, trees, or any individually-distinguishable specimen plant. Those are all "other" -- they should NOT be capped the way a large uniform lawn is. This distinction matters: a single ornamental grass plant, a flower bed, a shrub -- their actual captured color is exactly what should be preserved/correctable, not treated like a large-area lawn.

This does not need to be precise at the cell boundary -- it's scoping a conservative safety check, not a pixel mask. Sky is often irregular and broken up by rooflines, trees, or other obstructions into multiple disconnected patches -- classify each cell on its own merits, don't assume sky is only in one contiguous block. The same applies to lawn if present.

Return ONLY strict JSON, no markdown fences, no prose outside the JSON, as a single string, row by row, top to bottom, one character per cell: 'S' for sky, 'G' for grass (mowed lawn/turf only), '.' for other. Each row must be exactly {cols} characters.

{{"grid": ["<row 1, {cols} chars>", "<row 2, {cols} chars>", ... "<row {rows}, {cols} chars>"]}}"""


def _encode_for_vision(img) -> Optional[tuple]:
    h, w = img.shape[:2]
    scale = min(1.0, MAX_VISION_EDGE / float(max(h, w)))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii"), small.shape[:2]


def _call_vision_api(image_b64: str, media_type: str) -> dict:
    """Raw urllib.request call, no SDK -- same pattern as every other Vision
    module in this codebase."""
    last_col = chr(ord('A') + GRID_COLS - 1)
    prompt_text = SYSTEM_PROMPT.format(cols=GRID_COLS, rows=GRID_ROWS, last_col=last_col)
    body = json.dumps({
        "model": SKY_GRASS_MASK_MODEL,
        "max_tokens": 1536,  # sized for GRID_ROWS strings of GRID_COLS chars each,
        # plus JSON overhead -- comfortably more than 24x18=432 chars needs.
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
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

    if not text:
        raise ValueError(f"empty_response_text (stop_reason={payload.get('stop_reason')!r})")
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"no_json_object_found -- raw_text={text[:300]!r}")
    try:
        return json.JSONDecoder().raw_decode(text, brace_start)[0]
    except json.JSONDecodeError as e:
        raise ValueError(f"{e} -- raw_text={text[:300]!r}") from e


def identify_sky_vegetation(img) -> tuple:
    """
    Returns (result, report). result["grid"] is a list of GRID_ROWS strings,
    each GRID_COLS characters ('S'/'V'/'.'), or [] on any failure. Never
    raises. Callers must treat an empty/malformed grid as "apply no cap,"
    not as "nothing is sky/vegetation" -- see rasterize_sky_veg_mask.
    """
    report = {"enabled": SKY_GRASS_MASK_ENABLED, "called": False, "model": SKY_GRASS_MASK_MODEL, "error": None}

    if not SKY_GRASS_MASK_ENABLED:
        report["error"] = "disabled_via_env"
        return {"grid": [], "error": "disabled_via_env"}, report
    if not ANTHROPIC_API_KEY:
        logger.warning("level2_sky_grass_mask: missing ANTHROPIC_API_KEY, degrading to empty")
        report["error"] = "missing_ANTHROPIC_API_KEY"
        return {"grid": [], "error": "missing_ANTHROPIC_API_KEY"}, report

    try:
        enc = _encode_for_vision(img)
        if enc is None:
            report["error"] = "jpeg_encode_failed"
            return {"grid": [], "error": "jpeg_encode_failed"}, report
        image_b64, _sent_shape = enc

        report["called"] = True
        parsed = _call_vision_api(image_b64, "image/jpeg")

        grid = parsed.get("grid")
        if not isinstance(grid, list) or len(grid) != GRID_ROWS or any(
            not isinstance(row, str) or len(row) != GRID_COLS for row in grid
        ):
            report["error"] = f"malformed_grid (expected {GRID_ROWS}x{GRID_COLS}): {str(grid)[:200]!r}"
            return {"grid": [], "error": report["error"]}, report

        return {"grid": grid, "error": None}, report

    except Exception as e:  # noqa: BLE001 -- any failure must fall back, never crash the pipeline
        logger.warning(f"level2_sky_grass_mask: failed ({type(e).__name__}: {e}), degrading to empty")
        report["error"] = f"{type(e).__name__}: {e}"
        return {"grid": [], "error": report["error"]}, report


def rasterize_sky_veg_mask(grid: List[str], img_shape) -> np.ndarray:
    """
    Turns a GRID_ROWS x GRID_COLS character grid ('S'/'G'/anything else)
    into a raw (unfeathered) float32 mask, same H x W as img_shape, 1.0 in
    sky/grass cells, 0.0 elsewhere. 'G' means mowed lawn/turf ONLY (see
    module docstring and SYSTEM_PROMPT) -- ornamental grasses, shrubs,
    flowers, and individual plants are deliberately 'other' and never
    enter this mask, so they're never capped. Upsampled with
    nearest-neighbor (a grid cell is a hard category, not a value to
    interpolate) then left for the CALLER to feather -- same
    one-feather-step-only discipline as every other gate in this pipeline.

    An empty or malformed grid returns an all-zero mask -- "apply no cap,"
    per this module's contract, not "cap everything."
    """
    h, w = img_shape[:2]
    if not grid:
        return np.zeros((h, w), dtype=np.float32)

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    small = np.zeros((rows, cols), dtype=np.float32)
    for r, row_str in enumerate(grid):
        for c, ch in enumerate(row_str):
            if ch in ("S", "G"):
                small[r, c] = 1.0

    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
