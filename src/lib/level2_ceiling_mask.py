"""
level2_ceiling_mask.py

Vision-based identification of CEILING regions in interior photos --
exists for exactly one purpose: scoping oracleCorrection.smooth_deltas_
in_mask() to large, continuous overhead surfaces where Oracle's own
rendered small-scale variation produces visible blotchy correction,
without touching correction strength anywhere else in the frame.

WHY THIS EXISTS: Flux Kontext's interior prompt deliberately asks for
"believable luminance gradients" on the ceiling, so it doesn't render a
flat, sterile plane (a reasonable instruction for a standalone image).
But this pipeline uses Oracle purely as a source of PER-PIXEL DELTAS
against the Original -- illumination_delta = L_oracle - L_original,
applied directly. Flux's own invented ceiling variation doesn't
spatially match the Original's real lighting pattern, so the delta in
that region is largely Flux's own noise, not measured correction.
Confirmed on two separate real photos across two separate sessions --
not a one-off render artifact, a repeatable characteristic of how this
model handles this kind of surface under this prompt.

WHY A SEPARATE VISION CALL FROM level2_sky_grass_mask.py AND
level2_vision_recoverability.py: same category-separation discipline as
those two modules keep from each other. Sky/grass answers "what kind of
exterior surface is this" (scoped to exterior only). Recoverability
answers "is there enough data here to trust a correction at all" (any
scene). This answers a third, distinct question -- "is this pixel part
of a large continuous interior ceiling plane" -- specific to interior
photos and to a failure mode neither of the other two modules is built
to catch.

WHY NOT A CLASSICAL POSITION/COLOR HEURISTIC: this codebase has already
found two classical heuristics wrong on real photos for structurally
the same reason -- the interior/exterior HSV heuristic (level0_scene_
classifier.py) and the original bbox-based sky/grass mask (level2_sky_
grass_mask.py) before it was rewritten to a grid. A "top of frame" or
"low local variance" rule for ceiling would fail the same way: a
vaulted ceiling, an angled dormer ceiling, a bright uniform wall in a
wide shot, or a light-colored floor visible in a mirror would all
confuse a position/variance rule. This is a scene-level judgment a
classical rule can't make reliably, the same reason Vision replaced a
classical rule for sky/grass and for interior/exterior.

GRID CLASSIFICATION, NOT BOUNDING BOXES: same reasoning as level2_sky_
grass_mask.py's own choice -- a ceiling is often irregular (interrupted
by a beam, a fan, a skylight, recessed lighting, crown molding, or a
vaulted break), and a rectangle union handles that badly. A coarse grid,
classified cell by cell, handles irregular and interrupted shapes
without this failure mode by construction.

SHADOW MODE: ships computing and logging only, same phased-rollout
discipline as level0_scene_classifier.py and level2_vision_
recoverability.py before it. Unlike level2_sky_grass_mask.py's more
casual "both failure directions are low severity" reasoning, this
module gates behind CEILING_MASK_SHADOW_MODE explicitly (default true)
before its output is allowed to change real pixels -- a false positive
here (marking a wall as ceiling) would over-smooth real detail-recovery
work elsewhere in the frame, a higher-severity mistake than the sky/
grass module's own worst case, so a real batch should be reviewed
before this drives correction. ENABLED is a separate kill switch on top
of that. Any failure degrades to "no mask" -- the caller must treat an
empty/malformed grid as "apply no smoothing," never as "the whole photo
is ceiling."
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

CEILING_MASK_ENABLED = os.environ.get("LEVEL2_CEILING_MASK_ENABLED", "true").lower() not in ("false", "0", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CEILING_MASK_MODEL = os.environ.get("LEVEL2_CEILING_MASK_MODEL", "claude-haiku-4-5-20251001")
TIMEOUT_SECONDS = int(os.environ.get("LEVEL2_CEILING_MASK_TIMEOUT_SECONDS", "20"))
MAX_VISION_EDGE = 1568  # matches every other Vision module in this codebase

# Ships OFF (i.e. computed/logged, not applied) by default -- see module
# docstring for why this module's shadow-mode bar is higher than level2_
# sky_grass_mask.py's. Flip via LEVEL2_CEILING_MASK_SHADOW_MODE=false
# only after reviewing a real batch's ceiling_mask reports.
SHADOW_MODE = os.environ.get("LEVEL2_CEILING_MASK_SHADOW_MODE", "true").lower() not in ("false", "0", "")

# Same grid resolution as level2_sky_grass_mask.py -- reusing an already-
# validated resolution rather than introducing a third value with no
# real-photo evidence behind it.
GRID_COLS = int(os.environ.get("LEVEL2_CEILING_GRID_COLS", "24"))
GRID_ROWS = int(os.environ.get("LEVEL2_CEILING_GRID_ROWS", "18"))


SYSTEM_PROMPT = """You are looking at a real estate interior photograph, overlaid with a {cols}x{rows} grid (columns lettered A-{last_col}, rows numbered 1-{rows}, top-left cell is A1).

For EVERY cell, classify what's predominantly in it: "ceiling" or "other".

"ceiling" means a continuous, flat or near-flat overhead PAINTED surface -- the plane itself. A vaulted or angled ceiling still counts as ceiling; judge by whether the cell belongs to one continuous painted plane, not by its position in the frame.

Do NOT classify as "ceiling" -- these are "other" even when fully surrounded by ceiling cells: a ceiling beam, a skylight, a recessed light fixture housing or its visible glow bulb, a ceiling fan, an HVAC vent or diffuser, crown molding, a light fixture hanging below the plane, or any break in the continuous surface. Those are distinct features that should keep their own detail, not get smoothed together with the flat plane around them.

Walls, upper wall areas near the ceiling line, window tops, and door frames are "other" even if brightly lit -- only the overhead plane itself is "ceiling".

This does not need to be precise at the cell boundary -- it's scoping a smoothing pass on a large surface, not a pixel mask. A cell that's a mix should get whichever category covers more of that cell's area.

Return ONLY strict JSON, no markdown fences, no prose outside the JSON, as a single string, row by row, top to bottom, one character per cell: 'C' for ceiling, '.' for other. Each row must be exactly {cols} characters.

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
        "model": CEILING_MASK_MODEL,
        "max_tokens": 1536,  # sized for GRID_ROWS strings of GRID_COLS chars each,
        # plus JSON overhead -- matches level2_sky_grass_mask.py's identical sizing.
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


def identify_ceiling(img) -> tuple:
    """
    Returns (result, report). result["grid"] is a list of GRID_ROWS
    strings, each GRID_COLS characters ('C'/'.'), or [] on any failure.
    Never raises. Callers must treat an empty/malformed grid as "apply
    no smoothing," not as "nothing is ceiling" -- see
    rasterize_ceiling_mask.
    """
    report = {"enabled": CEILING_MASK_ENABLED, "called": False, "model": CEILING_MASK_MODEL,
              "shadowMode": SHADOW_MODE, "error": None}

    if not CEILING_MASK_ENABLED:
        report["error"] = "disabled_via_env"
        return {"grid": [], "error": "disabled_via_env"}, report
    if not ANTHROPIC_API_KEY:
        logger.warning("level2_ceiling_mask: missing ANTHROPIC_API_KEY, degrading to empty")
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
        logger.warning(f"level2_ceiling_mask: failed ({type(e).__name__}: {e}), degrading to empty")
        report["error"] = f"{type(e).__name__}: {e}"
        return {"grid": [], "error": report["error"]}, report


def rasterize_ceiling_mask(grid: List[str], img_shape) -> np.ndarray:
    """
    Turns a GRID_ROWS x GRID_COLS character grid ('C'/anything else) into
    a raw (unfeathered) float32 mask, same H x W as img_shape, 1.0 in
    ceiling cells, 0.0 elsewhere. Upsampled with nearest-neighbor (a grid
    cell is a hard category, not a value to interpolate) then left for
    the CALLER to feather -- oracleCorrection.smooth_deltas_in_mask does
    its own feathering internally, same one-feather-step-only discipline
    as every other gate in this pipeline.

    An empty or malformed grid returns an all-zero mask -- "apply no
    smoothing," per this module's contract, not "smooth everything."
    """
    h, w = img_shape[:2]
    if not grid:
        return np.zeros((h, w), dtype=np.float32)

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    small = np.zeros((rows, cols), dtype=np.float32)
    for r, row_str in enumerate(grid):
        for c, ch in enumerate(row_str):
            if ch == "C":
                small[r, c] = 1.0

    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
