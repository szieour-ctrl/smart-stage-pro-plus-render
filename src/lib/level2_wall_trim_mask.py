"""
level2_wall_trim_mask.py

Vision grid classifier identifying WALL and TRIM/MOLDING regions --
painted architectural surfaces whose true captured color must be held
near-constant through correction, as distinct from furniture, fabric,
and wood, which are explicitly allowed to shift color as part of
legitimate cast removal (see oracleGeneration.py's Color Temperature --
Independent of Brightness section).

WHY THIS EXISTS: built specifically to feed
oracleCorrection.apply_hue_fidelity_gate()'s architectural_mask
parameter -- see that function's docstring for the real-photo evidence
(IMG_8310 wall regions) that motivated it. Vision judges WHERE the wall
is; it never judges or touches color values -- that's still deterministic
OpenCV/numpy math in apply_hue_fidelity_gate itself. Same AB 723 posture
as every other Vision module in this pipeline: Vision may judge, never
generate or modify pixels.

CEILING IS DELIBERATELY EXCLUDED from this module's 'A' category --
level2_ceiling_mask.py already identifies ceiling for a different job
(delta smoothing against Flux's invented luminance noise, see
oracleCorrection.smooth_deltas_in_mask). Rather than duplicate that
Vision call here, callers needing the FULL architectural surface set for
apply_hue_fidelity_gate should take the UNION of this module's rasterized
mask and level2_ceiling_mask's rasterized mask. Two separate Vision calls
for the same physical surface, serving two separate downstream jobs, is
consistent with how this pipeline already treats sky/vegetation (one
call, saturation-cap job) and ceiling (separate call, smoothing job) as
independent modules even where real photos might have both in frame.

SHADOW MODE: ships gated the same as every other Vision-driven gate in
this pipeline (level2_vision_recoverability, level2_sky_grass_mask,
level2_ceiling_mask) -- compute and log agreement/coverage on a live
batch first, do not let it drive pixels until reviewed. See SHADOW_MODE
below and how oracleCorrection.run_oracle_driven_pipeline consumes it.

GRID RESOLUTION: reuses the 24x18 grid already validated for sky/
vegetation and ceiling identification, rather than inventing a third
resolution with no evidence behind it -- same discipline noted in the
ceiling mask module.
"""

import os
import json
import base64
import logging
import urllib.error
import urllib.request
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

WALL_TRIM_MASK_ENABLED = os.environ.get("LEVEL2_WALL_TRIM_MASK_ENABLED", "true").lower() not in ("false", "0", "")

# Shadow mode ON by default -- compute and log only, never drives pixels
# until explicitly turned off after a real batch is reviewed. Same
# contract and same env-var naming convention as
# level2_vision_recoverability.SHADOW_MODE and the sky/grass and ceiling
# mask modules.
SHADOW_MODE = os.environ.get("LEVEL2_WALL_TRIM_MASK_SHADOW_MODE", "true").lower() not in ("false", "0", "")

WALL_TRIM_MASK_MODEL = os.environ.get("LEVEL2_WALL_TRIM_MASK_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TIMEOUT_SECONDS = int(os.environ.get("LEVEL2_WALL_TRIM_MASK_TIMEOUT_SECONDS", "30"))
MAX_VISION_EDGE = int(os.environ.get("LEVEL2_WALL_TRIM_MASK_MAX_EDGE", "1024"))

GRID_COLS = 24
GRID_ROWS = 18

SYSTEM_PROMPT = """You are classifying a grid overlaid on a real estate interior photograph into a {rows}x{cols} grid of cells (row 1 to {rows} top to bottom, column A to {last_col} left to right).

Classify each cell as 'A' (architectural -- a painted WALL surface, or TRIM/MOLDING: baseboards, door casings, window casings, crown molding, wainscoting) or '.' (other -- everything else: furniture, floor, ceiling, windows/glass, artwork, mirrors, fixtures, decor, doors themselves excluding their trim, fabric, rugs).

Ceiling is NOT 'A' even though it is also painted and architectural -- it is handled by a separate classifier for a different purpose. Classify ceiling cells as '.' here.

A wall behind furniture, partially occluded, still counts as 'A' for the visible painted-surface portion of that cell -- judge each cell by whether the majority of its visible content is wall/trim paint, not by whether something is in front of it.

Door and window FRAMES/casings count as 'A' (trim). The door panel itself, or the window glass itself, does not -- '.'.

This does not need to be precise at the cell boundary -- it is scoping a color-fidelity safety check, not a pixel mask. Walls are often broken into multiple disconnected regions by furniture, doorways, or the camera angle -- classify each cell on its own merits, don't assume wall is only in one contiguous block.

Return ONLY strict JSON, no markdown fences, no prose outside the JSON, as a single string, row by row, top to bottom, one character per cell: 'A' for wall/trim, '.' for other. Each row must be exactly {cols} characters.

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
    module in this codebase (level0_scene_classifier.py, level2_diagnosis.py,
    level0_sky_vegetation_mask.py, level2_ceiling_mask.py)."""
    last_col = chr(ord('A') + GRID_COLS - 1)
    prompt_text = SYSTEM_PROMPT.format(cols=GRID_COLS, rows=GRID_ROWS, last_col=last_col)

    body = json.dumps({
        "model": WALL_TRIM_MASK_MODEL,
        "max_tokens": 1536,  # sized for GRID_ROWS strings of GRID_COLS chars each,
        # plus JSON overhead -- comfortably more than 24x18=432 chars needs.
        "system": prompt_text,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": "Classify this photo's grid."},
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

    # Same defensive extraction as level2_ceiling_mask.py's _call_vision_api
    # -- previously this was a bare json.loads(text.strip()), which raises
    # a JSONDecodeError carrying only a line/column position, with NO raw
    # model output captured anywhere in the resulting error. That's a worse
    # diagnostic gap than the malformed_grid truncation fixed below: on a
    # parse failure, there was literally nothing to look at. Finding the
    # first '{' and raw-decoding from there also tolerates a model response
    # with leading or trailing prose around the JSON object, which a bare
    # json.loads does not.
    if not text:
        raise ValueError(f"empty_response_text (stop_reason={payload.get('stop_reason')!r})")
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"no_json_object_found -- raw_text={text[:2000]!r}")
    try:
        return json.JSONDecoder().raw_decode(text, brace_start)[0]
    except json.JSONDecodeError as e:
        raise ValueError(f"{e} -- raw_text={text[:2000]!r}") from e


def identify_wall_trim(img) -> tuple:
    """
    Classifies a decoded image (cv2/numpy array, BGR) into a GRID_ROWS x
    GRID_COLS grid of 'A' (wall/trim) / '.' (other) cells via Vision.

    Returns (result, report). result["grid"] is a list of GRID_ROWS
    strings, each GRID_COLS characters ('A'/'.'), or [] on any failure.
    Never raises. Callers must treat an empty/malformed grid as "no
    architectural surface identified" -- see rasterize_wall_trim_mask,
    same "empty = apply nothing" contract as every other mask module
    here.
    """
    report = {
        "enabled": WALL_TRIM_MASK_ENABLED,
        "called": False,
        "model": WALL_TRIM_MASK_MODEL,
        "error": None,
    }

    if not WALL_TRIM_MASK_ENABLED:
        report["error"] = "disabled_via_env"
        return {"grid": [], "error": "disabled_via_env"}, report
    if not ANTHROPIC_API_KEY:
        logger.warning("level2_wall_trim_mask: missing ANTHROPIC_API_KEY, degrading to empty")
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
            report["error"] = f"malformed_grid (expected {GRID_ROWS}x{GRID_COLS}): {str(grid)[:2000]!r}"
            return {"grid": [], "error": report["error"]}, report

        n_wall_cells = sum(row.count("A") for row in grid)
        report["wall_trim_cell_fraction"] = n_wall_cells / float(GRID_ROWS * GRID_COLS)

        return {"grid": grid, "error": None}, report

    except Exception as e:  # noqa: BLE001 -- any failure must fall back, never crash the pipeline
        logger.warning(f"level2_wall_trim_mask: failed ({type(e).__name__}: {e}), degrading to empty")
        report["error"] = f"{type(e).__name__}: {e}"
        return {"grid": [], "error": report["error"]}, report


def rasterize_wall_trim_mask(grid: List[str], img_shape) -> np.ndarray:
    """
    Turns a GRID_ROWS x GRID_COLS character grid ('A'/anything else) into
    a raw (unfeathered) float32 mask, same H x W as img_shape, 1.0 in
    wall/trim cells, 0.0 elsewhere. Upsampled with nearest-neighbor (a
    grid cell is a hard category, not a value to interpolate) then left
    for the CALLER to feather -- same one-feather-step-only discipline as
    every other gate in this pipeline (see
    oracleCorrection.apply_saturation_cap /
    apply_hue_fidelity_gate's mask_grid_cols parameter, which this
    module's GRID_COLS is meant to be passed into directly).

    An empty or malformed grid returns an all-zero mask -- "apply
    nothing," per this module's contract, same as
    level0_sky_vegetation_mask.rasterize_sky_veg_mask and
    level2_ceiling_mask.rasterize_ceiling_mask.
    """
    h, w = img_shape[:2]

    if not grid or len(grid) != GRID_ROWS:
        return np.zeros((h, w), dtype=np.float32)

    cell_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    for r, row in enumerate(grid):
        if not isinstance(row, str) or len(row) != GRID_COLS:
            return np.zeros((h, w), dtype=np.float32)
        for c, ch in enumerate(row):
            if ch == "A":
                cell_grid[r, c] = 1.0

    mask = cv2.resize(cell_grid, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask
