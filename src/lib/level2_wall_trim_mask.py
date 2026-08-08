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

GRID DIMENSION TOLERANCE (added Aug 2026, IMG_8310 investigation):
Vision's adherence to "exactly GRID_ROWS x GRID_COLS" is not perfectly
deterministic call-to-call. A real production case: an 18x23 grid --
correct row count, every single row identically 23 characters, one
column narrow -- was previously discarded outright by a strict
`len(row) != GRID_COLS` equality check, even though it cleanly and
consistently identified three real vertical wall regions in the photo
(left wall, a center wall segment between furniture, right wall/window
trim). That grid was real, usable classification data, not garbage --
but the old check couldn't distinguish "one column short, otherwise
perfectly self-consistent" from "actually incoherent," and treated both
identically: degrade to empty, apply nothing.

identify_wall_trim() now accepts a grid whose actual dimensions are
within GRID_DIM_TOLERANCE of GRID_ROWS x GRID_COLS, PROVIDED every row
is still the same length as every other row (internal consistency is
still required -- inconsistent row lengths, or dimensions outside
tolerance, still degrade to empty, same contract as before). This widens
what counts as recoverable; it does not remove the safety net.

rasterize_wall_trim_mask() derives its pixel mapping from the grid's
ACTUAL shape, not the GRID_ROWS/GRID_COLS constants. This matters: if a
23-wide grid were resized as though it were 24-wide, every column
boundary after column 0 would be off by a fraction of a cell, compounding
rightward across the row -- a real, visible misalignment between the
mask and the true wall edge, i.e. exactly the kind of artifact this
tolerance change is meant to avoid introducing. Building the cell grid
at its own (rows, cols) and letting cv2.resize scale THAT to the image's
(w, h) keeps column 0 and the last column pinned to the true left/right
edges regardless of the exact column count in between.

Known follow-up, out of scope for this module: oracleCorrection.py's
apply_hue_fidelity_gate (and any other caller) that derives a feather
blur radius from `image_width / GRID_COLS` as "one cell in pixels" will
be very slightly off (~4% at a one-column miss) when fed a near-miss
grid, since the true per-cell pixel width is `image_width / actual_cols`.
This is a minor feather-radius inaccuracy, not a correctness bug, and
does not reproduce the banding artifact this change fixes -- but it's a
real, known imprecision worth closing in oracleCorrection.py at some
point rather than leaving implicit.

PROMPT STRUCTURE (added Aug 2026, second IMG_8310 investigation):
GRID_DIM_TOLERANCE (above) widened what counts as recoverable, but a
real production run on IMG_8310 after that patch still failed --
this time with row lengths of [24, 25, 28] WITHIN THE SAME grid, a
genuinely inconsistent response that tolerance can't and shouldn't
absorb (see the "rows disagree with EACH OTHER" branch in
identify_wall_trim -- that check is untouched by this change and still
correctly rejects that failure mode).

Real side-by-side data from that investigation: level2_ceiling_mask.py
puts its full instructions and the image TOGETHER in a single user
turn. This module previously put instructions in the API `system`
field and gave the user turn only "Classify this photo's grid." On the
same IMG_8310 photo, in the same session, ceiling's grid came back
correct in row count and off by exactly one character on a handful of
rows (a narrow, comprehensible miscount); this module's grid came back
with row lengths bouncing all over the place with no consistent
pattern (a much more severe failure). That is the most concrete,
evidence-backed lead available for why this module's grid quality is
worse than ceiling's on the same input -- not proof, since the two
classifiers ask about different visual categories, but a real
structural difference correlated with a real severity gap on real data.

_call_vision_api now puts the full instructional prompt and the image
together in one user-turn `content` list (text block first, then
image), with no separate `system` field -- matching level2_ceiling_mask.py's
structure. Nothing about the classification task, the grid contract,
or any of the failure-mode handling in identify_wall_trim /
rasterize_wall_trim_mask changed -- this is purely a request-structure
change to test whether it improves grid reliability, same as it
apparently does for ceiling. If a real batch shows this doesn't help,
the fix is reverting this one function, not questioning the tolerance
or consistency logic elsewhere in this file.

PER-ROW LENGTH TOLERANCE (added Aug 2026, same investigation as
PROMPT STRUCTURE above): the PROMPT STRUCTURE change was tested on a
real IMG_8310 run and worked -- 17 of 18 rows came back exactly
GRID_COLS characters. But the 18th row was one character short, in a
long run of repeated 'A's -- the same narrow miscounting failure this
module's own docstring already documents for level2_ceiling_mask.py on
this exact photo. Before this change, identify_wall_trim's "rows must
all match each other" check had zero tolerance: one short row killed
the entire grid, discarding 17 rows of clean, usable data over a
single character.

identify_wall_trim now computes the grid's majority (mode) row length
and pads or truncates any row within ROW_LEN_TOLERANCE characters of
that mode to match it, before the dimension-tolerance check against
GRID_ROWS x GRID_COLS runs. A short row is padded by repeating its own
trailing character (the best available guess for "which character was
undercounted" without inventing new information); a long row is
truncated. Rows further from the mode than ROW_LEN_TOLERANCE still
hard-reject the whole grid -- this is deliberately about forgiving a
narrow miscount on a few rows, not absorbing genuine chaos. The first
IMG_8310 investigation's wall_trim failure (row lengths 26-41 chars,
no consistent pattern, spread across many rows) would still correctly
reject under this tolerance -- verified against that real data, not
just reasoned about.

Known imprecision, not a correctness bug: padding with the row's own
last character is a heuristic, not a certainty -- on the real case
that motivated this, the short row's last character happened to be
'.', so the padded cell became '.' rather than 'A', even though the
long run of 'A's immediately before it makes 'A' the more likely true
value. This affects at most ROW_LEN_TOLERANCE cells out of the whole
grid (432 cells at 18x24) per malformed row, in the same direction the
module already errs (uncertain cells default toward "not wall" here,
consistent with this pipeline's existing caution around potentially
over-claiming architectural surface).
"""

import os
import json
import base64
import logging
import urllib.error
import urllib.request
from collections import Counter
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

# Max allowed absolute deviation, per axis, between a returned grid's
# actual row/column count and GRID_ROWS/GRID_COLS for that grid to be
# treated as recoverable near-miss data rather than malformed. See the
# module docstring's "GRID DIMENSION TOLERANCE" section for the real
# case (18x23) that motivated this. Deliberately conservative -- this is
# about tolerating a one-or-two-cell miss from an otherwise clean,
# self-consistent grid, not about accepting wildly wrong dimensions.
GRID_DIM_TOLERANCE = int(os.environ.get("LEVEL2_WALL_TRIM_MASK_DIM_TOLERANCE", "2"))

# Max allowed absolute deviation, per ROW, between an individual row's
# length and the grid's own majority row length, for that row to be
# padded/truncated to match rather than the whole grid being rejected
# as inconsistent. See the module docstring's "PER-ROW LENGTH
# TOLERANCE" section for the real case (IMG_8310, second investigation)
# that motivated this: 17 of 18 rows exactly correct, one row a single
# character short in a long same-character run -- previously this
# killed the entire grid via the "rows must all match each other"
# check, discarding 17 rows of clean data over one row's narrow
# miscount. Deliberately separate from GRID_DIM_TOLERANCE (which
# governs the grid's OVERALL shape against GRID_ROWS x GRID_COLS) --
# this one governs individual rows against each other, a different
# question. Same conservative-tolerance philosophy: forgive a narrow
# miscount, still reject genuine chaos (many rows, wildly different
# lengths -- see the wall_trim row-length data from the first IMG_8310
# investigation: 26-41 chars, no pattern -- that case has multiple rows
# far outside this tolerance and still correctly degrades to empty).
ROW_LEN_TOLERANCE = int(os.environ.get("LEVEL2_WALL_TRIM_MASK_ROW_LEN_TOLERANCE", "2"))

_VALID_CHARS = frozenset("A.")

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
    level0_sky_vegetation_mask.py, level2_ceiling_mask.py).

    PROMPT STRUCTURE (Aug 2026, second IMG_8310 investigation -- see module
    docstring's "PROMPT STRUCTURE" section for the real-photo evidence):
    instructions and image are combined into a single user-turn `content`
    list, no separate `system` field -- matching level2_ceiling_mask.py's
    structure, which produced measurably better grid consistency on the
    same real photo. Text block comes first, then the image, in that
    single user message."""
    last_col = chr(ord('A') + GRID_COLS - 1)
    prompt_text = SYSTEM_PROMPT.format(cols=GRID_COLS, rows=GRID_ROWS, last_col=last_col)

    body = json.dumps({
        "model": WALL_TRIM_MASK_MODEL,
        "max_tokens": 1536,  # sized for GRID_ROWS strings of GRID_COLS chars each,
        # plus JSON overhead -- comfortably more than 24x18=432 chars needs.
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
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
    Classifies a decoded image (cv2/numpy array, BGR) into a grid of 'A'
    (wall/trim) / '.' (other) cells via Vision, targeting GRID_ROWS x
    GRID_COLS but accepting a near-miss grid (see GRID_DIM_TOLERANCE and
    the module docstring's "GRID DIMENSION TOLERANCE" section) so long as
    it is internally consistent -- every row the same length as every
    other row.

    Returns (result, report). result["grid"] is a list of strings, each
    the same length as each other (GRID_COLS, or within tolerance of it),
    or [] on any failure. Never raises. Callers must treat an
    empty/malformed grid as "no architectural surface identified" -- see
    rasterize_wall_trim_mask, same "empty = apply nothing" contract as
    every other mask module here.

    report["dimension_mismatch"] is set (and logged) when an accepted
    grid's actual shape differs from GRID_ROWS x GRID_COLS, so a
    near-miss acceptance stays visible in diagnostics rather than being
    silently absorbed.
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

        if not isinstance(grid, list) or not grid or any(not isinstance(row, str) for row in grid):
            report["error"] = f"malformed_grid (not a non-empty list of strings): {str(grid)[:2000]!r}"
            return {"grid": [], "error": report["error"]}, report

        actual_rows = len(grid)
        row_len_list = [len(row) for row in grid]
        row_lengths = set(row_len_list)

        if len(row_lengths) != 1:
            # Rows disagree with each other on length. Before treating
            # this as chaotic garbage, check whether it's a narrow
            # near-miss on a few rows rather than genuine inconsistency
            # -- see ROW_LEN_TOLERANCE's docstring for the real case
            # this handles. Majority row length wins; any row within
            # ROW_LEN_TOLERANCE of it gets padded/truncated to match.
            # A row further off than that means the response really is
            # inconsistent -- still degrades to empty, same contract as
            # before, just with an actual tolerance window instead of
            # zero.
            mode_len, _count = Counter(row_len_list).most_common(1)[0]
            too_far = [i for i, n in enumerate(row_len_list) if abs(n - mode_len) > ROW_LEN_TOLERANCE]

            if too_far:
                report["error"] = (
                    f"malformed_grid (inconsistent row lengths {sorted(row_lengths)}, "
                    f"rows must all match each other): {str(grid)[:2000]!r}"
                )
                return {"grid": [], "error": report["error"]}, report

            normalized = []
            changed = []
            for i, row in enumerate(grid):
                diff = mode_len - len(row)
                if diff > 0:
                    # Short row -- pad by repeating its own last
                    # character. The real case this handles is a
                    # miscounted run of the SAME repeated character
                    # (e.g. one 'A' dropped from a long wall run), so
                    # extending with that row's own trailing character
                    # is the best available guess, not an arbitrary
                    # choice of 'A' or '.' that would bias the count
                    # toward wall or non-wall.
                    pad_char = row[-1] if row else "."
                    row = row + pad_char * diff
                    changed.append(f"row {i + 1}: padded +{diff} ({pad_char!r})")
                elif diff < 0:
                    row = row[:mode_len]
                    changed.append(f"row {i + 1}: truncated {-diff}")
                normalized.append(row)

            grid = normalized
            report["row_length_normalized"] = f"target_len={mode_len}: " + "; ".join(changed)
            logger.info(
                f"level2_wall_trim_mask: normalized row lengths {sorted(row_lengths)} "
                f"-> uniform {mode_len} ({report['row_length_normalized']})"
            )
            row_lengths = {mode_len}

        actual_cols = row_lengths.pop()

        bad_chars = {ch for row in grid for ch in row} - _VALID_CHARS
        if bad_chars:
            report["error"] = f"malformed_grid (invalid characters {sorted(bad_chars)}): {str(grid)[:2000]!r}"
            return {"grid": [], "error": report["error"]}, report

        row_delta = abs(actual_rows - GRID_ROWS)
        col_delta = abs(actual_cols - GRID_COLS)

        if row_delta > GRID_DIM_TOLERANCE or col_delta > GRID_DIM_TOLERANCE:
            report["error"] = (
                f"malformed_grid (got {actual_rows}x{actual_cols}, expected "
                f"{GRID_ROWS}x{GRID_COLS}, outside tolerance of {GRID_DIM_TOLERANCE}): "
                f"{str(grid)[:2000]!r}"
            )
            return {"grid": [], "error": report["error"]}, report

        if row_delta or col_delta:
            # Within tolerance but not an exact match -- accepted.
            # rasterize_wall_trim_mask derives pixel mapping from this
            # grid's ACTUAL shape, not the GRID_ROWS/GRID_COLS constants,
            # so this does not introduce cumulative column-boundary
            # drift -- see that function's docstring.
            report["dimension_mismatch"] = f"{actual_rows}x{actual_cols} vs expected {GRID_ROWS}x{GRID_COLS}"
            logger.info(
                f"level2_wall_trim_mask: accepted near-miss grid "
                f"{actual_rows}x{actual_cols} (expected {GRID_ROWS}x{GRID_COLS}, "
                f"within tolerance {GRID_DIM_TOLERANCE})"
            )

        n_wall_cells = sum(row.count("A") for row in grid)
        report["wall_trim_cell_fraction"] = n_wall_cells / float(actual_rows * actual_cols)

        return {"grid": grid, "error": None}, report

    except Exception as e:  # noqa: BLE001 -- any failure must fall back, never crash the pipeline
        logger.warning(f"level2_wall_trim_mask: failed ({type(e).__name__}: {e}), degrading to empty")
        report["error"] = f"{type(e).__name__}: {e}"
        return {"grid": [], "error": report["error"]}, report


def rasterize_wall_trim_mask(grid: List[str], img_shape) -> np.ndarray:
    """
    Turns a character grid ('A'/anything else) into a raw (unfeathered)
    float32 mask, same H x W as img_shape, 1.0 in wall/trim cells, 0.0
    elsewhere. Upsampled with nearest-neighbor (a grid cell is a hard
    category, not a value to interpolate) then left for the CALLER to
    feather -- same one-feather-step-only discipline as every other gate
    in this pipeline (see oracleCorrection.apply_saturation_cap /
    apply_hue_fidelity_gate's mask_grid_cols parameter).

    Pixel mapping is derived from the grid's ACTUAL shape (row count, and
    each row's length -- every row must match every other row) rather
    than the GRID_ROWS/GRID_COLS constants. This matters as of the
    GRID_DIM_TOLERANCE change in identify_wall_trim: that function can
    now hand this one a near-miss grid (e.g. 18x23, one column narrow).
    If this function built its cell array at a fixed (GRID_ROWS,
    GRID_COLS) regardless of the grid's real dimensions, or resized as
    though the source were 24 cells wide when it's actually 23, every
    column boundary after column 0 would be off by a fraction of a cell,
    compounding rightward across the row -- a real, visible misalignment
    between the mask and the true wall edge. Building cell_grid at the
    grid's own (rows, cols) and letting cv2.resize scale THAT to (w, h)
    keeps column 0 and the last column pinned to the true left/right
    edges regardless of the exact column count in between.

    An empty grid, or one with inconsistent row lengths, returns an
    all-zero mask -- "apply nothing," per this module's contract, same as
    level0_sky_vegetation_mask.rasterize_sky_veg_mask and
    level2_ceiling_mask.rasterize_ceiling_mask. This function performs
    its own consistency check independent of identify_wall_trim's, since
    it can be called directly with a raw grid by other callers in this
    pipeline.
    """
    h, w = img_shape[:2]

    if not grid or any(not isinstance(row, str) for row in grid):
        return np.zeros((h, w), dtype=np.float32)

    actual_rows = len(grid)
    row_lengths = {len(row) for row in grid}
    if len(row_lengths) != 1:
        return np.zeros((h, w), dtype=np.float32)

    actual_cols = row_lengths.pop()
    if actual_cols == 0:
        return np.zeros((h, w), dtype=np.float32)

    cell_grid = np.zeros((actual_rows, actual_cols), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "A":
                cell_grid[r, c] = 1.0

    mask = cv2.resize(cell_grid, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask
