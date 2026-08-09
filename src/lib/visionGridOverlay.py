"""
visionGridOverlay.py

Shared utility: draws an actual grid -- lines plus row/column labels --
onto an image before it's sent to a Vision grid-classification call.

WHY THIS EXISTS (Aug 2026, ceiling_mask / wall_trim_mask reliability
investigation): every Vision grid-classifier prompt in this pipeline
(level2_ceiling_mask.py, level2_wall_trim_mask.py, and by the same
pattern level0_sky_vegetation_mask.py) tells the model the image is
"overlaid with a {cols}x{rows} grid" and asks it to classify cell by
cell. But an audit of every file between image capture and the Vision
API call (see oracleRouter.py's own "DIAGNOSTIC ONLY" comment, added
for an unrelated investigation into a rendering artifact) found that
NONE of them ever drew a grid onto the image -- each module's own
_encode_for_vision() only resized and JPEG-encoded the raw photo. The
model was being told it was looking at a gridded image and never was.

This is the most concrete, evidence-backed explanation available for
why ceiling_mask and wall_trim_mask's grid classifications have shown
two full failure families on real batches: (1) severe over/under
coverage misreads (ceiling read as 90-100% of frame on 17/19 real
photos in one batch, against a normal room's real ~15-20%), and (2)
row-length/dimension chaos (grids coming back the wrong shape entirely,
or internally inconsistent row to row). Asking a model to partition an
image into a fixed cell grid with literally no visual reference for
where cell boundaries fall is exactly the kind of task that produces
both failure shapes: confident-but-wrong boundary guesses on large
uniform surfaces (ceiling), and inconsistent boundary guesses on busier
ones competing with real edges (wall/trim).

STATUS: this is a hypothesis-driven fix, not yet validated against a
real batch -- unlike most constants and thresholds elsewhere in this
pipeline, there is no "confirmed on real photo X" evidence behind this
change yet. Ship it, then re-run the same 19-photo batch (or similar)
and compare ceiling_mask / wall_trim_mask pass rates before and after.
If pass rates don't improve, the grid-boundary-visibility theory is
wrong or incomplete, and the real cause is still open.

DESIGN NOTES:
- Operates on a COPY of the input array (img.copy() first thing) --
  never mutates the caller's image in place. This matters because
  callers may pass through an array that aliases the original without
  an intervening copy (see both _encode_for_vision implementations:
  when no resize is needed, `small = img` is a direct reference, not a
  copy). Silently mutating a shared array here would be a real bug,
  not just bad practice.
- Line color (magenta, 255/0/255 in BGR since this codebase's images
  are OpenCV BGR throughout) chosen to be visually distinct from
  typical interior photo content -- walls, ceilings, wood tones, and
  most furniture fabric rarely land near pure magenta, unlike white,
  black, or gray gridlines which could be confused with real trim,
  shadow lines, or grout.
- Labels (column letters along the top, row numbers along the left)
  mirror the coordinate scheme each prompt already describes ("column A
  to {last_col} left to right", "row 1 to {rows} top to bottom") --
  giving the model the same labels it's asked to reason in, not just
  unlabeled boundary lines it would have to count itself.
- Thin (1px) anti-aliased lines -- a grid overlay that itself obscures
  meaningful fractions of cell content (paint color, texture) would
  trade one accuracy problem for another. This is meant to mark
  boundaries, not shade regions.
"""

import cv2
import numpy as np

# BGR, not RGB -- this codebase's images are OpenCV arrays throughout.
GRID_LINE_COLOR = (255, 0, 255)  # magenta
GRID_LINE_THICKNESS = 1
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.35
LABEL_THICKNESS = 1


def draw_grid_overlay(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Returns a NEW array (img.copy(), never the original) with a
    `rows` x `cols` grid drawn onto it: vertical/horizontal boundary
    lines at each cell edge, a full-image border, column letters
    (A, B, C...) along the top edge, and row numbers (1, 2, 3...)
    along the left edge.

    Cell boundaries are computed the same way rasterize_ceiling_mask /
    rasterize_wall_trim_mask map grid cells back to pixels (evenly
    spaced across the image's actual width/height), so the lines drawn
    here are the same boundaries those functions assume -- this
    overlay and the downstream rasterization stay geometrically
    consistent with each other by construction, not by coincidence.

    Safe on any rows/cols >= 1. Does nothing destructive on a 1x1 grid
    (draws only the border) -- not a real use case in this codebase,
    but not a crash either.
    """
    overlay = img.copy()
    h, w = overlay.shape[:2]

    for c in range(1, cols):
        x = int(round(c * w / cols))
        cv2.line(overlay, (x, 0), (x, h - 1), GRID_LINE_COLOR, GRID_LINE_THICKNESS, cv2.LINE_AA)

    for r in range(1, rows):
        y = int(round(r * h / rows))
        cv2.line(overlay, (0, y), (w - 1, y), GRID_LINE_COLOR, GRID_LINE_THICKNESS, cv2.LINE_AA)

    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), GRID_LINE_COLOR, GRID_LINE_THICKNESS, cv2.LINE_AA)

    cell_w = w / float(cols)
    cell_h = h / float(rows)

    for c in range(cols):
        letter = chr(ord('A') + c) if c < 26 else f"{c}"
        x = int(c * cell_w) + 2
        y = 12
        cv2.putText(overlay, letter, (x, y), LABEL_FONT, LABEL_FONT_SCALE,
                    GRID_LINE_COLOR, LABEL_THICKNESS, cv2.LINE_AA)

    for r in range(rows):
        num = str(r + 1)
        x = 2
        y = int(r * cell_h) + 12
        cv2.putText(overlay, num, (x, y), LABEL_FONT, LABEL_FONT_SCALE,
                    GRID_LINE_COLOR, LABEL_THICKNESS, cv2.LINE_AA)

    return overlay
