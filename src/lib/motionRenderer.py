#!/usr/bin/env python3
"""
motionRenderer.py — Smooth Ken Burns motion renderer for Smart Stage PRO Plus.

Replaces FFmpeg's zoompan filter. zoompan quantizes all crop-window positions
to integer pixels at the working-canvas size (960x540), then upscales 2x to
the output. For a 1024x576 source image this produces an alternating +1.9px /
+3.75px staircase in the output — confirmed as the "shaky" motion on real
footage. Raising fps or lowering zoom rate does not fix it; the cause is
spatial, not temporal.

This renderer computes every transform as a floating-point affine matrix at
full OUTPUT resolution (1920x1080), applied via cv2.warpAffine with INTER_CUBIC
interpolation. Sub-pixel accuracy, no staircase, no zoompan.

Preset reference:
  push_in          — dolly toward center / feature wall / front door
  pull_back        — reveal / drone boom-up approximation
  pan_left         — tracking left, flow to adjacent rooms, countertops
  pan_right        — tracking right, countertops, patio
  tilt_up          — high ceilings, chandeliers, fireplace, drone reveal feel
  tilt_down        — flooring reveal, tilework, soaking tub approach
  drift            — slow diagonal dolly (corner-to-corner, toward windows)
  float            — gentle sine-wave breathing, static-feeling shots
  luxury_parallax  — Kling continuation beat: ease-in diagonal settle
  static           — MLS clarity wide shot, no motion

Usage (called by motionPresets.js via child_process.spawn):
  python3 motionRenderer.py \
    --source   /tmp/job-xyz/frame_living.jpg \
    --preset   push_in \
    --duration 5.5 \
    --output   /tmp/job-xyz/clip_living.mp4 \
    --start-zoom 1.0 \
    --fps 25

Outputs a single JSON line to stdout on success:
  {"path": "/tmp/.../clip_living.mp4", "endingZoom": 1.5, "duration": 5.5}
"""

import argparse
import json
import math
import subprocess
import numpy as np
import cv2


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",     required=True,  help="Input image path")
    p.add_argument("--preset",     required=True,
                   choices=["push_in","pull_back","pan_left","pan_right",
                             "tilt_up","tilt_down","drift",
                             "float","luxury_parallax","static"])
    p.add_argument("--duration",   type=float, required=True)
    p.add_argument("--output",     required=True,  help="Output .mp4 path")
    p.add_argument("--start-zoom", type=float, default=1.0, dest="start_zoom")
    p.add_argument("--output-w",   type=int,   default=1920, dest="output_w")
    p.add_argument("--output-h",   type=int,   default=1080, dest="output_h")
    p.add_argument("--fps",        type=int,   default=25)
    return p.parse_args()


# ── IMAGE PREP ────────────────────────────────────────────────────────────────

def load_and_crop(source_path, out_w, out_h):
    """
    Load source image and center-crop to output aspect ratio.
    Works for landscape (3:2, 4:3), square, and portrait sources.
    Returns the cropped image at native resolution — NOT resized to output.
    The renderer upscales/crops per-frame via warpAffine.
    """
    img = cv2.imread(source_path)
    if img is None:
        raise ValueError(f"Could not load image: {source_path}")

    h, w = img.shape[:2]
    target_ratio = out_w / out_h  # 1.778 for 16:9

    if w / h > target_ratio:
        # Source wider than target — trim sides
        new_w = int(h * target_ratio)
        x = (w - new_w) // 2
        img = img[:, x : x + new_w]
    else:
        # Source taller than target — trim top/bottom
        new_h = int(w / target_ratio)
        y = (h - new_h) // 2
        img = img[y : y + new_h, :]

    return img


# ── EASING ────────────────────────────────────────────────────────────────────

def ease_in_out(t):
    """Smoothstep S-curve. Starts slow, peaks speed at midpoint, ends slow."""
    return t * t * (3.0 - 2.0 * t)

def ease_in(t):
    """Starts slow, accelerates. Used for Kling continuation settle."""
    return t * t

def ease_out(t):
    """Starts fast, decelerates. Used for tilt_down (gravity feel)."""
    return 1.0 - (1.0 - t) * (1.0 - t)


# ── MOTION CURVES ─────────────────────────────────────────────────────────────

def build_motion_curve(preset, n_frames, start_zoom):
    """
    Returns list of (zoom, pan_x_frac, pan_y_frac) per frame.

    zoom        — scale factor. 1.0 = full source fills output.
                  1.5 = zoomed in 1.5x (showing 1/1.5 of source).
    pan_x_frac  — horizontal center offset as fraction of source width.
                  0.0 = dead center. Positive = shift right (appears to pan left).
    pan_y_frac  — vertical center offset as fraction of source height.
                  0.0 = dead center. Positive = shift down (appears to tilt up).

    Pan/tilt range math:
      At zoom=1.25, available margin in each axis = 1 - 1/1.25 = 20% of source.
      We use 60% of that margin so there is always a safe buffer at the edges.
      PAN_RANGE = 0.20 * 0.60 = 0.12 of source width/height.

    Tilt range uses zoom=1.3 for more vertical travel without hitting edges:
      Available margin = 1 - 1/1.3 = 23% of source height.
      TILT_RANGE = 0.23 * 0.55 = ~0.127 of source height.
    """
    max_zoom  = min(start_zoom + 0.5, 1.8)
    min_zoom  = max(start_zoom - 0.5, 1.0)

    PAN_RANGE  = (1.0 - 1.0 / 1.25) * 0.60   # ~0.120 — horizontal travel
    TILT_RANGE = (1.0 - 1.0 / 1.30) * 0.55   # ~0.127 — vertical travel
    DRIFT_RANGE = (1.0 - 1.0 / 1.20) * 0.50  # ~0.083 — diagonal travel

    curve = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)  # 0.0 → 1.0

        if preset == "push_in":
            # Dolly toward center — front door, feature wall, bed
            zoom  = start_zoom + (max_zoom - start_zoom) * ease_in_out(t)
            pan_x = 0.0
            pan_y = 0.0

        elif preset == "pull_back":
            # Reveal / drone boom-up approximation — exterior, backyard wide
            zoom  = max_zoom - (max_zoom - min_zoom) * ease_in_out(t)
            pan_x = 0.0
            pan_y = 0.0

        elif preset == "pan_left":
            # Tracking left — countertops, patio, show adjacent rooms
            zoom  = 1.25
            pan_x = PAN_RANGE * (0.5 - ease_in_out(t))
            pan_y = 0.0

        elif preset == "pan_right":
            # Tracking right — countertops, vanities, patio
            zoom  = 1.25
            pan_x = PAN_RANGE * (ease_in_out(t) - 0.5)
            pan_y = 0.0

        elif preset == "tilt_up":
            # Pan vertically bottom → top.
            # Starts showing lower portion of frame (more floor/countertop),
            # rises to reveal ceiling, chandelier, high windows, architectural
            # height. ease_out gives a slight deceleration at the top — like
            # a camera operator tilting up and letting the shot breathe.
            zoom  = 1.30
            pan_x = 0.0
            pan_y = TILT_RANGE * (0.5 - ease_out(t))  # positive=down, so start low

        elif preset == "tilt_down":
            # Pan vertically top → bottom.
            # Starts at eye level / counter height, tilts down to reveal
            # flooring, tilework, soaking tub base. ease_in gives a deliberate
            # intentional feel — like a director pointing at a detail.
            zoom  = 1.30
            pan_x = 0.0
            pan_y = TILT_RANGE * (ease_in(t) - 0.5)   # negative=up, so start high

        elif preset == "drift":
            # Slow diagonal dolly — corner-to-corner, toward windows, patio.
            # Combines a gentle push-in with a lateral drift for a more
            # cinematic feel than a pure pan. ease_in_out keeps it deliberate.
            # Direction: left-to-right + slight push in (most common natural read).
            zoom  = 1.15 + 0.05 * ease_in_out(t)      # 1.15 → 1.20
            pan_x = DRIFT_RANGE * (ease_in_out(t) - 0.5)
            pan_y = -DRIFT_RANGE * 0.4 * ease_in_out(t)  # slight upward as we push

        elif preset == "float":
            # Gentle sine-wave breathing — dining rooms, flex spaces, wide shots
            # where deliberate directional movement would feel forced.
            zoom  = 1.1 + 0.03 * math.sin(2.0 * math.pi * t)
            pan_x = 0.0
            pan_y = 0.0

        elif preset == "luxury_parallax":
            # Kling continuation beat — ease_in so the camera appears to
            # settle after the vacant→staged transformation, then drifts.
            drift = (1.0 - 1.0 / 1.15) * 0.30
            t_in  = ease_in(t)
            zoom  = 1.15
            pan_x = drift * t_in
            pan_y = -drift * 0.6 * t_in

        else:  # static
            zoom  = 1.0
            pan_x = 0.0
            pan_y = 0.0

        curve.append((zoom, pan_x, pan_y))

    return curve


# ── PER-FRAME RENDERER ────────────────────────────────────────────────────────

def render_frame(img, zoom, pan_x_frac, pan_y_frac, out_w, out_h):
    """
    Render one video frame using a floating-point affine transform.
    cv2.warpAffine with INTER_CUBIC = sub-pixel bicubic interpolation.
    BORDER_REPLICATE avoids black borders if the crop window clips the edge.
    """
    src_h, src_w = img.shape[:2]

    # Crop window size in source pixels
    crop_w = src_w / zoom
    crop_h = src_h / zoom

    # Center of crop window with pan offset applied
    cx = src_w / 2.0 + pan_x_frac * src_w
    cy = src_h / 2.0 + pan_y_frac * src_h

    # Top-left of crop window (floating point — the key to smooth motion)
    x1 = cx - crop_w / 2.0
    y1 = cy - crop_h / 2.0

    # Affine: map source crop corners → output frame corners
    src_pts = np.float32([
        [x1,          y1         ],
        [x1 + crop_w, y1         ],
        [x1,          y1 + crop_h],
    ])
    dst_pts = np.float32([
        [0,     0    ],
        [out_w, 0    ],
        [0,     out_h],
    ])

    M = cv2.getAffineTransform(src_pts, dst_pts)

    return cv2.warpAffine(
        img, M, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ── FFMPEG PIPE ───────────────────────────────────────────────────────────────

def open_ffmpeg_pipe(output_path, fps, out_w, out_h):
    """Open FFmpeg subprocess accepting raw BGR24 frames on stdin."""
    cmd = [
        "ffmpeg", "-y",
        "-f",       "rawvideo",
        "-vcodec",  "rawvideo",
        "-s",       f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r",       str(fps),
        "-i",       "pipe:0",
        "-c:v",     "libx264",
        "-pix_fmt", "yuv420p",
        "-preset",  "fast",
        "-crf",     "18",
        "-movflags", "+faststart",
        "-threads", "2",
        output_path,
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    fps      = args.fps
    out_w    = args.output_w
    out_h    = args.output_h
    n_frames = round(args.duration * fps)

    img   = load_and_crop(args.source, out_w, out_h)
    curve = build_motion_curve(args.preset, n_frames, args.start_zoom)

    ending_zoom = curve[-1][0]

    proc = open_ffmpeg_pipe(args.output, fps, out_w, out_h)

    try:
        for zoom, pan_x, pan_y in curve:
            frame = render_frame(img, zoom, pan_x, pan_y, out_w, out_h)
            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed (exit {proc.returncode})")

    except Exception:
        proc.kill()
        raise

    print(json.dumps({
        "path":       args.output,
        "endingZoom": ending_zoom,
        "duration":   args.duration,
    }))


if __name__ == "__main__":
    main()
