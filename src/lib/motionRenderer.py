#!/usr/bin/env python3
"""
motionRenderer.py — Smooth Ken Burns motion renderer for Smart Stage PRO Plus.

Replaces FFmpeg's zoompan filter. zoompan quantizes all crop-window positions
to integer pixels at the working-canvas size (960×540), then upscales 2× to
the output. For a 1024×576 source image this produces an alternating +1.9px /
+3.75px staircase in the output — confirmed as the "shaky" motion on real
footage. Raising fps or lowering zoom rate does not fix it; the cause is
spatial, not temporal.

This renderer computes every transform as a floating-point affine matrix at
full OUTPUT resolution (1920×1080), applied via cv2.warpAffine with INTER_CUBIC
interpolation. Sub-pixel accuracy, no staircase, no zoompan.

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

Node.js reads this to get the endingZoom for carryZoom continuity.
"""

import argparse
import json
import math
import subprocess
import sys
import numpy as np
import cv2


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",     required=True,  help="Input image path")
    p.add_argument("--preset",     required=True,
                   choices=["push_in","pull_back","pan_left","pan_right",
                             "float","luxury_parallax","static"])
    p.add_argument("--duration",   type=float, required=True)
    p.add_argument("--output",     required=True,  help="Output .mp4 path")
    p.add_argument("--start-zoom", type=float, default=1.0,  dest="start_zoom")
    p.add_argument("--output-w",   type=int,   default=1920, dest="output_w")
    p.add_argument("--output-h",   type=int,   default=1080, dest="output_h")
    p.add_argument("--fps",        type=int,   default=25)
    return p.parse_args()


# ── IMAGE PREP ────────────────────────────────────────────────────────────────

def load_and_crop(source_path, out_w, out_h):
    """
    Load source image and center-crop to the output aspect ratio.
    Works correctly for landscape (3:2, 4:3), square, and portrait sources.
    Returns the cropped image at its native resolution — NOT resized to output.
    The renderer upscales/crops to output size per-frame via warpAffine.
    """
    img = cv2.imread(source_path)
    if img is None:
        raise ValueError(f"Could not load image: {source_path}")

    h, w = img.shape[:2]
    target_ratio = out_w / out_h  # 16:9 = 1.778

    if w / h > target_ratio:
        # Source is wider than target — trim left and right
        new_w = int(h * target_ratio)
        x = (w - new_w) // 2
        img = img[:, x : x + new_w]
    else:
        # Source is taller (or equal) — trim top and bottom
        new_h = int(w / target_ratio)
        y = (h - new_h) // 2
        img = img[y : y + new_h, :]

    return img  # aspect-correct, native resolution


# ── EASING ───────────────────────────────────────────────────────────────────

def ease_in_out(t):
    """Smoothstep S-curve. t in [0,1] → output in [0,1]."""
    return t * t * (3.0 - 2.0 * t)

def ease_out(t):
    """Ease out — starts fast, decelerates. Used for pull_back."""
    return 1.0 - (1.0 - t) * (1.0 - t)

def ease_in(t):
    """Ease in — starts slow, accelerates. Used for push_in start."""
    return t * t


# ── MOTION CURVES ─────────────────────────────────────────────────────────────

def build_motion_curve(preset, n_frames, start_zoom):
    """
    Returns a list of (zoom, pan_x_frac, pan_y_frac) tuples, one per frame.

    zoom        — crop scale factor. At zoom=1.0 the full source fits the output.
                  At zoom=1.5 we show 1/1.5 of the source, filling the output.
    pan_x_frac  — horizontal center offset as a fraction of source width.
                  0.0 = dead center. Positive = shift crop right (pan left).
    pan_y_frac  — vertical center offset as fraction of source height.
                  0.0 = dead center. Positive = shift crop down (pan up).

    Motion ease convention:
      push_in / pull_back use ease_in_out for cinematic feel.
      pan_left / pan_right use ease_in_out so they start and settle gently.
      float uses a sine wave — naturally smooth.
      luxury_parallax (Kling continuation) uses ease_in so it starts from
        nearly still — this is the "camera settling after the reveal" beat
        that makes the Kling → Ken Burns transition feel intentional, not abrupt.
    """
    max_zoom = min(start_zoom + 0.5, 1.8)
    min_zoom = max(start_zoom - 0.5, 1.0)

    # How far to pan as a fraction of source size (at fixed 1.25× zoom,
    # available pan range = 1 - 1/1.25 = 20% of source in each axis).
    # We use 60% of the available range so there's always a safe margin.
    PAN_RANGE = (1.0 - 1.0 / 1.25) * 0.6  # ~0.12 of source width

    curve = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)  # 0.0 → 1.0

        if preset == "push_in":
            zoom  = start_zoom + (max_zoom - start_zoom) * ease_in_out(t)
            pan_x = 0.0
            pan_y = 0.0

        elif preset == "pull_back":
            zoom  = max_zoom - (max_zoom - min_zoom) * ease_in_out(t)
            pan_x = 0.0
            pan_y = 0.0

        elif preset == "pan_left":
            zoom  = 1.25
            # Start right-of-center, move to left-of-center
            pan_x = PAN_RANGE * (0.5 - ease_in_out(t))
            pan_y = 0.0

        elif preset == "pan_right":
            zoom  = 1.25
            # Start left-of-center, move to right-of-center
            pan_x = PAN_RANGE * (ease_in_out(t) - 0.5)
            pan_y = 0.0

        elif preset == "float":
            # Gentle sine-wave breathing — starts and ends at rest
            zoom  = 1.1 + 0.03 * math.sin(2.0 * math.pi * t)
            pan_x = 0.0
            pan_y = 0.0

        elif preset == "luxury_parallax":
            # Continuation beat after a Kling clip.
            # ease_in so motion starts from nearly still — the camera "settles"
            # after the Kling transformation, then drifts deliberately.
            # Available drift range = (1 - 1/1.15) ≈ 13% of source.
            # We use 30% of that so the drift never exits the safe zone.
            drift = (1.0 - 1.0 / 1.15) * 0.30
            t_in  = ease_in(t)
            zoom  = 1.15
            pan_x = drift * t_in          # rightward drift
            pan_y = -drift * 0.6 * t_in   # slight upward drift

        else:  # static
            zoom  = 1.0
            pan_x = 0.0
            pan_y = 0.0

        curve.append((zoom, pan_x, pan_y))

    return curve


# ── PER-FRAME RENDERER ────────────────────────────────────────────────────────

def render_frame(img, zoom, pan_x_frac, pan_y_frac, out_w, out_h):
    """
    Render one video frame.

    Computes a floating-point affine transform that maps the desired crop
    window (zoom + pan) in the source image to the full output frame.
    cv2.warpAffine with INTER_CUBIC gives sub-pixel bicubic interpolation —
    no integer quantization, no staircase stepping.

    BORDER_REPLICATE fills any out-of-bounds pixels with the nearest edge
    pixel (avoids black borders if the crop window slightly exceeds the source).
    """
    src_h, src_w = img.shape[:2]

    # Size of the crop window IN SOURCE PIXELS.
    # At zoom=1.0: we show the full source. At zoom=1.5: we show 1/1.5 of it.
    crop_w = src_w / zoom
    crop_h = src_h / zoom

    # Center of the crop window, shifted by the pan offset.
    # pan_x_frac is a fraction of src_w, so pan moves in source-pixel units.
    cx = src_w / 2.0 + pan_x_frac * src_w
    cy = src_h / 2.0 + pan_y_frac * src_h

    # Top-left corner of the crop window (floating point — the key to smoothness)
    x1 = cx - crop_w / 2.0
    y1 = cy - crop_h / 2.0

    # Affine transform: 3-point correspondence
    #   source crop corners  →  output frame corners
    src_pts = np.float32([
        [x1,          y1         ],   # top-left
        [x1 + crop_w, y1         ],   # top-right
        [x1,          y1 + crop_h],   # bottom-left
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
    """
    Open an FFmpeg subprocess that accepts raw BGR24 frames on stdin
    and encodes them to H.264 MP4.

    preset=fast balances encoding speed and quality on Railway.
    crf=18 is visually lossless for web/social video.
    """
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
    args    = parse_args()
    fps     = args.fps
    out_w   = args.output_w
    out_h   = args.output_h
    n_frames = round(args.duration * fps)

    # Load and aspect-crop source image
    img = load_and_crop(args.source, out_w, out_h)

    # Build the full motion curve for this clip
    curve = build_motion_curve(args.preset, n_frames, args.start_zoom)

    # Ending zoom — used by renderPipeline.js as the next clip's start_zoom
    ending_zoom = curve[-1][0]

    # Open FFmpeg encoder
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

    # Success — print JSON for Node.js to parse
    print(json.dumps({
        "path":       args.output,
        "endingZoom": ending_zoom,
        "duration":   args.duration,
    }))


if __name__ == "__main__":
    main()
