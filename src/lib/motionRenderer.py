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
  soft_hold        — Reveal Presets opener (Classic Reveal, Luxury Drift):
                      near-static, barely-perceptible zoom. Deliberately
                      does not commit to a direction, so it can never
                      collide with whatever End Motion the continuation
                      phase picks.
  restrained_push  — Reveal Presets opener (Cinematic Reveal): a small
                      fraction of push_in's full range. Reads as the first
                      beat of one continuous push that the continuation
                      phase finishes — NOT a full move of its own. Only
                      safe to pair with a continuation that keeps pushing
                      forward; pull_back is excluded from every Reveal
                      Preset's End Motion list for exactly this reason.

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
                             "tilt_up","tilt_down","drift","pan_zoom",
                             "float","luxury_parallax","static",
                             "soft_hold","restrained_push"])
    p.add_argument("--duration",   type=float, required=True)
    p.add_argument("--output",     required=True,  help="Output .mp4 path")
    p.add_argument("--start-zoom", type=float, default=1.0, dest="start_zoom")
    p.add_argument("--output-w",   type=int,   default=1920, dest="output_w")
    p.add_argument("--output-h",   type=int,   default=1080, dest="output_h")
    p.add_argument("--fps",        type=int,   default=25)
    # NEW (July 20, 2026 — Sam's request): burns a small text badge into
    # every frame of this clip. Only ever passed for the Room Reveal
    # opener phase (soft_hold/restrained_push, before the wipe) with
    # "Original" — see motionPresets.js's applyMotionPreset and
    # renderPipeline.js's opener call site. Optional and unused by every
    # other caller (continuation phase, standalone Ken Burns), so nothing
    # else changes shape.
    p.add_argument("--label-text", default=None, dest="label_text")
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

        elif preset == "pan_zoom":
            # Classic "textbook" Ken Burns move — steady lateral pan combined
            # with a simultaneous zoom-in, so it reads as one cohesive
            # diagonal-feeling camera move rather than a pure dolly (push_in)
            # or a pure track (pan_right). Distinct from "drift": drift uses
            # a small, subtle zoom change (1.15→1.20) as a secondary touch on
            # top of a diagonal path; pan_zoom uses the full push_in zoom
            # range (start_zoom→max_zoom) paired with a full-width lateral
            # pan, so both axes read as equally deliberate. Direction is
            # left-to-right (positive pan_x convention, matching pan_right)
            # — the standard convention for this shot type.
            zoom  = start_zoom + (max_zoom - start_zoom) * ease_in_out(t)
            pan_x = PAN_RANGE * (ease_in_out(t) - 0.5)
            pan_y = 0.0

        elif preset == "float":
            # Gentle sine-wave breathing — dining rooms, flex spaces, wide shots
            # where deliberate directional movement would feel forced.
            zoom  = 1.1 + 0.03 * math.sin(2.0 * math.pi * t)
            pan_x = 0.0
            pan_y = 0.0

        elif preset == "luxury_parallax":
            # Kling continuation beat — ease_in zoom + diagonal settle.
            #
            # zoom now RAMPS from start_zoom up to start_zoom + SETTLE_AMOUNT,
            # rather than the previous flat zoom = 1.15 (hardcoded, ignoring
            # start_zoom entirely). That hardcoded value was the actual cause
            # of a visible jump cut at the Kling→Ken Burns stitch point: the
            # caller passes start_zoom=1.0 specifically because the source
            # image already IS Kling's exact last frame at full composition —
            # but the old code rendered its very first frame at 1.15 zoom
            # regardless, a real ~13% tighter crop than what Kling ended on,
            # at the same center point (pan_x/pan_y are 0 at t=0 either way,
            # which is why the discontinuity reads as "same focal point, just
            # closer" rather than a framing shift). Ramping from start_zoom
            # guarantees frame 0 is pixel-identical in composition to
            # whatever the caller passed in, then eases into the same total
            # "settle in" push as before — just anchored to the real starting
            # composition instead of an arbitrary constant.
            SETTLE_AMOUNT = 0.15
            t_in  = ease_in(t)
            zoom  = start_zoom + SETTLE_AMOUNT * t_in
            drift = (1.0 - 1.0 / zoom) * 0.30 if zoom > 1.0 else 0.0
            pan_x = drift * t_in
            pan_y = -drift * 0.6 * t_in

        elif preset == "soft_hold":
            # Reveal Presets opener — Classic Reveal, Luxury Drift.
            # Deliberately near-static: a tiny ease-in-out zoom creep,
            # much smaller than "float"'s already-subtle breathing, and
            # zero pan on either axis. The point is to read as "holding
            # on the vacant room" rather than any camera move at all, so
            # the wipe into the continuation phase never has to fight a
            # direction the opener already established.
            SOFT_HOLD_AMOUNT = 0.04
            zoom  = start_zoom + SOFT_HOLD_AMOUNT * ease_in_out(t)
            pan_x = 0.0
            pan_y = 0.0

        elif preset == "restrained_push":
            # Reveal Presets opener — Cinematic Reveal only. A small
            # fraction of push_in's full range (0.10 vs push_in's up to
            # 0.5+ zoom delta) — reads as the FIRST beat of one continuous
            # push that the continuation phase (almost always push_in)
            # finishes after the wipe, not a complete move on its own.
            # Only pair this with a continuation that keeps pushing
            # forward — never pull_back, which is why pull_back is
            # excluded from Cinematic Reveal's End Motion list.
            RESTRAINED_PUSH_AMOUNT = 0.10
            zoom  = start_zoom + RESTRAINED_PUSH_AMOUNT * ease_in_out(t)
            pan_x = 0.0
            pan_y = 0.0

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


# ── LABEL BADGE ───────────────────────────────────────────────────────────────

def draw_label_badge(frame, text):
    """
    Burns a text badge into the upper-left corner of a frame — e.g.
    "Original" on the Room Reveal opener, before the wipe. Semi-transparent
    dark background behind white text for legibility against any photo.
    Uses cv2 (already the renderer's own dependency — no new library needed).

    REVISED (July 21, 2026 — real render feedback: the opener wipes fast
    enough that the original bottom-left, smaller badge was missed ~95% of
    the time — Sam had to pause playback to find it at all). Moved to the
    UPPER section of the frame (out of the way of typical bottom-of-frame
    room content like floor/rugs, and the first place a viewer's eye lands)
    and enlarged substantially — font_scale 0.7→1.3, thickness 2→3 — so it
    reads clearly even across a sub-second opener before the wipe.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 3
    padding = 20

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x1, y1 = 32, 32
    x2, y2 = x1 + text_w + 2 * padding, y1 + text_h + 2 * padding

    # Semi-transparent dark rectangle behind the text — alpha-blended onto
    # a copy of the region rather than drawn opaque, so it reads as a badge
    # rather than a hard block over the photo.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), thickness=-1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    text_x = x1 + padding
    text_y = y2 - padding
    cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return frame


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
            if args.label_text:
                frame = draw_label_badge(frame, args.label_text)
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
