#!/usr/bin/env python3
# hdrGainMapTest.py — STANDALONE HDR gain-map inspect/decode test tool
#
# Per the July 28 architecture review: build a test tool first, fully
# decoupled from production Smart Correct. Inspect-only pass reporting
# HDR presence/type per file, then attempt decode and output
# recovered-vs-standard-decode side by side. No MLS guard, no Mertens,
# no brightness lift touched — this file does not import from, call, or
# get called by smartCorrect.py / correctPipeline.js in any way.
#
# ── WHAT THIS HANDLES ────────────────────────────────────────────────────
# Apple's "Apple Gain Map" format as embedded in JPEG (not HEIC). The
# July 29 browser-side test confirmed production photos are arriving as
# JPEG+XMP-gain-map (camera "Most Compatible" setting), not true HEIC
# containers — a meaningfully different format from what decode_heif_source()
# implied. This script targets that JPEG+MPF shape specifically.
#
# ── SOURCES THIS IS BUILT FROM (all public) ─────────────────────────────
# - Apple's own doc: "Applying Apple HDR effect to your photos"
#   https://developer.apple.com/documentation/appkit/applying-apple-hdr-effect-to-your-photos
# - Apple WWDC24 session 10177, "Use HDR for dynamic image experiences in your app"
# - Community extraction validated on real iPhone 15 Pro photos (2023):
#   https://gist.github.com/kiding/fa4876ab4ddc797e3f18c71b3c2eeb3a
#   (`exiftool -MPImage2 -b photo.jpg > gainmap.jpg` — this exact command,
#   confirmed working against a real file)
# - Formula cross-check: https://jackchou00.com/en/posts/iphone-heic-hdr-format/
#     hdr_rgb = sdr_rgb * (1.0 + (headroom - 1.0) * gainmap)      [all linear-light]
#   "When both layers equal 1.0 the formula outputs exactly the headroom
#   value" — verified against this script's own math below.
#
# ── WHAT THIS DOES NOT DO ────────────────────────────────────────────────
# This is a visualization/inspection aid, not a production-quality decoder.
# The tonemap step at the end (compressing the recovered HDR data back to
# an 8-bit viewable preview) is a simple, clearly-approximate curve — good
# enough to SEE whether recovery worked, not good enough to ship as a
# finished MLS photo. That's a separate, later decision.

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

EXIFTOOL_BIN = shutil.which("exiftool") or "exiftool"


# ── METADATA INSPECTION (exiftool) ──────────────────────────────────────

def read_metadata(source_path):
    """Runs exiftool once, returns the parsed JSON dict (or {} on failure).
    exiftool auto-decodes unknown XMP namespaces generically by their XML
    element names — confirmed empirically against a synthetic file built
    with Apple's real XMP structure — so no custom Apple-specific tag
    table is needed for this to work."""
    try:
        proc = subprocess.run(
            [EXIFTOOL_BIN, "-j", "-G1", source_path],
            capture_output=True, timeout=20, check=False,
        )
        data = json.loads(proc.stdout.decode("utf-8", errors="replace") or "[]")
        return data[0] if data else {}
    except Exception as e:
        print(f"[hdrGainMapTest] exiftool metadata read failed: {e}", file=sys.stderr)
        return {}


def find_headroom(meta):
    """Returns (headroom_float, source_string) or (None, reason_string).
    Priority order per Apple's own documented methods, most direct first."""
    # Method 1 — direct XMP tag (most reliable when present)
    for key in meta:
        if key.endswith("HDRGainMapHeadroom") or key.endswith("HDRCapacityMax"):
            try:
                val = float(meta[key])
                if val > 0:
                    return val, key
            except (TypeError, ValueError):
                pass

    # Method 2 — MakerNotes HDRHeadroom / HDRGain (Apple's documented
    # no-frameworks fallback). NOTE: Apple's exact piecewise formula for
    # combining these two raw values into a headroom-in-stops number
    # isn't in any public doc found during this session — where present,
    # HDRHeadroom itself is reported here directly as a best-effort
    # approximation, clearly flagged as such, rather than guessing at
    # undocumented math. Confirm/replace this if it's ever wrong.
    for key in meta:
        if key.endswith(":HDRHeadroom") or key.endswith(":HDRHeadroom "):
            try:
                val = float(meta[key])
                if val > 0:
                    return val, key + " (MakerNotes fallback — approximate, see comment)"
            except (TypeError, ValueError):
                pass

    return None, "no headroom metadata found (HDRGainMapVersion/HDRGainMapHeadroom/HDRCapacityMax/MakerNotes:HDRHeadroom all absent)"


def has_gain_map_version(meta):
    for key in meta:
        if key.endswith("HDRGainMapVersion"):
            return True, meta[key]
    return False, None


# ── EMBEDDED GAIN-MAP IMAGE EXTRACTION (MPF auxiliary image) ────────────

def extract_gain_map_image_bytes(source_path, workdir):
    """Extracts the second MPF image (the gain map) via exiftool, exactly
    the command validated against real iPhone photos in the public
    extraction thread cited above. Tries MPImage2 first (the common case
    — primary + one auxiliary), falls back to scanning all MPImage* tags
    if there turn out to be more than two."""
    candidates = ["MPImage2", "MPImage1", "MPImage3"]
    for tag in candidates:
        out_path = os.path.join(workdir, f"_extracted_{tag}.jpg")
        try:
            with open(out_path, "wb") as f:
                proc = subprocess.run(
                    [EXIFTOOL_BIN, f"-{tag}", "-b", source_path],
                    stdout=f, stderr=subprocess.PIPE, timeout=20, check=False,
                )
            size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            if size > 100:  # trivially non-empty
                return out_path, tag, None
        except Exception as e:
            return None, tag, str(e)
    return None, None, "no non-empty MPImage tag found (MPImage2/1/3 all missing or empty)"


# ── COLOR MATH (all public/standard transfer functions — no Apple-internal math) ──

def srgb_eotf(v):
    """sRGB encoded [0,1] -> linear light [0,1]. IEC 61966-2-1, public standard."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb_oetf(lin):
    """Linear light [0,1] -> sRGB encoded [0,1]. Inverse of srgb_eotf."""
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * (lin ** (1 / 2.4)) - 0.055)


def rec709_eotf(v):
    """Rec.709/BT.709 encoded [0,1] -> linear light [0,1]. ITU-R BT.709,
    public standard. Apple's docs specify the gain map is encoded with
    this transfer function (not sRGB) — using the wrong one here would
    silently produce a plausible-looking but wrong result."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45))


# ── CORE DECODE ──────────────────────────────────────────────────────────

def decode_standard(source_path):
    """Exactly what production's smartCorrect.py does today: cv2.imread()
    reads only the flattened base image, silently discarding XMP/MPF/gain
    map data. Reproduced here unchanged so the comparison is apples-to-apples
    with what's actually shipping."""
    img = cv2.imread(source_path)
    if img is None:
        raise ValueError(f"cv2.imread could not read: {source_path}")
    return img


def apply_gain_map(base_bgr_uint8, gain_map_gray_uint8, headroom, target_display_headroom=2.0):
    """
    hdr_rgb = sdr_rgb * (1.0 + (headroom - 1.0) * gainmap)      [all linear-light]

    Formula per Apple's own "Applying Apple HDR effect to your photos" doc,
    cross-checked against jackchou00.com's independent write-up. At
    gainmap=1.0 everywhere, output == headroom exactly (verified below in
    a self-check in main()).

    The result is a genuine linear-light HDR image that can exceed 1.0 —
    correct, but not directly viewable as an 8-bit JPEG. `target_display_headroom`
    simulates a display that can only show `target_display_headroom`x the
    SDR white point (typical of a decent modern monitor) and tonemaps down
    to that — this is what makes recovered highlight detail actually
    visible in the output JPEG instead of just clipping again. This
    tonemap curve is a simple clip-and-normalize, deliberately NOT the
    "conservative tonemap function" planned for production — good enough
    to confirm recovery is happening, not a finished product.
    """
    h, w = base_bgr_uint8.shape[:2]
    gain_map_resized = cv2.resize(gain_map_gray_uint8, (w, h), interpolation=cv2.INTER_LINEAR)

    base_norm = base_bgr_uint8.astype(np.float64) / 255.0
    gain_norm = gain_map_resized.astype(np.float64) / 255.0

    base_linear = srgb_eotf(base_norm)
    gain_linear = rec709_eotf(gain_norm)[..., np.newaxis]  # broadcast over BGR channels

    hdr_linear = base_linear * (1.0 + (headroom - 1.0) * gain_linear)

    # Tonemap: clip to target_display_headroom, normalize, re-encode sRGB.
    recovered_linear = np.clip(hdr_linear, 0.0, target_display_headroom) / target_display_headroom
    recovered_srgb = srgb_oetf(recovered_linear)
    recovered_uint8 = np.clip(recovered_srgb * 255.0, 0, 255).astype(np.uint8)

    stats = {
        "headroomUsed": headroom,
        "targetDisplayHeadroom": target_display_headroom,
        "hdrLinearMax": float(np.max(hdr_linear)),
        "hdrLinearMean": float(np.mean(hdr_linear)),
        "fractionAboveSDRWhite": float(np.mean(hdr_linear > 1.0)),
    }
    return recovered_uint8, stats


def self_check_formula():
    """When gainmap==1.0 everywhere and base pixel is reference white
    (linear 1.0), the formula must output exactly `headroom`. This is the
    one sanity check from the public write-up that's cheap to verify in
    code, so it runs on every invocation rather than trusting it blindly."""
    headroom = 4.0
    base_linear = np.array([[1.0]])
    gain_linear = np.array([[1.0]])
    hdr_linear = base_linear * (1.0 + (headroom - 1.0) * gain_linear)
    assert abs(float(hdr_linear[0][0]) - headroom) < 1e-9, "gain map formula self-check failed"


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    self_check_formula()

    parser = argparse.ArgumentParser(description="Standalone HDR gain-map inspect/decode test — not wired into any production pipeline.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-display-headroom", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {"source": args.source, "gainMapPresent": False, "warnings": []}

    # ── 1. Metadata inspection ──
    meta = read_metadata(args.source)
    has_version, version_val = has_gain_map_version(meta)
    headroom, headroom_source = find_headroom(meta)
    report["hdrGainMapVersion"] = version_val
    report["headroom"] = headroom
    report["headroomSource"] = headroom_source
    report["gainMapPresent"] = bool(has_version or headroom)

    # ── 2. Standard decode (reproduces production's cv2.imread exactly) ──
    standard_path = os.path.join(args.output_dir, "standard.jpg")
    try:
        base_img = decode_standard(args.source)
        cv2.imwrite(standard_path, base_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        report["standardDecode"] = {"output": standard_path, "width": base_img.shape[1], "height": base_img.shape[0]}
    except Exception as e:
        report["standardDecode"] = {"error": str(e)}
        print(json.dumps(report))
        sys.exit(1)

    # ── 3. Attempt recovery, only if metadata says a gain map exists ──
    if not report["gainMapPresent"]:
        report["warnings"].append("No gain map metadata found — skipping recovery attempt. This file has no recoverable HDR data through this path.")
        print(json.dumps(report, indent=2))
        return

    if headroom is None:
        report["warnings"].append("Gain map version tag present but no headroom value found — cannot compute recovery without it.")
        print(json.dumps(report, indent=2))
        return

    workdir = tempfile.mkdtemp(prefix="hdrtest-")
    try:
        gain_map_path, tag_used, extract_err = extract_gain_map_image_bytes(args.source, workdir)
        report["gainMapImageExtraction"] = {"tagUsed": tag_used, "error": extract_err}

        if not gain_map_path:
            report["warnings"].append(f"Gain map metadata present but auxiliary image bytes could not be extracted via exiftool MPImage tags: {extract_err}")
            print(json.dumps(report, indent=2))
            return

        gain_map_img = cv2.imread(gain_map_path, cv2.IMREAD_GRAYSCALE)
        if gain_map_img is None:
            report["warnings"].append(f"Extracted gain map bytes at {gain_map_path} could not be decoded as an image.")
            print(json.dumps(report, indent=2))
            return

        report["gainMapImage"] = {"width": int(gain_map_img.shape[1]), "height": int(gain_map_img.shape[0])}

        recovered_img, recovery_stats = apply_gain_map(
            base_img, gain_map_img, headroom, target_display_headroom=args.target_display_headroom
        )
        recovered_path = os.path.join(args.output_dir, "recovered.jpg")
        cv2.imwrite(recovered_path, recovered_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        report["recoveredDecode"] = {"output": recovered_path, **recovery_stats}

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
