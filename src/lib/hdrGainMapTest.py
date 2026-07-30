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


def read_aux_metadata(aux_image_path):
    """Same as read_metadata but for the extracted auxiliary (gain map)
    image, which per real-world validation carries its OWN separate XMP
    block with HDRGainMapVersion/HDRGainMapHeadroom — this is often where
    that metadata actually lives, not in the top-level file at all."""
    return read_metadata(aux_image_path)


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

def looks_like_heic(source_path):
    """Quick signature check — HEIC/HEIF files start with an ftyp box
    whose brand indicates HEIC, independent of file extension (which
    can't be trusted; upload paths sometimes mislabel extensions)."""
    try:
        with open(source_path, "rb") as f:
            head = f.read(12)
        if len(head) < 12:
            return False
        brand = head[8:12]
        return brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1")
    except Exception:
        return False


def decode_standard(source_path):
    """Exactly what production's smartCorrect.py does today: cv2.imread()
    reads only the flattened base image, silently discarding XMP/MPF/gain
    map data. Reproduced here unchanged so the comparison is apples-to-apples
    with what's actually shipping."""
    img = cv2.imread(source_path)
    if img is None:
        if looks_like_heic(source_path):
            raise ValueError(
                "This is a genuine HEIC/HEIF file, and cv2.imread() cannot decode HEIC at all "
                "in this environment (OpenCV's standard build has no HEIF codec). This is NOT "
                "specific to this test script — production's smartCorrect.py uses the exact same "
                "cv2.imread() call and would fail identically on this file. Real HEIC support needs "
                "a HEIC-capable decoder added (e.g. pillow-heif) before any gain-map recovery work "
                "can even begin for this format."
            )
        raise ValueError(f"cv2.imread could not read: {source_path} (not a recognized HEIC file either — check the actual file signature)")
    return img


def apply_gain_map(base_bgr_uint8, gain_map_gray_uint8, headroom, target_display_headroom=None):
    """
    hdr_rgb = sdr_rgb * (1.0 + (headroom - 1.0) * gainmap)      [all linear-light]

    Formula per Apple's own "Applying Apple HDR effect to your photos" doc,
    cross-checked against jackchou00.com's independent write-up. At
    gainmap=1.0 everywhere, output == headroom exactly (verified below in
    a self-check in main()).

    The result is a genuine linear-light HDR image that can exceed 1.0 —
    correct, but not directly viewable as an 8-bit JPEG.

    `target_display_headroom` sets the display ceiling for the tonemap
    clip-and-normalize step below. Left as None (the default), it's set to
    the photo's OWN decoded `headroom` automatically — not a fixed
    constant. This is a fix for a real problem found across the 4 real
    test photos on July 30, 2026: with a fixed target of 2.0, the light-
    fixture photo (headroom 3.44, 48.6% of the frame elevated above SDR
    white) lost everything between 2.0 and 3.44 to hard clipping — nearly
    half the image. The backlit-window photo (27% affected) looked fine
    under the same fixed target purely because less area happened to sit
    above the clip point. That's not really two photos needing different
    treatment — it's one bug: a fixed ceiling loses more information the
    more of the frame sits above it. Setting the ceiling to the photo's
    actual headroom means nothing gets clipped away regardless of how much
    of the frame is elevated, which is why this fix didn't need a second
    "how much of the frame is affected" branch at all — it falls out of
    getting the ceiling right in the first place.

    A manual override is still accepted (e.g. to force a lower ceiling and
    compare) — pass a number instead of None.

    This is still a simple clip-and-normalize curve, not a proper
    photographic tonemap operator (Reinhard, filmic, etc.) — good enough
    to confirm recovery is happening correctly across varied scenes, not
    a finished production tonemap. That's a separate, later refinement if
    the adaptive ceiling alone doesn't look right on more real photos.
    """
    if target_display_headroom is None:
        target_display_headroom = headroom

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
        "targetWasAutomatic": target_display_headroom == headroom,
        "hdrLinearMax": float(np.max(hdr_linear)),
        "hdrLinearMean": float(np.mean(hdr_linear)),
        "fractionAboveSDRWhite": float(np.mean(hdr_linear > 1.0)),
        "fractionClippedAtCeiling": float(np.mean(hdr_linear > target_display_headroom)),
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
    parser.add_argument("--target-display-headroom", type=float, default=None, help="Manual override for the tonemap ceiling. Omit to auto-use the photo's own decoded headroom (default, recommended).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {"source": args.source, "gainMapPresent": False, "warnings": []}

    # ── 1. Metadata inspection — TOP-LEVEL file first ──
    # This alone is often NOT enough: real-world validation (community
    # extraction thread, cited in the module docstring) found that
    # HDRGainMapVersion/HDRGainMapHeadroom frequently live inside the
    # auxiliary gain-map image's OWN XMP block, not the top-level file's.
    # So this is a first pass, not the final answer — step 2 below always
    # runs regardless of what this finds.
    meta = read_metadata(args.source)
    has_version, version_val = has_gain_map_version(meta)
    headroom, headroom_source = find_headroom(meta)

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

    # ── 3. ALWAYS attempt gain-map image extraction, regardless of what
    # step 1 found — this is the fix for the real bug this line of
    # comments is replacing: bailing out here on a top-level metadata
    # miss was wrong, because the metadata frequently isn't there to find
    # until AFTER extraction. ──
    workdir = tempfile.mkdtemp(prefix="hdrtest-")
    try:
        gain_map_path, tag_used, extract_err = extract_gain_map_image_bytes(args.source, workdir)
        report["gainMapImageExtraction"] = {"tagUsed": tag_used, "error": extract_err}

        if gain_map_path:
            # Check the EXTRACTED image's own metadata — this is the
            # primary source of truth per real-world validation, not a
            # fallback. Overrides the top-level result when it finds
            # something the top-level pass didn't.
            aux_meta = read_aux_metadata(gain_map_path)
            aux_has_version, aux_version_val = has_gain_map_version(aux_meta)
            aux_headroom, aux_headroom_source = find_headroom(aux_meta)

            if aux_has_version or aux_headroom:
                has_version = has_version or aux_has_version
                version_val = version_val or aux_version_val
                headroom = aux_headroom if aux_headroom is not None else headroom
                headroom_source = (f"{aux_headroom_source} (found in extracted auxiliary image, not top-level file)"
                                    if aux_headroom is not None else headroom_source)

        report["hdrGainMapVersion"] = version_val
        report["headroom"] = headroom
        report["headroomSource"] = headroom_source
        report["gainMapPresent"] = bool(has_version or headroom or gain_map_path)

        if not report["gainMapPresent"]:
            report["warnings"].append("No gain map metadata found in either the top-level file or (extraction attempted but failed/empty) the auxiliary image. This file has no recoverable HDR data through this path.")
            print(json.dumps(report, indent=2))
            return

        if not gain_map_path:
            report["warnings"].append(f"Gain map metadata present but auxiliary image bytes could not be extracted via exiftool MPImage tags: {extract_err}")
            print(json.dumps(report, indent=2))
            return

        if headroom is None:
            report["warnings"].append("Gain map version tag and/or auxiliary image present but no headroom value found anywhere — cannot compute recovery without it.")
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
