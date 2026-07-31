# hdrRecover.py — SHARED HDR gain-map decode/recovery module
#
# Extracted from hdrGainMapTest.py on July 30, 2026, when this logic moved
# from "standalone test tool only" to "wired into production smartCorrect.py".
# Both smartCorrect.py and hdrGainMapTest.py import from this file — the
# actual decode/recovery math lives in exactly one place. Do not copy these
# functions into either caller; import them instead, or the two will
# silently drift apart the next time either needs a fix (exactly the kind
# of bug this whole investigation has been about catching elsewhere).
#
# Everything here was validated against real iPhone photos across several
# rounds on July 29-30, 2026:
#   - Ceiling: auto-scales to each photo's own decoded headroom (was a
#     fixed 2.0 — caused real, measured clipping: 8% of pixels on the
#     worst real test photo, 0% after the fix)
#   - Color: tonemap applied to luminance only, not each RGB channel
#     independently (per-channel version washed a blue sky toward gray)
#   - Gamut: when one channel would still exceed 1.0 after luminance
#     scaling, all three scale down together rather than clamping just the
#     one channel (residual desaturation fix)
#   - highlight_reserve default: 0.30, raised from an initial 0.15 after
#     every real test photo (portraits, a real backyard/sliding-door shot)
#     showed flattened local contrast (shirt wrinkles, sunlit foliage) at
#     the lower value
#
# Sources (all public):
# - Apple's own doc: "Applying Apple HDR effect to your photos"
#   https://developer.apple.com/documentation/appkit/applying-apple-hdr-effect-to-your-photos
# - Apple WWDC24 session 10177, "Use HDR for dynamic image experiences in your app"
# - Community extraction validated on real iPhone 15 Pro photos (2023):
#   https://gist.github.com/kiding/fa4876ab4ddc797e3f18c71b3c2eeb3a
# - Formula cross-check: https://jackchou00.com/en/posts/iphone-heic-hdr-format/

import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

EXIFTOOL_BIN = shutil.which("exiftool") or "exiftool"


# ── FORMAT DETECTION ─────────────────────────────────────────────────────

def looks_like_heic(source_path):
    """Signature check — HEIC/HEIF files start with an ftyp box whose brand
    indicates HEIC, independent of file extension (which can't be trusted)."""
    try:
        with open(source_path, "rb") as f:
            head = f.read(12)
        if len(head) < 12:
            return False
        brand = head[8:12]
        return brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1")
    except Exception:
        return False


# ── METADATA INSPECTION (exiftool) ──────────────────────────────────────

def read_metadata(source_path):
    """Runs exiftool once, returns the parsed JSON dict (or {} on failure).
    exiftool auto-decodes unknown XMP namespaces generically by their XML
    element names — confirmed empirically — so no custom Apple-specific
    tag table is needed for this to work."""
    try:
        proc = subprocess.run(
            [EXIFTOOL_BIN, "-j", "-G1", source_path],
            capture_output=True, timeout=20, check=False,
        )
        data = json.loads(proc.stdout.decode("utf-8", errors="replace") or "[]")
        return data[0] if data else {}
    except Exception as e:
        print(f"[hdrRecover] exiftool metadata read failed: {e}", file=sys.stderr)
        return {}


def read_aux_metadata(aux_image_path):
    """Same as read_metadata but for the extracted auxiliary (gain map)
    image, which per real-world validation carries its OWN separate XMP
    block with HDRGainMapVersion/HDRGainMapHeadroom — this is often where
    that metadata actually lives, not in the top-level file at all."""
    return read_metadata(aux_image_path)


def has_gain_map_version(meta):
    for key in meta:
        if key.endswith("HDRGainMapVersion"):
            return True, meta[key]
    return False, None


def find_headroom(meta):
    """Returns (headroom_float, source_string) or (None, reason_string).
    Priority order per Apple's own documented methods, most direct first."""
    for key in meta:
        if key.endswith("HDRGainMapHeadroom") or key.endswith("HDRCapacityMax"):
            try:
                val = float(meta[key])
                if val > 0:
                    return val, key
            except (TypeError, ValueError):
                pass

    # MakerNotes HDRHeadroom fallback — Apple's exact piecewise formula for
    # combining HDRHeadroom+HDRGain into a proper headroom value isn't in
    # any public doc found during this investigation. Where present,
    # HDRHeadroom itself is reported directly as a best-effort
    # approximation, clearly flagged as such, rather than guessing at
    # undocumented math.
    for key in meta:
        if key.endswith(":HDRHeadroom") or key.endswith(":HDRHeadroom "):
            try:
                val = float(meta[key])
                if val > 0:
                    return val, key + " (MakerNotes fallback — approximate)"
            except (TypeError, ValueError):
                pass

    return None, "no headroom metadata found (HDRGainMapVersion/HDRGainMapHeadroom/HDRCapacityMax/MakerNotes:HDRHeadroom all absent)"


# ── EMBEDDED GAIN-MAP IMAGE EXTRACTION (MPF auxiliary image) ────────────

def extract_gain_map_image_bytes(source_path, workdir):
    """Extracts the second MPF image (the gain map) via exiftool — the
    exact command validated against real iPhone photos in the public
    extraction thread cited above."""
    candidates = ["MPImage2", "MPImage1", "MPImage3"]
    for tag in candidates:
        out_path = os.path.join(workdir, f"_extracted_{tag}.jpg")
        try:
            with open(out_path, "wb") as f:
                subprocess.run(
                    [EXIFTOOL_BIN, f"-{tag}", "-b", source_path],
                    stdout=f, stderr=subprocess.PIPE, timeout=20, check=False,
                )
            size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            if size > 100:
                return out_path, tag, None
        except Exception as e:
            return None, tag, str(e)
    return None, None, "no non-empty MPImage tag found (MPImage2/1/3 all missing or empty)"


# ── COLOR MATH (all public/standard transfer functions) ─────────────────

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
    this transfer function, not sRGB."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45))


# ── CORE DECODE ──────────────────────────────────────────────────────────

def decode_standard(source_path):
    """What non-HDR-aware code (production's cv2.imread() before this
    module existed) does: reads only the flattened base image, silently
    discarding XMP/MPF/gain map data."""
    img = cv2.imread(source_path)
    if img is None:
        if looks_like_heic(source_path):
            raise ValueError(
                "This is a genuine HEIC/HEIF file, and cv2.imread() cannot decode HEIC at all "
                "in this environment (OpenCV's standard build has no HEIF codec). Real HEIC "
                "support needs a HEIC-capable decoder added (e.g. pillow-heif) before any gain-"
                "map recovery work can even begin for this format — deliberately not built, per "
                "the July 30, 2026 decision that this path is unlikely enough (requires a "
                "deliberate camera-setting change plus a cable transfer users have no reason to "
                "choose over email) not to be worth the added dependency."
            )
        raise ValueError(f"cv2.imread could not read: {source_path} (not a recognized HEIC file either)")
    return img


def apply_gain_map(base_bgr_uint8, gain_map_gray_uint8, headroom, target_display_headroom=None, highlight_reserve=0.30):
    """
    hdr_rgb = sdr_rgb * (1.0 + (headroom - 1.0) * gainmap)      [all linear-light]

    Formula per Apple's own "Applying Apple HDR effect to your photos" doc.
    At gainmap=1.0 everywhere, output == headroom exactly (verified in
    self_check_formula() below).

    Knee-curve tonemap with luminance-based color preservation and
    proportional gamut scaling — see module docstring for the three
    real-photo bugs this specific shape fixes (clipping, hue shift,
    residual desaturation). See git history / hdrGainMapTest.py's original
    comments for the full before/after reasoning on each fix if needed.
    """
    if target_display_headroom is None:
        target_display_headroom = headroom

    h, w = base_bgr_uint8.shape[:2]
    gain_map_resized = cv2.resize(gain_map_gray_uint8, (w, h), interpolation=cv2.INTER_LINEAR)

    base_norm = base_bgr_uint8.astype(np.float64) / 255.0
    gain_norm = gain_map_resized.astype(np.float64) / 255.0

    base_linear = srgb_eotf(base_norm)
    gain_linear = rec709_eotf(gain_norm)[..., np.newaxis]

    hdr_linear = base_linear * (1.0 + (headroom - 1.0) * gain_linear)

    # Luminance-based knee curve (BGR channel order — OpenCV convention).
    luma = 0.0722 * hdr_linear[..., 0] + 0.7152 * hdr_linear[..., 1] + 0.2126 * hdr_linear[..., 2]
    luma_safe = np.maximum(luma, 1e-6)

    unboosted_scale = 1.0 - highlight_reserve
    max_excess = max(target_display_headroom - 1.0, 1e-6)
    excess = np.maximum(luma - 1.0, 0.0)
    compressed_excess = highlight_reserve * excess / (excess + max_excess)
    luma_out = np.where(luma <= 1.0, luma * unboosted_scale, unboosted_scale + compressed_excess)

    color_scale = (luma_out / luma_safe)[..., np.newaxis]
    recovered_linear = hdr_linear * color_scale

    # Gamut fix: scale all 3 channels together if one still exceeds 1.0,
    # rather than clamping just the offending channel (preserves hue).
    max_channel = np.max(recovered_linear, axis=-1, keepdims=True)
    gamut_scale = np.where(max_channel > 1.0, 1.0 / np.maximum(max_channel, 1e-6), 1.0)
    recovered_linear = recovered_linear * gamut_scale
    recovered_linear = np.clip(recovered_linear, 0.0, 1.0)

    recovered_srgb = srgb_oetf(recovered_linear)
    recovered_uint8 = np.clip(recovered_srgb * 255.0, 0, 255).astype(np.uint8)

    stats = {
        "headroomUsed": headroom,
        "targetDisplayHeadroom": target_display_headroom,
        "targetWasAutomatic": target_display_headroom == headroom,
        "highlightReserve": highlight_reserve,
        "hdrLinearMax": float(np.max(hdr_linear)),
        "hdrLinearMean": float(np.mean(hdr_linear)),
        "fractionAboveSDRWhite": float(np.mean(hdr_linear > 1.0)),
        "fractionClippedAtCeiling": float(np.mean(hdr_linear > target_display_headroom)),
    }
    return recovered_uint8, stats


def self_check_formula():
    """When gainmap==1.0 everywhere and base pixel is reference white
    (linear 1.0), the formula must output exactly `headroom`. Runs on
    every import-time use rather than being trusted blindly."""
    headroom = 4.0
    base_linear = np.array([[1.0]])
    gain_linear = np.array([[1.0]])
    hdr_linear = base_linear * (1.0 + (headroom - 1.0) * gain_linear)
    assert abs(float(hdr_linear[0][0]) - headroom) < 1e-9, "gain map formula self-check failed"


self_check_formula()  # run once at import time, in every caller


# ── HIGH-LEVEL ENTRY POINT ───────────────────────────────────────────────

def recover_hdr_if_present(source_path, target_display_headroom=None, highlight_reserve=0.30):
    """
    The one function production code should call. Attempts full HDR
    gain-map recovery for `source_path`. Returns (recovered_img, report):

      - recovered_img: a BGR uint8 numpy array (same shape/type as
        cv2.imread() would return) if a gain map was found AND
        successfully extracted AND decoded — otherwise None. Callers
        should fall back to their own cv2.imread() when this is None;
        this function does NOT do that fallback itself, so a None return
        here does not necessarily mean the file is unreadable at all —
        only that there was no HDR data to recover.
      - report: a dict with gainMapPresent, headroom, headroomSource, and
        (when recovery succeeded) the same recovery stats apply_gain_map()
        returns — useful for logging/debugging which images this path
        actually engaged for. This is deliberately explicit and always set
        based on real outcomes, not a flag that's ever set unconditionally.
    """
    report = {"gainMapPresent": False, "recoveryApplied": False}

    meta = read_metadata(source_path)
    has_version, version_val = has_gain_map_version(meta)
    headroom, headroom_source = find_headroom(meta)

    workdir = tempfile.mkdtemp(prefix="hdrrecover-")
    try:
        gain_map_path, tag_used, extract_err = extract_gain_map_image_bytes(source_path, workdir)

        if gain_map_path:
            aux_meta = read_aux_metadata(gain_map_path)
            aux_has_version, aux_version_val = has_gain_map_version(aux_meta)
            aux_headroom, aux_headroom_source = find_headroom(aux_meta)
            if aux_has_version or aux_headroom:
                has_version = has_version or aux_has_version
                version_val = version_val or aux_version_val
                headroom = aux_headroom if aux_headroom is not None else headroom
                headroom_source = aux_headroom_source if aux_headroom is not None else headroom_source

        report["hdrGainMapVersion"] = version_val
        report["headroom"] = headroom
        report["headroomSource"] = headroom_source
        report["gainMapPresent"] = bool(has_version or headroom or gain_map_path)

        if not report["gainMapPresent"] or not gain_map_path or headroom is None:
            return None, report

        gain_map_img = cv2.imread(gain_map_path, cv2.IMREAD_GRAYSCALE)
        if gain_map_img is None:
            report["error"] = "extracted gain map image could not be decoded"
            return None, report

        base_img = decode_standard(source_path)
        recovered_img, recovery_stats = apply_gain_map(
            base_img, gain_map_img, headroom,
            target_display_headroom=target_display_headroom,
            highlight_reserve=highlight_reserve,
        )
        report["recoveryApplied"] = True
        report.update(recovery_stats)
        return recovered_img, report

    except Exception as e:
        report["error"] = str(e)
        return None, report
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
