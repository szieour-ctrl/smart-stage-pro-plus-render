#!/usr/bin/env python3
# hdrGainMapTest.py — STANDALONE HDR gain-map inspect/decode test CLI
#
# As of July 30, 2026, this is a thin wrapper around hdrRecover.py, the
# shared module now also used by production smartCorrect.py. All the
# actual decode/recovery logic lives there — see that file's docstring
# for the full history of what's been validated and fixed. This file's
# only job is: parse CLI args, call the shared functions, write
# standard.jpg / recovered.jpg, print the JSON report. Output format is
# unchanged from before the refactor, so the existing Railway route
# (/hdr-gainmap-test) and tester page keep working without any changes.

import argparse
import json
import os
import sys

import cv2

from hdrRecover import (
    read_metadata, read_aux_metadata, has_gain_map_version, find_headroom,
    extract_gain_map_image_bytes, decode_standard, apply_gain_map,
)


def main():
    parser = argparse.ArgumentParser(description="Standalone HDR gain-map inspect/decode test — not wired into any production pipeline.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-display-headroom", type=float, default=None, help="Manual override for the tonemap ceiling. Omit to auto-use the photo's own decoded headroom (default, recommended).")
    parser.add_argument("--highlight-reserve", type=float, default=0.30, help="Fraction of output range (0-1) reserved for compressed highlights. Default 0.30, updated July 30, 2026 after every real test photo (portraits and a real backyard/sliding-door shot) needed more than the original 0.15 default to avoid flattening local contrast in bright/boosted areas (shirt wrinkles, sunlit foliage, curtain fabric). Tune by eye if a specific photo still looks off.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {"source": args.source, "gainMapPresent": False, "warnings": []}

    meta = read_metadata(args.source)
    has_version, version_val = has_gain_map_version(meta)
    headroom, headroom_source = find_headroom(meta)

    standard_path = os.path.join(args.output_dir, "standard.jpg")
    try:
        base_img = decode_standard(args.source)
        cv2.imwrite(standard_path, base_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        report["standardDecode"] = {"output": standard_path, "width": base_img.shape[1], "height": base_img.shape[0]}
    except Exception as e:
        report["standardDecode"] = {"error": str(e)}
        print(json.dumps(report))
        sys.exit(1)

    import shutil
    import tempfile
    workdir = tempfile.mkdtemp(prefix="hdrtest-")
    try:
        gain_map_path, tag_used, extract_err = extract_gain_map_image_bytes(args.source, workdir)
        report["gainMapImageExtraction"] = {"tagUsed": tag_used, "error": extract_err}

        if gain_map_path:
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
            base_img, gain_map_img, headroom,
            target_display_headroom=args.target_display_headroom,
            highlight_reserve=args.highlight_reserve,
        )
        recovered_path = os.path.join(args.output_dir, "recovered.jpg")
        cv2.imwrite(recovered_path, recovered_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        report["recoveredDecode"] = {"output": recovered_path, **recovery_stats}

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
