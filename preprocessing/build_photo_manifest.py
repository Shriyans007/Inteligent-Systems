"""
Task 4 - Manifest for real handwritten photos

Makes a manifest.csv (image,label) for a folder of real photos, so they can be
run through cnn_model/evaluate_uploads.py and scored with benchmark_metrics.py
the same way as the MNIST benchmark set.

Name each photo with its correct number first, e.g. 527.jpg or 527_02.jpg.
Anything after the first _ is ignored, so the same number can be photographed
more than once.

Run from the repo root:
  python3 -m preprocessing.build_photo_manifest data/photos
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path

# the same image types the API accepts
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LABEL_PATTERN = re.compile(r"^(\d+)(?:_.*)?$")


def label_from_name(path: Path) -> str | None:
    # kept as a string so leading zeros like 007 aren't lost
    match = LABEL_PATTERN.match(Path(path).stem)
    return match.group(1) if match else None


def build_photo_manifest(folder, output=None):
    """Returns the manifest path, the rows written and the files that were skipped."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Cannot find folder at {folder}")
    output = Path(output) if output else folder / "manifest.csv"

    rows, skipped = [], []
    for path in sorted(p for p in folder.iterdir() if p.is_file()):
        suffix = path.suffix.lower()
        if suffix == ".heic":
            skipped.append((path.name, "HEIC isn't supported, export it as JPG"))
            continue
        if suffix not in IMAGE_TYPES:
            continue
        label = label_from_name(path)
        if label is None:
            skipped.append((path.name, "name doesn't start with the number"))
            continue
        # evaluate_uploads reads image paths relative to the manifest's folder
        relative = Path(os.path.relpath(path, output.parent)).as_posix()
        rows.append({"image": relative, "label": label})

    if not rows:
        raise ValueError(f"No photos named with their number were found in {folder}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return output, rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an image,label manifest for a folder of real handwritten photos.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to save the manifest (default: manifest.csv inside the folder)")
    args = parser.parse_args()

    try:
        manifest, rows, skipped = build_photo_manifest(args.folder, args.output)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}")

    print(f"Saved {len(rows)} photos to {manifest}")
    for length, count in sorted(Counter(len(row["label"]) for row in rows).items()):
        print(f"  {length}-digit numbers: {count}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")


if __name__ == "__main__":
    main()
