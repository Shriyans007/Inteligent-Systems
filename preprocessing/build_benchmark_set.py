"""
Task 4 - Benchmark test image set

Builds labelled single and multi-digit number images from the MNIST test split,
plus a manifest.csv that cnn_model/evaluate_uploads.py can read.

Run from the repo root:
  python3 -m preprocessing.build_benchmark_set
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from preprocessing.image_acquisition import build_number_from_digits

DEFAULT_COUNTS = {1: 100, 2: 100, 3: 100, 4: 100}


def make_number_image(digit_images, height: int = 56, spacing: int = 8, margin: int = 16) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for i, digit in enumerate(digit_images):
            # flip to dark ink on white so it looks more like a real photo
            inverted = 255 - np.asarray(digit, dtype=np.uint8)
            Image.fromarray(inverted).save(folder / f"digit_{i:02d}.png")
        number = build_number_from_digits(str(folder), height=height, spacing=spacing)
    return ImageOps.expand(number, border=margin, fill=255)


def build_benchmark_set(images, labels, output, counts=None, height: int = 56,
                        spacing: int = 8, margin: int = 16, seed: int = 42) -> Path:
    counts = counts or DEFAULT_COUNTS
    needed = sum(length * amount for length, amount in counts.items())
    if needed > len(images):
        raise ValueError(f"Need {needed} digit images but only {len(images)} were given")

    output = Path(output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # fixed seed so the same set comes out every time, each digit only used once
    order = np.random.default_rng(seed).permutation(len(images))
    position = 0
    rows = []
    for length in sorted(counts):
        for number_index in range(counts[length]):
            picked = order[position:position + length]
            position += length
            label = "".join(str(int(labels[i])) for i in picked)
            image = make_number_image([images[i] for i in picked], height, spacing, margin)
            name = f"len{length}_{number_index:04d}_{label}.png"
            image.save(image_dir / name)
            rows.append({"image": f"images/{name}", "label": label})

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _parse_counts(text: str) -> dict:
    counts = {}
    for part in text.split(","):
        length, amount = part.split(":")
        counts[int(length)] = int(amount)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the labelled benchmark image set from MNIST test digits.")
    parser.add_argument("--output", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--counts", type=_parse_counts, default=DEFAULT_COUNTS,
                        help="Images per number length, e.g. 1:100,2:100,3:100,4:100")
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--spacing", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from cnn_model.models import _keras

    # test split only, the models were trained on the training split
    (_, _), (images, labels) = _keras().datasets.mnist.load_data()
    manifest = build_benchmark_set(images, labels, args.output, args.counts,
                                   height=args.height, spacing=args.spacing, seed=args.seed)
    print(f"Saved {sum(args.counts.values())} benchmark images and {manifest}")
    for length in sorted(args.counts):
        print(f"  {length}-digit numbers: {args.counts[length]}")


if __name__ == "__main__":
    main()
