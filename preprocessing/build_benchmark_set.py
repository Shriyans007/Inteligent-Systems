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


def _blank_columns(tile: np.ndarray, from_left: bool) -> int:
    columns = tile.min(axis=0) < 255
    if not from_left:
        columns = columns[::-1]
    return int(np.argmax(columns)) if columns.any() else tile.shape[1]


def _jitter_tiles(tiles, jitter: int, rng) -> list:
    # real handwriting isn't evenly spaced or on a perfect line, so shift each
    # digit up/down a bit and make each gap wider or tighter at random
    shifted = []
    for tile in tiles:
        top = int(rng.integers(0, jitter + 1))
        shifted.append(np.pad(tile, ((top, jitter - top), (0, 0)), constant_values=255))
    tiles = shifted
    for i in range(1, len(tiles)):
        gap = int(rng.integers(-jitter, jitter + 1))
        if gap > 0:
            tiles[i] = np.pad(tiles[i], ((0, 0), (gap, 0)), constant_values=255)
        elif gap < 0:
            # only trim blank columns, from both sides of the gap, so tight
            # gaps can make digits touch without cutting any ink off
            trim_left = min(-gap, _blank_columns(tiles[i], from_left=True))
            tiles[i] = tiles[i][:, trim_left:]
            trim_right = min(-gap - trim_left, _blank_columns(tiles[i - 1], from_left=False))
            if trim_right:
                tiles[i - 1] = tiles[i - 1][:, :-trim_right]
    return tiles


def make_number_image(digit_images, height: int = 56, spacing: int = 8, margin: int = 16,
                      jitter: int = 0, rng=None) -> Image.Image:
    # flip to dark ink on white so it looks more like a real photo
    tiles = [255 - np.asarray(digit, dtype=np.uint8) for digit in digit_images]
    if jitter:
        tiles = _jitter_tiles(tiles, jitter, rng or np.random.default_rng())
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for i, tile in enumerate(tiles):
            Image.fromarray(tile).save(folder / f"digit_{i:02d}.png")
        number = build_number_from_digits(str(folder), height=height, spacing=spacing)
    return ImageOps.expand(number, border=margin, fill=255)


def build_benchmark_set(images, labels, output, counts=None, height: int = 56,
                        spacing: int = 8, margin: int = 16, seed: int = 42, jitter: int = 0) -> Path:
    counts = counts or DEFAULT_COUNTS
    needed = sum(length * amount for length, amount in counts.items())
    if needed > len(images):
        raise ValueError(f"Need {needed} digit images but only {len(images)} were given")

    output = Path(output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # fixed seed so the same set comes out every time, each digit only used once
    order = np.random.default_rng(seed).permutation(len(images))
    # separate generator so jitter doesn't change which digits get picked
    jitter_rng = np.random.default_rng(seed + 1)
    position = 0
    rows = []
    for length in sorted(counts):
        for number_index in range(counts[length]):
            picked = order[position:position + length]
            position += length
            label = "".join(str(int(labels[i])) for i in picked)
            image = make_number_image([images[i] for i in picked], height, spacing, margin, jitter, jitter_rng)
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
    parser.add_argument("--jitter", type=int, default=0,
                        help="Max random up/down shift and extra gap per digit in pixels (0 = neat, the default)")
    args = parser.parse_args()

    from cnn_model.models import _keras

    # test split only, the models were trained on the training split
    (_, _), (images, labels) = _keras().datasets.mnist.load_data()
    manifest = build_benchmark_set(images, labels, args.output, args.counts,
                                   height=args.height, spacing=args.spacing, seed=args.seed, jitter=args.jitter)
    print(f"Saved {sum(args.counts.values())} benchmark images and {manifest}")
    for length in sorted(args.counts):
        print(f"  {length}-digit numbers: {args.counts[length]}")


if __name__ == "__main__":
    main()
