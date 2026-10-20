"""Build a benchmark of numbers whose digits touch, from MNIST test digits.

Person A's benchmark leaves gaps between digits, so it cannot show whether a
segmenter can separate touching digits. This set can, and it uses the same
manifest format, so cnn_model.benchmark_all runs on it unchanged:

  python -m segmentation_model.build_touching_set
  python -m cnn_model.benchmark_all --manifest data/touching/manifest.csv --output artifacts/benchmark/touching
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps

from .data import ink_crop, make_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("data/touching"))
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from cnn_model.models import _keras
    (_, _), (digits, labels) = _keras().datasets.mnist.load_data()
    # touching=1.0: every neighbouring pair touches or overlaps.
    images, _, _, numbers = make_dataset(digits, labels, args.count, args.seed, touching=1.0,
                                         min_digits=2, max_digits=4)
    folder = args.output / "images"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (image, number) in enumerate(zip(images, numbers)):
        # Dark ink on white, twice the size, with a border: closer to a scanned or drawn image.
        image = ink_crop(image)
        picture = Image.fromarray(255 - image).resize((image.shape[1] * 2, image.shape[0] * 2),
                                                      Image.Resampling.LANCZOS)
        picture = ImageOps.expand(picture, border=16, fill=255)
        name = f"touching_{index:04d}_{number}.png"
        picture.save(folder / name)
        rows.append({"image": f"images/{name}", "label": number})
    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} touching-digit numbers and {manifest}")


if __name__ == "__main__":
    main()
