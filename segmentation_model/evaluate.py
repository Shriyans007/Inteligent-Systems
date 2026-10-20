"""Measure the trained cut-point CNN by the pieces it actually produces.

Counting raw cut columns is misleading: a cut in an empty margin, or two cuts in the
same wide gap, change nothing because empty strips are thrown away. So each test
image goes through segment_learned(), exactly as in the API, and the number of
pieces it returns is compared with the number of digits.

Run from the repo root (after python -m segmentation_model.train):
  python -m segmentation_model.evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from cnn_model.models import _keras

from .data import make_dataset
from .segment import segment_learned


def test_sets(seed: int, count: int) -> dict:
    """Numbers built from MNIST test digits, which the model never sees in training."""
    (_, _), (digits, labels) = _keras().datasets.mnist.load_data()
    return {
        "mixed_gaps_and_touching": make_dataset(digits, labels, count, seed + 2, touching=0.4, min_digits=2),
        "all_touching": make_dataset(digits, labels, count, seed + 3, touching=1.0, min_digits=2),
    }


def piece_counts(model, images, numbers) -> dict:
    """Share of numbers cut into the right number of pieces, too many, or too few."""
    right = too_many = too_few = 0
    for image, number in zip(images, numbers):
        # Dark ink on white, like an uploaded or drawn image.
        pieces = len(segment_learned(Image.fromarray(255 - image), model))
        right += pieces == len(number)
        too_many += pieces > len(number)
        too_few += pieces < len(number)
    total = len(numbers)
    return {"right_number_of_pieces": right / total, "too_many_pieces": too_many / total,
            "too_few_pieces": too_few / total, "images": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/segmenter/cutpoint_cnn.keras"))
    parser.add_argument("--count", type=int, default=2000, help="Test numbers per test set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = _keras().models.load_model(args.model, compile=False)
    results = {f"test_{name}": piece_counts(model, data[0], data[3])
               for name, data in test_sets(args.seed, args.count).items()}
    result_path = args.model.parent / "result.json"
    record = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    record.update(results)
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
