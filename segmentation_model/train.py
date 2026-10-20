"""Train the cut-point CNN on synthetic number images built from MNIST digits.

Run from the repo root:
  python -m segmentation_model.train
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cnn_model.models import _keras

from .data import make_dataset
from .model import build_cutpoint_model, compile_cutpoint_model
from .segment import find_cuts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=int, default=15000, help="Synthetic training images")
    parser.add_argument("--val", type=int, default=2000, help="Synthetic validation images")
    parser.add_argument("--test", type=int, default=2000, help="Synthetic test images per test set")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/segmenter"))
    return parser.parse_args()


def batches(keras, images, labels, batch_size: int = 64):
    """Feed uint8 images as 0-1 floats, one batch at a time, to keep memory low."""

    class Batches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()

        def __len__(self):
            return (len(images) + batch_size - 1) // batch_size

        def __getitem__(self, index):
            part = slice(index * batch_size, (index + 1) * batch_size)
            return (images[part, :, :, None].astype(np.float32) / 255.0,
                    labels[part, None, :, None])   # model output shape: (1, width, 1)

    return Batches()


def score(model, images, boundaries, tolerance: float = 3.0) -> dict:
    """How often the predicted cuts match the true digit boundaries."""
    probabilities = model.predict(images[..., None].astype(np.float32) / 255.0, verbose=0)[:, 0, :, 0]
    right_count = matched = predicted = actual = 0
    for row, truth in zip(probabilities, boundaries):
        cuts = find_cuts(row)
        right_count += len(cuts) == len(truth)
        unused = list(truth)
        for cut in cuts:
            near = [b for b in unused if abs(b - cut) <= tolerance]
            if near:
                unused.remove(near[0])
                matched += 1
        predicted += len(cuts)
        actual += len(truth)
    return {"right_number_of_pieces": right_count / len(boundaries),
            "cut_precision": matched / predicted if predicted else 1.0,
            "cut_recall": matched / actual if actual else 1.0,
            "images": len(boundaries)}


def main() -> None:
    args = parse_args()
    keras = _keras()
    keras.utils.set_random_seed(args.seed)
    (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()
    # Training and validation numbers use different MNIST training digits; the test
    # numbers use MNIST test digits, which the model never sees during training.
    cut = int(len(train_digits) * 0.9)
    x_train, y_train, _, _ = make_dataset(train_digits[:cut], train_labels[:cut], args.train, args.seed)
    x_val, y_val, _, _ = make_dataset(train_digits[cut:], train_labels[cut:], args.val, args.seed + 1)

    model = compile_cutpoint_model(build_cutpoint_model())
    started = time.perf_counter()
    history = model.fit(batches(keras, x_train, y_train), validation_data=batches(keras, x_val, y_val),
                        epochs=args.epochs, verbose=2,
                        callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    seconds = time.perf_counter() - started

    mixed = make_dataset(test_digits, test_labels, args.test, args.seed + 2, touching=0.4, min_digits=2)
    touching = make_dataset(test_digits, test_labels, args.test, args.seed + 3, touching=1.0, min_digits=2)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "cutpoint_cnn.keras"
    model.save(model_path)
    result = {
        "model": "Cut-point CNN",
        "training_seconds": round(seconds, 2),
        "epochs_completed": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "parameters": int(model.count_params()),
        "train_images": args.train,
        "seed": args.seed,
        "test_mixed_gaps_and_touching": score(model, mixed[0], mixed[2]),
        "test_all_touching": score(model, touching[0], touching[2]),
        "model_path": str(model_path),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
