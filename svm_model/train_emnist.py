"""Train the RBF SVM on EMNIST ByClass (62 letters and digits) for the character route."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.svm import SVC

from text_recognition.common import CHAR_MODEL
from text_recognition.emnist import FILES, check_data, load, model_paths, read_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/emnist/byclass"))
    parser.add_argument("--output", type=Path, default=CHAR_MODEL.parent)
    parser.add_argument("--train-samples", type=int, default=20000,
                        help="RBF SVM training time grows quickly with the number of samples")
    parser.add_argument("--test-samples", type=int, default=20000,
                        help="Random EMNIST test images to score; 0 scores all of them (slow)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample(images: np.ndarray, labels: np.ndarray, size: int, seed: int):
    """Keep a fixed random subset. 0, or a size bigger than the split, keeps everything."""
    if not size or size >= len(labels):
        return images, labels
    keep = np.random.default_rng(seed).choice(len(labels), size=size, replace=False)
    return images[keep], labels[keep]


def flatten(images: np.ndarray) -> np.ndarray:
    """Turn each 28x28 character into 784 values between 0 and 1, the input an SVM expects."""
    return images.reshape(len(images), -1).astype("float32") / 255.0


def save_tables(folder: Path, labels: np.ndarray, predictions: np.ndarray, mapping: dict) -> None:
    """Write the same per-character and confusion tables as the EMNIST networks."""
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(mapping)))
    with (folder / "confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted"] + [mapping[i] for i in range(len(mapping))])
        for i, row in enumerate(matrix):
            writer.writerow([mapping[i], *row])
    with (folder / "per_character.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "character", "samples", "accuracy"])
        for i, row in enumerate(matrix):
            writer.writerow([i, mapping[i], int(row.sum()), float(row[i] / row.sum()) if row.sum() else ""])


def main() -> None:
    args = parse_args()
    check_data(args.data)
    mapping = read_mapping(args.data / FILES["mapping"])

    # load() already turns the raw EMNIST images upright, the same as for the networks.
    train_images, train_labels = load(args.data, "train")
    train_images, train_labels = sample(train_images, train_labels, args.train_samples, args.seed)
    missing = sorted(set(mapping) - set(train_labels.tolist()))
    if missing:
        raise ValueError(f"The training sample has no examples of {[mapping[i] for i in missing]}; "
                         "use a larger --train-samples.")

    # probability=True gives the API a confidence for every character, like the networks.
    model = SVC(kernel="rbf", gamma="scale", probability=True, random_state=args.seed)
    print(f"Training on {len(train_labels)} EMNIST images; this can take several minutes...")
    started = time.perf_counter()
    model.fit(flatten(train_images), train_labels)
    training_seconds = time.perf_counter() - started

    # The test split is opened only after training, as in text_recognition.emnist.
    test_images, test_labels = load(args.data, "test")
    test_images, test_labels = sample(test_images, test_labels, args.test_samples, args.seed)
    x_test = flatten(test_images)
    print(f"Scoring {len(test_labels)} EMNIST test images...")
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    folder, keras_path, mapping_path = model_paths("svm", args.output)
    model_path = keras_path.with_suffix(".joblib")
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    mapping_path.write_text(json.dumps(mapping, indent=2))
    save_tables(folder, test_labels, predictions, mapping)

    result = {
        "model": "SVM (RBF)",
        "model_key": "svm",
        # Accuracy uses predict(), the same decision the API shows for each character.
        "test_accuracy": float((predictions == test_labels).mean()),
        "test_loss": float(log_loss(test_labels, probabilities, labels=model.classes_)),
        "test_samples": len(test_labels),
        "train_samples": len(train_labels),
        "training_seconds": round(training_seconds, 2),
        "seed": args.seed,
        "kernel": "rbf",
        "model_path": str(model_path),
    }
    (folder / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
