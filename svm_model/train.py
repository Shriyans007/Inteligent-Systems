"""Train the RBF SVM baseline (Person D, Model B) and save it for the API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=10000,
                        help="RBF SVM training time grows quickly with the number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/svm"))
    return parser.parse_args()


def flatten(images: np.ndarray) -> np.ndarray:
    """Turn each 28x28 image into 784 values between 0 and 1, the input an SVM expects."""
    return images.reshape(len(images), -1).astype("float32") / 255.0


def load_mnist(train_samples: int, seed: int):
    from cnn_model.models import _keras
    (x_train, y_train), (x_test, y_test) = _keras().datasets.mnist.load_data()
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(x_train), size=min(train_samples, len(x_train)), replace=False)
    return flatten(x_train[keep]), y_train[keep], flatten(x_test), y_test


def main() -> None:
    args = parse_args()
    x_train, y_train, x_test, y_test = load_mnist(args.train_samples, args.seed)

    # probability=True lets the API return a full 10-class vector, like the Keras models do.
    model = SVC(kernel="rbf", gamma="scale", probability=True, random_state=args.seed)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started

    predictions = model.predict(x_test)
    record = {
        "model": "SVM (RBF)",
        "model_key": "svm",
        "test_accuracy": float((predictions == y_test).mean()),
        "per_class_accuracy": {str(d): float((predictions[y_test == d] == d).mean()) for d in range(10)},
        "training_seconds": round(training_seconds, 2),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "seed": args.seed,
        "kernel": "rbf",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "svm_mnist.joblib"
    joblib.dump(model, model_path)
    record["model_path"] = str(model_path)
    (args.output / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
