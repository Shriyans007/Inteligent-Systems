"""Create repeatable evaluation evidence for a saved Keras classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .models import _keras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/cnn_mnist.keras"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()
    keras = _keras()
    (_, _), (images, labels) = keras.datasets.mnist.load_data()
    images = images.astype("float32")[..., None] / 255.0
    probabilities = keras.models.load_model(args.model).predict(images, verbose=0)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(labels, predicted, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predicted)
    errors = np.where(predicted != labels)[0]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(args.output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    np.savetxt(
        args.output / "misclassified_samples.csv",
        np.column_stack([errors, labels[errors], predicted[errors], probabilities[errors].max(axis=1)]),
        delimiter=",", header="test_index,true_label,predicted_label,confidence", comments="",
    )
    print(f"Accuracy: {report['accuracy']:.4f}; errors: {len(errors)}")


if __name__ == "__main__":
    main()

