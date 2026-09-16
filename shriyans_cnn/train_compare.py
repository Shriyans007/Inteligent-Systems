"""Train and compare a dense baseline and CNN on the same MNIST split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from .models import _keras, build_cnn, build_mlp, compile_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use 10,000/2,000 samples for a fast demo")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keras = _keras()
    random.seed(args.seed)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    try:
        keras.config.enable_op_determinism()
    except AttributeError:
        pass

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    if args.quick:
        x_train, y_train = x_train[:10000], y_train[:10000]
        x_test, y_test = x_test[:2000], y_test[:2000]
    x_train = x_train.astype("float32")[..., None] / 255.0
    x_test = x_test.astype("float32")[..., None] / 255.0
    args.output.mkdir(parents=True, exist_ok=True)

    results = []
    for name, builder in (("MLP baseline", build_mlp), ("CNN selected", build_cnn)):
        model = compile_model(builder())
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.5),
        ]
        started = time.perf_counter()
        history = model.fit(
            x_train, y_train, validation_split=0.1, epochs=args.epochs,
            batch_size=args.batch_size, callbacks=callbacks, verbose=2,
        )
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        elapsed = time.perf_counter() - started
        predictions = model.predict(x_test, verbose=0).argmax(axis=1)
        per_class = {
            str(digit): float((predictions[y_test == digit] == digit).mean())
            for digit in range(10)
        }
        record = {
            "model": name,
            "test_accuracy": float(accuracy), "test_loss": float(loss),
            "training_seconds": round(elapsed, 2), "parameters": int(model.count_params()),
            "epochs_completed": len(history.history["loss"]), "per_class_accuracy": per_class,
        }
        results.append(record)
        model.save(args.output / ("cnn_mnist.keras" if name.startswith("CNN") else "mlp_mnist.keras"))
        (args.output / f"{model.name}_history.json").write_text(
            json.dumps(history.history, indent=2), encoding="utf-8"
        )

    (args.output / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "test_accuracy", "test_loss", "training_seconds", "parameters", "epochs_completed"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fields})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

