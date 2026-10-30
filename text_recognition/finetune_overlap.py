"""Fine-tune the character classifier so it can read overlapped and shared-stroke letters.

The classifier was trained on clean, isolated EMNIST characters. Crops from
overlapping handwriting (a "t" whose crossbar is shared with the next "t", an
"a" written over an "e") contain neighbour ink it has never seen. This script
continues training the existing model on a 50/50 mix of clean EMNIST and
synthetic overlap crops, then reports accuracy on both so you can confirm
clean accuracy has not dropped.

    python -m text_recognition.finetune_overlap --data data/emnist/byclass

The original model file is left untouched; the result is saved next to it as
emnist_cnn_overlap.keras.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from segmentation.overlap_crops import overlap_crops
from segmentation.synthetic_lines import EmnistSource, StubSource
from segmentation.unet_resnet import _keras
from text_recognition.common import CHAR_MODEL

CHAR_MODEL_OVERLAP = CHAR_MODEL.with_name("emnist_cnn_overlap.keras")


def accuracy(model, x, y):
    return float((model.predict(x, batch_size=256, verbose=0).argmax(axis=1) == y).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/emnist/byclass"))
    parser.add_argument("--base", type=Path, default=CHAR_MODEL, help="trained classifier to start from")
    parser.add_argument("--output", type=Path, default=CHAR_MODEL_OVERLAP)
    parser.add_argument("--lines", type=int, default=6000, help="synthetic training lines (about 11 crops each)")
    parser.add_argument("--clean", type=int, default=70000, help="clean EMNIST images mixed in")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--stub", action="store_true", help="smoke test with OpenCV letters (no EMNIST needed)")
    args = parser.parse_args()
    keras = _keras()

    if args.stub:
        train_source = test_source = StubSource()
        mapping = {i: c for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")}
        clean_x = clean_y = clean_test_x = clean_test_y = None
    else:
        from text_recognition.emnist import FILES, check_data, load, read_mapping
        check_data(args.data)
        mapping = read_mapping(args.data / FILES["mapping"])
        train_images, train_labels = load(args.data, "train")
        test_images, test_labels = load(args.data, "test")
        train_source = EmnistSource(train_images, train_labels, mapping)
        test_source = EmnistSource(test_images, test_labels, mapping)
        rng = np.random.default_rng(0)
        pick = rng.permutation(len(train_labels))[:args.clean]
        clean_x, clean_y = train_images[pick].astype(np.float32) / 255, train_labels[pick]
        pick = rng.permutation(len(test_labels))[:10000]
        clean_test_x, clean_test_y = test_images[pick].astype(np.float32) / 255, test_labels[pick]
    char_to_label = {c: i for i, c in mapping.items()}

    print("Building synthetic overlap crops ...")
    over_x, over_y = overlap_crops(train_source, char_to_label, args.lines, seed=1)
    test_over_x, test_over_y = overlap_crops(test_source, char_to_label, max(100, args.lines // 6), seed=2)
    if clean_x is None:
        x, y = over_x, over_y
    else:
        x, y = np.concatenate([clean_x, over_x]), np.concatenate([clean_y, over_y])
    order = np.random.default_rng(3).permutation(len(y))
    x, y = x[order], y[order]
    print(f"{len(over_y)} overlap crops + {0 if clean_y is None else len(clean_y)} clean images")

    model = keras.models.load_model(args.base)
    if model.optimizer is None:           # saved without training state
        model.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:                                 # keep the team's own loss; just lower the learning rate
        model.optimizer.learning_rate = args.learning_rate
    result = {"overlap_accuracy_before": accuracy(model, test_over_x, test_over_y)}
    if clean_test_x is not None:
        result["clean_accuracy_before"] = accuracy(model, clean_test_x, clean_test_y)
    print(result)

    model.fit(x, y, batch_size=128, epochs=args.epochs, validation_split=0.05,
              callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

    result["overlap_accuracy_after"] = accuracy(model, test_over_x, test_over_y)
    if clean_test_x is not None:
        result["clean_accuracy_after"] = accuracy(model, clean_test_x, clean_test_y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    (args.output.parent / "overlap_finetune_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
