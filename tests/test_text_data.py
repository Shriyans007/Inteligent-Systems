"""Small EMNIST fixtures test label mapping and image orientation without a download."""

import gzip
import struct

import numpy as np

from text_recognition.emnist import BUILDERS, model_paths, orient, read_idx, read_mapping
from text_recognition.augmentation import augment_batch
from text_recognition.confusions import analyse


def test_four_character_architectures_have_distinct_artifacts(tmp_path):
    paths = [model_paths(key, tmp_path) for key in BUILDERS]
    assert len(paths) == 4 and len({model for _, model, _ in paths}) == 4
    assert model_paths("cnn", tmp_path)[1] == tmp_path / "emnist_cnn.keras"
    assert model_paths("resnet", tmp_path, quick=True)[1] == tmp_path / "resnet" / "quick" / "emnist_resnet.keras"
    assert model_paths("cnn_case_balanced", tmp_path)[1] == tmp_path / "cnn_case_balanced" / "emnist_cnn_case_balanced.keras"
    assert model_paths("cnn_light_aug", tmp_path)[1] != model_paths("cnn", tmp_path)[1]
    assert model_paths("cnn_deep_aug", tmp_path)[1] != model_paths("cnn_light_aug", tmp_path)[1]


def test_emnist_mapping_and_idx(tmp_path):
    mapping = tmp_path / "mapping.txt"
    mapping.write_text("\n".join(f"{i} {ord('0') + i}" for i in range(62)))
    assert len(read_mapping(mapping)) == 62
    path = tmp_path / "images.gz"
    with gzip.open(path, "wb") as file:
        file.write(b"\0\0\x08\x03" + struct.pack(">III", 1, 28, 28) + bytes(28 * 28))
    assert read_idx(path).shape == (1, 28, 28)
    img = np.zeros((1, 28, 28), dtype=np.uint8)
    img[0, 2, 5] = 255
    assert orient(img).shape == (1, 28, 28, 1)
    assert orient(img)[0, 5, 2, 0] == 255  # transpose, without mirroring


def test_augmentation_keeps_shape_and_ink_polarity():
    images = np.zeros((2, 28, 28, 1), dtype=np.float32)
    images[:, 8:20, 12:16, 0] = 1
    changed = augment_batch(images, seed=42)
    assert changed.shape == images.shape
    assert np.array_equal(changed, augment_batch(images, seed=42))
    assert changed.min() >= 0 and changed.max() <= 1
    assert changed[:, 8:20, 12:16, 0].sum() > 30
    assert images[:, 0, 0, 0].sum() == changed[:, 0, 0, 0].sum() == 0


def test_confusion_summary_counts_both_case_directions(tmp_path):
    import csv
    labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    matrix = tmp_path / "confusion_matrix.csv"
    with matrix.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *labels])
        for label in labels:
            values = [0] * 62
            values[labels.index(label)] = 10
            if label == "c": values[labels.index("C")] = 3
            if label == "C": values[labels.index("c")] = 1
            writer.writerow([label, *values])
    analyse(matrix, tmp_path / "diagnosis")
    with (tmp_path / "diagnosis" / "case_pairs.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert next(row for row in rows if row["letter"] == "c")["lower_to_upper"] == "3"
    assert next(row for row in rows if row["letter"] == "c")["upper_to_lower"] == "1"


def test_deeper_character_model_shape():
    import pytest
    pytest.importorskip("tensorflow")
    from text_recognition.emnist_model import build_deep_cnn
    model = build_deep_cnn()
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 62)
