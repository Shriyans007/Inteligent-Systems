"""Small EMNIST fixtures test label mapping and image orientation without a download."""

import gzip
import struct

import numpy as np

from text_recognition.emnist import BUILDERS, model_paths, orient, read_idx, read_mapping


def test_four_character_architectures_have_distinct_artifacts(tmp_path):
    paths = [model_paths(key, tmp_path) for key in BUILDERS]
    assert len(paths) == 4 and len({model for _, model, _ in paths}) == 4
    assert model_paths("cnn", tmp_path)[1] == tmp_path / "emnist_cnn.keras"
    assert model_paths("resnet", tmp_path, quick=True)[1] == tmp_path / "resnet" / "quick" / "emnist_resnet.keras"


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
