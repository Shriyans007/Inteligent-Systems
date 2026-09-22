"""
Tests for the segmentation module, verified against:
  - preprocessing.image_acquisition.build_number_from_digits() (real tile-composite format)
  - cnn_model.image_utils.prepare_digit_image() (real CNN input prep)

These are integration-style tests deliberately built against the team's real
functions rather than mocks, since the tile-composite pixel format has real
quirks (see segmenter.py docstring) that a synthetic mock wouldn't catch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from PIL import Image

from segmentation import segment, crops_to_model_input
from preprocessing.image_acquisition import build_number_from_digits


def _make_digit_tile(digit: str) -> np.ndarray:
    """MNIST-style tile: black background, white digit stroke."""
    canvas = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(canvas, digit, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return canvas


def _build_composite(tmp_path, digits: str):
    folder = tmp_path / "digits"
    folder.mkdir()
    for i, d in enumerate(digits):
        Image.fromarray(_make_digit_tile(d)).save(folder / f"digit_{i}_label{d}.png")
    return build_number_from_digits(str(folder), height=28, spacing=4)


def test_finds_correct_number_of_tiles(tmp_path):
    composite = _build_composite(tmp_path, "504")
    crops = segment(composite, method="connected_components")
    assert len(crops) == 3


def test_tiles_are_left_to_right_ordered(tmp_path):
    composite = _build_composite(tmp_path, "19")
    crops = segment(composite)
    xs = [c.bbox[0] for c in crops]
    assert xs == sorted(xs)


def test_gaps_between_tiles_are_not_treated_as_characters(tmp_path):
    # regression test for the inversion-heuristic bug: white gap columns
    # between tiles must NOT be picked up as their own "characters"
    composite = _build_composite(tmp_path, "123")
    crops = segment(composite)
    assert len(crops) == 3, (
        f"expected 3 digit crops, got {len(crops)} -- gaps may be "
        f"leaking through as extra regions"
    )


def test_contours_method_agrees_with_connected_components(tmp_path):
    composite = _build_composite(tmp_path, "77")
    cc_crops = segment(composite, method="connected_components")
    contour_crops = segment(composite, method="contours")
    assert len(cc_crops) == len(contour_crops) == 2


def test_crops_convert_to_valid_cnn_input(tmp_path):
    composite = _build_composite(tmp_path, "42")
    crops = segment(composite)
    tensors = crops_to_model_input(crops)
    assert len(tensors) == 2
    for t in tensors:
        assert t.shape == (1, 28, 28, 1)
        assert t.dtype == np.float32
        assert 0.0 <= t.min() and t.max() <= 1.0


def test_empty_canvas_returns_no_crops():
    blank = np.full((28, 100), 255, dtype=np.uint8)
    crops = segment(blank)
    assert crops == []


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for fn in [
        test_finds_correct_number_of_tiles,
        test_tiles_are_left_to_right_ordered,
        test_gaps_between_tiles_are_not_treated_as_characters,
        test_contours_method_agrees_with_connected_components,
        test_crops_convert_to_valid_cnn_input,
    ]:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    test_empty_canvas_returns_no_crops()
    print("All tests passed.")
