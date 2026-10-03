"""
Tests for the segmentation module, verified against:
  - preprocessing.image_acquisition.build_number_from_digits() (real tile-composite format)
  - preprocessing.prepare_mnist_digit() (real CNN input prep)

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
from PIL import ImageOps

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


def test_touching_digit_group_splits_at_narrow_bridge():
    # A thin bridge joins two otherwise separate digit-shaped marks.
    canvas = np.full((90, 130), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (15, 15), (44, 65), 0, 5)
    cv2.rectangle(canvas, (55, 15), (84, 65), 0, 5)
    cv2.line(canvas, (43, 40), (56, 40), 0, 2)
    assert len(segment(canvas)) == 1  # original opt-out behaviour
    crops = segment(canvas, split_touching=True)
    assert len(crops) == 2
    assert crops[0].bbox[0] < crops[1].bbox[0]
    assert crops[0].image.max() == crops[1].image.max() == 255


def test_single_wide_digit_is_not_split():
    canvas = np.full((90, 130), 255, dtype=np.uint8)
    cv2.putText(canvas, '7', (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 2.1, 0, 5)
    assert len(segment(canvas, split_touching=True)) == 1


def test_windows_opencv_threshold_error_keeps_multiple_strokes(monkeypatch):
    canvas = np.full((80, 200), 255, dtype=np.uint8)
    cv2.line(canvas, (24, 12), (24, 65), 0, 6)
    cv2.line(canvas, (140, 12), (140, 65), 0, 6)
    expected = [crop.bbox for crop in segment(canvas)]
    assert len(expected) == 2

    def unavailable(*args, **kwargs):
        raise cv2.error('Unknown C++ exception from OpenCV code')

    monkeypatch.setattr(cv2, 'threshold', unavailable)
    assert [crop.bbox for crop in segment(canvas)] == expected


def test_windows_opencv_connected_components_error_keeps_multiple_strokes(monkeypatch):
    canvas = np.full((80, 200), 255, dtype=np.uint8)
    cv2.line(canvas, (24, 12), (24, 65), 0, 6)
    cv2.line(canvas, (140, 12), (140, 65), 0, 6)
    expected = [crop.bbox for crop in segment(canvas)]
    assert len(expected) == 2

    def unavailable(*args, **kwargs):
        raise cv2.error('Unknown C++ exception from OpenCV code')

    monkeypatch.setattr(cv2, 'connectedComponentsWithStats', unavailable)
    assert [crop.bbox for crop in segment(canvas)] == expected


def test_real_nine_keeps_the_whole_digit_on_both_backgrounds():
    """The black hole within 9 must not be mistaken for the digit itself."""
    from pathlib import Path
    from preprocessing import prepare_mnist_digit

    source = Path(__file__).resolve().parents[2] / "preprocessing/sample_digits/digit_4_label9.png"
    dark = Image.open(source).convert("L")
    for image in (dark, ImageOps.invert(dark)):
        crops = segment(image)
        assert len(crops) == 1
        assert crops[0].bbox[2] >= 12 and crops[0].bbox[3] >= 18
        np.testing.assert_allclose(crops_to_model_input(crops)[0], prepare_mnist_digit(image))


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
