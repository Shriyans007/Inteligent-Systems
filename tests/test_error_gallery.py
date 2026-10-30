"""Tests for error_gallery.py, using small blank images as the "uploads"."""

import pytest
from PIL import Image

from preprocessing.error_gallery import CELL_HEIGHT, CELL_WIDTH, make_error_gallery, wrong_cases


def _cases(tmp_path, pairs):
    cases = []
    for i, (expected, predicted) in enumerate(pairs):
        path = tmp_path / f"{i}_{expected}.png"
        Image.new("L", (30 * len(expected), 28), color=255).save(path)
        cases.append({"image": str(path), "expected": expected, "predicted": predicted})
    return cases


def test_wrong_cases_skips_right_answers_and_labels_the_cause(tmp_path):
    cases = _cases(tmp_path, [("5", "5"), ("42", "47"), ("308", "38")])
    assert [c["error_type"] for c in wrong_cases(cases)] == ["classifier", "segmentation"]
    assert [c["expected"] for c in wrong_cases(cases, only="segmentation")] == ["308"]


def test_gallery_has_one_cell_per_wrong_prediction(tmp_path):
    cases = _cases(tmp_path, [("1", "7"), ("22", "27"), ("333", "33"), ("4", "4"), ("55", "5")])
    path = make_error_gallery(cases, tmp_path / "out" / "errors.png", columns=2)
    # 4 wrong answers in 2 columns is 2 rows
    assert Image.open(path).size == (2 * CELL_WIDTH, 2 * CELL_HEIGHT)


def test_limit_caps_how_many_are_drawn(tmp_path):
    cases = _cases(tmp_path, [(str(d), "0") for d in range(1, 9)])
    path = make_error_gallery(cases, tmp_path / "errors.png", limit=3, columns=4)
    assert Image.open(path).size == (4 * CELL_WIDTH, CELL_HEIGHT)


def test_missing_image_still_gets_a_cell(tmp_path):
    cases = [{"image": str(tmp_path / "gone.png"), "expected": "9", "predicted": "8"}]
    path = make_error_gallery(cases, tmp_path / "errors.png")
    assert path.exists()


def test_raises_when_nothing_is_wrong(tmp_path):
    with pytest.raises(ValueError):
        make_error_gallery(_cases(tmp_path, [("5", "5")]), tmp_path / "errors.png")
