"""Tests for report_figures.py, using small drawn digits in a temp folder."""

import cv2
import numpy as np
import pytest
from PIL import Image

from preprocessing.report_figures import acquisition_figure, preprocessing_steps_figure


def _digit_folder(tmp_path, digits="5041"):
    folder = tmp_path / "digits"
    folder.mkdir()
    for i, d in enumerate(digits):
        tile = np.zeros((28, 28), dtype=np.uint8)
        cv2.putText(tile, d, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        Image.fromarray(tile).save(folder / f"digit_{i}_label{d}.png")
    return folder


def test_steps_figure_has_a_row_per_digit_and_a_column_per_method(tmp_path):
    folder = _digit_folder(tmp_path)
    path = preprocessing_steps_figure(folder, tmp_path / "out" / "steps.png", count=3, scale=4)
    width, height = Image.open(path).size
    # 3 methods across and 3 digits down, each 28*4 pixels plus gaps and a header
    assert width == 3 * (28 * 4 + 10) + 10
    assert height == 30 + 3 * (28 * 4 + 10) + 10


def test_thresholded_columns_only_have_black_and_white(tmp_path):
    folder = _digit_folder(tmp_path, "8")
    path = preprocessing_steps_figure(folder, tmp_path / "steps.png", count=1, scale=1)
    sheet = np.array(Image.open(path))
    fixed = sheet[40:68, 48:76]
    assert set(np.unique(fixed)) <= {0, 255}


def test_acquisition_figure_is_wide_enough_for_the_joined_number(tmp_path):
    folder = _digit_folder(tmp_path)
    path = acquisition_figure(folder, tmp_path / "acq.png", count=4, height=40, spacing=6)
    # the joined number is 4 * 40 + 3 * 6 = 178 wide, plus padding on each side
    assert Image.open(path).width >= 178 + 2 * 12


def test_errors_for_a_folder_with_no_digits(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        preprocessing_steps_figure(empty, tmp_path / "steps.png")
