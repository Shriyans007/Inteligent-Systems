"""Basic tests for preprocess_pipeline() - checks the grayscale-only,
fixed-threshold and Otsu paths all still work as the preprocessing code
keeps changing."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from preprocessing import preprocess_pipeline


def _sample_digit_image():
    image = Image.new("L", (80, 100), 255)
    ImageDraw.Draw(image).line((40, 15, 40, 85), fill=0, width=10)
    return image


def test_pipeline_grayscale_only_shape_and_range():
    result = preprocess_pipeline(_sample_digit_image(), size=(28, 28), binarize=False)
    assert result.shape == (28, 28)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() <= result.max() <= 1.0
    assert len(np.unique(result)) > 2


def test_pipeline_fixed_threshold_binarize_only_has_0_and_1():
    result = preprocess_pipeline(
        _sample_digit_image(), size=(28, 28), binarize=True, method="fixed", threshold=128
    )
    assert result.shape == (28, 28)
    assert set(np.unique(result).tolist()).issubset({0.0, 1.0})


def test_pipeline_otsu_binarize_only_has_0_and_1():
    result = preprocess_pipeline(_sample_digit_image(), size=(28, 28), binarize=True, method="otsu")
    assert result.shape == (28, 28)
    assert set(np.unique(result).tolist()).issubset({0.0, 1.0})


def test_pipeline_respects_custom_size():
    result = preprocess_pipeline(_sample_digit_image(), size=(14, 14), binarize=False)
    assert result.shape == (14, 14)


def test_pipeline_default_binarize_is_false():
    default_result = preprocess_pipeline(_sample_digit_image(), size=(28, 28))
    explicit_grayscale = preprocess_pipeline(_sample_digit_image(), size=(28, 28), binarize=False)
    assert np.array_equal(default_result, explicit_grayscale)