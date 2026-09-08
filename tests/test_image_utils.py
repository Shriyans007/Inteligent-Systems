import numpy as np
from PIL import Image, ImageDraw

from preprocessing import prepare_mnist_digit, preprocess_pipeline


def test_prepare_digit_image_has_expected_shape_and_range():
    image = Image.new("L", (80, 100), 255)
    ImageDraw.Draw(image).line((40, 15, 40, 85), fill=0, width=10)
    result = prepare_mnist_digit(image)
    assert result.shape == (1, 28, 28, 1)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() <= result.max() <= 1.0
    assert result.sum() > 0


def test_blank_image_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="visible digit"):
        prepare_mnist_digit(Image.new("L", (28, 28), 255))


def test_core_pipeline_returns_model_ready_values():
    image = Image.new("RGB", (40, 60), "white")
    result = preprocess_pipeline(image, size=(28, 28), binarize=False)
    assert result.shape == (28, 28)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() <= result.max() <= 1.0
