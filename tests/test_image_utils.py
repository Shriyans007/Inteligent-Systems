import numpy as np
from PIL import Image, ImageDraw

from shriyans_cnn.image_utils import prepare_digit_image


def test_prepare_digit_image_has_expected_shape_and_range():
    image = Image.new("L", (80, 100), 255)
    ImageDraw.Draw(image).line((40, 15, 40, 85), fill=0, width=10)
    result = prepare_digit_image(image)
    assert result.shape == (1, 28, 28, 1)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() <= result.max() <= 1.0
    assert result.sum() > 0


def test_blank_image_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="visible digit"):
        prepare_digit_image(Image.new("L", (28, 28), 255))

