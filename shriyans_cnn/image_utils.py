"""Image conversion shared by the GUI and inference code."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def prepare_digit_image(source: str | Image.Image) -> np.ndarray:
    """Convert a digit image to an MNIST-like, centred 28x28 float tensor.

    The foreground is automatically inverted when the source is dark-on-light.
    The digit is cropped, resized inside a 20x20 box and centred. This reduces
    the train/deployment mismatch for uploaded phone images.
    """
    image = Image.open(source) if isinstance(source, str) else source.copy()
    gray = ImageOps.grayscale(image)
    pixels = np.asarray(gray, dtype=np.uint8)
    if float(pixels.mean()) > 127:
        pixels = 255 - pixels

    mask = pixels > max(20, int(pixels.max()) * 0.15)
    if not mask.any():
        raise ValueError("The selected image does not contain a visible digit.")
    rows, cols = np.where(mask)
    cropped = Image.fromarray(pixels[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1])
    width, height = cropped.size
    scale = min(20 / width, 20 / height)
    resized = cropped.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    return np.asarray(canvas, dtype=np.float32)[None, ..., None] / 255.0

