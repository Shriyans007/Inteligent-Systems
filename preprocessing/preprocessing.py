"""Task 1 - image preprocessing."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str) -> Image.Image:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find image at {path}")
    return Image.open(path)


def grayscale_image(image: Image.Image) -> Image.Image:
    return image.convert("L")


def resize_image(image: Image.Image, size: tuple = (28, 28)) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)


def binarize_image(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Apply a fixed threshold: bright pixels become white and dark pixels black."""
    array = np.array(image)
    binary = np.where(array >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def binarize_image_otsu(image: Image.Image) -> Image.Image:
    """Apply Otsu thresholding, which selects a threshold for each image."""
    from skimage.filters import threshold_otsu

    array = np.array(image)
    threshold = threshold_otsu(array)
    binary = np.where(array >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def to_normalized_array(image: Image.Image) -> np.ndarray:
    """Convert pixel values from 0-255 integers to 0-1 float values."""
    return np.array(image).astype(np.float32) / 255.0


def preprocess_pipeline(
    image: Image.Image,
    size: tuple = (28, 28),
    binarize: bool = False,
    method: str = "fixed",
    threshold: int = 128,
) -> np.ndarray:
    """Run grayscale, resize, optional binarisation and normalisation."""
    processed = grayscale_image(image)
    processed = resize_image(processed, size)
    if binarize:
        if method == "otsu":
            processed = binarize_image_otsu(processed)
        else:
            processed = binarize_image(processed, threshold)
    return to_normalized_array(processed)


def prepare_mnist_digit(source: str | Image.Image) -> np.ndarray:
    """Prepare an uploaded or segmented digit for the trained CNN.

    This deployment wrapper first crops and centres the visible digit. It then
    calls the team's selected preprocessing pipeline using grayscale values,
    because the preprocessing experiment found this worked better than the
    fixed and Otsu thresholding options.
    """
    image = load_image(source) if isinstance(source, str) else source.copy()
    grayscale = grayscale_image(image)
    pixels = np.asarray(grayscale, dtype=np.uint8)

    # MNIST has a bright digit on a dark background. Most uploaded drawings
    # have the opposite format, so invert them when the background is bright.
    if float(pixels.mean()) > 127:
        pixels = 255 - pixels

    mask = pixels > max(20, int(pixels.max()) * 0.15)
    if not mask.any():
        raise ValueError("The selected image does not contain a visible digit.")

    rows, columns = np.where(mask)
    cropped = Image.fromarray(
        pixels[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    )
    width, height = cropped.size
    scale = min(20 / width, 20 / height)
    resized = cropped.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )

    centred = Image.new("L", (28, 28), 0)
    centred.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    processed = preprocess_pipeline(centred, size=(28, 28), binarize=False)
    return processed[None, ..., None]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <path_to_image>")
        sys.exit(1)

    original = load_image(sys.argv[1])
    result = preprocess_pipeline(original, size=(28, 28), binarize=False)

    print(f"Loaded image: {sys.argv[1]}")
    print(f"Original size: {original.size}, mode: {original.mode}")
    print(f"Processed array shape: {result.shape}, dtype: {result.dtype}")
    print(f"Value range: [{result.min():.3f}, {result.max():.3f}]")
