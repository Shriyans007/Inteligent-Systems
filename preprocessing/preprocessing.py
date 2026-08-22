"""
Task 1 - Image Preprocessing

Functions to prepare an input image (or a single digit sub-image) for the
ML model: resizing, grayscaling, and optional binarization.
"""

from pathlib import Path
import numpy as np
from PIL import Image


def load_image(path: str) -> Image.Image:
    """Load an image from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find image at {path}")
    return Image.open(path)


def grayscale_image(image: Image.Image) -> Image.Image:
    """Convert an image to single-channel grayscale."""
    return image.convert("L")


def resize_image(image: Image.Image, size: tuple = (28, 28)) -> Image.Image:
    """Resize an image to match the model's expected input size (MNIST = 28x28)."""
    return image.resize(size, Image.LANCZOS)


def binarize_image(image: Image.Image, threshold: int = 128) -> Image.Image:
    """
    Convert a grayscale image to pure black/white using a fixed threshold.
    Pixels >= threshold become white (255), below become black (0).
    """
    array = np.array(image)
    binary = np.where(array >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def to_normalized_array(image: Image.Image) -> np.ndarray:
    """Convert a grayscale image to a numpy array of floats in [0, 1], ready for a model."""
    array = np.array(image).astype(np.float32) / 255.0
    return array


def preprocess_pipeline(
    image: Image.Image,
    size: tuple = (28, 28),
    binarize: bool = False,
    threshold: int = 128,
) -> np.ndarray:
    """
    Full preprocessing pipeline: grayscale -> resize -> (optional binarize) -> normalize.
    Returns a numpy array ready to feed into a model.
    """
    img = grayscale_image(image)
    img = resize_image(img, size)
    if binarize:
        img = binarize_image(img, threshold)
    return to_normalized_array(img)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <path_to_image>")
        sys.exit(1)

    img = load_image(sys.argv[1])
    processed = preprocess_pipeline(img, size=(28, 28), binarize=False)
    print(f"Loaded image: {sys.argv[1]}")
    print(f"Original size: {img.size}, mode: {img.mode}")
    print(f"Processed array shape: {processed.shape}, dtype: {processed.dtype}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
