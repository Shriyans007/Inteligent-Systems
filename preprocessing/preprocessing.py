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
    return image.resize(size, Image.LANCZOS)


def binarize_image(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Fixed-threshold binarization: >= threshold becomes white, else black."""
    array = np.array(image)
    binary = np.where(array >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def binarize_image_otsu(image: Image.Image) -> Image.Image:
    """Binarize using Otsu's method - picks the threshold per image instead of a fixed number."""
    from skimage.filters import threshold_otsu

    array = np.array(image)
    thresh = threshold_otsu(array)
    binary = np.where(array >= thresh, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def to_normalized_array(image: Image.Image) -> np.ndarray:
    return np.array(image).astype(np.float32) / 255.0


def preprocess_pipeline(
    image: Image.Image,
    size: tuple = (28, 28),
    binarize: bool = False,
    method: str = "fixed",
    threshold: int = 128,
) -> np.ndarray:
    """grayscale -> resize -> optional binarize (fixed/otsu) -> normalize."""
    img = grayscale_image(image)
    img = resize_image(img, size)
    if binarize:
        if method == "otsu":
            img = binarize_image_otsu(img)
        else:
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