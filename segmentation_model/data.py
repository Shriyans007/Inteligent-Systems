"""Synthetic number images with known digit boundaries, for training the cut-point CNN.

MNIST digits are cropped to their ink, resized and placed side by side, sometimes
with a gap and sometimes touching or overlapping. Since the code places every
digit itself, it knows exactly where each boundary is, so nothing has to be
labelled by hand.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

HEIGHT = 32         # every image the model sees is 32 pixels tall
WIDTH = 192         # training width; the model itself accepts any width
DIGIT_HEIGHT = 24   # tallest digit, leaving a margin above and below like MNIST
BAND = 1            # columns either side of a boundary that also count as "cut here"


def ink_crop(digit: np.ndarray, level: int = 20) -> np.ndarray:
    """Cut away the empty border around a digit's ink."""
    rows = np.where(digit.max(axis=1) > level)[0]
    cols = np.where(digit.max(axis=0) > level)[0]
    if not len(rows):
        return digit
    return digit[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))


def compose(digits, rng: np.random.Generator, touching: float = 0.4):
    """Place digits left to right on a black canvas, white ink like MNIST.

    Returns (image, cut labels per column, boundary positions), or None if the
    digits do not fit in the training width.
    """
    canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    boundaries, gaps = [], []
    x, right = int(rng.integers(2, 16)), None
    for digit in digits:
        piece = ink_crop(np.asarray(digit, dtype=np.uint8))
        if rng.random() < 0.3:
            # Thicker strokes, closer to a pen drawn on the GUI canvas.
            piece = np.asarray(Image.fromarray(piece).filter(ImageFilter.MaxFilter(3)))
        piece = resize_to_height(piece, int(rng.integers(18, DIGIT_HEIGHT + 1)))
        h, w = piece.shape
        if right is not None:
            # A negative gap makes digits touch or overlap. This is the case the rule-based
            # segmenter can't cut, and basically the main reason to learn where cuts go.
            # Gaps go up to 30 columns. The first model only saw gaps up to 10, so it never
            # learnt to cut the wider gaps in real images and merged neighbouring digits.
            gap = int(rng.integers(-6, 0)) if rng.random() < touching else int(rng.integers(1, 31))
            x = right + gap
            boundaries.append((right + x) / 2)   # middle of the gap, or of the overlap
            if gap > 0:
                gaps.append((right, x))
        if x < 0 or x + w > WIDTH:
            return None
        y = int(np.clip((HEIGHT - h) // 2 + rng.integers(-2, 3), 0, HEIGHT - h))
        region = canvas[y:y + h, x:x + w]
        np.maximum(region, piece, out=region)   # overlapping ink merges instead of overwriting
        right = x + w

    labels = np.zeros(WIDTH, dtype=np.float32)
    for boundary in boundaries:
        centre = int(round(boundary))
        labels[max(0, centre - BAND):centre + BAND + 1] = 1
    for start, end in gaps:
        # Every empty column between two digits is a fine place to cut, however wide the gap is.
        labels[start:end] = 1
    return canvas, labels, boundaries


def make_dataset(images, digit_labels, count: int, seed: int, touching: float = 0.4,
                 min_digits: int = 1, max_digits: int = 5):
    """Build `count` number images; returns images, cut labels, boundaries and the true numbers."""
    rng = np.random.default_rng(seed)
    x = np.zeros((count, HEIGHT, WIDTH), dtype=np.uint8)
    y = np.zeros((count, WIDTH), dtype=np.float32)
    boundaries, numbers = [], []
    while len(numbers) < count:
        picked = rng.integers(0, len(images), int(rng.integers(min_digits, max_digits + 1)))
        made = compose([images[i] for i in picked], rng, touching)
        if made is None:
            continue
        i = len(numbers)
        x[i], y[i], cuts = made
        boundaries.append(cuts)
        numbers.append("".join(str(int(digit_labels[j])) for j in picked))
    return x, y, boundaries, numbers
