"""Turn the cut-point CNN's column probabilities into digit crops for the recognition model."""

from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image

from segmentation.segmenter import CharacterCrop, _binary_for_detection, _to_grayscale_array

from .data import DIGIT_HEIGHT, HEIGHT

MARGIN = 4   # blank columns added either side, like the margin in the training images


def find_cuts(probabilities: np.ndarray, threshold: float = 0.5) -> List[float]:
    """Each run of columns above the threshold is one cut, placed at the run's centre."""
    cuts, start = [], None
    for column, above in enumerate(np.asarray(probabilities) > threshold):
        if above and start is None:
            start = column
        elif not above and start is not None:
            cuts.append((start + column - 1) / 2)
            start = None
    if start is not None:
        cuts.append((start + len(probabilities) - 1) / 2)
    return cuts


def model_input(gray: np.ndarray, ink: np.ndarray):
    """Crop to the ink, make it white on black and 32 pixels tall, as in training.

    Returns (image, ink box (top, bottom, left, right), resized width), or None if blank.
    """
    rows, cols = np.where(ink)
    if not len(rows):
        return None
    top, bottom, left, right = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    bright = 255 - gray if np.median(border) > 127 else gray   # the model expects white ink
    crop = bright[top:bottom, left:right].astype(np.uint8)
    width = max(1, round(crop.shape[1] * DIGIT_HEIGHT / crop.shape[0]))
    resized = np.asarray(Image.fromarray(crop).resize((width, DIGIT_HEIGHT), Image.Resampling.LANCZOS))
    canvas = np.zeros((HEIGHT, width + 2 * MARGIN), dtype=np.uint8)
    offset = (HEIGHT - DIGIT_HEIGHT) // 2
    canvas[offset:offset + DIGIT_HEIGHT, MARGIN:MARGIN + width] = resized
    return canvas, (top, bottom, left, right), width


def segment_learned(image, model, threshold: float = 0.5, padding: int = 2,
                    min_ink: float = 0.02) -> List[CharacterCrop]:
    """Cut a number image where the model predicts digit boundaries.

    Returns the same CharacterCrop objects as segmentation.segment(), so the crops go
    through exactly the same preprocessing and recognition afterwards.
    """
    gray = _to_grayscale_array(image)
    ink = _binary_for_detection(gray) > 0
    prepared = model_input(gray, ink)
    if prepared is None:
        return []
    canvas, (top, bottom, left, right), width = prepared
    batch = canvas[None, :, :, None].astype(np.float32) / 255.0
    probabilities = np.asarray(model.predict(batch, verbose=0))[0, 0, :, 0]

    # Map each cut from the resized image back to columns of the original image.
    scale = (right - left) / width
    cuts = [left + (cut - MARGIN) * scale for cut in find_cuts(probabilities, threshold)]
    edges = [left] + sorted(int(round(c)) for c in cuts if left < c < right) + [right]

    crops = []
    total_ink = ink[top:bottom, left:right].sum()
    for start, end in zip(edges[:-1], edges[1:]):
        part = ink[:, start:end]
        if end <= start or part.sum() < min_ink * total_ink:
            continue   # an empty strip or a sliver of a neighbour, not a digit
        rows, cols = np.where(part)
        y0, y1 = rows.min(), rows.max() + 1
        x0, x1 = start + cols.min(), start + cols.max() + 1
        crop = gray[max(0, y0 - padding):y1 + padding, max(start, x0 - padding):min(end, x1 + padding)]
        crops.append(CharacterCrop(image=crop, bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                                   index=len(crops)))
    return crops
