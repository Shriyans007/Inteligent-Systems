"""Find individual characters in a line of handwriting with the ResNet U-Net.

    line image -> normalise scale -> U-Net -> centre peaks -> one box per character

Overlapping characters get separate boxes because each has its own peak. The
boxes may overlap each other, and crops include a little of the shared ink.
The optional overlap fine-tuning uses the same crop width
(text_recognition/finetune_overlap.py).
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segmentation.segmenter import _binary_for_detection
from segmentation.unet_resnet import DIVISOR, LINE_HEIGHT, STRIDE, UNET_MODEL, load_unet

TARGET_MEDIAN_HEIGHT = 26      # typical glyph height the network was trained on (pixels)
CROP_MARGIN = 0.08             # extra width on each side; validated on held-out EMNIST line crops


@dataclass
class CharBox:
    x: int          # tight box of the ink inside the character's window (image coordinates)
    y: int
    w: int
    h: int
    cx: float       # predicted centre and half-width (image coordinates)
    half_width: float
    score: float


def pad_to_multiple(image, multiple=DIVISOR, vertical_context=8, minimum=32):
    """Pad a float image so H and W are multiples of ``multiple``. Returns (padded, top, left)."""
    h, w = image.shape
    top = vertical_context
    padded_h = max(minimum, -(-(h + 2 * vertical_context) // multiple) * multiple)
    padded_w = max(minimum, -(-w // multiple) * multiple)
    out = np.zeros((padded_h, padded_w), np.float32)
    out[top:top + h, :w] = image
    return out, top, 0


def decode_peaks(heat, half_width, threshold=0.3, max_overlap=0.62, weak_overlap=0.45, weak_score=0.6):
    """Turn network output maps into (cx, half_width, score) in model-pixel coordinates.

    A peak is a local maximum of the column profile. Letters with several humps
    or stems (m, h, b, d) can produce a second, smaller peak inside the same
    letter, so a weaker peak is dropped when its predicted box sits mostly
    inside a stronger peak's box (more than ``max_overlap`` of the narrower box,
    or more than ``weak_overlap`` if its score is also below ``weak_score``).
    Genuinely overlapping neighbours overlap less than that and are both kept.
    """
    profile = heat.max(axis=0)
    rows = heat.argmax(axis=0)
    candidates = []
    for i, value in enumerate(profile):
        if value < threshold:
            continue
        lo, hi = max(0, i - 2), min(len(profile), i + 3)
        if (i > lo and value <= profile[lo:i].max()) or value < profile[i + 1:hi].max(initial=0):
            continue
        hw = max(2.0, float(half_width[rows[i], i]) * LINE_HEIGHT)
        candidates.append(((i + 0.5) * STRIDE, hw, float(value)))

    def overlap(a, b):
        inter = min(a[0] + a[1], b[0] + b[1]) - max(a[0] - a[1], b[0] - b[1])
        return max(0.0, inter) / (2 * min(a[1], b[1]))

    kept = []
    for cand in sorted(candidates, key=lambda c: -c[2]):
        if all(not (overlap(cand, k) > max_overlap or (overlap(cand, k) > weak_overlap and cand[2] < weak_score))
               for k in kept):
            kept.append(cand)
    return sorted(kept)


def window_bounds(cx, half_width, width, margin=CROP_MARGIN):
    """Columns [x0, x1) of a character's window, widened by ``margin`` on each side."""
    pad = half_width * 2 * margin
    return (int(max(0, np.floor(cx - half_width - pad))),
            int(min(width, np.ceil(cx + half_width + pad))))


def window_crop(gray, cx, half_width, margin=CROP_MARGIN):
    """Crop a character window (with margin) from a gray line image; rows are trimmed to the ink.

    Used by inference and by training-crop generation, so both see identical crops.
    ``gray`` may be white-on-black or black-on-white; ink is whatever differs from the border.
    """
    h, w = gray.shape[:2]
    x0, x1 = window_bounds(cx, half_width, w, margin)
    if x1 - x0 < 2:
        return None
    window = gray[:, x0:x1]
    background = float(np.median(np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))))
    difference = np.abs(window.astype(np.float32) - background)
    rows = np.flatnonzero((difference > 0.15 * max(float(difference.max()), 1e-6)).any(axis=1))
    if not len(rows):
        return None
    y0, y1 = max(0, rows.min() - 1), min(h, rows.max() + 2)
    return window[y0:y1]


def tight_strip_crop(gray, left, right, padding=2):
    """Digit-style crop: assign each column to one glyph, then trim to its ink.

    This is deliberately separate from the overlapping U-Net windows. A cut
    through shared ink can remove a true stroke, so it must be evaluated before
    becoming the default.
    """
    gray = np.asarray(gray)
    h, w = gray.shape[:2]
    x0, x1 = max(0, int(np.floor(left))), min(w, int(np.ceil(right)))
    if x1 <= x0:
        return None
    mask = _binary_for_detection(gray)[:, x0:x1] > 0
    if mask.sum() < 6:
        return None
    rows, cols = np.where(mask)
    bx0, bx1 = x0 + int(cols.min()), x0 + int(cols.max()) + 1
    by0, by1 = int(rows.min()), int(rows.max()) + 1
    crop_x0, crop_x1 = max(x0, bx0 - padding), min(x1, bx1 + padding)
    crop_y0, crop_y1 = max(0, by0 - padding), min(h, by1 + padding)
    return gray[crop_y0:crop_y1, crop_x0:crop_x1], (bx0, by0, bx1 - bx0, by1 - by0)


def tight_character_crop(gray, detection, left_cut, right_cut, margin=CROP_MARGIN):
    """Use the U-Net's own width, clipped at neighbour cuts, then box its ink.

    A whole Voronoi strip may contain background noise or a distant neighbour.
    The predicted half-width gives each character its own size; the hard cuts
    ensure the resulting crops never reuse columns from adjacent detections.
    """
    x0, x1 = window_bounds(detection.cx, detection.half_width, gray.shape[1], margin)
    x0, x1 = max(x0, left_cut), min(x1, right_cut)
    return tight_strip_crop(gray, x0, x1)


def tight_boundaries(boxes, width):
    """One cut between adjacent U-Net detections, respecting their widths."""
    cuts = [0]
    for left, right in zip(boxes[:-1], boxes[1:]):
        proposed = (left.cx + left.half_width + right.cx - right.half_width) / 2
        # Keep the cut between centres even if width estimates cross wildly.
        cut = int(round(np.clip(proposed, left.cx + 1, right.cx - 1)))
        cuts.append(max(cuts[-1] + 1, min(width - 1, cut)))
    cuts.append(width)
    return cuts


def remove_disconnected_neighbours(gray, boxes, index):
    """Hide ink owned solely by another U-Net character centre.

    Ink spanning multiple centres is retained in both crops because connected
    components cannot reliably separate letters sharing a stroke.
    """
    binary = _binary_for_detection(gray)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 2 or len(boxes) <= 1:
        return gray
    centres = np.asarray([b.cx for b in boxes])
    excluded = np.zeros(count, dtype=bool)
    for component in range(1, count):
        x, _, width, _, _ = stats[component]
        owners = np.flatnonzero((centres >= x) & (centres < x + width))
        if not len(owners):
            owners = np.array([np.argmin(abs(centres - (x + width / 2)))])
        excluded[component] = index not in owners
    if not excluded.any():
        return gray
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    cleaned = gray.copy()
    cleaned[excluded[labels]] = np.uint8(np.median(border))
    return cleaned


class UNetSegmenter:
    def __init__(self, model_path=UNET_MODEL, threshold=0.3, model=None):
        self.model = model if model is not None else load_unet(model_path)
        self.threshold = threshold

    @staticmethod
    def available(model_path=UNET_MODEL):
        return Path(model_path).is_file()

    def find_characters(self, gray):
        """Return a left-to-right list of CharBox for a grayscale line image (uint8)."""
        gray = np.asarray(gray)
        binary = _binary_for_detection(gray)
        if not binary.any():
            return []
        ink = cv2.GaussianBlur(binary, (3, 3), 0).astype(np.float32) / 255

        # Rescale so a typical glyph is the size the network was trained on.
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, count)
                   if stats[i, cv2.CC_STAT_HEIGHT] >= 0.2 * binary.shape[0]
                   and stats[i, cv2.CC_STAT_AREA] >= 12]
        scale = TARGET_MEDIAN_HEIGHT / float(np.median(heights)) if heights else LINE_HEIGHT / binary.shape[0]
        scale = float(np.clip(scale, 0.3, 4.0))
        resized = cv2.resize(ink, (max(8, round(ink.shape[1] * scale)), max(8, round(ink.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        padded, top, left = pad_to_multiple(resized)

        output = np.asarray(self.model(padded[None, ..., None], training=False))[0]
        peaks = decode_peaks(output[..., 0], output[..., 1], self.threshold)

        boxes = []
        for cx_model, hw_model, score in peaks:
            cx, hw = (cx_model - left) / scale, hw_model / scale
            x0, x1 = int(max(0, round(cx - hw))), int(min(binary.shape[1], round(cx + hw) + 1))
            part = binary[:, x0:x1] > 0
            if part.sum() < 6:      # a peak with no ink under it is a false alarm
                continue
            rows, cols = np.flatnonzero(part.any(axis=1)), np.flatnonzero(part.any(axis=0))
            boxes.append(CharBox(x0 + int(cols.min()), int(rows.min()), int(cols.max() - cols.min() + 1),
                                 int(rows.max() - rows.min() + 1), float(cx), float(hw), score))
        return boxes

    def crops(self, gray, boxes=None, margin=CROP_MARGIN, tight=False, with_boxes=False):
        """Raw grayscale PIL crops (one per box) for the character classifier."""
        gray = np.asarray(gray)
        boxes = self.find_characters(gray) if boxes is None else boxes
        result = []
        cuts = tight_boundaries(boxes, gray.shape[1]) if tight and boxes else None
        for i, b in enumerate(boxes):
            if tight:
                item = tight_character_crop(gray, b, cuts[i], cuts[i + 1])
                if item is None:
                    continue
                crop, bbox = item
            else:
                crop = window_crop(remove_disconnected_neighbours(gray, boxes, i), b.cx, b.half_width, margin)
                bbox = (b.x, b.y, b.w, b.h)
            if crop is not None:
                image = Image.fromarray(crop)
                result.append((image, bbox) if with_boxes else image)
        return result
