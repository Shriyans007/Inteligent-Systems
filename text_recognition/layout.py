"""Segment separated handwriting while retaining spaces and line breaks."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from segmentation.segmenter import _binary_for_detection


@dataclass
class Glyph:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    before: str = ""


def _boxes(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [tuple(map(int, stats[i, :4])) for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= max(8, mask.size // 30000)]


def _line_groups(boxes):
    # Group tall strokes by their vertical centres; attach dots to the closest
    # stroke on the same line instead of treating a dot as another text line.
    heights = [h for _, _, _, h in boxes]
    typical = float(np.median(heights))
    groups = []
    for box in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
        x, y, w, h = box
        center = y + h / 2
        candidates = [(abs(center - np.median([b[1] + b[3] / 2 for b in group])), group)
                      for group in groups]
        if candidates and min(candidates, key=lambda item: item[0])[0] < typical * 0.65:
            min(candidates, key=lambda item: item[0])[1].append(box)
        else:
            groups.append([box])
    return sorted(groups, key=lambda group: min(b[1] for b in group))


def _attach_marks(boxes):
    # An i/j dot has a similar horizontal position to its stem. Merge its
    # bounding box with that stem without altering the original image pixels.
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    if not boxes:
        return boxes
    median_area = np.median([w * h for _, _, w, h in boxes])
    kept = []
    for x, y, w, h in boxes:
        if w * h < median_area * 0.25:
            matches = [i for i, (sx, sy, sw, sh) in enumerate(kept)
                       if max(x, sx) < min(x + w, sx + sw) and
                       0 <= max(sy - y - h, y - sy - sh) < max(h, sh)]
            if matches:
                i = min(matches, key=lambda n: abs((kept[n][0] + kept[n][2] / 2) - (x + w / 2)))
                sx, sy, sw, sh = kept[i]
                x0, y0 = min(x, sx), min(y, sy)
                kept[i] = (x0, y0, max(x + w, sx + sw) - x0, max(y + h, sy + sh) - y0)
                continue
        kept.append((x, y, w, h))
    return sorted(kept)


def segment_text(image: Image.Image) -> list[Glyph]:
    """Return glyph crops and separators for clearly separated handwriting.

    Short stroke gaps are closed only for detection. Crops contain the original
    grayscale pixels, so the classifier receives the untouched handwriting.
    """
    gray = np.asarray(image.convert("L"))
    binary = _binary_for_detection(gray)
    if not np.any(binary):
        return []
    distances = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    width = int(np.clip(round(2 * np.percentile(distances[distances > 0], 75)), 3, 9))
    if width % 2 == 0:
        width += 1
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1, width), np.uint8))
    boxes = _boxes(joined)
    if not boxes:
        return []
    result = []
    for line_index, group in enumerate(_line_groups(boxes)):
        glyphs = _attach_marks(group)
        widths = [w for _, _, w, _ in glyphs]
        typical_width = float(np.median(widths))
        gaps = [max(0, glyphs[i][0] - (glyphs[i - 1][0] + glyphs[i - 1][2]))
                for i in range(1, len(glyphs))]
        positive = [gap for gap in gaps if gap > 0]
        small_gap = min(positive) if positive else 0
        threshold = (0.9 * typical_width if small_gap >= 1.1 * typical_width
                     else max(1.1 * typical_width, 2.8 * small_gap))
        for i, (x, y, w, h) in enumerate(glyphs):
            before = "\n" if line_index and i == 0 else ""
            if i and gaps[i - 1] > threshold:
                before = " "
            margin = 2
            crop = Image.fromarray(gray[max(0, y-margin):min(gray.shape[0], y+h+margin),
                                        max(0, x-margin):min(gray.shape[1], x+w+margin)])
            result.append(Glyph(crop, (x, y, w, h), before))
    return result
