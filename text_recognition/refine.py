"""Carefully fix a character that got split across two U-Net crops.

The EMNIST classifier can sometimes see a whole character in a connected
component even when the segmenter put two centres inside it. A merge is only
accepted when the whole component is narrow enough to be one glyph and the
selected classifier clearly prefers it to both of the original crops. The rule
doesn't use a word list or any character-specific fixes.
"""

import cv2
import numpy as np
from PIL import Image

from segmentation.segmenter import _binary_for_detection
from text_recognition.common import prepare_character
from text_recognition.layout import Glyph


def refine_split_components(image, glyphs, prepared, probabilities, classifier):
    """Return (glyphs, tensors, scores), only merging false splits that are well supported."""
    if len(glyphs) < 2:
        return glyphs, prepared, probabilities

    gray = np.asarray(image.convert("L"))
    count, _, stats, _ = cv2.connectedComponentsWithStats(_binary_for_detection(gray), 8)
    candidates = []
    for x, y, width, height, area in stats[1:count]:
        if area < 12 or width / height >= 0.85:
            continue
        owners = []
        for i, glyph in enumerate(glyphs):
            gx, gy, gw, gh = glyph.bbox
            if x <= gx + gw / 2 < x + width and y <= gy + gh / 2 < y + height:
                owners.append(i)
        if len(owners) != 2 or owners[1] != owners[0] + 1 or glyphs[owners[1]].before:
            continue
        crop = Image.fromarray(gray[y:y + height, x:x + width])
        try:
            tensor = prepare_character(crop)
        except ValueError:
            continue
        candidates.append((owners[0], Glyph(crop, (int(x), int(y), int(width), int(height)),
                                             glyphs[owners[0]].before), tensor))

    if not candidates:
        return glyphs, prepared, probabilities
    candidate_scores = classifier.predict(np.stack([item[2] for item in candidates]), verbose=0)
    replacements = {}
    for (i, glyph, tensor), scores in zip(candidates, candidate_scores):
        confidence = float(scores.max())
        first, second = probabilities[i], probabilities[i + 1]
        if (confidence >= 0.95 and float(second.max()) < 0.85
                and confidence > max(float(first.max()), float(second.max())) + 0.04
                and int(scores.argmax()) not in (int(first.argmax()), int(second.argmax()))):
            replacements[i] = (glyph, tensor, scores)

    if not replacements:
        return glyphs, prepared, probabilities
    output_glyphs, output_tensors, output_scores = [], [], []
    i = 0
    while i < len(glyphs):
        if i in replacements:
            glyph, tensor, scores = replacements[i]
            output_glyphs.append(glyph)
            output_tensors.append(tensor)
            output_scores.append(scores)
            i += 2
        else:
            output_glyphs.append(glyphs[i])
            output_tensors.append(prepared[i])
            output_scores.append(probabilities[i])
            i += 1
    return output_glyphs, output_tensors, np.stack(output_scores)
