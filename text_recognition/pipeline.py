"""Read a handwritten image: segment -> classify -> fix upper/lowercase -> text.

    from text_recognition.pipeline import recognise_text
    result = recognise_text(pil_image)
    result["text"]          # e.g. "Butterfly"
    result["glyphs"]        # per character: char, confidence, bbox

The classifier is the overlap fine-tuned model if it exists
(text_recognition/finetune_overlap.py), otherwise the original EMNIST model.
"""
import json

import numpy as np

from text_recognition.case import resolve_case
from text_recognition.common import CHAR_MAPPING, CHAR_MODEL, prepare_character
from text_recognition.layout import segment_text

CHAR_MODEL_OVERLAP = CHAR_MODEL.with_name("emnist_cnn_overlap.keras")
_CACHE = {}


def load_character_model(prefer_overlap=True):
    """(model, {label: character}) -- cached after the first call."""
    path = CHAR_MODEL_OVERLAP if prefer_overlap and CHAR_MODEL_OVERLAP.is_file() else CHAR_MODEL
    if path not in _CACHE:
        from segmentation.unet_resnet import _keras
        mapping = {int(k): v for k, v in json.loads(CHAR_MAPPING.read_text()).items()}
        _CACHE[path] = (_keras().models.load_model(path), mapping)
    return _CACHE[path]


def recognise_text(image, method="auto", model=None, mapping=None):
    """Recognise handwriting in a PIL image. ``method`` is passed to ``segment_text``."""
    glyphs = segment_text(image, method=method)
    if not glyphs:
        return {"text": "", "glyphs": []}
    if model is None:
        model, mapping = load_character_model()

    tensors, kept = [], []
    for glyph in glyphs:
        try:
            tensors.append(prepare_character(glyph.image))
            kept.append(glyph)
        except ValueError:                      # crop with no visible ink
            continue
    if not tensors:
        return {"text": "", "glyphs": []}
    probabilities = model.predict(np.stack(tensors), verbose=0)
    chars = [mapping[int(i)] for i in probabilities.argmax(axis=1)]

    # Letters such as c/C, o/O, s/S look identical once resized; use their size on the line.
    line_ids = np.cumsum([g.before == "\n" for g in kept]).tolist()
    chars = resolve_case(chars, [g.bbox for g in kept], line_ids)

    text = "".join(g.before + c for g, c in zip(kept, chars))
    return {"text": text,
            "glyphs": [{"char": c, "confidence": float(p.max()), "bbox": g.bbox}
                       for g, c, p in zip(kept, chars, probabilities)]}
