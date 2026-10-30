"""A complete connected glyph may replace two uncertain U-Net detections."""

import numpy as np
from PIL import Image

from text_recognition import refine
from text_recognition.layout import Glyph


class CertainMergedGlyph:
    def predict(self, batch, verbose=0):
        scores = np.zeros((len(batch), 62), np.float32)
        scores[:, 16] = 0.96
        return scores


def test_tight_repair_merges_two_weak_parts_of_one_connected_glyph(monkeypatch):
    monkeypatch.setattr(refine, "prepare_character", lambda image: np.zeros((28, 28, 1), np.float32))
    pixels = np.full((50, 60), 255, np.uint8)
    pixels[10:40, 10:43] = 0  # one connected, fairly wide glyph
    image = Image.fromarray(pixels)
    glyphs = [Glyph(image.crop((10, 10, 27, 40)), (10, 10, 17, 30)),
              Glyph(image.crop((27, 10, 43, 40)), (27, 10, 16, 30))]
    scores = np.zeros((2, 62), np.float32)
    scores[0, 1], scores[1, 2] = 0.54, 0.22
    result, _, _ = refine.refine_split_components(
        image, glyphs, [np.zeros((28, 28, 1), np.float32)] * 2, scores,
        CertainMergedGlyph(), max_width_ratio=1.15, min_merge_confidence=0.85,
        min_confidence_gain=0.15, max_first_confidence=0.85)
    assert len(result) == 1
    assert result[0].bbox == (10, 10, 33, 30)


def test_tight_repair_keeps_two_confident_predictions(monkeypatch):
    monkeypatch.setattr(refine, "prepare_character", lambda image: np.zeros((28, 28, 1), np.float32))
    pixels = np.full((50, 60), 255, np.uint8)
    pixels[10:40, 10:43] = 0
    image = Image.fromarray(pixels)
    glyphs = [Glyph(image.crop((10, 10, 27, 40)), (10, 10, 17, 30)),
              Glyph(image.crop((27, 10, 43, 40)), (27, 10, 16, 30))]
    scores = np.zeros((2, 62), np.float32)
    scores[0, 1], scores[1, 2] = 0.93, 0.89
    result, _, _ = refine.refine_split_components(
        image, glyphs, [np.zeros((28, 28, 1), np.float32)] * 2, scores,
        CertainMergedGlyph(), max_width_ratio=1.15, min_merge_confidence=0.85,
        min_confidence_gain=0.15, max_first_confidence=0.85)
    assert len(result) == 2
