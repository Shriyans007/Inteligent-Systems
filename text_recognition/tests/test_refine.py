import numpy as np
from PIL import Image, ImageDraw

from text_recognition.common import prepare_character
from text_recognition.layout import Glyph
from text_recognition.refine import refine_split_components


class CandidateClassifier:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, batch, verbose=0):
        assert len(batch) == 1
        return self.scores[None, :]


def test_complete_connected_ink_repairs_a_split_without_changing_the_next_glyph():
    image = Image.new("L", (120, 110), 255)
    draw = ImageDraw.Draw(image)
    draw.line([(20, 96), (20, 16), (36, 46), (53, 16), (53, 96)], fill=0, width=5)
    draw.line([(95, 18), (95, 97)], fill=0, width=5)
    glyphs = [Glyph(image.crop((17, 12, 43, 100)), (18, 14, 25, 84)),
              Glyph(image.crop((39, 12, 57, 100)), (39, 14, 18, 84)),
              Glyph(image.crop((91, 14, 99, 101)), (92, 15, 7, 85))]
    tensors = [prepare_character(g.image) for g in glyphs]
    scores = np.zeros((3, 62), dtype=np.float32)
    scores[0, 1], scores[1, 2], scores[2, 4] = 0.92, 0.77, 0.99
    merged_scores = np.zeros(62, dtype=np.float32)
    merged_scores[3] = 0.98

    refined, prepared, probabilities = refine_split_components(
        image, glyphs, tensors, scores, CandidateClassifier(merged_scores))

    assert len(refined) == len(prepared) == len(probabilities) == 2
    assert int(probabilities[0].argmax()) == 3
    assert refined[1] is glyphs[2]
    assert refined[0].bbox[2] > glyphs[0].bbox[2]


def test_ambiguous_complete_ink_does_not_force_a_merge():
    image = Image.new("L", (70, 110), 255)
    ImageDraw.Draw(image).line([(20, 96), (20, 16), (36, 46), (53, 16), (53, 96)],
                               fill=0, width=5)
    glyphs = [Glyph(image.crop((17, 12, 43, 100)), (18, 14, 25, 84)),
              Glyph(image.crop((39, 12, 57, 100)), (39, 14, 18, 84))]
    tensors = [prepare_character(g.image) for g in glyphs]
    scores = np.zeros((2, 62), dtype=np.float32)
    scores[0, 1], scores[1, 2] = 0.92, 0.77
    candidate = np.zeros(62, dtype=np.float32)
    candidate[1] = 0.99  # the complete component repeats the first prediction

    refined, prepared, probabilities = refine_split_components(
        image, glyphs, tensors, scores, CandidateClassifier(candidate))

    assert refined is glyphs
    assert prepared is tensors
    assert probabilities is scores
