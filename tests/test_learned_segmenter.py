"""The learned segmenter: training labels, turning cuts into crops, and the API option."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import api.main as api
from segmentation import segment
from segmentation_model.data import compose
from segmentation_model.segment import find_cuts, segment_learned


class MiddleCut:
    """Stands in for the CNN: always predicts one cut in the middle of the image."""

    def predict(self, batch, verbose=0):
        width = batch.shape[2]
        probabilities = np.zeros((len(batch), 1, width, 1), dtype=np.float32)
        probabilities[:, 0, width // 2, 0] = 0.9
        return probabilities


def block_digit(width):
    digit = np.zeros((28, 28), dtype=np.uint8)
    digit[4:24, 14 - width // 2:14 + width // 2] = 255
    return digit


def touching_pair():
    """Two strokes joined by a bar: one blob, so the rule-based segmenter sees one piece."""
    image = Image.new("L", (120, 60), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 45, 50), outline=0, width=5)
    draw.rectangle((45, 10, 70, 50), outline=0, width=5)
    return image


def test_compose_labels_one_cut_per_boundary():
    rng = np.random.default_rng(0)
    image, labels, boundaries = compose([block_digit(10)] * 3, rng, touching=1.0)
    assert image.shape == (32, 192) and len(boundaries) == 2
    assert len(find_cuts(labels)) == 2


def test_wide_gaps_are_labelled_as_one_cut():
    rng = np.random.default_rng(3)
    image, labels, boundaries = compose([block_digit(10)] * 2, rng, touching=0.0)
    inked = np.where(image.max(axis=0) > 0)[0]
    between = [c for c in range(inked[0], inked[-1]) if image[:, c].max() == 0]
    assert between and labels[between].all()
    assert len(find_cuts(labels)) == len(boundaries) == 1


def test_find_cuts_returns_the_centre_of_each_run():
    assert find_cuts(np.array([0, 0.9, 0.9, 0.9, 0, 0, 0.8, 0])) == [2.0, 6.0]
    assert find_cuts(np.zeros(10)) == []


def test_learned_cuts_split_digits_the_rules_cannot():
    image = touching_pair()
    assert len(segment(image)) == 1
    crops = segment_learned(image, MiddleCut())
    assert len(crops) == 2 and crops[0].bbox[0] < crops[1].bbox[0]


def test_api_runs_the_learned_segmenter_when_trained(tmp_path, monkeypatch):
    model_file = tmp_path / "cutpoint_cnn.keras"
    paths = {key: tmp_path / f"{key}.keras" for key in api.MODEL_PATHS}
    paths["cnn"].touch()

    class Digit7:
        def predict(self, batch, verbose=0):
            probabilities = np.zeros((len(batch), 10), dtype=np.float32)
            probabilities[:, 7] = 1
            return probabilities

    monkeypatch.setattr(api, "MODEL_PATHS", paths)
    monkeypatch.setattr(api, "_models", {"cnn": Digit7()})
    monkeypatch.setattr(api, "SEGMENTER_MODEL_PATH", model_file)
    monkeypatch.setattr(api, "_segmenter_models", {"cutpoint": MiddleCut()})
    buffer = io.BytesIO()
    touching_pair().save(buffer, format="PNG")

    with TestClient(api.app) as client:
        def send():
            return client.post("/api/predict", data={"model": "cnn", "segmentation": "learned_cuts"},
                               files={"files": ("pair.png", buffer.getvalue(), "image/png")})
        def learned_listing():
            listed = client.get("/api/segmenters").json()["segmenters"]
            return next(item for item in listed if item["key"] == "learned_cuts")
        untrained, listed_before = send(), learned_listing()
        model_file.touch()
        trained, listed_after = send(), learned_listing()

    assert untrained.status_code == 503 and listed_before["available"] is False
    assert listed_after["available"] is True
    assert trained.status_code == 200
    assert trained.json()["number"] == "77"
    assert trained.json()["segmentation_label"] == "Learned cut-point CNN"


def test_cutpoint_model_gives_one_probability_per_column():
    from segmentation_model.model import build_cutpoint_model
    model = build_cutpoint_model()
    for width in (57, 192):
        output = model.predict(np.zeros((1, 32, width, 1), dtype=np.float32), verbose=0)
        assert output.shape == (1, 1, width, 1)
