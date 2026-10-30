"""The EMNIST SVM must work through the same character route as the networks."""

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import api.main as api

CHARACTERS = [str(d) for d in range(10)] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
MAPPING = {str(index): character for index, character in enumerate(CHARACTERS)}


class StubCharacterSVM:
    """Knows only A (10) and B (11): predict() decides B while predict_proba() leans to A."""

    def __init__(self):
        self.classes_ = np.array([10, 11])
        self.seen_shape = None

    def predict_proba(self, flat):
        self.seen_shape = flat.shape
        return np.tile([0.6, 0.4], (len(flat), 1))

    def predict(self, flat):
        return np.full(len(flat), 11)


def test_sklearn_probabilities_fill_all_62_character_columns():
    batch = np.zeros((1, 28, 28, 1), dtype=np.float32)
    probabilities = api.predict_probabilities(StubCharacterSVM(), "svm", batch, num_classes=62)
    assert probabilities.shape == (1, 62)
    assert probabilities[0, 10] == pytest.approx(0.6) and probabilities[0, 11] == pytest.approx(0.4)


def test_character_route_runs_the_svm(tmp_path, monkeypatch):
    model_file, mapping_file = tmp_path / "emnist_svm.joblib", tmp_path / "mapping.json"
    model_file.touch()
    mapping_file.write_text(json.dumps(MAPPING))
    stub = StubCharacterSVM()
    monkeypatch.setattr(api, "CHARACTER_PATHS", {"svm": (model_file, mapping_file)})
    monkeypatch.setattr(api, "_extension_models", {"svm": (stub, MAPPING)})
    image = Image.new("L", (70, 70), 255)
    ImageDraw.Draw(image).line((20, 10, 30, 55), fill=0, width=5)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    with TestClient(api.app) as client:
        listed = client.get("/api/extension-models").json()["character_models"]
        response = client.post("/api/recognise-text", data={"model": "svm"},
                               files={"file": ("b.png", buffer.getvalue(), "image/png")})
    assert listed == [{"key": "svm", "label": "SVM (RBF)", "available": True}]
    assert response.status_code == 200
    assert stub.seen_shape == (1, 784)
    body = response.json()
    # The character follows predict(); its confidence is that character's probability.
    assert body["text"] == "B" and body["model_label"] == "SVM (RBF)"
    assert body["predictions"][0]["confidence"] == pytest.approx(0.4)
