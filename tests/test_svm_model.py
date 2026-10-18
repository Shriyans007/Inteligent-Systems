"""The SVM baseline must plug into the same prediction path as the Keras models."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import api.main as api


class StubSVM:
    """Behaves like scikit-learn: flat input, predict_proba, classes_ in its own order."""

    def __init__(self):
        self.classes_ = np.array([3, 7])
        self.seen_shape = None

    def predict_proba(self, flat):
        self.seen_shape = flat.shape
        return np.tile([0.25, 0.75], (len(flat), 1))


def test_sklearn_models_get_flat_input_and_ten_class_output():
    stub = StubSVM()
    probabilities = api.predict_probabilities(stub, "svm", np.zeros((2, 28, 28, 1), dtype=np.float32))
    assert stub.seen_shape == (2, 784)
    assert probabilities.shape == (2, 10)
    # Columns follow classes_, so 0.75 must land on digit 7, not on column 1.
    assert probabilities[0, 7] == 0.75 and probabilities[0, 3] == 0.25
    assert np.isclose(probabilities[0].sum(), 1.0)


def test_svm_is_a_digit_model_but_not_a_character_model():
    assert "svm" in api.MODEL_PATHS
    assert "svm" not in api.CHARACTER_PATHS


def test_predict_endpoint_runs_the_svm(tmp_path, monkeypatch):
    paths = {key: tmp_path / f"{key}.model" for key in api.MODEL_PATHS}
    paths["svm"].touch()
    monkeypatch.setattr(api, "MODEL_PATHS", paths)
    monkeypatch.setattr(api, "_models", {"svm": StubSVM()})
    image = Image.new("L", (40, 40), 255)
    ImageDraw.Draw(image).line((12, 8, 23, 31), fill=0, width=5)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    with TestClient(api.app) as client:
        response = client.post("/api/predict", data={"model": "svm"},
                               files={"files": ("digit.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["number"] == "7"
    assert response.json()["model_label"] == "SVM (RBF)"
