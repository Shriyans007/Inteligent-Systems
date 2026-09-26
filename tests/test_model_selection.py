"""Model choice must map to trained local files and identify its prediction."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import api.main as api


class StubClassifier:
    def __init__(self, digit):
        self.digit = digit

    def predict(self, batch, verbose=0):
        probabilities = np.zeros((len(batch), 10), dtype=np.float32)
        probabilities[:, self.digit] = 1
        return probabilities


def test_selects_requested_model_and_rejects_missing_or_unknown(tmp_path, monkeypatch):
    paths = {key: tmp_path / f"{key}.keras" for key in api.MODEL_PATHS}
    paths["cnn"].touch()
    paths["resnet"].touch()
    monkeypatch.setattr(api, "MODEL_PATHS", paths)
    monkeypatch.setattr(api, "_models", {"cnn": StubClassifier(1), "resnet": StubClassifier(9)})
    image = Image.new("L", (40, 40), 255)
    ImageDraw.Draw(image).line((12, 8, 23, 31), fill=0, width=5)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    with TestClient(api.app) as client:
        listed = {item["key"]: item["available"] for item in client.get("/api/models").json()["models"]}
        assert listed == {"cnn": True, "mlp": False, "lenet5": False, "resnet": True}
        def send(choice):
            return client.post("/api/predict", data={"model": choice},
                               files={"files": ("digit.png", image_bytes, "image/png")})
        cnn = send("cnn")
        resnet = send("resnet")
        assert cnn.status_code == 200 and cnn.json()["number"] == "1"
        assert resnet.status_code == 200 and resnet.json()["number"] == "9"
        assert resnet.json()["model_label"] == "Small ResNet"
        assert send("mlp").status_code == 503
        assert send("../../other").status_code == 400
