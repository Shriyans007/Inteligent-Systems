"""The API reports where each piece was cut from, so the GUI can draw the segmentation."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import api.main as api
from segmentation_model.segment import segment_learned


class MiddleCut:
    def predict(self, batch, verbose=0):
        width = batch.shape[2]
        probabilities = np.zeros((len(batch), 1, width, 1), dtype=np.float32)
        probabilities[:, 0, width // 2, 0] = 0.9
        return probabilities


class Digit4:
    def predict(self, batch, verbose=0):
        probabilities = np.zeros((len(batch), 10), dtype=np.float32)
        probabilities[:, 4] = 1
        return probabilities


def two_strokes():
    image = Image.new("L", (160, 70), 255)
    draw = ImageDraw.Draw(image)
    draw.line((30, 10, 30, 60), fill=0, width=6)
    draw.line((120, 10, 120, 60), fill=0, width=6)
    return image


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_learned_segmenter_reports_cuts_and_profile():
    details = {}
    crops = segment_learned(two_strokes(), MiddleCut(), details=details)
    assert len(crops) == 2 and len(details["cuts"]) == 1
    assert 30 < details["cuts"][0] < 120
    assert max(p for _, p in details["profile"]) == 0.9


def test_predict_returns_boxes_inside_the_uploaded_image(tmp_path, monkeypatch):
    paths = {key: tmp_path / f"{key}.keras" for key in api.MODEL_PATHS}
    paths["cnn"].touch()
    model_file = tmp_path / "cutpoint_cnn.keras"
    model_file.touch()
    monkeypatch.setattr(api, "MODEL_PATHS", paths)
    monkeypatch.setattr(api, "_models", {"cnn": Digit4()})
    monkeypatch.setattr(api, "SEGMENTER_MODEL_PATH", model_file)
    monkeypatch.setattr(api, "_segmenter_models", {"cutpoint": MiddleCut()})

    with TestClient(api.app) as client:
        def send(method):
            return client.post("/api/predict", data={"model": "cnn", "segmentation": method},
                               files={"files": ("two.png", png(two_strokes()), "image/png")}).json()
        rules, learned = send("components"), send("learned_cuts")

    for body in (rules, learned):
        assert body["images"][0]["width"] == 160 and body["images"][0]["height"] == 70
        assert [item["image"] for item in body["predictions"]] == [0, 0]
        for item in body["predictions"]:
            x, y, w, h = item["bbox"]
            assert 0 <= x and x + w <= 160 and 0 <= y and y + h <= 70
    assert "cuts" not in rules["images"][0]
    assert len(learned["images"][0]["cuts"]) == 1 and learned["images"][0]["profile"]
