"""The chosen segmentation method must reach segment() on both recognition routes."""

import io
import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import api.main as api


class StubClassifier:
    def __init__(self, classes, winner):
        self.classes, self.winner = classes, winner

    def predict(self, batch, verbose=0):
        probabilities = np.zeros((len(batch), self.classes), dtype=np.float32)
        probabilities[:, self.winner] = 1
        return probabilities


def drawing():
    image = Image.new("L", (70, 70), 255)
    ImageDraw.Draw(image).line((20, 10, 30, 55), fill=0, width=5)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def record_segment(monkeypatch):
    """Wrap the real segmenter so the test can see which options each request used."""
    calls = []
    real = api.segment

    def recording(image, **options):
        calls.append(options)
        return real(image, **options)

    monkeypatch.setattr(api, "segment", recording)
    return calls


def test_segmenters_endpoint_lists_methods_and_defaults():
    with TestClient(api.app) as client:
        body = client.get("/api/segmenters").json()
    assert [item["key"] for item in body["segmenters"]] == list(api.SEGMENTERS)
    assert body["defaults"] == {"numbers": "components_split", "characters": "components"}


def test_digit_route_uses_the_chosen_segmenter(tmp_path, monkeypatch):
    paths = {key: tmp_path / f"{key}.keras" for key in api.MODEL_PATHS}
    paths["cnn"].touch()
    monkeypatch.setattr(api, "MODEL_PATHS", paths)
    monkeypatch.setattr(api, "_models", {"cnn": StubClassifier(10, 4)})
    calls = record_segment(monkeypatch)

    with TestClient(api.app) as client:
        def send(**form):
            return client.post("/api/predict", data={"model": "cnn", **form},
                               files={"files": ("digit.png", drawing(), "image/png")})
        default = send()
        contours = send(segmentation="contours")
        unknown = send(segmentation="nope")

    assert default.status_code == 200 and default.json()["segmentation"] == "components_split"
    assert calls[0] == {"method": "connected_components", "split_touching": True}
    assert contours.status_code == 200 and contours.json()["segmentation_label"] == "Contours"
    assert calls[1] == {"method": "contours", "split_touching": False}
    assert unknown.status_code == 400


def test_character_route_uses_the_chosen_segmenter(tmp_path, monkeypatch):
    model_file, mapping_file = tmp_path / "char.keras", tmp_path / "mapping.json"
    model_file.touch()
    mapping_file.write_text(json.dumps({"11": "B"}))
    monkeypatch.setattr(api, "CHARACTER_PATHS", {"cnn": (model_file, mapping_file)})
    monkeypatch.setattr(api, "_extension_models", {"cnn": (StubClassifier(62, 11), {"11": "B"})})
    calls = record_segment(monkeypatch)

    with TestClient(api.app) as client:
        def send(**form):
            return client.post("/api/recognise-text", data=form,
                               files={"file": ("b.png", drawing(), "image/png")})
        default = send()
        chosen = send(segmentation="contours_split")

    assert default.status_code == 200 and default.json()["segmentation"] == "components"
    assert calls[0] == {"method": "connected_components", "split_touching": False}
    assert chosen.status_code == 200 and chosen.json()["text"] == "B"
    assert calls[1] == {"method": "contours", "split_touching": True}
