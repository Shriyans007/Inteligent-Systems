"""Exercise the real upload/segmentation route with a stub classifier."""

import csv

import numpy as np
from PIL import Image, ImageDraw

from cnn_model.evaluate_uploads import evaluate_manifest


def test_labelled_upload_evaluation_uses_api_and_segmentation(tmp_path, monkeypatch):
    import api.main as api

    class StubClassifier:
        def predict(self, batch, verbose=0):
            assert batch.shape[1:] == (28, 28, 1)
            assert batch.dtype == np.float32
            probabilities = np.zeros((len(batch), 10), dtype=np.float32)
            probabilities[:, 0] = 1
            return probabilities

    monkeypatch.setattr(api, "get_model", lambda: StubClassifier())
    image = Image.new("L", (90, 40), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 8, 22, 32), fill=0)
    draw.rectangle((55, 8, 65, 32), fill=0)
    image.save(tmp_path / "two.png")
    manifest = tmp_path / "samples.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label"])
        writer.writerow(["two.png", "00"])
    output = tmp_path / "out.json"
    evaluate_manifest(manifest, output)
    import json
    result = json.loads(output.read_text())
    assert result["multi_digit"] == 1
    assert result["cases"][0]["predicted"] == "00"
