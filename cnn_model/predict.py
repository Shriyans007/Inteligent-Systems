"""Prediction service used by the GUI and later segmentation integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .image_utils import prepare_digit_image
from .models import _keras


class DigitPredictor:
    def __init__(self, model_path: str | Path = "artifacts/cnn_mnist.keras") -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run the training command first.")
        self.model = _keras().models.load_model(path)

    def predict(self, image_path: str | Path) -> tuple[int, float, np.ndarray]:
        probabilities = self.model.predict(prepare_digit_image(str(image_path)), verbose=0)[0]
        digit = int(probabilities.argmax())
        return digit, float(probabilities[digit]), probabilities

