"""FastAPI service for handwritten digit predictions."""

from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

from cnn_model.models import _keras
from preprocessing import prepare_mnist_digit
from segmentation import crops_to_model_input, segment

MODEL_PATH = Path(os.getenv("HNRS_MODEL_PATH", "artifacts/cnn_mnist.keras"))

app = FastAPI(title="HNRS Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model = None


def get_model():
    """Load the model once, on the first prediction request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Train it with "
                "python -m cnn_model.train_compare --epochs 12 --output artifacts"
            )
        _model = _keras().models.load_model(MODEL_PATH)
    return _model


@app.on_event("startup")
def warm_up_model() -> None:
    """Load the trained CNN when the API starts so the first request is faster."""
    if MODEL_PATH.exists():
        model = get_model()
        model.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    """Report API availability separately from model availability."""
    return {"status": "ok", "model_ready": MODEL_PATH.exists(), "model_path": str(MODEL_PATH)}


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)) -> dict:
    """Recognise one digit or an ordered list of pre-segmented digit images."""
    if not 1 <= len(files) <= 30:
        raise HTTPException(status_code=400, detail="Upload between 1 and 30 digit images.")

    try:
        model = get_model()
        prepared_images = []
        filenames = []
        for upload in files:
            if upload.content_type not in {"image/png", "image/jpeg", "image/bmp", "image/webp"}:
                raise HTTPException(status_code=415, detail=f"{upload.filename} is not a supported image.")
            content = await upload.read()
            if len(content) > 5 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds the 5 MB limit.")
            image = Image.open(io.BytesIO(content))
            crops = segment(image)
            if crops:
                prepared_images.extend(crops_to_model_input(crops))
                filenames.extend(f"{upload.filename} #{crop.index + 1}" for crop in crops)
            else:
                # Some MNIST-style images have a dark full-image background,
                # so segmentation may correctly find no separate dark region.
                prepared_images.append(prepare_mnist_digit(image))
                filenames.append(upload.filename)

        # Run one batch prediction rather than calling TensorFlow once per file.
        batch = np.concatenate(prepared_images, axis=0)
        batch_probabilities = model.predict(batch, verbose=0)
        predictions = []
        for position, (filename, probabilities) in enumerate(zip(filenames, batch_probabilities)):
            digit = int(probabilities.argmax())
            predictions.append(
                {
                    "position": position,
                    "filename": filename,
                    "digit": digit,
                    "confidence": float(probabilities[digit]),
                    "probabilities": [float(value) for value in probabilities],
                }
            )
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "number": "".join(str(item["digit"]) for item in predictions),
        "average_confidence": sum(item["confidence"] for item in predictions) / len(predictions),
        "lowest_confidence": min(item["confidence"] for item in predictions),
        "predictions": predictions,
    }
