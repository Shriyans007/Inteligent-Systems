"""FastAPI service for handwritten digit predictions."""

from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from cnn_model.models import _keras
from preprocessing import prepare_mnist_digit

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


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    """Report API availability separately from model availability."""
    return {"status": "ok", "model_ready": MODEL_PATH.exists(), "model_path": str(MODEL_PATH)}


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)) -> dict:
    """Recognise one digit or an ordered list of pre-segmented digit images."""
    if not 1 <= len(files) <= 30:
        raise HTTPException(status_code=400, detail="Upload between 1 and 30 digit images.")

    predictions = []
    try:
        model = get_model()
        for position, upload in enumerate(files):
            if upload.content_type not in {"image/png", "image/jpeg", "image/bmp", "image/webp"}:
                raise HTTPException(status_code=415, detail=f"{upload.filename} is not a supported image.")
            content = await upload.read()
            if len(content) > 5 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds the 5 MB limit.")
            image = Image.open(io.BytesIO(content))
            probabilities = model.predict(prepare_mnist_digit(image), verbose=0)[0]
            digit = int(probabilities.argmax())
            predictions.append(
                {
                    "position": position,
                    "filename": upload.filename,
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
