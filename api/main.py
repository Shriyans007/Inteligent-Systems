"""FastAPI service for handwritten digit predictions."""

from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

from cnn_model.models import _keras
from preprocessing import prepare_mnist_digit
from segmentation import crops_to_model_input, segment

MODEL_PATHS = {
    "cnn": Path("artifacts/cnn_mnist.keras"),
    "mlp": Path("artifacts/mlp_mnist.keras"),
    "lenet5": Path("artifacts/models/lenet5/lenet5_mnist.keras"),
    "resnet": Path("artifacts/models/resnet/resnet_mnist.keras"),
}
MODEL_LABELS = {"cnn": "Shallow CNN", "mlp": "MLP", "lenet5": "LeNet-5", "resnet": "Small ResNet"}
# Keep the previously documented environment override for known models.
_configured_path = Path(os.getenv("HNRS_MODEL_PATH", str(MODEL_PATHS["cnn"])))
DEFAULT_MODEL = next((key for key, path in MODEL_PATHS.items() if path == _configured_path), "cnn")

app = FastAPI(title="HNRS Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_models = {}


def get_model(model_key: str = DEFAULT_MODEL):
    """Load only known saved models and reuse each instance on later requests."""
    if model_key not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Unknown model selection.")
    path = MODEL_PATHS[model_key]
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"{MODEL_LABELS[model_key]} is not trained at {path}.")
    if model_key not in _models:
        _models[model_key] = _keras().models.load_model(path)
    return _models[model_key]


@app.on_event("startup")
def warm_up_model() -> None:
    """Load the trained CNN when the API starts so the first request is faster."""
    if MODEL_PATHS[DEFAULT_MODEL].is_file():
        model = get_model(DEFAULT_MODEL)
        model.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    """Report API availability separately from model availability."""
    return {"status": "ok", "model_ready": MODEL_PATHS[DEFAULT_MODEL].is_file(),
            "model_path": str(MODEL_PATHS[DEFAULT_MODEL])}


@app.get("/api/models")
def available_models() -> dict:
    """Let the GUI disable models that have not been trained locally."""
    return {"default_model": DEFAULT_MODEL,
            "models": [{"key": key, "label": MODEL_LABELS[key], "available": path.is_file()}
                       for key, path in MODEL_PATHS.items()]}


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...), model: str = Form(DEFAULT_MODEL)) -> dict:
    """Recognise one digit or an ordered list of pre-segmented digit images."""
    if not 1 <= len(files) <= 30:
        raise HTTPException(status_code=400, detail="Upload between 1 and 30 digit images.")

    try:
        classifier = get_model(model)
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
        batch_probabilities = classifier.predict(batch, verbose=0)
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
        "model": model,
        "model_label": MODEL_LABELS[model],
        "number": "".join(str(item["digit"]) for item in predictions),
        "average_confidence": sum(item["confidence"] for item in predictions) / len(predictions),
        "lowest_confidence": min(item["confidence"] for item in predictions),
        "predictions": predictions,
    }
