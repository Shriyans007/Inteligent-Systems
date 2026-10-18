"""FastAPI service for handwritten digit and character predictions."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

from cnn_model.models import _keras
from preprocessing import prepare_mnist_digit
from segmentation import crops_to_model_input, segment
from text_recognition.common import prepare_character
from text_recognition.emnist import model_paths

# These paths belong to the original ten-digit classifiers.
MODEL_PATHS = {
    "cnn": Path("artifacts/cnn_mnist.keras"),
    "mlp": Path("artifacts/mlp_mnist.keras"),
    "lenet5": Path("artifacts/models/lenet5/lenet5_mnist.keras"),
    "resnet": Path("artifacts/models/resnet/resnet_mnist.keras"),
    "svm": Path("artifacts/models/svm/svm_mnist.joblib"),
}
MODEL_LABELS = {"cnn": "Shallow CNN", "mlp": "MLP", "lenet5": "LeNet-5", "resnet": "Small ResNet",
                "svm": "SVM (RBF)"}
# These come from scikit-learn, not Keras: they load with joblib and take flat 784-pixel input.
SKLEARN_MODELS = {"svm"}

ACCURACY_SOURCES = {
    "cnn": (Path("artifacts/comparison.json"), "CNN selected"),
    "mlp": (Path("artifacts/comparison.json"), "MLP baseline"),
    "lenet5": (Path("artifacts/models/lenet5/result.json"), None),
    "resnet": (Path("artifacts/models/resnet/result.json"), None),
    "svm": (Path("artifacts/models/svm/result.json"), None),
}


def model_accuracy(model_key: str):
    """Read the saved test accuracy so the GUI can set a threshold from it."""
    path, label = ACCURACY_SOURCES.get(model_key, (None, None))
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    records = data if isinstance(data, list) else [data]
    for row in records:
        if label is None or row.get("model") == label:
            value = row.get("test_accuracy")
            return float(value) if value is not None else None
    return None

# Make sure you keep the previously documented environment override for known models.
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

# Loading a Keras file takes time, so keep each model around after its first use.
_models = {}


def get_model(model_key: str = DEFAULT_MODEL):
    """Load only known saved models and reuse each instance on later requests."""
    if model_key not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Unknown model selection.")
    path = MODEL_PATHS[model_key]
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"{MODEL_LABELS[model_key]} is not trained at {path}.")
    if model_key not in _models:
        if model_key in SKLEARN_MODELS:
            import joblib
            _models[model_key] = joblib.load(path)
        else:
            _models[model_key] = _keras().models.load_model(path)
    return _models[model_key]


def predict_probabilities(classifier, model_key: str, batch: np.ndarray) -> np.ndarray:
    """Return an (n, 10) probability array, whichever library the model comes from."""
    if model_key in SKLEARN_MODELS:
        # The SVM was trained on flat rows of 784 pixels, not 28x28x1 images.
        scores = classifier.predict_proba(batch.reshape(len(batch), -1))
        # predict_proba orders its columns by classes_, so place each one under its digit.
        probabilities = np.zeros((len(batch), 10), dtype=np.float32)
        probabilities[:, classifier.classes_.astype(int)] = scores
        return probabilities
    return classifier.predict(batch, verbose=0)


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
            "models": [{"key": key, "label": MODEL_LABELS[key], "available": path.is_file(),
                        "accuracy": model_accuracy(key)}
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
            # The segmenter handles a whole number; each crop then gets the
            # same 28x28 input preparation as an uploaded single digit.
            crops = segment(image, split_touching=True)
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
        batch_probabilities = predict_probabilities(classifier, model, batch)
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


# The SVM only knows digits, so it is not offered as a letter-and-digit model.
CHARACTER_PATHS = {key: model_paths(key)[1:] for key in MODEL_PATHS if key not in SKLEARN_MODELS}
_extension_models = {}


@app.get("/api/extension-models")
def extension_models() -> dict:
    return {"character_models": [{"key": key, "label": MODEL_LABELS[key],
                                  "available": path.is_file() and meta.is_file()}
                                 for key, (path, meta) in CHARACTER_PATHS.items()]}


def get_extension_model(model_key="cnn"):
    if model_key not in CHARACTER_PATHS:
        raise HTTPException(status_code=400, detail="Unknown character model selection.")
    path, metadata = CHARACTER_PATHS[model_key]
    if not path.is_file() or not metadata.is_file():
        raise HTTPException(status_code=503, detail=f"Train the character {model_key} model first; expected {path} and {metadata}.")
    if model_key not in _extension_models:
        _extension_models[model_key] = (_keras().models.load_model(path), json.loads(metadata.read_text()))
    return _extension_models[model_key]


@app.post("/api/recognise-text")
async def recognise_text(file: UploadFile = File(...), model: str = Form("cnn"),
                         mode: str | None = Form(None)) -> dict:
    if mode is not None and mode != "character":
        raise HTTPException(status_code=400, detail="Only character recognition is available.")
    if file.content_type not in {"image/png", "image/jpeg", "image/bmp", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, BMP or WebP image.")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The image exceeds the 5 MB limit.")
    try:
        classifier, mapping = get_extension_model(model)
        # Verify the upload before opening it again for preprocessing.
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            crops = segment(image)
            if not crops:
                raise ValueError("The image does not contain visible writing.")
            if len(crops) > 30:
                raise ValueError("Draw no more than 30 separated characters in one image.")
            # Use the full image for one character, or classify separated
            # crops in left-to-right order for a longer drawing.
            prepared = [prepare_character(image)] if len(crops) == 1 else [prepare_character(Image.fromarray(c.image)) for c in crops]
        probabilities = classifier.predict(np.stack(prepared), verbose=0)
        predictions = []
        for position, scores in enumerate(probabilities):
            index = int(np.argmax(scores))
            predictions.append({"position": position, "character": mapping[str(index)], "confidence": float(scores[index])})
        return {"mode": "character", "model": model, "model_label": MODEL_LABELS[model],
                "text": "".join(item["character"] for item in predictions),
                "confidence": sum(item["confidence"] for item in predictions)/len(predictions),
                "predictions": predictions}
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
