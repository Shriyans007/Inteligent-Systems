"""FastAPI service for handwritten digit and character predictions."""

from __future__ import annotations

import io
import json
import os
import base64
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

from cnn_model.models import _keras
from preprocessing import prepare_mnist_digit
from segmentation import crops_to_model_input, segment
from text_recognition.common import CHAR_MODEL, prepare_character
from text_recognition.emnist import model_paths
from text_recognition.refine import refine_split_components
from text_recognition.layout import Glyph, segment_text
from segmentation.unet_resnet import UNET_MODEL

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


def predict_probabilities(classifier, model_key: str, batch: np.ndarray,
                          num_classes: int = 10) -> np.ndarray:
    """Return an (n, num_classes) probability array, whichever library the model comes from."""
    if model_key in SKLEARN_MODELS:
        # The SVM was trained on flat rows of 784 pixels, not 28x28x1 images.
        scores = classifier.predict_proba(batch.reshape(len(batch), -1))
        # predict_proba orders its columns by classes_, so place each one under its class.
        probabilities = np.zeros((len(batch), num_classes), dtype=np.float32)
        probabilities[:, classifier.classes_.astype(int)] = scores
        return probabilities
    return classifier.predict(batch, verbose=0)


def predicted_digits(classifier, model_key: str, batch: np.ndarray,
                     probabilities: np.ndarray) -> np.ndarray:
    """Use the same SVM decision rule as its reported test accuracy."""
    if model_key in SKLEARN_MODELS:
        return classifier.predict(batch.reshape(len(batch), -1)).astype(int)
    return probabilities.argmax(axis=1)


# Segmentation methods the GUI can choose: each one is a set of options for segment().
SEGMENTERS = {
    "components_split": ("Connected components + split touching",
                         {"method": "connected_components", "split_touching": True}),
    "components": ("Connected components", {"method": "connected_components", "split_touching": False}),
    "contours_split": ("Contours + split touching", {"method": "contours", "split_touching": True}),
    "contours": ("Contours", {"method": "contours", "split_touching": False}),
    # Not rule-based: a trained CNN predicts where one digit ends and the next begins.
    "learned_cuts": ("Learned cut-point CNN", {"method": "learned"}),
}
SEGMENTER_MODEL_PATH = Path("artifacts/models/segmenter/cutpoint_cnn.keras")
_segmenter_models = {}


def segmenter_available(key: str) -> bool:
    """Rule-based methods always work; the learned one needs its trained model file."""
    return SEGMENTERS[key][1]["method"] != "learned" or SEGMENTER_MODEL_PATH.is_file()


def run_segmenter(image, options: dict, details: dict | None = None):
    """Cut an image into character crops with the chosen method."""
    if options["method"] != "learned":
        return segment(image, **options)
    if not SEGMENTER_MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Train the segmentation model first: "
                                                    "python -m segmentation_model.train")
    if "cutpoint" not in _segmenter_models:
        # compile=False: the model is only used for prediction, so its custom loss is not needed.
        _segmenter_models["cutpoint"] = _keras().models.load_model(SEGMENTER_MODEL_PATH, compile=False)
    from segmentation_model.segment import segment_learned
    return segment_learned(image, _segmenter_models["cutpoint"], details=details)
# Each route keeps the segmentation it used before the choice existed.
DEFAULT_SEGMENTATION = {"numbers": "components_split", "characters": "components"}


def segmentation_options(choice: str | None, route: str) -> tuple[str, dict]:
    """Return the chosen segmenter's key and its segment() options."""
    key = choice or DEFAULT_SEGMENTATION[route]
    if key not in SEGMENTERS:
        raise HTTPException(status_code=400, detail="Unknown segmentation method.")
    return key, SEGMENTERS[key][1]


@app.on_event("startup")
def warm_up_model() -> None:
    """Load the trained CNN when the API starts so the first request is faster."""
    if MODEL_PATHS[DEFAULT_MODEL].is_file():
        model = get_model(DEFAULT_MODEL)
        batch = np.zeros((1, 28, 28, 1), dtype=np.float32)
        predict_probabilities(model, DEFAULT_MODEL, batch)


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


@app.get("/api/segmenters")
def segmenters() -> dict:
    """Let the GUI list the segmentation methods and each route's default."""
    return {"defaults": DEFAULT_SEGMENTATION,
            "segmenters": [{"key": key, "label": label, "available": segmenter_available(key)}
                           for key, (label, _) in SEGMENTERS.items()]}


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...), model: str = Form(DEFAULT_MODEL),
                  segmentation: str | None = Form(None)) -> dict:
    """Recognise one digit or an ordered list of pre-segmented digit images."""
    segmenter, segment_options = segmentation_options(segmentation, "numbers")
    if not 1 <= len(files) <= 30:
        raise HTTPException(status_code=400, detail="Upload between 1 and 30 digit images.")

    try:
        classifier = get_model(model)
        prepared_images = []
        filenames = []
        boxes, image_info = [], []
        for upload in files:
            if upload.content_type not in {"image/png", "image/jpeg", "image/bmp", "image/webp"}:
                raise HTTPException(status_code=415, detail=f"{upload.filename} is not a supported image.")
            content = await upload.read()
            if len(content) > 5 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds the 5 MB limit.")
            image = Image.open(io.BytesIO(content))
            # The segmenter handles a whole number; each crop then gets the
            # same 28x28 input preparation as an uploaded single digit.
            details = {}
            crops = run_segmenter(image, segment_options, details)
            # The upload's size and each piece's box let the GUI draw the segmentation.
            image_info.append({"filename": upload.filename, "width": image.width, "height": image.height,
                               **details})
            if crops:
                prepared_images.extend(crops_to_model_input(crops))
                filenames.extend(f"{upload.filename} #{crop.index + 1}" for crop in crops)
                boxes.extend((len(image_info) - 1, crop.bbox) for crop in crops)
            else:
                # Some MNIST-style images have a dark full-image background,
                # so segmentation may correctly find no separate dark region.
                prepared_images.append(prepare_mnist_digit(image))
                filenames.append(upload.filename)
                boxes.append((len(image_info) - 1, (0, 0, image.width, image.height)))

        # Run one batch prediction rather than calling TensorFlow once per file.
        batch = np.concatenate(prepared_images, axis=0)
        batch_probabilities = predict_probabilities(classifier, model, batch)
        digits = predicted_digits(classifier, model, batch, batch_probabilities)
        predictions = []
        for position, (filename, probabilities, predicted, (image_index, bbox)) in enumerate(
                zip(filenames, batch_probabilities, digits, boxes)):
            digit = int(predicted)
            predictions.append(
                {
                    "position": position,
                    "filename": filename,
                    "image": image_index,
                    "bbox": [int(value) for value in bbox],
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
        "segmentation": segmenter,
        "segmentation_label": SEGMENTERS[segmenter][0],
        "number": "".join(str(item["digit"]) for item in predictions),
        "average_confidence": sum(item["confidence"] for item in predictions) / len(predictions),
        "lowest_confidence": min(item["confidence"] for item in predictions),
        "images": image_info,
        "predictions": predictions,
    }


def character_paths(model_key: str):
    """Model and mapping files for a letters-and-digits model."""
    _, model_path, mapping_path = model_paths(model_key)
    if model_key in SKLEARN_MODELS:
        # svm_model.train_emnist saves the SVM with joblib beside its mapping.
        model_path = model_path.with_suffix(".joblib")
    return model_path, mapping_path


CHARACTER_PATHS = {key: character_paths(key) for key in MODEL_PATHS}
for key in ("cnn_case_balanced", "cnn_light_aug", "cnn_deep_aug", "mlp_large"):
    CHARACTER_PATHS[key] = model_paths(key)[1:]
CHARACTER_PATHS["cnn_overlap"] = (CHAR_MODEL.with_name("emnist_cnn_overlap.keras"), model_paths("cnn")[2])
CHARACTER_PATHS["cnn_unet_tight"] = (CHAR_MODEL.with_name("emnist_cnn_unet_tight.keras"), model_paths("cnn")[2])
CHARACTER_LABELS = {**MODEL_LABELS, "cnn_case_balanced": "Case-balanced CNN (experimental)",
                    "cnn_light_aug": "CNN with light augmentation (experimental)",
                    "cnn_deep_aug": "Deeper CNN with augmentation (experimental)",
                    "mlp_large": "Larger MLP (256 + 128)",
                    "cnn_overlap": "Overlap fine-tuned CNN (experimental)",
                    "cnn_unet_tight": "CNN for tight U-Net crops (experimental)"}
_extension_models = {}


def crop_data_url(image: Image.Image) -> str:
    """Encode a small glyph preview in the response; no image is stored on disk."""
    buffer = io.BytesIO()
    image.convert("L").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/api/extension-models")
def extension_models() -> dict:
    return {"character_models": [{"key": key, "label": CHARACTER_LABELS[key],
                                  "available": path.is_file() and meta.is_file()}
                                 for key, (path, meta) in CHARACTER_PATHS.items()],
            "segmenters": {"classical": True, "unet": UNET_MODEL.is_file(),
                           "unet_tight": UNET_MODEL.is_file()}}


def get_extension_model(model_key="cnn"):
    if model_key not in CHARACTER_PATHS:
        raise HTTPException(status_code=400, detail="Unknown character model selection.")
    path, metadata = CHARACTER_PATHS[model_key]
    if not path.is_file() or not metadata.is_file():
        raise HTTPException(status_code=503, detail=f"Train the character {model_key} model first; expected {path} and {metadata}.")
    if model_key not in _extension_models:
        if model_key in SKLEARN_MODELS:
            import joblib
            classifier = joblib.load(path)
        else:
            classifier = _keras().models.load_model(path)
        _extension_models[model_key] = (classifier, json.loads(metadata.read_text()))
    return _extension_models[model_key]


@app.post("/api/recognise-text")
async def recognise_text(file: UploadFile = File(...), model: str = Form("cnn"),
                         mode: str | None = Form(None),
                         segmenter: str | None = Form(None),
                         segmentation: str | None = Form(None),
                         show_crops: bool = Form(False)) -> dict:
    if mode is not None and mode != "character":
        raise HTTPException(status_code=400, detail="Only character recognition is available.")
    method = segmenter or os.getenv("HNRS_CHARACTER_SEGMENTER", "classical")
    if method not in {"classical", "unet", "unet_tight"}:
        raise HTTPException(status_code=400, detail="Choose classical, unet or unet_tight character segmentation.")
    if method in {"unet", "unet_tight"} and not UNET_MODEL.is_file():
        raise HTTPException(status_code=503, detail=f"Train the full U-Net first; expected {UNET_MODEL}. The smoke model is not used by the GUI.")
    if segmentation is not None and segmenter is not None:
        raise HTTPException(status_code=400, detail="Choose one character segmentation method.")
    if segmentation is not None:
        legacy_key, legacy_options = segmentation_options(segmentation, "characters")
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
            if segmentation is None:
                glyphs = segment_text(image, method=method)
            else:
                # Preserve the digit-style segmentation experiment as an explicit
                # API option while the GUI uses the text-aware character routes.
                glyphs = [Glyph(Image.fromarray(crop.image), crop.bbox)
                          for crop in run_segmenter(image, legacy_options)]
            if not glyphs:
                raise ValueError("The image does not contain visible writing.")
            if len(glyphs) > 30:
                raise ValueError("Draw no more than 30 separated characters in one image.")
            prepared = [prepare_character(glyph.image) for glyph in glyphs]
            batch = np.stack(prepared)
            probabilities = predict_probabilities(classifier, model, batch, num_classes=len(mapping))
            if segmentation is None and method in {"unet", "unet_tight"} and model not in SKLEARN_MODELS:
                # Tight strips can split a broad single glyph when U-Net puts
                # two centres inside one connected stroke. Only join adjacent
                # low-confidence pieces when their complete ink component is
                # confidently classified as one character.
                repair_options = ({"max_width_ratio": 1.15,
                                   "min_merge_confidence": 0.85,
                                   "min_confidence_gain": 0.15,
                                   "max_first_confidence": 0.85}
                                  if method == "unet_tight" else {})
                glyphs, prepared, probabilities = refine_split_components(
                    image, glyphs, prepared, probabilities, classifier, **repair_options)
                batch = np.stack(prepared)
        indexes = predicted_digits(classifier, model, batch, probabilities)
        predictions = []
        for position, (glyph, scores, index) in enumerate(zip(glyphs, probabilities, indexes)):
            index = int(index)
            item = {"position": position, "character": mapping[str(index)],
                    "confidence": float(scores[index]), "before": glyph.before, "bbox": glyph.bbox}
            if show_crops:
                item["crop_preview"] = crop_data_url(glyph.image)
                item["model_input_preview"] = crop_data_url(
                    Image.fromarray(np.uint8(np.clip(prepared[position][..., 0], 0, 1) * 255)))
            predictions.append(item)
        return {"mode": "character", "model": model, "model_label": CHARACTER_LABELS[model],
                "segmenter": method, "glyph_count": len(glyphs),
                "segmentation": legacy_key if segmentation is not None else method,
                "segmentation_label": SEGMENTERS[legacy_key][0] if segmentation is not None else
                                      ("U-Net tight crops" if method == "unet_tight" else
                                       "U-Net" if method == "unet" else "Classical"),
                "text": "".join(item["before"] + item["character"] for item in predictions),
                "confidence": sum(item["confidence"] for item in predictions)/len(predictions),
                "predictions": predictions}
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
