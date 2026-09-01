"""Keras model definitions used for the controlled MNIST comparison."""

from __future__ import annotations


def _keras():
    """Import TensorFlow only when model code is used, keeping utilities lightweight."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training/inference. Run: pip install -r requirements.txt"
        ) from exc
    return keras


def build_mlp(num_classes: int = 10):
    """Return a simple dense baseline for a fair comparison with the CNN."""
    keras = _keras()
    return keras.Sequential(
        [
            keras.layers.Input((28, 28, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="mlp_baseline",
    )


def build_cnn(num_classes: int = 10):
    """Return the selected CNN architecture for handwritten digit recognition."""
    keras = _keras()
    return keras.Sequential(
        [
            keras.layers.Input((28, 28, 1)),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.40),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_selected",
    )


def compile_model(model, learning_rate: float = 1e-3):
    """Compile a classifier with settings shared by both experiments."""
    keras = _keras()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
