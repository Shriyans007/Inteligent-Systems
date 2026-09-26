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


def build_lenet5(num_classes: int = 10):
    """LeNet-5 style classifier adapted from 32x32 to 28x28 MNIST inputs.

    The original used 32x32 inputs, tanh and average pooling. Here, same
    padding on the first convolution retains its feature-map layout on 28x28.
    """
    keras = _keras()
    return keras.Sequential(
        [
            keras.layers.Input((28, 28, 1)),
            keras.layers.Conv2D(6, 5, padding="same", activation="tanh"),
            keras.layers.AveragePooling2D(pool_size=2),
            keras.layers.Conv2D(16, 5, activation="tanh"),
            keras.layers.AveragePooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(120, activation="tanh"),
            keras.layers.Dense(84, activation="tanh"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="lenet5",
    )


def _residual_block(keras, inputs, filters: int, stride: int = 1):
    """Two convolutions plus an identity or projected shortcut."""
    shortcut = inputs
    x = keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False)(inputs)
        shortcut = keras.layers.BatchNormalization()(shortcut)
    x = keras.layers.Add()([x, shortcut])
    return keras.layers.Activation("relu")(x)


def build_resnet(num_classes: int = 10):
    """Small residual network: two blocks at 16 and two at 32 channels.

    Full-size ResNets are designed for much larger images. This uses genuine
    skip connections, but fewer blocks and filters for MNIST on a CPU.
    """
    keras = _keras()
    inputs = keras.layers.Input((28, 28, 1))
    x = keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = _residual_block(keras, x, 16)
    x = _residual_block(keras, x, 16)
    x = _residual_block(keras, x, 32, stride=2)
    x = _residual_block(keras, x, 32)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="small_resnet")


def compile_model(model, learning_rate: float = 1e-3):
    """Compile a classifier with settings shared by both experiments."""
    keras = _keras()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
