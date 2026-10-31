"""Keras model definitions for the MNIST model comparison."""

from __future__ import annotations


def _keras():
    """Only import TensorFlow when model code is actually used, so the other tools stay light."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training/inference. Run: pip install -r requirements.txt"
        ) from exc
    return keras


def build_mlp(num_classes: int = 10):
    """A simple dense baseline, so that the CNN has something fair to be compared against."""
    keras = _keras()
    # Flattening throws away the image layout, which is what makes this a plain dense baseline.
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


def build_mlp_large(num_classes: int = 10):
    """A two-layer MLP, used to check whether more capacity helps on EMNIST."""
    keras = _keras()
    return keras.Sequential(
        [
            keras.layers.Input((28, 28, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="mlp_large",
    )


def build_cnn(num_classes: int = 10):
    """The CNN we picked for handwritten digit recognition."""
    keras = _keras()
    # num_classes is 10 for MNIST and 62 when the same layout is trained on EMNIST.
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
    """LeNet-5 style model, adapted from 32x32 inputs to 28x28 MNIST.

    The original used 32x32 inputs, tanh and average pooling. Here the first
    convolution uses same padding, so the feature maps keep their layout on 28x28.
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
    """Two convolutions plus a shortcut (identity, or projected when the shape changes)."""
    shortcut = inputs
    x = keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    if stride != 1 or inputs.shape[-1] != filters:
        # Make the shortcut the same size as the convolution path before adding them.
        shortcut = keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False)(inputs)
        shortcut = keras.layers.BatchNormalization()(shortcut)
    x = keras.layers.Add()([x, shortcut])
    return keras.layers.Activation("relu")(x)


def build_resnet(num_classes: int = 10):
    """Small residual network: two blocks at 16 channels and two at 32.

    Full-size ResNets are made for much bigger images. This one still has real
    skip connections, just fewer blocks and filters so that it trains on a CPU.
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
    """Compile a model with the settings both experiments share."""
    keras = _keras()
    # All four models predict a single class index for each image.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
