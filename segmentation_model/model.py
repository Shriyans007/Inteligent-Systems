"""The cut-point CNN: for every column of a number image, how likely is a cut between two digits?"""

from __future__ import annotations

from cnn_model.models import _keras

from .data import HEIGHT

POSITIVE_WEIGHT = 4.0    # cut columns are now about 11% of columns (wide gaps), so a smaller boost


def build_cutpoint_model():
    keras = _keras()
    layers = keras.layers
    inputs = keras.Input((HEIGHT, None, 1))   # any width, so numbers of any length fit
    x = inputs
    for filters in (32, 64, 64):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 1))(x)   # halve the height only, so every column survives
    # The remaining 4 rows are summarised into one feature vector per column.
    x = layers.Conv2D(64, (HEIGHT // 8, 1), activation="relu")(x)
    # Wide kernels along the row let each column see its neighbours on both sides.
    x = layers.Conv2D(64, (1, 9), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (1, 9), padding="same", dilation_rate=(1, 2), activation="relu")(x)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)   # shape (1, width, 1): cut probability
    return keras.Model(inputs, outputs, name="cutpoint_cnn")


def weighted_binary_crossentropy(y_true, y_pred):
    """Binary cross-entropy where a missed cut column counts POSITIVE_WEIGHT times more."""
    ops = _keras().ops
    y_pred = ops.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -(POSITIVE_WEIGHT * y_true * ops.log(y_pred) + (1 - y_true) * ops.log(1 - y_pred))
    return ops.mean(loss)


def compile_cutpoint_model(model, learning_rate: float = 1e-3):
    keras = _keras()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=weighted_binary_crossentropy)
    return model
