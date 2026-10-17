"""Image preparation shared by EMNIST training and the API."""
from pathlib import Path

CHAR_MODEL = Path("artifacts/text/character/emnist_cnn.keras")
CHAR_MAPPING = Path("artifacts/text/character/mapping.json")


def prepare_character(image):
    # EMNIST is white ink on black, as is the existing MNIST digit model.
    from preprocessing import prepare_mnist_digit
    return prepare_mnist_digit(image)[0]
