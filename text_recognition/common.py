"""Image preparation shared by EMNIST training and the API."""
from pathlib import Path

CHAR_MODEL = Path("artifacts/text/character/emnist_cnn.keras")
CHAR_MAPPING = Path("artifacts/text/character/mapping.json")


def prepare_character(image):
    # EMNIST is white ink on black, the same as the MNIST images the digit models use.
    from preprocessing import prepare_mnist_digit
    return prepare_mnist_digit(image)[0]
