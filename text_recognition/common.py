"""Image preparation shared by training and the API."""
from pathlib import Path
import numpy as np
from PIL import Image

CHAR_MODEL = Path("artifacts/text/character/emnist_cnn.keras")
WORD_MODEL = Path("artifacts/text/word/iam_crnn.keras")
CHAR_MAPPING = Path("artifacts/text/character/mapping.json")
WORD_VOCAB = Path("artifacts/text/word/vocabulary.json")
# Words need room across the image; a whole word should not be shrunk to 28x28.
HEIGHT, WIDTH = 64, 384


def prepare_character(image):
    # EMNIST is white ink on black, as is the existing MNIST digit model.
    from preprocessing import prepare_mnist_digit
    return prepare_mnist_digit(image)[0]


def prepare_word(image, width=WIDTH):
    """Keep the word's proportions; add background padding to a fixed canvas."""
    gray = image.convert("L")
    pixels = np.asarray(gray)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if np.median(border) < 128:  # dark background, bright ink
        gray = Image.fromarray(255 - pixels)
        pixels = np.asarray(gray)
    mask = pixels < 235
    if not mask.any():
        raise ValueError("The image does not contain visible writing.")
    yy, xx = np.where(mask)
    # Crop empty margins first, then resize without stretching the writing.
    cropped = gray.crop((max(0, xx.min()-3), max(0, yy.min()-3),
                         min(gray.width, xx.max()+4), min(gray.height, yy.max()+4)))
    scale = min((width-16)/cropped.width, (HEIGHT-12)/cropped.height)
    size = (max(1, round(cropped.width*scale)), max(1, round(cropped.height*scale)))
    resized = cropped.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, HEIGHT), 255)
    canvas.paste(resized, (8, (HEIGHT-size[1])//2))
    # The network sees bright ink on a dark background after this inversion.
    return (1 - np.asarray(canvas, dtype=np.float32)/255)[..., None]
