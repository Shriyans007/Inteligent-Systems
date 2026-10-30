"""
Task 1 - Test image helper

Downloads a handful of MNIST digit images (via scikit-learn's fetch_openml)
and saves them to disk, so preprocessing.py and image_acquisition.py have
real images to run against.
"""

from pathlib import Path
import numpy as np
from PIL import Image

# next to this file, where the committed sample_digits folder already is
OUT_DIR = Path(__file__).resolve().parent / "sample_digits"


def save_sample_digits(n: int = 10):
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    images = mnist.data.reshape(-1, 28, 28).astype(np.uint8)
    labels = mnist.target

    OUT_DIR.mkdir(exist_ok=True)

    for i in range(n):
        img = Image.fromarray(images[i])
        label = labels[i]
        img.save(OUT_DIR / f"digit_{i}_label{label}.png")

    print(f"Saved {n} images to {OUT_DIR}/")


if __name__ == "__main__":
    save_sample_digits(10)
