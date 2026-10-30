"""
Task 1 - Figures for the preprocessing and image acquisition section

Makes two pictures from the sample digits in preprocessing/sample_digits:
  preprocessing_steps.png  - the same digits with grayscale only, the fixed
                             threshold and Otsu's threshold side by side
  acquisition_example.png  - separate digit images and the number image that
                             build_number_from_digits() joins them into

Run from the repo root:
  python3 -m preprocessing.report_figures
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preprocessing.image_acquisition import build_number_from_digits
from preprocessing.preprocessing import preprocess_pipeline

HERE = Path(__file__).resolve().parent
VARIANTS = [
    ("Grayscale only", {"binarize": False}),
    ("Fixed threshold (128)", {"binarize": True, "method": "fixed"}),
    ("Otsu threshold", {"binarize": True, "method": "otsu"}),
]
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


def _font(size: int = 14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # older Pillow only has the small fixed-size font
        return ImageFont.load_default()


def _digit_paths(folder: Path, count: int) -> list:
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_TYPES)
    if not paths:
        raise FileNotFoundError(f"No digit images found in {folder}")
    return paths[:count]


def _to_image(array: np.ndarray, scale: int) -> Image.Image:
    # nearest neighbour so each of the 28x28 pixels stays a clear square
    image = Image.fromarray((array * 255).astype(np.uint8))
    return image.resize((image.width * scale, image.height * scale), Image.NEAREST)


def preprocessing_steps_figure(folder, output, count: int = 4, scale: int = 5) -> Path:
    paths = _digit_paths(folder, count)
    cell = 28 * scale
    header, gap = 30, 10
    width = len(VARIANTS) * (cell + gap) + gap
    height = header + len(paths) * (cell + gap) + gap
    sheet = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(sheet)
    font = _font()

    for col, (name, options) in enumerate(VARIANTS):
        x = gap + col * (cell + gap)
        draw.text((x, 8), name, fill=0, font=font)
        for row, path in enumerate(paths):
            with Image.open(path) as digit:
                processed = preprocess_pipeline(digit, size=(28, 28), **options)
            y = header + gap + row * (cell + gap)
            sheet.paste(_to_image(processed, scale), (x, y))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def acquisition_figure(folder, output, count: int = 4, height: int = 56, spacing: int = 8) -> Path:
    paths = _digit_paths(folder, count)
    with tempfile.TemporaryDirectory() as tmp:
        # copy just the chosen digits so the joined number matches the top row
        for i, path in enumerate(paths):
            shutil.copy(path, Path(tmp) / f"digit_{i:02d}{path.suffix.lower()}")
        number = build_number_from_digits(tmp, height=height, spacing=spacing)

    tiles = []
    for path in paths:
        with Image.open(path) as digit:
            tiles.append(digit.convert("L").resize((height, height), Image.LANCZOS))

    font = _font()
    pad, label = 12, 24
    width = max(number.width, len(tiles) * (height + pad) - pad) + 2 * pad
    total_height = 2 * label + 2 * height + 3 * pad
    sheet = Image.new("L", (width, total_height), color=200)
    draw = ImageDraw.Draw(sheet)

    draw.text((pad, pad // 2), "Digit images in the folder", fill=0, font=font)
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (pad + i * (height + pad), label + pad // 2))
    y = label + height + pad
    draw.text((pad, y), "Joined by build_number_from_digits()", fill=0, font=font)
    sheet.paste(number, (pad, y + label))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the figures for the preprocessing and acquisition section.")
    parser.add_argument("--digits", type=Path, default=HERE / "sample_digits")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--count", type=int, default=4, help="How many sample digits to use")
    args = parser.parse_args()

    try:
        steps = preprocessing_steps_figure(args.digits, args.output / "preprocessing_steps.png", args.count)
        acquisition = acquisition_figure(args.digits, args.output / "acquisition_example.png", args.count)
    except FileNotFoundError as error:
        raise SystemExit(f"Error: {error}")
    print(f"Saved {steps}")
    print(f"Saved {acquisition}")


if __name__ == "__main__":
    main()
