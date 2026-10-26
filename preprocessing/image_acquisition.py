"""
Task 1 - Image Acquisition

Gets the input "number" image in one of two ways:
  1. Automatically build it from a folder of individual digit images.
  2. Load a number image directly from a single image file.
"""

import re
from pathlib import Path
from PIL import Image, ImageOps


def load_number_image(path: str) -> Image.Image:
    """Load a number image directly from a file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find image at {path}")
    with Image.open(path) as image:
        # phone cameras often save photos sideways with a rotation tag instead
        # of actually rotating the pixels, so apply that tag here
        upright = ImageOps.exif_transpose(image)
        # read the pixels now so the file isn't left open afterwards
        upright.load()
    return upright


def _natural_sort_key(path: Path):
    """Sort key that reads the numbers in a filename as numbers, so digit_10.png
    comes after digit_9.png instead of straight after digit_1.png."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def build_number_from_digits(folder: str, height: int = 28, spacing: int = 4) -> Image.Image:
    """
    Combine individual digit images in a folder into one horizontal number image.
    Files are read in filename order, so name them so that order matches the
    number, e.g. digit_0.png, digit_1.png, digit_2.png -> "012". Numbers in the
    names are compared as numbers, so digit_10.png still comes after digit_9.png.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Cannot find folder at {folder}")

    paths = sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
        key=_natural_sort_key,
    )
    if not paths:
        raise FileNotFoundError(f"No digit images found in {folder}")

    digits = []
    for p in paths:
        img = Image.open(p).convert("L")
        ratio = height / img.height
        new_width = max(1, round(img.width * ratio))
        digits.append(img.resize((new_width, height), Image.LANCZOS))

    total_width = sum(d.width for d in digits) + spacing * (len(digits) - 1)
    canvas = Image.new("L", (total_width, height), color=255)

    x = 0
    for d in digits:
        canvas.paste(d, (x, 0))
        x += d.width + spacing

    return canvas


def main(argv=None) -> Path:
    import argparse

    parser = argparse.ArgumentParser(description="Get a number image from a file or a folder of digit images.")
    parser.add_argument("mode", choices=["file", "folder"],
                        help="file = load one image, folder = join the digit images in a folder")
    parser.add_argument("target", help="Path to the image file or the digit folder")
    parser.add_argument("--output", default="acquired_number.png")
    parser.add_argument("--height", type=int, default=28, help="Digit height when joining a folder")
    parser.add_argument("--spacing", type=int, default=4, help="Gap between digits when joining a folder")
    args = parser.parse_args(argv)

    try:
        if args.mode == "file":
            img = load_number_image(args.target)
        else:
            img = build_number_from_digits(args.target, height=args.height, spacing=args.spacing)
    except FileNotFoundError as error:
        raise SystemExit(f"Error: {error}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved: {out_path}")
    print(f"Size: {img.size}, mode: {img.mode}")
    return out_path


if __name__ == "__main__":
    main()
