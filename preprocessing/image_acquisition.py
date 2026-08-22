"""
Task 1 - Image Acquisition

Gets the input "number" image in one of two ways:
  1. Automatically build it from a folder of individual digit images.
  2. Load a number image directly from a single image file.
"""

from pathlib import Path
from PIL import Image


def load_number_image(path: str) -> Image.Image:
    """Load a number image directly from a file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find image at {path}")
    return Image.open(path)


def build_number_from_digits(folder: str, height: int = 28, spacing: int = 4) -> Image.Image:
    """
    Combine individual digit images in a folder into one horizontal number image.
    Files are read in sorted filename order, so name them so that order matches
    the number, e.g. digit_0.png, digit_1.png, digit_2.png -> "012".
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Cannot find folder at {folder}")

    paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python image_acquisition.py file <path_to_image>")
        print("  python image_acquisition.py folder <path_to_digit_folder>")
        sys.exit(1)

    mode, target = sys.argv[1], sys.argv[2]

    if mode == "file":
        img = load_number_image(target)
    elif mode == "folder":
        img = build_number_from_digits(target)
    else:
        print("First argument must be 'file' or 'folder'")
        sys.exit(1)

    out_path = "acquired_number.png"
    img.save(out_path)
    print(f"Saved: {out_path}")
    print(f"Size: {img.size}, mode: {img.mode}")
