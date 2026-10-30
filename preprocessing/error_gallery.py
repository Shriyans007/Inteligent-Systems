"""
Task 4 - Error gallery

Puts the images a model got wrong into one picture, each with what it should
have read and what it actually read. Red borders are segmentation errors (wrong
number of digits) and orange borders are classifier errors (right count, wrong
digit). Much quicker than opening the failed images one by one.

Run from the repo root after evaluate_uploads.py:
  python3 -m preprocessing.error_gallery --results artifacts/benchmark_svm.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from preprocessing.benchmark_metrics import load_results

CELL_WIDTH, CELL_HEIGHT, IMAGE_HEIGHT = 220, 96, 60
BORDER_COLOURS = {"segmentation": (220, 40, 40), "classifier": (240, 150, 20)}


def wrong_cases(cases, only=None):
    wrong = []
    for case in cases:
        if case["expected"] == case["predicted"]:
            continue
        kind = "classifier" if len(case["expected"]) == len(case["predicted"]) else "segmentation"
        if only is None or kind == only:
            wrong.append({**case, "error_type": kind})
    return wrong


def _thumbnail(path: Path) -> Image.Image:
    try:
        image = Image.open(path).convert("L")
    except OSError:
        # still show the case even if the image has been moved or deleted
        return Image.new("L", (IMAGE_HEIGHT, IMAGE_HEIGHT), color=200)
    ratio = IMAGE_HEIGHT / image.height
    size = (max(1, min(CELL_WIDTH - 16, round(image.width * ratio))), IMAGE_HEIGHT)
    return image.resize(size, Image.LANCZOS)


def make_error_gallery(cases, output, only=None, limit=40, columns=4) -> Path:
    wrong = wrong_cases(cases, only)[:limit]
    if not wrong:
        raise ValueError("No wrong predictions to show")

    rows = (len(wrong) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * CELL_WIDTH, rows * CELL_HEIGHT), "white")
    draw = ImageDraw.Draw(sheet)
    for i, case in enumerate(wrong):
        x, y = (i % columns) * CELL_WIDTH, (i // columns) * CELL_HEIGHT
        draw.rectangle([x + 2, y + 2, x + CELL_WIDTH - 3, y + CELL_HEIGHT - 3],
                       outline=BORDER_COLOURS[case["error_type"]], width=3)
        thumb = _thumbnail(Path(case["image"])).convert("RGB")
        sheet.paste(thumb, (x + 8, y + 8))
        draw.text((x + 8, y + IMAGE_HEIGHT + 12), f"expected {case['expected']}, got {case['predicted']}", fill="black")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a picture of the images a model got wrong.")
    parser.add_argument("--results", type=Path, required=True, help="One results file from evaluate_uploads.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: artifacts/benchmark/<model>_errors.png")
    parser.add_argument("--only", choices=["segmentation", "classifier"], default=None)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    try:
        (model, cases), = load_results([args.results]).items()
        output = args.output or Path("artifacts/benchmark") / f"{model}_errors.png"
        path = make_error_gallery(cases, output, args.only, args.limit)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}")
    shown = min(len(wrong_cases(cases, args.only)), args.limit)
    print(f"Saved {shown} wrong predictions for {model} to {path}")


if __name__ == "__main__":
    main()
