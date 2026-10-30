"""Compare every available GUI character model on labelled word images.

Run from the repository root:
    py compare_character_models.py --manifest data/character_examples/manifest.csv

Uses the same FastAPI route as the GUI, without requiring a running server.
Only character models are included: MNIST digit-only models cannot score words.
"""

import argparse
import csv
import mimetypes
import re
from pathlib import Path


FIELDS = [
    "row_type", "model_key", "model_label", "segmenter", "image", "expected",
    "prediction", "expected_count", "glyph_count", "confidence", "exact_raw",
    "edits_raw", "exact_normalized", "edits_normalized", "word_accuracy_raw",
    "cer_raw", "word_accuracy_normalized", "cer_normalized", "wrong_crop_count",
    "api_errors", "error",
]


def edit_distance(expected, predicted):
    previous = list(range(len(predicted) + 1))
    for i, actual in enumerate(expected, 1):
        current = [i]
        for j, guess in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (actual != guess)))
        previous = current
    return previous[-1]


def normalized(text):
    """Ignore case and spaces for a second metric; retain digits and letters."""
    return re.sub(r"\s+", "", text).casefold()


def load_examples(manifest):
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"image", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs image,label columns.")
        examples = []
        for record in reader:
            expected = record["label"].strip()
            image = manifest.parent / record["image"]
            if not expected or not image.is_file():
                raise ValueError(f"Missing label or image: {record!r}")
            examples.append((image, expected, image.read_bytes()))
    if not examples:
        raise ValueError("The manifest has no labelled images.")
    return examples


def compare(manifest, output, segmenter):
    # Import here so --help works even outside the project environment.
    from fastapi.testclient import TestClient
    from api.main import app

    examples = load_examples(manifest)
    rows = []
    summary = []
    with TestClient(app) as client:
        status_response = client.get("/api/extension-models")
        status_response.raise_for_status()
        status = status_response.json()
        if not status.get("segmenters", {}).get(segmenter, False):
            raise ValueError(f"The {segmenter} segmenter is not available. "
                             "Choose --segmenter classical, or train the U-Net.")
        models = [item for item in status["character_models"] if item["available"]]
        if not models:
            raise ValueError("No trained EMNIST character models are available.")
        total_characters = sum(len(expected) for _, expected, _ in examples)
        normalized_characters = sum(len(normalized(expected)) for _, expected, _ in examples)

        for item in models:
            model_rows = []
            print(f"Testing {item['label']} ({item['key']}) on {len(examples)} images...", flush=True)
            for image, expected, data in examples:
                content_type = mimetypes.guess_type(image.name)[0] or "image/png"
                response = client.post(
                    "/api/recognise-text",
                    data={"model": item["key"], "segmenter": segmenter},
                    files={"file": (image.name, data, content_type)},
                )
                result = response.json()
                success = response.status_code == 200
                prediction = str(result.get("text", "")) if success else ""
                expected_count = sum(not char.isspace() for char in expected)
                row = {
                    "row_type": "image", "model_key": item["key"],
                    "model_label": item["label"], "segmenter": segmenter,
                    "image": image.name, "expected": expected,
                    "prediction": prediction, "expected_count": expected_count,
                    "glyph_count": result.get("glyph_count", "") if success else "",
                    "confidence": result.get("confidence", "") if success else "",
                    "exact_raw": int(success and prediction == expected),
                    "edits_raw": edit_distance(expected, prediction),
                    "exact_normalized": int(success and normalized(prediction) == normalized(expected)),
                    "edits_normalized": edit_distance(normalized(expected), normalized(prediction)),
                    "error": "" if success else f"HTTP {response.status_code}: {result.get('detail', result)}",
                }
                model_rows.append(row)
                rows.append(row)

            totals = {
                "row_type": "SUMMARY", "model_key": item["key"],
                "model_label": item["label"], "segmenter": segmenter,
                "exact_raw": sum(int(r["exact_raw"]) for r in model_rows),
                "edits_raw": sum(int(r["edits_raw"]) for r in model_rows),
                "exact_normalized": sum(int(r["exact_normalized"]) for r in model_rows),
                "edits_normalized": sum(int(r["edits_normalized"]) for r in model_rows),
                "wrong_crop_count": sum(r["glyph_count"] != r["expected_count"]
                                        for r in model_rows if not r["error"]),
                "api_errors": sum(bool(r["error"]) for r in model_rows),
            }
            totals["word_accuracy_raw"] = totals["exact_raw"] / len(examples)
            totals["cer_raw"] = totals["edits_raw"] / total_characters
            totals["word_accuracy_normalized"] = totals["exact_normalized"] / len(examples)
            totals["cer_normalized"] = totals["edits_normalized"] / normalized_characters
            rows.append(totals)
            summary.append(totals)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {output} ({len(examples)} images x {len(models)} models).")
    print("Model | exact raw | CER raw | exact normalized | CER normalized | wrong crop count | API errors")
    for row in sorted(summary, key=lambda r: (r["cer_normalized"], r["cer_raw"])):
        print(f"{row['model_label']} | {row['exact_raw']}/{len(examples)} | "
              f"{row['cer_raw']:.1%} | {row['exact_normalized']}/{len(examples)} | "
              f"{row['cer_normalized']:.1%} | {row['wrong_crop_count']} | {row['api_errors']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/text/segmentation/all_character_models.csv"))
    parser.add_argument("--segmenter", choices=("classical", "unet", "unet_tight"),
                        default="unet_tight", help="Same segmentation for every model (default: unet_tight)")
    args = parser.parse_args()
    compare(args.manifest, args.output, args.segmenter)


if __name__ == "__main__":
    main()
