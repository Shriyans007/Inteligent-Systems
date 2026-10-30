"""Score original and tight U-Net crops on the same labelled real handwriting.

Manifest CSV columns: image,label. Image paths are relative to the CSV file.
This calls the FastAPI route in-process, so the server does not need to run.
The two setups differ in both crop geometry and the separately fine-tuned CNN.
"""
import argparse
import csv
import json
import mimetypes
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


SETUPS = {"original": ("cnn", "unet"),
          "tight": ("cnn_unet_tight", "unet_tight")}


def edit_distance(expected, predicted):
    previous = list(range(len(predicted) + 1))
    for i, actual in enumerate(expected, 1):
        current = [i]
        for j, guess in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (actual != guess)))
        previous = current
    return previous[-1]


def evaluate(manifest, output):
    manifest, output = Path(manifest), Path(output)
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"image", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs image,label columns.")
        examples = list(reader)
    if not examples:
        raise ValueError("The manifest has no labelled images.")

    rows = []
    with TestClient(app) as client:
        for example in examples:
            image = manifest.parent / example["image"]
            expected = example["label"].strip()
            if not expected or not image.is_file():
                raise ValueError(f"Missing label or image: {example!r}")
            data = image.read_bytes()
            content_type, _ = mimetypes.guess_type(image.name)
            row = {"image": str(image), "expected": expected,
                   "expected_count": sum(not char.isspace() for char in expected)}
            for name, (model, segmenter) in SETUPS.items():
                response = client.post("/api/recognise-text",
                                       data={"model": model, "segmenter": segmenter},
                                       files={"file": (image.name, data, content_type or "image/png")})
                result = response.json()
                prediction = result.get("text", "") if response.status_code == 200 else ""
                row[f"{name}_prediction"] = prediction
                row[f"{name}_count"] = result.get("glyph_count", 0) if response.status_code == 200 else 0
                row[f"{name}_exact"] = int(prediction == expected and response.status_code == 200)
                row[f"{name}_edits"] = edit_distance(expected, prediction)
                row[f"{name}_error"] = "" if response.status_code == 200 else str(result.get("detail", result))
            rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total_chars = sum(len(row["expected"]) for row in rows)
    summary = {"images": len(rows), "characters": total_chars}
    for name in SETUPS:
        summary[name] = {
            "exact_words": sum(row[f"{name}_exact"] for row in rows),
            "exact_word_accuracy": sum(row[f"{name}_exact"] for row in rows) / len(rows),
            "character_error_rate": sum(row[f"{name}_edits"] for row in rows) / total_chars,
            "wrong_crop_count": sum(row[f"{name}_count"] != row["expected_count"] for row in rows),
            "api_errors": sum(bool(row[f"{name}_error"]) for row in rows),
        }
    summary["paired_outcomes"] = {
        "tight_fewer_character_errors": sum(row["tight_edits"] < row["original_edits"] for row in rows),
        "original_fewer_character_errors": sum(row["original_edits"] < row["tight_edits"] for row in rows),
        "same_character_errors": sum(row["original_edits"] == row["tight_edits"] for row in rows),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Per-image results: {csv_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/text/segmentation/tight_crop_paired"))
    args = parser.parse_args()
    evaluate(args.manifest, args.output)


if __name__ == "__main__":
    main()
