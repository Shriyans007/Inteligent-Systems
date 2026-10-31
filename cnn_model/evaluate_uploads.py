"""Test the running API on labelled image files of single digits and whole numbers."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from pathlib import Path


def evaluate_manifest(manifest: Path, output: Path, model: str = "cnn", segmentation: str | None = None):
    # Call the same API route as the React GUI, so segmentation and preprocessing are included.
    from fastapi.testclient import TestClient
    from api.main import app

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"image", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs image,label columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no images")

    details = []
    with TestClient(app) as client:
        for row in rows:
            path = Path(row["image"])
            if not path.is_absolute():
                path = manifest.parent / path
            # Labels can be whole numbers, kept as strings so that leading zeros aren't lost.
            label = row["label"].strip()
            if not label or not label.isdigit():
                raise ValueError(f"Invalid digit label: {label!r}")
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            with path.open("rb") as handle:
                # Send the model choice the same way the React dropdown does.
                form = {"model": model, **({"segmentation": segmentation} if segmentation else {})}
                response = client.post("/api/predict", data=form,
                                       files={"files": (path.name, handle, content_type)})
            response.raise_for_status()
            predicted = response.json()["number"]
            details.append({"image": str(path), "expected": label, "predicted": predicted,
                            "correct": predicted == label, "length_match": len(predicted) == len(label)})
    summary = {
        "model": model,
        # The API says which method it used, including its default if none was picked.
        "segmentation": response.json().get("segmentation"),
        "images": len(details),
        "single_digit": sum(len(row["expected"]) == 1 for row in details),
        "multi_digit": sum(len(row["expected"]) > 1 for row in details),
        "exact_matches": sum(row["correct"] for row in details),
        "length_mismatches": sum(not row["length_match"] for row in details),
        "cases": details,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
    print(f"Saved case-level results to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV containing image,label columns")
    parser.add_argument("--output", type=Path, default=Path("artifacts/upload_evaluation.json"))
    parser.add_argument("--model", default="cnn", help="Model key listed by /api/models, e.g. cnn or svm")
    parser.add_argument("--segmentation", help="Segmentation key listed by /api/segmenters; default: the API's choice")
    args = parser.parse_args()
    evaluate_manifest(args.manifest, args.output, args.model, args.segmentation)


if __name__ == "__main__":
    main()
