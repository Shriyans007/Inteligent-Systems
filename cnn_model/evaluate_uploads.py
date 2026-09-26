"""Measure the deployed API on labelled single- and multi-digit image files."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from pathlib import Path


def evaluate_manifest(manifest: Path, output: Path):
    # TestClient executes the same /api/predict route used by React, including
    # segmentation, shared preprocessing and the default trained CNN.
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
            label = row["label"].strip()
            if not label or not label.isdigit():
                raise ValueError(f"Invalid digit label: {label!r}")
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            with path.open("rb") as handle:
                response = client.post("/api/predict", files={"files": (path.name, handle, content_type)})
            response.raise_for_status()
            predicted = response.json()["number"]
            details.append({"image": str(path), "expected": label, "predicted": predicted,
                            "correct": predicted == label, "length_match": len(predicted) == len(label)})
    summary = {
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
    args = parser.parse_args()
    evaluate_manifest(args.manifest, args.output)


if __name__ == "__main__":
    main()
