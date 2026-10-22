"""Train a case decision from labelled, separated handwriting images.

The EMNIST CNN remains the character classifier. This optional model learns
when a visually similar upper/lower/digit label fits the original word layout.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import prepare_character
from .emnist import model_paths
from .layout import segment_text

DEFAULT_OUTPUT = Path("artifacts/text/case_context")
SPECIAL = {"0": "0Oo", "O": "0Oo", "o": "0Oo",
           "1": "1Il", "I": "1Il", "l": "1Il",
           "5": "5Ss", "S": "5Ss", "s": "5Ss"}


def alternatives(char):
    if char in SPECIAL:
        return tuple(SPECIAL[char])
    if char.isalpha():
        return (char.upper(), char.lower())
    return (char,)


def manifest(path):
    if not path.is_file():
        raise FileNotFoundError(f"Labelled case examples missing: {path}. Save examples in the GUI or create image,text,writer CSV.")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"image", "text"}.issubset(reader.fieldnames):
            raise ValueError("Case labels CSV must have image,text columns (writer optional).")
        rows = []
        for number, row in enumerate(reader, 2):
            relative = Path(row["image"] or "")
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Row {number}: image needs a relative path inside {path.parent}.")
            image = path.parent / relative
            if not image.is_file():
                raise FileNotFoundError(f"Row {number}: image missing: {image}")
            if not row["text"] or not row["text"].strip():
                raise ValueError(f"Row {number}: text is empty.")
            rows.append({"image": str(image), "text": row["text"], "writer": row.get("writer", "")})
    if not rows:
        raise ValueError("Case labels CSV contains no examples.")
    return rows


def split(rows):
    """Use whole writers when all samples have IDs and at least three exist."""
    rng = np.random.default_rng(42)
    writers = {r["writer"] for r in rows}
    writer_split = "" not in writers and len(writers) >= 3
    if writer_split:
        ids = sorted(writers)
        rng.shuffle(ids)
        buckets = [[r for r in rows if r["writer"] == writer] for writer in ids]
    else:
        buckets = [[rows[i]] for i in rng.permutation(len(rows))]
    if len(buckets) < 7:
        raise ValueError("Need at least seven independent images or writers for train/validation/test; collect more labelled examples.")
    first = max(1, int(len(buckets) * 0.7))
    second = max(first + 1, int(len(buckets) * 0.85))
    if second >= len(buckets):
        second = len(buckets) - 1
    flatten = lambda groups: [row for bucket in groups for row in bucket]
    return (flatten(buckets[:first]), flatten(buckets[first:second]),
            flatten(buckets[second:]), "writer" if writer_split else "image")


def features(scores, glyphs, mapping):
    """Make candidate rows using original geometry, CNN probabilities and position."""
    chars = [mapping[str(int(i))] for i in np.argmax(scores, axis=1)]
    indexes = {v: int(k) for k, v in mapping.items()}
    output = []
    for i, (char, glyph) in enumerate(zip(chars, glyphs)):
        start = i
        while start and glyphs[start]["before"] == "":
            start -= 1
        end = i + 1
        while end < len(glyphs) and glyphs[end]["before"] == "":
            end += 1
        token = glyphs[start:end]
        heights = np.array([g["bbox"][3] for g in token], dtype=float)
        width = np.array([g["bbox"][2] for g in token], dtype=float)
        h = glyph["bbox"][3]
        y = glyph["bbox"][1]
        bottom = y + h
        token_bottom = max(g["bbox"][1] + g["bbox"][3] for g in token)
        token_height = max(heights)
        for candidate in alternatives(char):
            if candidate not in indexes:
                continue
            p = float(scores[i, indexes[candidate]])
            output.append((i, candidate, [
                np.log(max(p, 1e-7)), np.log(max(float(scores[i].max()), 1e-7)),
                h / max(np.median(heights), 1), h / max(token_height, 1),
                glyph["bbox"][2] / max(np.median(width), 1),
                (token_bottom - bottom) / max(token_height, 1),
                (y - min(g["bbox"][1] for g in token)) / max(token_height, 1),
                float(i == start), float(i == end - 1), float(end - start),
                float(candidate.isupper()), float(candidate.islower()), float(candidate.isdigit()),
                float(char == candidate),
                float(sum(c.islower() for c in chars[start:end])) / (end - start),
                float(sum(c.isdigit() for c in chars[start:end])) / (end - start),
            ]))
    return chars, output


def decode(chars, candidates, classifier):
    """Score all alternatives; never substitute a different letter family."""
    if not candidates:
        return chars
    likelihood = classifier.predict_proba(np.asarray([item[2] for item in candidates]))[:, 1]
    chosen = chars.copy()
    best = {}
    for (index, candidate, _), score in zip(candidates, likelihood):
        if index not in best or score > best[index][0]:
            best[index] = (score, candidate)
    for index, (_, candidate) in best.items():
        chosen[index] = candidate
    return chosen


def records(rows, model, mapping):
    output = []
    for row in rows:
        with Image.open(row["image"]) as image:
            glyphs = segment_text(image)
        boxes = [{"bbox": glyph.bbox, "before": glyph.before} for glyph in glyphs]
        if not glyphs or len(glyphs) > 30:
            output.append({**row, "glyphs": boxes, "status": "blank_or_too_many"})
            continue
        scores = model.predict(np.stack([prepare_character(g.image) for g in glyphs]), verbose=0)
        if scores.shape != (len(glyphs), 62):
            raise ValueError(f"Expected ({len(glyphs)}, 62) model scores, got {scores.shape}.")
        expected = [c for c in row["text"] if not c.isspace()]
        status = "ok" if len(expected) == len(glyphs) else "glyph_count_mismatch"
        output.append({**row, "glyphs": boxes, "scores": scores, "status": status})
    return output


def distance(a, b):
    old = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, old[j] + 1, old[j - 1] + (ca != cb)))
        old = new
    return old[-1]


def measure(rows, mapping, classifier=None):
    items = []
    for row in rows:
        if "scores" in row:
            raw, candidates = features(row["scores"], row["glyphs"], mapping)
            changed = decode(raw, candidates, classifier) if classifier else raw
            separators = [g["before"] for g in row["glyphs"]]
            join = lambda chars: "".join(before + char for before, char in zip(separators, chars))
            raw_text, changed_text = join(raw), join(changed)
        else:
            raw_text = changed_text = ""
        items.append({"image": row["image"], "writer": row["writer"], "actual": row["text"],
                      "raw": raw_text, "context": changed_text, "status": row["status"]})
    size = sum(len(item["actual"]) for item in items)
    def summary(key):
        return {"exact_accuracy": sum(item[key] == item["actual"] for item in items) / len(items),
                "cer": sum(distance(item[key], item["actual"]) for item in items) / size}
    return {"samples": len(items), "raw": summary("raw"), "context": summary("context"),
            "status": dict(Counter(item["status"] for item in items))}, items


def fit(rows, mapping):
    x, y = [], []
    skipped = Counter()
    for row in rows:
        if row["status"] != "ok":
            skipped[row["status"]] += 1
            continue
        expected = [c for c in row["text"] if not c.isspace()]
        if any(char not in mapping.values() for char in expected):
            skipped["unsupported_character"] += 1
            continue
        chars, candidates = features(row["scores"], row["glyphs"], mapping)
        if any(target not in alternatives(raw) for target, raw in zip(expected, chars)):
            skipped["different_letter_or_segmentation"] += 1
            continue
        for index, candidate, vector in candidates:
            x.append(vector)
            y.append(int(candidate == expected[index]))
    if len(x) < 60 or len(set(y)) < 2:
        raise ValueError(f"Only {len(x)} usable candidate rows; collect more varied, correctly segmented case examples. Skipped: {dict(skipped)}")
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    classifier.fit(x, y)
    return classifier, dict(skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "train", "evaluate"])
    parser.add_argument("--manifest", type=Path, default=Path("data/case_benchmark/labels.csv"))
    parser.add_argument("--model", choices=["cnn", "mlp", "lenet5", "resnet", "cnn_case_balanced"], default="cnn")
    parser.add_argument("--models", type=Path, default=Path("artifacts/text/character"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = manifest(args.manifest)
    if args.action == "check":
        print(f"Found {len(rows)} labelled images; {len({r['writer'] for r in rows if r['writer']})} writer IDs.")
        return
    _, model_path, mapping_path = model_paths(args.model, args.models)
    if not model_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError(f"Train character {args.model} first; expected {model_path} and {mapping_path}.")
    from cnn_model.models import _keras
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if len(mapping) != 62:
        raise ValueError(f"Expected 62 character labels in {mapping_path}.")
    model = _keras().models.load_model(model_path)
    output = args.output / args.model
    if args.action == "train":
        train, validation, test, split_type = split(rows)
        train_records = records(train, model, mapping)
        context, skipped = fit(train_records, mapping)
        validation_result, _ = measure(records(validation, model, mapping), mapping, context)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(context, output / "case_context.joblib")
        metadata = {"model": args.model, "split": split_type, "train": len(train),
                    "validation": len(validation), "test_reserved": len(test),
                    "skipped_training": skipped, "validation_results": validation_result,
                    "test_images": [r["image"] for r in test],
                    "enabled": validation_result["context"]["cer"] < validation_result["raw"]["cer"]}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        print("Held-out test remains untouched. Run evaluate once after reviewing validation.")
        return
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file() or not (output / "case_context.joblib").is_file():
        raise FileNotFoundError(f"Train the separate case context first; expected {metadata_path} and case_context.joblib.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model"] != args.model:
        raise ValueError("The case context was trained with another character model.")
    tests = {str(Path(path).resolve()) for path in metadata["test_images"]}
    heldout = [row for row in rows if str(Path(row["image"]).resolve()) in tests]
    if len(heldout) != len(tests):
        raise ValueError("Some reserved test images are missing from the manifest. Restore them before evaluation.")
    result, examples = measure(records(heldout, model, mapping), mapping,
                               joblib.load(output / "case_context.joblib"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, ["image", "writer", "actual", "raw", "context", "status"])
        writer.writeheader(); writer.writerows(examples)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
