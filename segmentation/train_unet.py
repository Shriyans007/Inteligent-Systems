"""Train and evaluate the character U-Net on synthetic overlapping handwriting.

Real EMNIST characters are pasted into lines (see synthetic_lines.py), so no
extra dataset is needed beyond the EMNIST files you already use:

    python -m segmentation.train_unet train    --data data/emnist/byclass --steps 3000
    python -m segmentation.train_unet evaluate --data data/emnist/byclass

Add --stub to use OpenCV-drawn letters instead of EMNIST (a quick smoke test only).
Everything is sized for a laptop CPU: about 0.8M parameters, batch 8, 64-pixel lines.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from segmentation.segmenter import _binary_for_detection, segment_connected_components

from segmentation.unet_resnet import UNET_MODEL, _keras, build_unet_resnet, centre_loss
from segmentation.unet_segmenter import UNetSegmenter


def load_source(args, split):
    if args.stub:
        return StubSource()
    # Reuse the team's EMNIST loader so orientation and class mapping match the classifier.
    from text_recognition.emnist import FILES, check_data, load, read_mapping
    check_data(args.data)
    images, labels = load(args.data, split)
    return EmnistSource(images, labels, read_mapping(args.data / FILES["mapping"]))


def train(source, steps, batch_size, width, output, seed=0, learning_rate=2e-3, resume=False):
    keras = _keras()
    rng = np.random.default_rng(seed)
    if resume and output.is_file():
        print(f"Continuing from {output}")
        model = keras.models.load_model(output, compile=False)
    else:
        model = build_unet_resnet()
    schedule = keras.optimizers.schedules.CosineDecay(learning_rate, steps, alpha=0.02)
    model.compile(optimizer=keras.optimizers.Adam(schedule), loss=centre_loss)
    print(f"U-Net ResNet: {model.count_params():,} parameters")
    started, running = time.time(), None
    for step in range(1, steps + 1):
        x, y, _ = make_batch(source, batch_size, rng, width)
        loss = float(np.asarray(model.train_on_batch(x, y)).mean())
        running = loss if running is None else 0.95 * running + 0.05 * loss
        if step % 50 == 0 or step == steps:
            print(f"step {step}/{steps}  loss {running:.3f}  {time.time() - started:.0f}s", flush=True)
        if step % 500 == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            model.save(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"Saved {output}")
    return model


def _match(glyphs, boxes):
    """Greedy one-to-one matching of predicted centres to true centres."""
    used, hits = set(), 0
    for g in glyphs:
        best = min((abs(b.cx - g.cx), i) for i, b in enumerate(boxes) if i not in used) if len(used) < len(boxes) else None
        if best and best[0] <= 0.4 * g.w:
            used.add(best[1])
            hits += 1
    return hits


def evaluate(segmenter, source, lines=200, width=384, seed=123):
    """Detection precision/recall and exact-count accuracy, split by how much the text overlaps."""
    rng = np.random.default_rng(seed)
    stats = {name: dict(lines=0, exact=0, cc_exact=0, hits=0, truth=0, predicted=0)
             for name in ("all", "overlapped", "separate")}
    for n in range(lines):
        text = random_text(rng, source.available, max_chars=int(rng.integers(8, 28)))
        overlapped = n % 2 == 0          # half the lines overlap, half are cleanly separated
        image, glyphs = render_line(source, text, rng, width, overlap=overlapped)
        if not glyphs:
            continue
        gray = (augment_line(image, rng) * 255).astype(np.uint8)
        boxes = segmenter.find_characters(gray)
        hits = _match(glyphs, boxes)
        cc = len(segment_connected_components(_binary_for_detection(gray)))
        kind = "overlapped" if overlapped else "separate"
        for name in ("all", kind):
            s = stats[name]
            s["lines"] += 1
            s["exact"] += int(len(boxes) == len(glyphs) == hits)
            s["cc_exact"] += int(cc == len(glyphs))
            s["hits"] += hits
            s["truth"] += len(glyphs)
            s["predicted"] += len(boxes)
    result = {}
    for name, s in stats.items():
        if s["lines"]:
            result[name] = {"lines": s["lines"],
                            "unet_exact_line_accuracy": round(s["exact"] / s["lines"], 3),
                            "connected_components_exact_count": round(s["cc_exact"] / s["lines"], 3),
                            "character_recall": round(s["hits"] / max(s["truth"], 1), 3),
                            "character_precision": round(s["hits"] / max(s["predicted"], 1), 3)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--data", type=Path, default=Path("data/emnist/byclass"))
    parser.add_argument("--output", type=Path, default=UNET_MODEL)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--resume", action="store_true", help="continue training the saved model (use a new --seed-offset run for more steps)")
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stub", action="store_true", help="draw letters with OpenCV instead of using EMNIST")
    args = parser.parse_args()
    if args.action == "train":
        train(load_source(args, "train"), args.steps, args.batch, args.width, args.output,
              seed=args.seed, learning_rate=args.learning_rate, resume=args.resume)
    # Held-out EMNIST test characters are used for the final numbers.
    result = evaluate(UNetSegmenter(args.output), load_source(args, "test"), width=args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "segmentation_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
