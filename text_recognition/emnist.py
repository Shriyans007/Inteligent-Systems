"""Read official EMNIST ByClass IDX archives and train a separate 62-way CNN."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import struct
import numpy as np
from cnn_model.models import build_cnn, compile_model, _keras
from .common import CHAR_MODEL, CHAR_MAPPING

FILES = {"train_images": "emnist-byclass-train-images-idx3-ubyte.gz",
         "train_labels": "emnist-byclass-train-labels-idx1-ubyte.gz",
         "test_images": "emnist-byclass-test-images-idx3-ubyte.gz",
         "test_labels": "emnist-byclass-test-labels-idx1-ubyte.gz",
         "mapping": "emnist-byclass-mapping.txt"}


def check_data(folder):
    missing = [name for name in FILES.values() if not (folder/name).is_file()]
    if missing:
        raise FileNotFoundError(f"EMNIST ByClass missing in {folder}: {', '.join(missing)}")


def read_idx(path):
    with gzip.open(path, "rb") as file:
        header = file.read(4)
        if len(header) != 4 or header[:2] != b"\0\0" or header[2] != 8:
            raise ValueError(f"Invalid IDX byte header: {path}")
        shape = tuple(struct.unpack(">I", file.read(4))[0] for _ in range(header[3]))
        data = np.frombuffer(file.read(), dtype=np.uint8)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"Incorrect IDX size: {path}")
    return data.reshape(shape)


def read_mapping(path):
    entries = [line.split() for line in path.read_text().splitlines() if line.strip()]
    mapping = {int(key): chr(int(code)) for key, code in entries}
    if set(mapping) != set(range(62)) or len(set(mapping.values())) != 62:
        raise ValueError("Expected 62 distinct EMNIST ByClass mapping entries (0-61).")
    return mapping


def orient(images):
    # IDX reads rows in the inverse orientation of the published character images.
    # tfds uses transpose followed by horizontal flip; here that is a 90-degree rotation.
    return np.rot90(images, k=-1, axes=(1, 2)).copy()[..., None]


def load(folder, split):
    images = read_idx(folder/FILES[f"{split}_images"])
    labels = read_idx(folder/FILES[f"{split}_labels"])
    if len(images) != len(labels) or images.shape[1:] != (28, 28):
        raise ValueError("Mismatched EMNIST images and labels or unexpected shape.")
    return orient(images), labels.astype(np.int64)


def evaluate(model, x, y, mapping, output):
    from sklearn.metrics import confusion_matrix
    output.mkdir(parents=True, exist_ok=True)
    x = x.astype(np.float32)/255
    loss, accuracy = model.evaluate(x, y, batch_size=256, verbose=0)
    pred = model.predict(x, batch_size=256, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(62))
    with (output/"confusion_matrix.csv").open("w", newline="") as file:
        writer = csv.writer(file); writer.writerow(["actual/predicted"]+[mapping[i] for i in range(62)])
        for i, row in enumerate(cm): writer.writerow([mapping[i], *row])
    with (output/"per_character.csv").open("w", newline="") as file:
        writer = csv.writer(file); writer.writerow(["label", "character", "samples", "accuracy"])
        for i, row in enumerate(cm): writer.writerow([i, mapping[i], int(row.sum()), float(row[i]/row.sum()) if row.sum() else ""])
    result = {"test_loss": float(loss), "test_accuracy": float(accuracy), "test_samples": len(y)}
    (output/"results.json").write_text(json.dumps(result, indent=2))
    print(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "preview", "train", "evaluate"])
    parser.add_argument("--data", type=Path, default=Path("data/emnist/byclass"))
    parser.add_argument("--output", type=Path, default=CHAR_MODEL.parent)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--quick", action="store_true", help="Use small train/test subsets in a separate output directory")
    args = parser.parse_args()
    check_data(args.data)
    mapping = read_mapping(args.data/FILES["mapping"])
    if args.action == "check":
        print("EMNIST files and 62-character mapping found. Verify the orientation by viewing samples before full training.")
        return
    output = args.output / "quick" if args.quick else args.output
    model_path = output/CHAR_MODEL.name
    if args.action == "evaluate":
        if not model_path.is_file(): raise FileNotFoundError(f"Train the EMNIST model first: {model_path}")
        model = _keras().models.load_model(model_path)
        x, y = load(args.data, "test")
        evaluate(model, x, y, mapping, output)
        return
    x, y = load(args.data, "train")
    if args.action == "preview":
        from PIL import Image, ImageDraw
        output.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("L", (8*84, 4*98), 255)
        pen = ImageDraw.Draw(canvas)
        for n, index in enumerate(np.linspace(0, len(y)-1, 32, dtype=int)):
            char = Image.fromarray(x[index, ..., 0]).resize((70, 70))
            canvas.paste(char, ((n%8)*84, (n//8)*98))
            pen.text(((n%8)*84, (n//8)*98+72), mapping[int(y[index])], fill=0)
        destination = output/"orientation_preview.png"
        canvas.save(destination)
        print(f"Open {destination} and confirm the letters are upright before training.")
        return
    # Fixed permutation; test is never used for callbacks or validation.
    order = np.random.default_rng(42).permutation(len(y))
    if args.quick: order = order[:4096]
    cut = int(len(order)*0.9)
    train, validation = order[:cut], order[cut:]
    model = compile_model(build_cnn(62))
    output.mkdir(parents=True, exist_ok=True)
    (output/CHAR_MAPPING.name).write_text(json.dumps(mapping, indent=2))
    keras = _keras()
    class Batches(keras.utils.PyDataset):
        def __init__(self, indices):
            super().__init__()
            self.indices = indices
        def __len__(self): return (len(self.indices)+127)//128
        def __getitem__(self, index):
            selected = self.indices[index*128:(index+1)*128]
            return x[selected].astype(np.float32)/255, y[selected]
    model.fit(Batches(train), validation_data=Batches(validation),
              epochs=args.epochs if not args.quick else min(args.epochs, 2),
              callbacks=[_keras().callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    model.save(model_path)
    test_x, test_y = load(args.data, "test")
    if args.quick: test_x, test_y = test_x[:512], test_y[:512]
    evaluate(model, test_x, test_y, mapping, output)

if __name__ == "__main__": main()
