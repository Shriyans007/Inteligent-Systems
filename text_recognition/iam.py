"""IAM word images: writer-grouped CRNN/CTC training and held-out evaluation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cnn_model.models import _keras
from .common import HEIGHT, WIDTH, WORD_MODEL, WORD_VOCAB, prepare_word


def read_samples(root, validate_images=False):
    labels = root / 'ascii' / 'words.txt'
    if not labels.is_file():
        raise FileNotFoundError(f'IAM labels missing: {labels}. Extract ascii.tgz inside {root}.')
    words = root / 'words'
    if not words.is_dir():
        raise FileNotFoundError(f'IAM images missing: {words}. Extract words.tgz inside {root}.')
    forms = root / 'ascii' / 'forms.txt'
    if not forms.is_file():
        raise FileNotFoundError(f'IAM writer IDs missing: {forms}. Extract ascii.tgz; writer-independent splitting requires forms.txt.')
    writers = {}
    for line in forms.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'): continue
        parts = line.split()
        if len(parts) < 2: raise ValueError(f'Invalid IAM forms.txt entry: {line}')
        writers[parts[0]] = parts[1]
    samples = []
    missing = 0
    for line in labels.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'): continue
        parts = line.split(maxsplit=8)
        if len(parts) < 8: raise ValueError(f'Invalid IAM words.txt entry: {line}')
        identifier, status, transcription = parts[0], parts[1], parts[-1]
        if status in {'er', 'err'}: continue
        form = '-'.join(identifier.split('-')[:2])
        if form not in writers: raise ValueError(f'Missing writer ID for form {form}')
        section = identifier.split('-')[0]
        path = words / section / form / f'{identifier}.png'
        if not path.is_file():
            missing += 1
            continue
        if transcription and transcription != '#': samples.append((path, transcription, writers[form]))
    if missing: print(f'Skipped {missing} word entries with missing PNG images.')
    if validate_images:
        valid, invalid = [], []
        for path, transcription, writer in samples:
            try:
                with Image.open(path) as image:
                    prepare_word(image)
            except (OSError, ValueError) as exc:
                invalid.append((path, str(exc)))
            else:
                valid.append((path, transcription, writer))
        samples = valid
        if invalid:
            print(f'Skipped {len(invalid)} blank or unreadable IAM word images.')
            for path, reason in invalid[:5]:
                print(f'  {path}: {reason}')
    if not samples: raise ValueError(f'No usable IAM word images found under {words}.')
    return samples


def split_writers(samples):
    rng = np.random.default_rng(42)
    ids = sorted({writer for _, _, writer in samples})
    if len(ids) < 3: raise ValueError('At least three distinct writers are needed for train/validation/test.')
    rng.shuffle(ids)
    train_end = min(len(ids)-2, max(1, int(.8*len(ids))))
    valid_end = min(len(ids)-1, max(train_end+1, int(.9*len(ids))))
    groups = [set(ids[:train_end]), set(ids[train_end:valid_end]), set(ids[valid_end:])]
    if any(not group for group in groups): raise ValueError('Not enough writer IDs to split three ways.')
    return [[s for s in samples if s[2] in group] for group in groups]


def build_model(classes):
    keras = _keras()
    inp = keras.Input((HEIGHT, WIDTH, 1))
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.Permute((2, 1, 3))(x)
    x = keras.layers.Reshape((WIDTH//4, (HEIGHT//4)*64))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
    out = keras.layers.Dense(classes+1, activation='softmax')(x)  # final class is the CTC blank
    return keras.Model(inp, out, name='iam_word_crnn')


def batches(samples, vocab, batch_size, shuffle=False):
    order = np.arange(len(samples))
    if shuffle: np.random.default_rng(42).shuffle(order)
    for start in range(0, len(order), batch_size):
        current = [samples[i] for i in order[start:start+batch_size]]
        images, labels = [], []
        for path, label, _ in current:
            with Image.open(path) as image: images.append(prepare_word(image))
            labels.append([vocab.index(char) for char in label])
        padded = np.zeros((len(labels), max(map(len, labels))), dtype=np.int32)
        for i, label in enumerate(labels): padded[i, :len(label)] = label
        yield np.array(images, dtype=np.float32), padded, np.array([len(x) for x in labels], dtype=np.int32), [x[1] for x in current]


def ctc_loss(keras, labels, probabilities, lengths):
    import tensorflow as tf
    batch = tf.shape(probabilities)[0]
    input_lengths = tf.fill((batch, 1), tf.shape(probabilities)[1])
    return tf.reduce_mean(keras.backend.ctc_batch_cost(labels, probabilities, input_lengths,
                                                        tf.reshape(lengths, (-1, 1))))


def decode(keras, probabilities, vocab):
    import tensorflow as tf
    lengths = np.full((len(probabilities),), probabilities.shape[1])
    decoded, _ = keras.backend.ctc_decode(probabilities, lengths, greedy=True)
    return [''.join(vocab[int(i)] for i in row if int(i) >= 0) for row in decoded[0].numpy()]


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    return previous[-1]


def evaluate(model, samples, vocab, output, name='test'):
    keras = _keras()
    errors, characters, wrong, rows = 0, 0, 0, []
    for images, _, _, labels in batches(samples, vocab, 32):
        predictions = decode(keras, model.predict(images, verbose=0), vocab)
        for actual, predicted in zip(labels, predictions):
            errors += distance(actual, predicted); characters += len(actual)
            wrong += actual != predicted
            rows.append((actual, predicted))
    output.mkdir(parents=True, exist_ok=True)
    with (output/f'{name}_predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file); writer.writerow(['actual', 'predicted']); writer.writerows(rows)
    metrics = {'samples': len(rows), 'cer': errors/characters, 'wer': wrong/len(rows),
               'exact_word_accuracy': 1-wrong/len(rows)}
    (output/f'{name}_results.json').write_text(json.dumps(metrics, indent=2))
    print(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'train', 'evaluate', 'evaluate-examples'])
    parser.add_argument('--data', type=Path, default=Path('data/iam'))
    parser.add_argument('--output', type=Path, default=WORD_MODEL.parent)
    parser.add_argument('--manifest', type=Path, help='CSV with image,label columns for evaluate-examples')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--quick', action='store_true', help='Use small subsets and save under output/quick')
    args = parser.parse_args()
    output = args.output/'quick' if args.quick else args.output
    model_path = output/WORD_MODEL.name
    if args.action == 'evaluate-examples':
        if not args.manifest: parser.error('--manifest is required for evaluate-examples')
        if not model_path.is_file(): raise FileNotFoundError(f'Train the IAM word model first: {model_path}')
        samples = []
        with args.manifest.open(newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                path = args.manifest.parent/row['image']
                if not path.is_file(): raise FileNotFoundError(path)
                samples.append((path, row['label'], 'external'))
        if not samples: raise ValueError('The example manifest has no rows.')
        vocab = json.loads((output/WORD_VOCAB.name).read_text())
        evaluate(_keras().models.load_model(model_path), samples, vocab, output, 'examples')
        return
    samples = read_samples(args.data, validate_images=True)
    train, valid, test = split_writers(samples)
    vocab = sorted({char for _, label, _ in samples for char in label})  # fixed IAM character inventory; no images or gradients from held-out writers
    if args.action == 'check':
        print(f'IAM words: {len(samples)}; writer-disjoint train/validation/test: {len(train)}/{len(valid)}/{len(test)}')
        print(f'Words containing digits: {sum(any(c.isdigit() for c in label) for _, label, _ in samples)}')
        print('Review this count before claiming mixed letter-and-number recognition.')
        return
    if args.action == 'evaluate':
        if not model_path.is_file(): raise FileNotFoundError(f'Train the IAM word model first: {model_path}')
        saved_vocab = json.loads((output/WORD_VOCAB.name).read_text())
        evaluate(_keras().models.load_model(model_path), test, saved_vocab, output)
        return
    keras = _keras()
    import tensorflow as tf
    tf.random.set_seed(42)
    if args.quick: train, valid, test = train[:128], valid[:32], test[:32]
    if any(not section for section in (train, valid, test)): raise ValueError('Empty IAM split.')
    # Repeated adjacent labels require extra CTC time steps. Limit excessively long labels.
    capacity = WIDTH//4
    def fits(label): return len(label)+sum(a==b for a, b in zip(label, label[1:])) <= capacity
    train, valid, test = [[s for s in section if fits(s[1])] for section in (train, valid, test)]
    model = build_model(len(vocab))
    optimizer = keras.optimizers.Adam(1e-3)
    output.mkdir(parents=True, exist_ok=True)
    (output/WORD_VOCAB.name).write_text(json.dumps(vocab, indent=2))
    best, patience = float('inf'), 0
    for epoch in range(min(args.epochs, 2) if args.quick else args.epochs):
        losses = []
        for images, labels, lengths, _ in batches(train, vocab, 32, shuffle=True):
            with tf.GradientTape() as tape:
                probabilities = model(images, training=True)
                loss = ctc_loss(keras, labels, probabilities, lengths)
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            losses.append(float(loss))
        validation = []
        for images, labels, lengths, _ in batches(valid, vocab, 32):
            validation.append(float(ctc_loss(keras, labels, model(images, training=False), lengths)))
        score = float(np.mean(validation))
        print(f'Epoch {epoch+1}: train CTC {np.mean(losses):.4f}, validation CTC {score:.4f}')
        if score < best:
            best, patience = score, 0
            model.save(model_path)
        else:
            patience += 1
            if patience >= 3: break
    evaluate(keras.models.load_model(model_path), test, vocab, output)

if __name__ == '__main__': main()
