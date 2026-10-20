# Handwritten Number Recognition System (HNRS)

COS30018 Intelligent Systems - Project Assignment Option B.

## Team
- Jaspreet Singh - 105342118 (Person A: preprocessing, acquisition and evaluation)
- Akhila (Person B: segmentation and integration)
- Shriyans (Person C: CNN experiments, core GUI and research extension)
- Zadeed Haque - 106382225 (Person D: alternative ML model and advanced GUI)

## Current components

- `preprocessing/` - acquisition and preprocessing investigation
- `cnn_model/` - controlled MLP/CNN MNIST experiment, evaluation and inference
- `frontend/` - responsive React/Vite image upload and confidence-aware prediction interface
- `api/` - FastAPI bridge between the React interface and trained CNN
- `extension/` - safe arithmetic-expression evaluator for the B + C extension
- `tests/` - automated checks for preprocessing contracts and expression safety

See [`docs/SHRIYANS_CONTRIBUTION.md`](docs/SHRIYANS_CONTRIBUTION.md) for setup, tutor-demonstration commands, evidence requirements and the segmentation integration contract.

## LeNet-5 and small ResNet experiments (Person C)

The existing `cnn_model.train_compare` command trains the MLP baseline and shallow CNN.
The MLP flattens the 28×28 image and uses a dense hidden layer, covering the
Week 7 lab's feedforward-network approach (its layer sizes and optimiser differ).
LeNet-5 uses convolution, average pooling and dense layers; its first layer uses
same padding to adapt the original 32×32 design to the project's 28×28 inputs.
The small ResNet uses four residual blocks and a projected shortcut where its
channel count and resolution change. It has fewer blocks and filters than a
full-size ResNet so it can be trained on a Windows CPU.

From the repository root in PowerShell, activate the existing environment:

```powershell
& "C:\venvs\hnrs\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
```

Optional short pipeline checks (their results are **not** final evidence):

```powershell
python -m cnn_model.train_variants --model lenet5 --quick --epochs 2
python -m cnn_model.train_variants --model resnet --quick --epochs 2
```

Full training uses the same MNIST sample order, 10% validation split, 0–1
normalisation, Adam optimiser, batch size, seed and stopping callbacks as the
existing MLP/CNN comparison. The 10,000 test images are evaluated only after
training; `--epochs 12` is an upper bound because early stopping can finish
sooner. Run one command at a time:

```powershell
python -m cnn_model.train_variants --model lenet5 --epochs 12
python -m cnn_model.train_variants --model resnet --epochs 12
```

The saved models and histories go to `artifacts/models/lenet5/` and
`artifacts/models/resnet/`; each has `result.json` with measured test loss,
accuracy, per-class accuracy, parameters and elapsed train/evaluation time.
These commands do not replace `artifacts/cnn_mnist.keras` or change the GUI's
model. Generate separate confusion matrices and error lists:

```powershell
python -m cnn_model.evaluate --model artifacts/models/lenet5/lenet5_mnist.keras --output artifacts/models/lenet5/evaluation
python -m cnn_model.evaluate --model artifacts/models/resnet/resnet_mnist.keras --output artifacts/models/resnet/evaluation
python -m cnn_model.compare_four
```

The last command reads the existing full-run `artifacts/comparison.json` and
the two new full-run `result.json` files, then writes
`artifacts/four_model_comparison.csv`. Confirm your existing MLP/CNN JSON came
from your **full** MNIST run, rather than `artifacts/quick_check`; older result
files did not record the seed or sample counts. Review test results alongside
real uploaded-digit performance before choosing which model to integrate.
For Task 4, check the complete React → FastAPI → segmentation → shared
preprocessing → prediction flow with labelled single-digit and multi-digit
images. Save the actual predictions, expected labels and errors as evidence;
MNIST accuracy alone does not establish system performance on uploaded images.
The API continues to load the trained shallow CNN by default.

### Choosing a trained model in the GUI

After running the backend and React frontend, use **Select model** above the
recognition result to compare saved MLP, shallow CNN, LeNet-5 and small ResNet
models on the same image. Models without a saved `.keras` file appear disabled.
The result shows which model produced the prediction. Changing the selection
clears the old result; press **Recognise number** again to run the newly
selected model. The first request for a model may take longer while the API
loads it; subsequent requests reuse it. The CNN remains the default selection
unless `HNRS_MODEL_PATH` specifies one of the other known saved-model paths.
This selector does not retrain a model or change the actual MNIST comparison.

When you have real labelled single- and multi-digit images, create a CSV
manifest such as `data/upload_examples.csv` (paths relative to that CSV):

```csv
image,label
single_9.png,9
number_527.png,527
```

Run the current GUI/API model through the full prediction route and save its
actual outcomes:

```powershell
python -m cnn_model.evaluate_uploads --manifest data/upload_examples.csv
```

This produces `artifacts/upload_evaluation.json` with predictions, exact
matches and segmentation length mismatches for each file. Include enough
representative images for a meaningful Task 4 assessment, and separately
inspect the React display in your browser. The upload command uses the CNN
unless you pass `--model` with another key from `/api/models` (for example
`--model svm`); it does not change the GUI's default model.

## SVM baseline (Person D)

A scikit-learn support vector machine is the second model family alongside the
Keras networks. It sees each digit as a flat row of 784 pixels rather than a
28x28 image, and `predict_proba` supplies the class probabilities shown in the
GUI. Train it from the repo root:

```bash
python -m svm_model.train
```

This trains an RBF-kernel `SVC` on 10,000 MNIST training images, tests it on
the 10,000 MNIST test images, and saves `artifacts/models/svm/svm_mnist.joblib`
with `result.json`. Restart the API and choose **SVM (RBF)** under
**Digits only (MNIST)**. The GUI's warning threshold starts at 50% confidence
and can be changed with the slider. The MNIST test accuracy is shown separately:
it measures a dataset, not the confidence of an individual upload. SVM digit
predictions use the same `predict()` decision rule as its reported test accuracy;
the probability bars show its separately calibrated `predict_proba()` scores.
Those two SVM outputs may occasionally rank digits differently.

### Comparison with the CNN on the benchmark set

Both models run through the same API route, so segmentation and preprocessing
are identical and only the classifier differs:

```bash
python -m preprocessing.build_benchmark_set
python -m cnn_model.evaluate_uploads --manifest data/benchmark/manifest.csv --model cnn --output artifacts/benchmark_cnn.json
python -m cnn_model.evaluate_uploads --manifest data/benchmark/manifest.csv --model svm --output artifacts/benchmark_svm.json
```

Results from one run (seed 42, 400 numbers of 1-4 digits built from MNIST test digits):

| | Shallow CNN | SVM (RBF) |
|---|---|---|
| MNIST test accuracy (single 28x28 digits) | 98.90% | 96.21% |
| Digit accuracy through the full pipeline (987 digits in correctly split numbers) | 98.07% | 93.82% |
| Whole number correct, all 400 | 93.75% (375) | 84.00% (336) |
| 1-digit numbers | 93% | 91% |
| 2-digit numbers | 97% | 86% |
| 3-digit numbers | 93% | 79% |
| 4-digit numbers | 92% | 80% |
| Segmentation length mismatches | 6 | 6 |

- Both models had the same 6 length mismatches on the same images, so those
  errors come from segmentation, not from either classifier.
- Every digit in a number must be right, so number accuracy falls with length
  roughly as digit accuracy to the power of the length (0.938^4 is about 0.77
  for the SVM). Small per-digit gaps become large gaps on longer numbers.
- The SVM loses more accuracy between clean MNIST and segmented crops (2.4
  points against 0.8 for the CNN). The likely reason is that a pixel-by-pixel
  model has little tolerance for the small shifts that segmentation and
  re-centring introduce, whereas convolution and pooling absorb them.

### Letters and digits (EMNIST)

The same SVM can also be trained on EMNIST ByClass (62 classes: 10 digits, 26
upper-case and 26 lower-case letters) and chosen under **Letters and digits**:

```bash
python -m svm_model.train_emnist
```

It uses the networks' EMNIST loader, so images are oriented the same way, and
saves `emnist_svm.joblib`, `mapping.json`, `results.json`, `per_character.csv`
and `confusion_matrix.csv` under `artifacts/text/character/svm/`. Training uses
a random 20,000 of the 697,932 training images, because RBF SVM training time
grows quickly with data size. Scoring uses a random 20,000 of the 116,323 test
images (`--test-samples 0` scores all of them, slowly); the networks are scored
on the full test set.

Results from one run (seed 42, 188 s training):

| | SVM (RBF) |
|---|---|
| Test accuracy, all 62 classes | 77.89% |
| Digits (9,946 test images) | 93.15% |
| Upper-case letters (5,384) | 67.38% |
| Lower-case letters (4,670) | 57.49% |
| Accuracy if upper and lower case of a letter count as the same | 81.58% |

- The most common errors are characters that look the same in another class:
  l read as 1 (401 times), O as 0 (352), I as 1 (223) and 0 as O (163).
- 738 of the 4,422 errors only confuse the case of the same letter, such as m
  and M or u and U. Without context a lone handwritten c and C have the same
  shape, so the model favours the class it saw more often.
- Lower-case c, f, i, m, o, s and u were never predicted correctly in this run.
  EMNIST ByClass is unbalanced: digits are about half of the images (9,946 of
  these 20,000), so the SVM leans towards digits and capitals when shapes overlap.

## Benchmark: every digit model with every segmentation method

The GUI and `/api/predict` accept a `segmentation` choice: connected components or
contours, each with or without splitting touching digits. `GET /api/segmenters`
lists them. One command runs every trained digit model with every segmentation
method on the 400-number benchmark set, through the real API, and summarises the
runs with `preprocessing.benchmark_metrics`:

```bash
python -m preprocessing.build_benchmark_set
python -m cnn_model.benchmark_all
```

Results are saved in `artifacts/benchmark/all/` (`benchmark_comparison.csv`,
`benchmark_summary.json` and one file per run). These results replace the
two-model table above: the networks have since been trained in full (12 epochs
with early stopping), and SVM digits now come from `predict()`.

Results from one run (seed 42). Every segmentation method gave each model the same
number of fully correct numbers, so the table shows one row per model (character
error rate is for the methods without splitting; with splitting it is 0.001 higher
because of one image):

| Model | MNIST test accuracy | Numbers fully correct (of 400) | Digit accuracy | Character error rate | Seconds per 400 |
|---|---|---|---|---|---|
| Small ResNet | 98.63% | 94.75% (379) | 98.48% | 0.021 | ~8.8 |
| Shallow CNN | 99.34% | 94.50% (378) | 98.28% | 0.023 | ~8.5 |
| LeNet-5 | 98.78% | 93.75% (375) | 98.07% | 0.025 | ~8.4 |
| MLP | 97.86% | 89.00% (356) | 95.95% | 0.047 | ~8.4 |
| SVM (RBF) | 96.21% | 82.50% (330) | 93.21% | 0.073 | ~2.6 |

- **Segmentation choice made no difference on this benchmark.** The benchmark
  places digits with gaps between them, so no digits touch and both methods find
  the same blobs. It therefore cannot show the value of splitting touching
  digits; hand-drawn input in the GUI is needed for that.
- **All 6 segmentation errors are over-segmentation**, the same 6 images for every
  model and method: a digit drawn in separate strokes is cut into two pieces
  (9 read as 91 and 99, 4 as 41, 75 as 754, 5985 as 59875, 2573 as 25673). On
  5985, splitting touching digits cut a piece again (598275), which made it worse.
- **Models without convolution lose most from clean MNIST to segmented crops:**
  digit accuracy fell 3.00 points for the SVM and 1.91 for the MLP, against 0.15
  to 1.06 for the three convolutional networks. Convolution and pooling tolerate
  the small shifts that cropping introduces; pixel-by-pixel models do not.
- **ResNet, CNN and LeNet-5 are within 4 numbers of each other**, too close to rank
  on 400 images. The CNN is best on MNIST, ResNet on the benchmark.
- **The SVM decision rule matters.** Switching SVM digits from the highest
  `predict_proba` score to `predict()` changed 10 benchmark predictions: 6
  became wrong and none became right (84.00% to 82.50%). `predict()` stays because
  it is the rule the reported 96.21% test accuracy was measured with.
- The SVM is about three times faster per request, because it skips TensorFlow.

## Research extension: letters and digits

The four existing MNIST classifiers (MLP, shallow CNN, LeNet-5 and small
ResNet) recognise digits only. The same architectures can be trained separately
with 62 outputs on EMNIST ByClass to recognise individual letters and digits.
Those character models have different weights and files from the ten-class
MNIST models. The default MNIST CNN is unchanged.

The interface has one drawing area for handwriting. Choose an available
**Letters and digits** model for separated characters (including numbers), or
an available **Digits only** model for the original MNIST number prediction
path. Uploaded images use the same selected path. It joins separate character
crops from left to right; connected handwriting may be treated as one crop,
and spaces, whole words and lines are not inferred.

Keep the five EMNIST ByClass files locally under `data/emnist/byclass/` (this directory is gitignored):

```text
data/emnist/byclass/emnist-byclass-mapping.txt
data/emnist/byclass/emnist-byclass-train-images-idx3-ubyte.gz
data/emnist/byclass/emnist-byclass-train-labels-idx1-ubyte.gz
data/emnist/byclass/emnist-byclass-test-images-idx3-ubyte.gz
data/emnist/byclass/emnist-byclass-test-labels-idx1-ubyte.gz
```

Run these commands from the project root in PowerShell with the `(hnrs)`
environment:

```powershell
& "C:\venvs\hnrs\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
python -m text_recognition.emnist check
python -m text_recognition.emnist preview
```

Check that the preview characters look upright and match their labels. The
optional quick run uses a separate output folder and is not a final result:

```powershell
python -m text_recognition.emnist train --model cnn --quick --epochs 2
```

Train each character model locally, one at a time, if its saved weights are
missing or you want to rerun that experiment:

```powershell
python -m text_recognition.emnist train --model cnn --epochs 12
python -m text_recognition.emnist train --model mlp --epochs 12
python -m text_recognition.emnist train --model lenet5 --epochs 12
python -m text_recognition.emnist train --model resnet --epochs 12
```

Re-evaluate and compare saved results:

```powershell
python -m text_recognition.emnist evaluate --model cnn
python -m text_recognition.emnist evaluate --model mlp
python -m text_recognition.emnist evaluate --model lenet5
python -m text_recognition.emnist evaluate --model resnet
python -m text_recognition.emnist compare
```

The character CNN saves to `artifacts/text/character/emnist_cnn.keras`, with
`mapping.json`, `results.json`, `per_character.csv` and
`confusion_matrix.csv` beside it. The other models save under
`artifacts/text/character/<model>/`; comparison writes
`artifacts/text/character/comparison.csv`. The training data is split into
training and validation sets, while the EMNIST test set is held out until
evaluation. Only locally measured results belong in these output files.

Start FastAPI and React in separate PowerShell terminals:

```powershell
& "C:\venvs\hnrs\Scripts\Activate.ps1"
uvicorn api.main:app
```

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:5173`. Use **Draw here** or **Upload image**, then
select a trained model and press **Recognise**. **Clear drawing** erases the
canvas; **Reset** clears the current input and result. Missing model files are
shown as unavailable; train the chosen model locally and refresh the page.
The original digit models still use `/api/predict`, while the EMNIST character
models use `/api/recognise-text`.

Dataset reference: [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset).

## Requirements

- **Python 3.10–3.13.** TensorFlow has no release for 3.14 yet, and `preprocessing.py`
  uses `X | Y` type-annotation syntax that requires 3.10 or newer.
- **Node.js 18 or newer** for the React interface.

Create the environment with a supported interpreter explicitly, for example
`py -3.12 -m venv .venv` on Windows or `python3.12 -m venv .venv` on macOS.
