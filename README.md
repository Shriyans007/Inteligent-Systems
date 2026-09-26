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
inspect the React display in your browser. The upload command evaluates the
API's currently configured model; it does not change the default model.

## Research extension: characters and handwritten words

The existing ten-class MNIST number models remain under `artifacts/` and use
`/api/predict`. The new **character** CNN reuses the shallow CNN architecture
with 62 new outputs and its own EMNIST training. The new **word** CRNN uses
convolutional features, a bidirectional LSTM and CTC to transcribe a single
word without cutting it into individual characters. Neither extension model is
trained or enabled until you run its training command locally. A word is
resized with its aspect ratio intact and padded to 64×384. This is a word
recogniser, not a line/page recogniser; spaces are not supported in Word mode.

Put the local datasets in these folders (all of `data/` is gitignored):

```text
data/emnist/byclass/emnist-byclass-mapping.txt
data/emnist/byclass/emnist-byclass-train-images-idx3-ubyte.gz
data/emnist/byclass/emnist-byclass-train-labels-idx1-ubyte.gz
data/emnist/byclass/emnist-byclass-test-images-idx3-ubyte.gz
data/emnist/byclass/emnist-byclass-test-labels-idx1-ubyte.gz
data/iam/ascii/words.txt
data/iam/ascii/forms.txt
data/iam/words/<section>/<form>/<word-id>.png
```

`forms.txt` comes from IAM `ascii.tgz`. It maps forms to writers. The loader
uses that mapping to keep writers separate in an 80/10/10 approximate split
with seed 42, then skips IAM segmentation errors and missing PNGs. The split
is deterministic. The IAM character inventory is fixed from the corpus labels;
only training writer images update model weights, and validation writers select
an epoch. The held-out test writers are used for final metrics. This split
will differ from IAM's official benchmark split, so compare results only when
the evaluation protocol matches.

Run these commands in **PowerShell from the repository root**. Your existing
`(hnrs)` Python environment can be activated with:

```powershell
& "C:\venvs\hnrs\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
```

Check your extraction and orientation **before full training**:

```powershell
python -m text_recognition.emnist check
python -m text_recognition.emnist preview
python -m text_recognition.iam check
```

Open `artifacts\text\character\orientation_preview.png` and confirm the
characters look upright and match the printed labels. IAM `check` reports the
number of samples containing digits: review that count before claiming the
word model covers mixed letters and numbers. Character and word commands
accept `--data` for a different local dataset directory and `--output` for a
different output directory. If extracted IAM files are nested one extra level,
move them to the layout above rather than pointing at the archives.

Optional small pipeline runs save under separate `quick` directories and are
**not final results**:

```powershell
python -m text_recognition.emnist train --quick --epochs 2
python -m text_recognition.iam train --quick --epochs 2
```

Run full training yourself, one command at a time. Twelve epochs is an upper
bound for EMNIST early stopping and an upper bound for IAM validation stopping:

```powershell
python -m text_recognition.emnist train --epochs 12
python -m text_recognition.iam train --epochs 12
```

Re-run held-out evaluation after training if needed:

```powershell
python -m text_recognition.emnist evaluate
python -m text_recognition.iam evaluate
```

EMNIST writes `artifacts/text/character/emnist_cnn.keras`, `mapping.json`,
`results.json`, `per_character.csv` and `confusion_matrix.csv`. Its training
uses a fixed 90/10 training/validation split from the official training IDX;
the official test IDX is held out until evaluation. IAM writes
`artifacts/text/word/iam_crnn.keras`, `vocabulary.json`, `test_results.json`
and `test_predictions.csv`. It reports character error rate (total character
edits divided by reference characters), word error rate (fraction of incorrect
words) and exact-word accuracy. These files contain measured results only
after local training/evaluation. Generated models and data are ignored by Git.

To test separately collected mixed letter-and-number **single words**, save
labelled images and a local `data/mixed_words.csv`:

```csv
image,label
room42.png,Room42
ai2026.png,AI2026
```

Then run:

```powershell
python -m text_recognition.iam evaluate-examples --manifest data/mixed_words.csv
```

This writes `examples_results.json` and `examples_predictions.csv` beside the
IAM model. IAM word data may contain few numeric examples, so this separate
evaluation matters. It cannot make an unsupported character part of the model's
vocabulary; check `vocabulary.json` after training. Real camera photos and
mouse drawings should be tested separately from cropped dataset images.

Start the API and GUI in separate PowerShell terminals from the repository
root; activate `(hnrs)` in the API terminal:

```powershell
& "C:\venvs\hnrs\Scripts\Activate.ps1"
uvicorn api.main:app
```

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:5173`. Select Numbers, Character or Word and then
Upload image or Draw here. Clear drawing erases the canvas; Reset clears the
current result and input. Number model selection still uses the trained MNIST
models, with CNN as the default. Character and Word show “Train the model
first” until their `.keras` and mapping files exist; restart the backend and
refresh the GUI after local training. Drawing uses the same API and image
preprocessing as uploading in the selected mode. Record labelled real-image
predictions before claiming performance outside the held-out datasets.

Dataset and method references: [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset),
[IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database),
[Keras IAM word-recognition example](https://keras.io/examples/vision/handwriting_recognition/).
