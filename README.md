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
