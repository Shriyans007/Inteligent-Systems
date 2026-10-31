# Handwritten Number Recognition System (HNRS)

COS30018 Intelligent Systems, Project Option B, Group 8.

HNRS reads a handwritten number from an uploaded image or from a drawing in the browser. The image gets split into single digits, each digit is prepared so it looks like an MNIST image, and a trained model reads it. You can choose between five digit models (shallow CNN, MLP, LeNet-5, small ResNet and an SVM) and a few different segmentation methods. For our research extension it can also read separated letters and digits, using models trained on EMNIST.

## Team

| Person | Name | Main work |
|---|---|---|
| A | Jaspreet Singh | Preprocessing, image acquisition, benchmark and testing |
| B | Akhila Gollamudi | Segmentation and integration |
| C | Shriyans Simhadri | CNN models, core GUI and API, research extension |
| D | Zadeed Haque | SVM model, learned segmenter, advanced GUI features |

## What's in the repo

| Folder | What it has |
|---|---|
| `preprocessing/` | Image acquisition, preprocessing, the benchmark set and the scripts that score it |
| `segmentation/` | Rule-based digit segmentation (connected components or contours) and the U-Net character segmenter |
| `segmentation_model/` | The learned cut-point CNN for splitting touching digits |
| `cnn_model/` | Training and evaluating the MLP, CNN, LeNet-5 and ResNet, plus the upload and benchmark evaluation |
| `svm_model/` | The SVM for digits and for EMNIST characters |
| `text_recognition/` | The letters and digits extension (EMNIST models, line layout, crop fixes) |
| `extension/` | A safe arithmetic expression evaluator from our first extension idea |
| `api/` | FastAPI backend that loads the models and runs predictions |
| `frontend/` | React GUI |
| `tests/` | Automated tests |
| `docs/` | Individual contribution notes |

## Setup

You need **Python 3.10 to 3.13** (TensorFlow doesn't support 3.14 yet) and **Node.js 18 or newer** for the GUI.

macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The trained models and datasets aren't in git (`artifacts/`, `data/` and model files are gitignored), so each person trains the models on their own machine. MNIST downloads by itself the first time a script needs it. EMNIST has to be downloaded by hand for the extension (see the section on it below).

## Training the models

Run these from the repo root. Everything gets saved in `artifacts/`. Adding `--quick` to the neural network commands does a short run, which is good for checking things work, but those numbers aren't the real results.

| Model | Command |
|---|---|
| MLP and shallow CNN | `python -m cnn_model.train_compare --epochs 12` |
| LeNet-5 | `python -m cnn_model.train_variants --model lenet5 --epochs 12` |
| Small ResNet | `python -m cnn_model.train_variants --model resnet --epochs 12` |
| SVM | `python -m svm_model.train` |
| Learned cut-point segmenter | `python -m segmentation_model.train` |

You only need the shallow CNN to run the app, since it's the default model. The others show up in the GUI once they're trained.

## Running the app

Start the API in one terminal (from the repo root, with the venv active):

```bash
uvicorn api.main:app --reload
```

Then start the GUI in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Pick **Draw here** or **Upload image** (PNG, JPG, BMP or WebP, up to 5 MB), choose a model and a segmentation method, and press **Recognise**. The result shows each digit with its probability bars, the boxes where the image was cut, and an amber warning on any digit below the confidence slider. Models or segmenters that haven't been trained on your machine are greyed out.

## Tests

```bash
python -m pytest
```

## Benchmark

MNIST test accuracy only shows how a model does on single, clean digits, so we also test whole numbers through the full system. The benchmark set is 400 numbers (100 each of 1, 2, 3 and 4 digits) built from MNIST test digits with a fixed seed, so every model gets exactly the same images.

```bash
python -m preprocessing.build_benchmark_set
python -m cnn_model.benchmark_all
```

`benchmark_all` runs every trained digit model with every segmentation method through the real API route, then scores them with `preprocessing.benchmark_metrics` (exact accuracy, digit accuracy, character error rate, and segmentation errors vs model errors). The results are saved in `artifacts/benchmark/all/`, with one file per model and segmenter in `runs/`. To see the images a model got wrong, for example the CNN with the default segmenter:

```bash
python -m preprocessing.error_gallery --results artifacts/benchmark/all/runs/cnn__components_split.json
```

## Results

From one run (seed 42). Training again gives slightly different numbers.

| Model | MNIST test accuracy | Whole numbers right (of 400) | Digit accuracy | Character error rate |
|---|---|---|---|---|
| Small ResNet | 98.63% | 94.75% (379) | 98.48% | 0.021 |
| Shallow CNN | 99.34% | 94.50% (378) | 98.28% | 0.023 |
| LeNet-5 | 98.78% | 93.75% (375) | 98.07% | 0.025 |
| MLP | 97.86% | 89.00% (356) | 95.95% | 0.047 |
| SVM (RBF) | 96.21% | 82.50% (330) | 93.21% | 0.073 |

- ResNet, the CNN and LeNet-5 are within 4 numbers of each other, which is too close to rank on 400 images. The shallow CNN is the default since it has the best MNIST accuracy.
- The MLP and SVM flatten the image, and they lose the most accuracy going from clean MNIST to cropped digits. Convolution handles the small shifts from cropping a lot better.
- With the rule-based segmenters, the same 6 images get the wrong digit count for every model. They're all a digit drawn in separate strokes that gets cut into two pieces.

The learned cut-point segmenter mostly helps when digits touch:

| Benchmark | Rule-based (components + split) | Learned cut-point CNN |
|---|---|---|
| Normal, 400 numbers: CNN | 94.50% | 95.50% |
| Normal, 400 numbers: SVM | 82.50% | 83.75% |
| Touching, 300 numbers: CNN | 9.67% | 64.00% |
| Touching, 300 numbers: SVM | 8.33% | 50.00% |

Preprocessing was picked by training the same small MLP on 5,000 MNIST images with each option. Grayscale only got 92.8%, a fixed threshold got 91.0% and Otsu got 91.2%, so the final system uses grayscale only.

## Research extension: letters and digits

The same model types can be trained with 62 outputs on EMNIST ByClass (10 digits, 26 capital and 26 lowercase letters), so that the app can read separated letters as well as digits. Put the five EMNIST ByClass files in `data/emnist/byclass/`:

```text
emnist-byclass-mapping.txt
emnist-byclass-train-images-idx3-ubyte.gz
emnist-byclass-train-labels-idx1-ubyte.gz
emnist-byclass-test-images-idx3-ubyte.gz
emnist-byclass-test-labels-idx1-ubyte.gz
```

Then check the files and train a character model:

```bash
python -m text_recognition.emnist check
python -m text_recognition.emnist train --model cnn --epochs 12
python -m text_recognition.emnist evaluate --model cnn
```

`--model` also takes `mlp`, `lenet5` and `resnet`, and `python -m svm_model.train_emnist` trains the SVM version. Trained character models show up under **Letters and digits** in the GUI. Touching letters can be split with the U-Net segmenter, which is trained with `python -m segmentation.train_unet train --data data/emnist/byclass --steps 3000`. This part is experimental, and it works best when the letters are clearly separated.

More detail on every part, including how each method was chosen, is in our project report.
