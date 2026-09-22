# Shriyans (Person C) - contribution and verification

## Implemented scope

- Reproducible MNIST experiment comparing a dense MLP baseline with the selected CNN.
- CNN architecture with two convolution layers, pooling and dropout.
- Early stopping, learning-rate reduction, fixed seed and identical data for a fair comparison.
- Evaluation outputs: overall/per-class accuracy, loss, confusion matrix and misclassified cases.
- Integrated Person A's official preprocessing pipeline into the CNN and API input path.
- Responsive React.js GUI for image upload, preview, multi-digit display and confidence feedback.
- FastAPI prediction service connecting the browser interface to the trained CNN.
- Safe arithmetic evaluator for the Person B + C extension; it handles precedence and brackets without `eval()`.
- Automated unit tests for preprocessing and expression safety.

## How this meets the marking scheme

The assignment allocates 14 marks to investigating and comparing ML methods. The experiment controls the dataset split, optimiser, batch size and stopping rule, then saves machine-readable evidence instead of relying on one reported accuracy. The GUI covers Shriyans's core interface task and exposes confidence, including the weakest digit, so uncertain results are visible. The API now calls Person A's `preprocess_pipeline` through `prepare_mnist_digit`, giving the team one preprocessing implementation instead of two competing versions.

## Commands used for tutor demonstration

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m cnn_model.train_compare --quick --epochs 2 --output artifacts/quick_check
python -m cnn_model.train_compare --epochs 12 --output artifacts
python -m cnn_model.evaluate --model artifacts/cnn_mnist.keras
pytest -q

# Terminal 1 - prediction API
uvicorn api.main:app --reload

# Terminal 2 - React interface
cd frontend
npm install
npm run dev
```

The quick run is only a pipeline check. Do not use quick-run values as final report results.

## Evidence to retain for the sprint report

1. Screenshot the terminal showing both models complete and the final JSON result.
2. Include `artifacts/comparison.csv` as the model-comparison table.
3. Convert `artifacts/evaluation/confusion_matrix.csv` into a labelled heatmap for the report.
4. Discuss at least two frequent confusion pairs using `misclassified_samples.csv` rather than only quoting accuracy.
5. Screenshot the React GUI with an uploaded digit and confidence output.
6. Link the Git commits for the model experiment, GUI, extension and tests.

Do not invent results. Final accuracy and timing depend on the hardware and must come from the full run.

## Integration contract for Person B

Person B should pass each segmented digit crop to `prepare_mnist_digit` from the `preprocessing` package in left-to-right order. That function crops and centres each digit, then calls Person A's selected `preprocess_pipeline(..., binarize=False)` before CNN prediction. The React GUI currently accepts multiple pre-segmented files as an interim path. During Phase 3, the API can call Person B's segmentation function first without redesigning the interface.
