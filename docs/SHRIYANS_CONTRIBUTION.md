# Shriyans (Person C) - contribution and verification

## Implemented scope

- Reproducible MNIST experiment comparing a dense MLP baseline with the selected CNN.
- CNN architecture with convolution, batch normalisation, pooling and dropout.
- Early stopping, learning-rate reduction, fixed seed and identical data for a fair comparison.
- Evaluation outputs: overall/per-class accuracy, loss, confusion matrix and misclassified cases.
- Shared upload preprocessing that converts real images to the 28 x 28 MNIST convention.
- Core desktop GUI for image upload, preview, model loading, multi-digit display and confidence feedback.
- Safe arithmetic evaluator for the Person B + C extension; it handles precedence and brackets without `eval()`.
- Automated unit tests for preprocessing and expression safety.

## How this meets the marking scheme

The assignment allocates 14 marks to investigating and comparing ML methods. The experiment controls the dataset split, optimiser, batch size and stopping rule, then saves machine-readable evidence instead of relying on one reported accuracy. The GUI covers Shriyans's core interface task and exposes confidence, including the weakest digit, so uncertain results are visible. The model and UI are separated behind `DigitPredictor`, allowing Person B's segmentation output to use the same prediction component during integration.

## Commands used for tutor demonstration

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m shriyans_cnn.train_compare --quick --epochs 2 --output artifacts/quick_check
python -m shriyans_cnn.train_compare --epochs 12 --output artifacts
python -m shriyans_cnn.evaluate --model artifacts/cnn_mnist.keras
python -m gui.app --model artifacts/cnn_mnist.keras
pytest -q
```

The quick run is only a pipeline check. Do not use quick-run values as final report results.

## Evidence to retain for the sprint report

1. Screenshot the terminal showing both models complete and the final JSON result.
2. Include `artifacts/comparison.csv` as the model-comparison table.
3. Convert `artifacts/evaluation/confusion_matrix.csv` into a labelled heatmap for the report.
4. Discuss at least two frequent confusion pairs using `misclassified_samples.csv` rather than only quoting accuracy.
5. Screenshot the GUI with an uploaded digit and confidence output.
6. Link the Git commits for the model experiment, GUI, extension and tests.

Do not invent results. Final accuracy and timing depend on the hardware and must come from the full run.

## Integration contract for Person B

Person B should pass each segmented digit crop through `prepare_digit_image`, then call the loaded model once for the full batch. The current GUI accepts multiple pre-segmented image files as an interim path. During Phase 3, replace that file list with left-to-right crops from the segmentation module; the GUI does not need to be redesigned.

