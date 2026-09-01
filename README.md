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
