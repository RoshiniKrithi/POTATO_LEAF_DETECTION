# Potato Leaf Disease Detection

Production-ready training, evaluation, export, and inference pipeline with a **Flask web application** for real-time potato leaf disease detection. Supports auto-detection of dataset layouts and transfer learning with modern backbones.

## Features

### 🌐 Web Application
- **Flask-based web interface** for easy image upload and prediction
- User authentication (login/register)
- Real-time disease detection using ONNX Runtime
- Prediction history with search and CSV export
- Responsive, modern UI with Bootstrap
- Secure session management.

### 🤖 Machine Learning Pipeline
- Auto-detects dataset at `--data-dir` with support for:
  1. `train/val(/test)/<class>/*.jpg`
  2. `<data>/<class>/*.jpg` (stratified 80/10/10 split written to disk)
  3. `images/*.jpg` + `labels.csv (filename,label)`
- Transfer learning (EfficientNet, ResNet50, MobileNetV3) using pretrained weights
- Mixed precision (AMP), cosine warm restarts, AdamW
- Class imbalance: class weights, optional oversampler, focal loss
- TensorBoard logging, checkpoints (epoch and best), resume
- EDA utility (classes, counts, sizes, corrupted files, imbalance suggestions)
- Evaluation: metrics (acc/prec/recall/F1 macro & per-class), confusion matrix PNG, optional ROC-AUC
- Exports: TorchScript, ONNX (dynamic axes)
- Grad-CAM visualization CLI
- Unit tests and a quick smoke test

## Quickstart

### Web Application Setup

```bash
# 1) Create virtual environment
python -m venv .venv

# 2) Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Export your trained model to ONNX (if not already done)
python scripts/export_to_onnx.py --checkpoint output/checkpoints/best_model.pth --model-name efficientnet_b2 --num-classes 3 --output model/potato_disease_model.onnx

# 5) Run the Flask application
python app.py

# 6) Open browser and go to http://127.0.0.1:5000
# Default login: admin / admin123
```

### Training Pipeline

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) (Optional) Inspect dataset
python -m potato_leaf_detection.utils.eda_dataset --data-dir ./potato_dataset --out ./output/eda

# 3) Train
python train.py --data-dir ./potato_dataset --model efficientnet_b2 --epochs 60 --batch-size 32 --lr 2e-4 --amp

# 4) Resume example
python train.py --data-dir ./potato_dataset --model efficientnet_b2 --epochs 60 --batch-size 32 --lr 2e-4 --amp \
  --resume ./output/checkpoints/ckpt_epoch_30.pth

# 5) Evaluate (generates confusion matrix + CSV of preds)
python evaluate.py --data-dir ./potato_dataset --checkpoint ./output/best_model.pth --split val --out ./output/eval

# 6) Export
python exports/torchscript_export.py --checkpoint ./output/best_model.pth --out ./output/exports/model_ts.pt
python exports/onnx_export.py --checkpoint ./output/best_model.pth --out ./output/exports/model.onnx

# 7) Inference on images folder
python infer.py --model ./output/best_model.pth --images ./some_images --out ./preds.csv --device cuda

# 8) Grad-CAM
python utils/gradcam_cli.py --checkpoint ./output/best_model.pth --images ./some_images --out ./output/gradcam
```

## Dataset Layouts
- Canonical: `potato_dataset/train/<class>/*.jpg`, `potato_dataset/val/<class>/*.jpg`, optional `test/<class>/*.jpg`.
- Single-folder per class: `potato_dataset/<class>/*.jpg` (will be split to `train/val/test` and saved as CSV split files).
- Flat with CSV: `potato_dataset/images/*.jpg` and `potato_dataset/labels.csv` with `filename,label`.

If images are present without labels, you will be prompted to create labels interactively or automatic labeling will be attempted if filenames contain class names.

## Reproducibility
- Deterministic seeds and cuDNN controls via `--seed` (default 42)
- All hyperparameters saved to `output/experiment.yaml`

## Models
- Default: `efficientnet_b2`
- Options: `resnet50`, `efficientnet_b0..b4`, `mobilenet_v3_large`

## Outputs
- `output/` contains TensorBoard logs, checkpoints, confusion matrix PNG, metrics JSON/CSV, and exports.

## Web App Features

- **Login/Register**: Secure user authentication with password hashing
- **Image Upload**: Drag-and-drop or click to upload potato leaf images
- **Real-time Prediction**: Get instant disease detection results with confidence scores
- **History**: View all past predictions with search functionality
- **CSV Export**: Download prediction history as CSV file
- **Multi-user Support**: Each user has their own prediction history

## Project Structure

```
potato_leaf_detection/
├── app.py                 # Flask web application
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── infer.py              # Inference script
├── templates/            # HTML templates
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── home.html
│   └── history.html
├── static/               # Static assets
│   ├── css/
│   ├── js/
│   └── uploads/
├── models/               # Model definitions
├── datasets/             # Dataset utilities
├── utils/                # Utility functions
├── scripts/              # Export scripts
└── model/                # Trained models (ONNX)
```

## Tests
```bash
pytest -q
```

## License
MIT

