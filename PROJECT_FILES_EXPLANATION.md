# 📚 Complete Project Files Explanation

This document explains every file in the Potato Leaf Disease Detection project, what it does, and how it fits into the overall system.

---

## 🎯 **MAIN APPLICATION FILES**

### **1. `app.py` - Flask Web Application**
**Purpose:** Main web server that handles user requests and serves the web interface.

**Key Functions:**
- **`init_db()`** - Creates SQLite database tables (users, predictions)
- **`ensure_default_user()`** - Creates default admin user if no users exist
- **`load_trained_model()`** - Loads ONNX model for inference
- **`preprocess_image()`** - Resizes and normalizes images to 380x380, converts to NCHW format
- **`predict_image()`** - Runs inference using ONNX Runtime, returns prediction and confidence
- **`login_required()`** - Decorator to protect routes that need authentication

**Routes:**
- `/` - Redirects to login or home
- `/login` - User login page (GET/POST)
- `/register` - User registration page (GET/POST)
- `/logout` - Logs out user
- `/home` - Main page with image upload form
- `/predict` - API endpoint for image prediction (POST)
- `/history` - Shows user's prediction history
- `/history/download` - Downloads history as CSV
- `/monitor` - Shows available checkpoint files

**How it works:**
1. User uploads image → Flask receives it
2. Image is preprocessed (resize, normalize)
3. ONNX Runtime runs inference
4. Result saved to database
5. JSON response sent to frontend

---

### **2. `train.py` - Model Training Script**
**Purpose:** Trains the deep learning model using PyTorch.

**Key Functions:**
- **`parse_args()`** - Parses command-line arguments (epochs, batch size, learning rate, etc.)
- **`FocalLoss`** - Custom loss function for handling class imbalance
- **`main()`** - Main training loop

**What it does:**
1. Loads dataset and creates data loaders
2. Creates model (EfficientNet-B2 by default)
3. Sets up optimizer (AdamW) and scheduler (CosineAnnealingWarmRestarts)
4. Trains for specified epochs:
   - Forward pass through model
   - Calculate loss
   - Backward pass (gradient computation)
   - Update weights
5. Validates on validation set each epoch
6. Saves checkpoints every N epochs
7. Saves best model based on F1 score
8. Logs metrics to TensorBoard and SQLite database

**Output:**
- Checkpoint files in `output/checkpoints/`
- Best model: `output/checkpoints/best_model.pth`
- TensorBoard logs in `output/tb/`
- Training logs in `output/train.log`

**Usage:**
```bash
python train.py --data-dir ./potato_dataset --epochs 60 --batch-size 32 --lr 2e-4
```

---

### **3. `evaluate.py` - Model Evaluation Script**
**Purpose:** Evaluates trained model on test/validation set and generates metrics.

**Key Functions:**
- Loads trained checkpoint
- Runs inference on test/validation set
- Computes metrics (accuracy, precision, recall, F1-score)
- Generates confusion matrix visualization
- Saves predictions to CSV

**What it does:**
1. Loads model from checkpoint
2. Runs inference on all images in test/val split
3. Collects predictions and true labels
4. Calculates metrics:
   - Overall accuracy
   - Per-class precision, recall, F1
   - Macro-averaged F1
   - Confusion matrix
5. Saves confusion matrix as PNG image
6. Saves predictions to CSV file

**Output:**
- `output/eval/confusion_matrix.png` - Visual confusion matrix
- `output/eval/predictions.csv` - All predictions with confidence scores
- `output/eval/metrics.json` - Numerical metrics

**Usage:**
```bash
python evaluate.py --checkpoint output/checkpoints/best_model.pth --split val
```

---

### **4. `infer.py` - Batch Inference Script**
**Purpose:** Runs inference on a folder of images using trained model.

**Key Functions:**
- Loads model checkpoint
- Processes all images in a folder
- Saves predictions to CSV

**What it does:**
1. Loads model from checkpoint
2. Iterates through all images in specified folder
3. Preprocesses each image
4. Runs inference
5. Saves results to CSV with filename and prediction

**Output:**
- `preds.csv` - Predictions for all images

**Usage:**
```bash
python infer.py --model output/checkpoints/best_model.pth --images ./test_images --out predictions.csv
```

---

## 🗂️ **MODEL & DATASET FILES**

### **5. `models/factory.py` - Model Factory**
**Purpose:** Creates model instances based on model name.

**Key Functions:**
- **`create_model()`** - Factory function that creates different model architectures

**Supported Models:**
- `efficientnet_b0` through `efficientnet_b4`
- `resnet50`
- `mobilenet_v3_large`

**What it does:**
1. Takes model name and number of classes
2. Creates appropriate model architecture
3. Replaces final layer for classification
4. Returns model and default image size

**Example:**
```python
model, image_size = create_model("efficientnet_b2", num_classes=3, pretrained=True)
```

---

### **6. `datasets/leaf_dataset.py` - Dataset Loading**
**Purpose:** Handles loading and preprocessing of image datasets.

**Key Functions:**
- **`detect_dataset_layout()`** - Auto-detects dataset structure (per-class folders, CSV, etc.)
- **`build_dataloaders()`** - Creates PyTorch DataLoaders for train/val/test
- **`stratified_split_and_save()`** - Splits dataset into train/val/test with stratification

**Dataset Layouts Supported:**
1. **Canonical:** `train/<class>/*.jpg`, `val/<class>/*.jpg`
2. **Per-class:** `<class>/*.jpg` (auto-splits)
3. **CSV-based:** `images/*.jpg` + `labels.csv`

**What it does:**
1. Scans dataset directory
2. Detects layout automatically
3. Creates train/val/test splits (if needed)
4. Applies data augmentation (training) or normalization (validation)
5. Returns DataLoaders ready for training

---

## 🛠️ **UTILITY FILES**

### **7. `utils/checkpoint.py` - Checkpoint Management**
**Purpose:** Saves and loads model checkpoints.

**Key Functions:**
- **`save_checkpoint()`** - Saves model state, optimizer, scheduler, epoch, etc.
- **`load_checkpoint()`** - Loads checkpoint from file
- **`save_hparams_yaml()`** - Saves hyperparameters to JSON file

**What it saves:**
- Model weights (`model.state_dict()`)
- Optimizer state
- Scheduler state
- Current epoch
- Best metric value
- Gradient scaler (for mixed precision)

---

### **8. `utils/metrics.py` - Metrics Calculation**
**Purpose:** Computes classification metrics.

**Key Functions:**
- **`compute_classification_metrics()`** - Calculates accuracy, precision, recall, F1
- **`per_class_report()`** - Per-class metrics breakdown
- **`confusion_matrix_array()`** - Creates confusion matrix
- **`try_multiclass_roc_auc()`** - Calculates ROC-AUC (if applicable)

**Metrics Calculated:**
- Overall accuracy
- Precision, Recall, F1-score (per-class and macro-averaged)
- Confusion matrix

---

### **9. `utils/transforms.py` - Image Transformations**
**Purpose:** Defines data augmentation and preprocessing transforms.

**Key Functions:**
- **`build_transforms()`** - Creates transform pipelines for train/val/test

**Training Transforms:**
- Random horizontal flip
- Random rotation
- Color jitter
- Normalization

**Validation/Test Transforms:**
- Resize to model input size
- Normalization

**What it does:**
- Creates different transform pipelines for training (with augmentation) and validation (without)
- Uses Albumentations library for efficient augmentation

---

### **10. `utils/logging_utils.py` - Logging Setup**
**Purpose:** Configures logging for the application.

**Key Functions:**
- **`setup_logger()`** - Creates logger that writes to both console and file

**What it does:**
- Sets up Python logging
- Logs to both console (stdout) and file
- Formats log messages with timestamps

---

### **11. `utils/seed.py` - Reproducibility**
**Purpose:** Sets random seeds for reproducibility.

**Key Functions:**
- **`set_seed()`** - Sets seeds for Python, NumPy, PyTorch, CUDA

**What it does:**
- Sets random seeds to ensure reproducible results
- Important for scientific experiments

---

### **12. `utils/db.py` - Experiment Database**
**Purpose:** Logs training metrics to SQLite database.

**Key Functions:**
- **`ExperimentDB`** - Class for managing experiment logging
- **`log_epoch()`** - Logs epoch metrics
- **`log_checkpoint()`** - Logs checkpoint information

**What it does:**
- Creates SQLite database for tracking experiments
- Logs training metrics (loss, accuracy, F1) per epoch
- Tracks checkpoint saves

---

### **13. `utils/eda_dataset.py` - Exploratory Data Analysis**
**Purpose:** Analyzes dataset before training.

**Key Functions:**
- Analyzes class distribution
- Checks image sizes
- Detects corrupted files
- Suggests class imbalance solutions

**What it does:**
1. Scans all images in dataset
2. Counts images per class
3. Checks image dimensions
4. Identifies corrupted files
5. Generates report with recommendations

**Usage:**
```bash
python -m potato_leaf_detection.utils.eda_dataset --data-dir ./potato_dataset
```

---

### **14. `utils/gradcam_cli.py` - Grad-CAM Visualization**
**Purpose:** Generates Grad-CAM visualizations to see what the model focuses on.

**Key Functions:**
- Generates heatmaps showing which parts of image the model uses for prediction

**What it does:**
1. Loads model and image
2. Runs forward and backward pass
3. Extracts gradients from last convolutional layer
4. Creates heatmap overlay on image
5. Saves visualization

**Usage:**
```bash
python utils/gradcam_cli.py --checkpoint best_model.pth --images ./test_images
```

---

## 📤 **EXPORT FILES**

### **15. `exports/onnx_export.py` - ONNX Export**
**Purpose:** Exports PyTorch model to ONNX format for web app.

**Key Functions:**
- Loads PyTorch checkpoint
- Converts to ONNX format
- Saves as `.onnx` file

**What it does:**
1. Loads trained PyTorch model
2. Creates dummy input
3. Exports to ONNX using `torch.onnx.export()`
4. Saves ONNX file

**Usage:**
```bash
python exports/onnx_export.py --checkpoint best_model.pth --out model.onnx
```

---

### **16. `exports/torchscript_export.py` - TorchScript Export**
**Purpose:** Exports model to TorchScript format (alternative to ONNX).

**Key Functions:**
- Converts PyTorch model to TorchScript
- Saves as `.pt` file

**What it does:**
- Similar to ONNX export but uses TorchScript format
- Useful for mobile deployment

---

### **17. `scripts/export_to_onnx.py` - ONNX Conversion Script**
**Purpose:** Standalone script to convert checkpoint to ONNX.

**Key Functions:**
- Loads checkpoint
- Creates model
- Exports to ONNX

**Usage:**
```bash
python scripts/export_to_onnx.py --checkpoint best_model.pth --model-name efficientnet_b2 --num-classes 3
```

---

### **18. `scripts/convert_onnx_to_h5.py` - ONNX to Keras Converter**
**Purpose:** Converts ONNX model to TensorFlow/Keras `.h5` format (not used in final app).

**Note:** This was created during development but we ended up using ONNX Runtime instead.

---

## 🎨 **WEB INTERFACE FILES**

### **19. `templates/base.html` - Base Template**
**Purpose:** Base HTML template that other pages extend.

**What it contains:**
- HTML structure (head, body)
- Bootstrap CSS and JavaScript
- Navigation bar (Home, History, Logout)
- Flash message display
- Footer

**How it works:**
- Other templates extend this using `{% extends "base.html" %}`
- Provides consistent layout across all pages

---

### **20. `templates/login.html` - Login Page**
**Purpose:** User authentication page.

**What it contains:**
- Username and password input fields
- Login form
- Link to registration page
- Demo account info

**Functionality:**
- Submits credentials to `/login` route
- Shows error messages if login fails
- Redirects to home on success

---

### **21. `templates/register.html` - Registration Page**
**Purpose:** New user registration page.

**What it contains:**
- Username input
- Password and confirm password fields
- Registration form
- Link back to login

**Functionality:**
- Validates password match
- Checks username uniqueness
- Creates new user account
- Redirects to login after registration

---

### **22. `templates/home.html` - Main Upload Page**
**Purpose:** Image upload and prediction interface.

**What it contains:**
- File upload input
- Image preview area
- "Analyze Leaf" button
- Results display panel

**Functionality:**
- Shows image preview before upload
- Submits image to `/predict` endpoint via AJAX
- Displays prediction results (Healthy/Diseased)
- Shows confidence score and description

---

### **23. `templates/history.html` - Prediction History**
**Purpose:** Shows user's past predictions.

**What it contains:**
- Table with all predictions
- Search bar
- Download CSV button
- Image thumbnails

**Functionality:**
- Displays all user's predictions
- Search/filter functionality
- Color-coded badges (green=Healthy, red=Diseased)
- CSV download

---

### **24. `templates/monitor.html` - Checkpoint Monitor**
**Purpose:** Shows available checkpoint files (for developers).

**What it contains:**
- List of checkpoint files
- File sizes and dates
- Download links

---

## 🎨 **STATIC FILES**

### **25. `static/css/styles.css` - Custom Styling**
**Purpose:** Custom CSS for modern, beautiful UI.

**What it contains:**
- Custom color scheme (green theme for agriculture)
- Card styling with shadows
- Responsive design
- Button and form styling
- Animation effects

**Key Features:**
- Modern gradient backgrounds
- Rounded corners
- Smooth transitions
- Mobile-responsive

---

### **26. `static/js/main.js` - Frontend JavaScript**
**Purpose:** Handles client-side interactions.

**Key Functions:**
- Image preview before upload
- Form submission via AJAX
- Loading spinner display
- Result display

**What it does:**
1. Shows image preview when file selected
2. Submits form to `/predict` endpoint
3. Shows loading spinner during prediction
4. Displays results when prediction completes
5. Handles errors gracefully

---

### **27. `static/js/monitor.js` - Monitor Page JavaScript**
**Purpose:** JavaScript for checkpoint monitor page (if needed).

---

## ⚙️ **CONFIGURATION FILES**

### **28. `requirements.txt` - Python Dependencies**
**Purpose:** Lists all Python packages needed for the project.

**Key Dependencies:**
- `torch`, `torchvision` - PyTorch for training
- `flask`, `werkzeug` - Web framework
- `onnxruntime` - ONNX model inference
- `numpy`, `pandas` - Data processing
- `Pillow` - Image processing
- `albumentations` - Data augmentation
- `timm` - Model architectures

**Usage:**
```bash
pip install -r requirements.txt
```

---

### **29. `setup.py` - Package Configuration**
**Purpose:** Makes project installable as Python package.

**What it does:**
- Defines package metadata
- Lists dependencies
- Allows installation with `pip install -e .`

---

### **30. `.gitignore` - Git Ignore Rules**
**Purpose:** Tells Git which files to ignore.

**What it excludes:**
- Virtual environments
- Model files (too large)
- Database files
- Cache files
- Log files

---

## 🧪 **TEST FILES**

### **31. `tests/test_dataset.py` - Dataset Tests**
**Purpose:** Unit tests for dataset loading functionality.

**What it tests:**
- Dataset layout detection
- DataLoader creation
- Image loading

---

### **32. `tests/test_smoke.py` - Smoke Tests**
**Purpose:** Quick tests to verify basic functionality.

**What it tests:**
- Can import modules
- Can create model
- Basic functionality works

---

## 📝 **DOCUMENTATION FILES**

### **33. `README.md` - Project Documentation**
**Purpose:** Main project documentation.

**What it contains:**
- Project overview
- Features list
- Installation instructions
- Usage examples
- Project structure

---

### **34. `WEB_APP_GUIDE.md` - Web App Guide**
**Purpose:** Detailed guide for setting up and using the web application.

**What it contains:**
- Setup instructions
- Running the app
- Using features
- Troubleshooting

---

### **35. `GITHUB_PUSH_SCHEDULE.md` - GitHub Upload Guide**
**Purpose:** Step-by-step guide for uploading to GitHub.

---

## 🔧 **SUPPORTING FILES**

### **36. `__init__.py` - Package Initialization**
**Purpose:** Makes directories Python packages.

**What it does:**
- Empty file that marks directory as Python package
- Allows imports like `from potato_leaf_detection.models import ...`

---

### **37. `ai_edge_litert/interpreter.py` - TFLite Shim**
**Purpose:** Compatibility shim for ONNX conversion tools.

**Note:** Created during development for model conversion, not used in final app.

---

## 📊 **DATA FILES (Not in Git)**

### **38. `database.db` - SQLite Database**
**Purpose:** Stores users and predictions.

**Tables:**
- `users` - User accounts
- `predictions` - All predictions made through web app

**Note:** Not committed to Git (in `.gitignore`)

---

### **39. `model/potato_disease_model.onnx` - Trained Model**
**Purpose:** ONNX format model used by web app.

**Note:** Not committed to Git (too large)

---

### **40. `output/checkpoints/*.pth` - Training Checkpoints**
**Purpose:** Saved model states during training.

**Note:** Not committed to Git (too large)

---

## 🔄 **HOW FILES WORK TOGETHER**

### **Training Flow:**
1. `train.py` uses `models/factory.py` to create model
2. `datasets/leaf_dataset.py` loads images
3. `utils/transforms.py` applies augmentation
4. Training loop runs, saves checkpoints via `utils/checkpoint.py`
5. Metrics logged via `utils/db.py` and `utils/metrics.py`

### **Web App Flow:**
1. User visits `app.py` (Flask server)
2. `templates/*.html` render pages
3. `static/css/styles.css` styles pages
4. `static/js/main.js` handles interactions
5. User uploads image → `app.py` preprocesses
6. `app.py` loads ONNX model → runs inference
7. Result saved to `database.db`
8. History shown from database

### **Export Flow:**
1. `train.py` saves PyTorch checkpoint
2. `scripts/export_to_onnx.py` converts to ONNX
3. ONNX file used by `app.py` for web inference

---

## 📋 **Summary**

| Category | Files | Purpose |
|----------|-------|---------|
| **Web App** | `app.py`, `templates/*`, `static/*` | User interface and API |
| **Training** | `train.py`, `models/*`, `datasets/*` | Model training pipeline |
| **Evaluation** | `evaluate.py`, `infer.py` | Model testing and inference |
| **Utilities** | `utils/*` | Helper functions |
| **Export** | `exports/*`, `scripts/*` | Model format conversion |
| **Config** | `requirements.txt`, `setup.py` | Project configuration |
| **Tests** | `tests/*` | Unit and smoke tests |
| **Docs** | `README.md`, `*.md` | Documentation |

---

This completes the explanation of all files in your project! 🎉

