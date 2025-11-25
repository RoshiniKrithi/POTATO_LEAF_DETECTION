# 📅 GitHub Upload Schedule - Day-by-Day Plan

## 📋 Overview
This guide breaks down exactly what files to push each day to upload your Potato Leaf Disease Detection project to GitHub.

---

## 📁 Files to Include/Exclude

### ✅ **INCLUDE in GitHub:**
- All Python source files (`.py`)
- HTML templates (`.html`)
- CSS and JavaScript files
- Configuration files (`requirements.txt`, `setup.py`)
- Documentation (`README.md`, `WEB_APP_GUIDE.md`)
- Scripts folder (export utilities)
- `.gitignore` file

### ❌ **EXCLUDE from GitHub:**
- Virtual environment (`.venv/`)
- Model files (`.pth`, `.onnx`, `.h5`) - too large
- Database file (`database.db`)
- Uploaded images (`static/uploads/*`)
- Checkpoints folder (`output/checkpoints/*`)
- Log files (`*.log`)
- Temporary CSV files (`predictions_*.csv`)

---

## 📅 DAY 1: Initial Setup & Configuration (30 minutes)

### **What to do:**
1. Create GitHub account (if you don't have one)
2. Initialize Git repository
3. Create and commit essential configuration files

### **Files to push:**
```
.gitignore
README.md
requirements.txt
setup.py
```

### **Commands:**
```bash
# Navigate to project directory
cd D:\potato_leaf_detection

# Initialize Git (if not already done)
git init

# Configure Git (replace with your info)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add configuration files
git add .gitignore
git commit -m "Add .gitignore to exclude unnecessary files"

git add README.md
git commit -m "Add comprehensive README with project documentation"

git add requirements.txt
git commit -m "Add Python dependencies file"

git add setup.py
git commit -m "Add package setup configuration"
```

### **Checklist:**
- [ ] Git initialized
- [ ] `.gitignore` created and committed
- [ ] `README.md` updated and committed
- [ ] `requirements.txt` committed
- [ ] `setup.py` committed (if exists)

---

## 📅 DAY 2: Create GitHub Repository & Push Base Files (30 minutes)

### **What to do:**
1. Create new repository on GitHub
2. Connect local repository to GitHub
3. Push initial commits

### **Steps on GitHub:**
1. Go to https://github.com
2. Click **"New repository"** (green button)
3. Repository name: `potato-leaf-disease-detection`
4. Description: `Flask web application for potato leaf disease detection using PyTorch and ONNX Runtime`
5. Visibility: **Public** (or Private if you prefer)
6. **DO NOT** check "Initialize with README" (we already have one)
7. Click **"Create repository"**

### **Commands:**
```bash
# Connect to GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/potato-leaf-disease-detection.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### **Checklist:**
- [ ] GitHub repository created
- [ ] Local repo connected to GitHub
- [ ] Initial files pushed successfully
- [ ] Can see files on GitHub website

---

## 📅 DAY 3: Push Flask Web Application (45 minutes)

### **What to do:**
Add the complete Flask web application with all routes and functionality.

### **Files to push:**
```
app.py
templates/
  - base.html
  - login.html
  - register.html
  - home.html
  - history.html
  - monitor.html
static/
  - css/
    - styles.css
  - js/
    - main.js
    - monitor.js
```

### **Commands:**
```bash
# Add Flask application
git add app.py
git commit -m "Add Flask web application with ONNX Runtime inference, user authentication, and prediction history"

# Add HTML templates
git add templates/
git commit -m "Add HTML templates (login, register, home, history pages)"

# Add static assets
git add static/css/
git commit -m "Add custom CSS styling for modern UI"

git add static/js/
git commit -m "Add JavaScript for image preview and interactive features"

# Push to GitHub
git push origin main
```

### **Checklist:**
- [ ] `app.py` committed
- [ ] All templates committed
- [ ] CSS files committed
- [ ] JavaScript files committed
- [ ] All changes pushed to GitHub

---

## 📅 DAY 4: Push Training & ML Pipeline (45 minutes)

### **What to do:**
Add all machine learning training scripts, utilities, and model definitions.

### **Files to push:**
```
train.py
evaluate.py
infer.py
models/
  - __init__.py
  - factory.py
datasets/
  - __init__.py
  - leaf_dataset.py
utils/
  - __init__.py
  - checkpoint.py
  - metrics.py
  - transforms.py
  - logging_utils.py
  - seed.py
  - db.py
  - eda_dataset.py
  - gradcam_cli.py
exports/
  - __init__.py
  - onnx_export.py
  - torchscript_export.py
scripts/
  - export_to_onnx.py
  - convert_onnx_to_h5.py
  - export_checkpoint_to_h5.py
```

### **Commands:**
```bash
# Add training script
git add train.py
git commit -m "Add PyTorch training script with EfficientNet support"

# Add evaluation script
git add evaluate.py
git commit -m "Add model evaluation script with metrics and confusion matrix"

# Add inference script
git add infer.py
git commit -m "Add batch inference script for image prediction"

# Add model definitions
git add models/
git commit -m "Add model factory and architecture definitions"

# Add dataset utilities
git add datasets/
git commit -m "Add dataset loading and preprocessing utilities"

# Add utility modules
git add utils/
git commit -m "Add utility modules (checkpoint, metrics, transforms, logging, etc.)"

# Add export scripts
git add exports/
git commit -m "Add model export scripts (ONNX, TorchScript)"

# Add conversion scripts
git add scripts/
git commit -m "Add model conversion scripts (PyTorch to ONNX)"

# Push to GitHub
git push origin main
```

### **Checklist:**
- [ ] Training script committed
- [ ] Evaluation script committed
- [ ] Model definitions committed
- [ ] Dataset utilities committed
- [ ] All utility modules committed
- [ ] Export scripts committed
- [ ] All changes pushed to GitHub

---

## 📅 DAY 5: Documentation, Tests & Final Polish (1 hour)

### **What to do:**
Add remaining documentation, tests, and finalize the repository.

### **Files to push:**
```
WEB_APP_GUIDE.md
GITHUB_UPLOAD_SCHEDULE.md (or GITHUB_PUSH_SCHEDULE.md)
tests/
  - __init__.py
  - test_dataset.py
  - test_smoke.py
__init__.py
```

### **Commands:**
```bash
# Add documentation
git add WEB_APP_GUIDE.md
git commit -m "Add web application setup and usage guide"

git add GITHUB_PUSH_SCHEDULE.md
git commit -m "Add GitHub upload schedule documentation"

# Add tests
git add tests/
git commit -m "Add unit tests and smoke tests"

# Add package init file
git add __init__.py
git commit -m "Add package initialization file"

# Push everything
git push origin main
```

### **Final Steps on GitHub:**
1. **Add repository topics/tags:**
   - Go to your repository on GitHub
   - Click on the gear icon (⚙️) next to "About"
   - Add topics: `flask`, `pytorch`, `onnx`, `machine-learning`, `potato-disease-detection`, `computer-vision`, `deep-learning`, `image-classification`

2. **Add repository description:**
   - Update the description if needed

3. **Create a release (optional but recommended):**
   - Go to "Releases" → "Create a new release"
   - Tag: `v1.0.0`
   - Title: `Initial Release - Potato Leaf Disease Detection Web App`
   - Description:
     ```
     ## Features
     - Flask web application for real-time disease detection
     - PyTorch training pipeline with EfficientNet
     - ONNX Runtime inference
     - User authentication and prediction history
     - Modern, responsive UI
     ```

4. **Verify repository:**
   - Check all files are present
   - Verify no sensitive data is exposed
   - Test cloning the repository in a new location

### **Checklist:**
- [ ] All documentation committed
- [ ] Tests committed
- [ ] Repository topics added
- [ ] Release created (optional)
- [ ] Repository verified and complete

---

## 📊 Complete File Structure to Push

```
potato_leaf_detection/
├── .gitignore                    ✅ Day 1
├── README.md                     ✅ Day 1
├── requirements.txt              ✅ Day 1
├── setup.py                      ✅ Day 1
├── __init__.py                   ✅ Day 5
│
├── app.py                        ✅ Day 3
│
├── train.py                      ✅ Day 4
├── evaluate.py                   ✅ Day 4
├── infer.py                      ✅ Day 4
│
├── templates/                    ✅ Day 3
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── home.html
│   ├── history.html
│   └── monitor.html
│
├── static/                       ✅ Day 3
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── main.js
│       └── monitor.js
│
├── models/                       ✅ Day 4
│   ├── __init__.py
│   └── factory.py
│
├── datasets/                     ✅ Day 4
│   ├── __init__.py
│   └── leaf_dataset.py
│
├── utils/                        ✅ Day 4
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── metrics.py
│   ├── transforms.py
│   ├── logging_utils.py
│   ├── seed.py
│   ├── db.py
│   ├── eda_dataset.py
│   └── gradcam_cli.py
│
├── exports/                      ✅ Day 4
│   ├── __init__.py
│   ├── onnx_export.py
│   └── torchscript_export.py
│
├── scripts/                      ✅ Day 4
│   ├── export_to_onnx.py
│   ├── convert_onnx_to_h5.py
│   └── export_checkpoint_to_h5.py
│
├── tests/                        ✅ Day 5
│   ├── __init__.py
│   ├── test_dataset.py
│   └── test_smoke.py
│
└── Documentation/                ✅ Day 5
    ├── WEB_APP_GUIDE.md
    └── GITHUB_PUSH_SCHEDULE.md
```

---

## 🚫 Files NOT to Push (Already in .gitignore)

```
.venv/                    ❌ Virtual environment
output/checkpoints/*.pth  ❌ Model checkpoints (too large)
model/*.onnx              ❌ ONNX models (too large)
model/*.h5                ❌ Keras models (too large)
database.db               ❌ Database file
static/uploads/*          ❌ Uploaded images
*.log                     ❌ Log files
predictions_*.csv         ❌ Temporary CSV files
__pycache__/             ❌ Python cache
*.pyc                     ❌ Compiled Python
```

---

## 📝 Quick Reference Commands

### **Check what will be committed:**
```bash
git status
```

### **See what files are tracked:**
```bash
git ls-files
```

### **Check if file is ignored:**
```bash
git check-ignore -v <filename>
```

### **Undo last commit (if needed):**
```bash
git reset --soft HEAD~1
```

### **View commit history:**
```bash
git log --oneline
```

---

## ⏱️ Time Estimate

- **Day 1:** 30 minutes
- **Day 2:** 30 minutes
- **Day 3:** 45 minutes
- **Day 4:** 45 minutes
- **Day 5:** 1 hour

**Total:** ~3.5 hours over 5 days

---

## ✅ Final Verification Checklist

Before considering the upload complete:

- [ ] All source code files are on GitHub
- [ ] No sensitive data (passwords, API keys) in code
- [ ] No large model files (use Git LFS if needed)
- [ ] README is comprehensive
- [ ] Repository has proper description and topics
- [ ] Can clone and run the project from GitHub
- [ ] All commits have meaningful messages

---

## 🎉 You're Done!

Your project is now on GitHub and ready to share with the world!

**Repository URL:** `https://github.com/YOUR_USERNAME/potato-leaf-disease-detection`

Good luck! 🚀

