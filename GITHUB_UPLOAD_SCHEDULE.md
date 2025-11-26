# 📅 GitHub Upload Schedule - 5 Day Plan

## Day 1: Preparation & Setup (30-45 minutes)

### Tasks:
1. **Create `.gitignore` file** ✅ (Already created)
   - Exclude virtual environment, cache files, model files, database
   - Keep only source code and essential files

2. **Review project structure**
   - Check what files should be included/excluded
   - Remove any sensitive data (API keys, passwords in code)

3. **Clean up temporary files**
   ```bash
   # Remove any test files, temporary CSVs, etc.
   ```

4. **Initialize Git repository** (if not already done)
   ```bash
   git init
   git config user.name "Your Name"
   git config user.email "your.email@example.com"
   ```

### Deliverables:
- ✅ `.gitignore` file created
- ✅ Project cleaned up
- ✅ Git initialized

---

## Day 2: Create GitHub Repository & First Commit (30 minutes)

### Tasks:
1. **Create GitHub account** (if you don't have one)
   - Go to https://github.com
   - Sign up for free account

2. **Create new repository on GitHub**
   - Click "New repository"
   - Name: `potato-leaf-disease-detection` (or your preferred name)
   - Description: "Potato Leaf Disease Detection Web App with Flask and ONNX Runtime"
   - Visibility: Public (or Private if preferred)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)

3. **Make first commit locally**
   ```bash
   git add .gitignore
   git commit -m "Add .gitignore file"
   
   git add README.md
   git commit -m "Add README documentation"
   
   git add requirements.txt
   git commit -m "Add requirements.txt"
   ```

4. **Connect to GitHub and push**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/potato-leaf-disease-detection.git
   git branch -M main
   git push -u origin main
   ```

### Deliverables:
- ✅ GitHub repository created
- ✅ First commit pushed to GitHub

---

## Day 3: Add Core Application Files (45 minutes)

### Tasks:
1. **Add Flask application files**
   ```bash
   git add app.py
   git commit -m "Add Flask web application with ONNX Runtime inference"
   
   git add templates/
   git commit -m "Add HTML templates (login, register, home, history)"
   
   git add static/
   git commit -m "Add CSS and JavaScript files"
   ```

2. **Add training scripts**
   ```bash
   git add train.py
   git commit -m "Add PyTorch training script"
   
   git add evaluate.py
   git commit -m "Add evaluation script"
   
   git add infer.py
   git commit -m "Add inference script"
   ```

3. **Add utility modules**
   ```bash
   git add utils/
   git commit -m "Add utility modules (checkpoint, metrics, transforms, etc.)"
   
   git add models/
   git commit -m "Add model factory and architecture definitions"
   
   git add datasets/
   git commit -m "Add dataset loading utilities"
   ```

4. **Push to GitHub**
   ```bash
   git push origin main
   ```

### Deliverables:
- ✅ All core application files committed
- ✅ All commits pushed to GitHub

---

## Day 4: Documentation & Scripts (1 hour)

### Tasks:
1. **Update README.md** with web app information
   - Add web app section
   - Include setup instructions
   - Add screenshots (optional)
   - Add deployment instructions

2. **Add export scripts** (if not already added)
   ```bash
   git add scripts/
   git commit -m "Add model export scripts (ONNX conversion)"
   ```

3. **Create setup instructions file**
   ```bash
   # Create SETUP.md or add to README
   git add SETUP.md  # or update README
   git commit -m "Add detailed setup instructions"
   ```

4. **Add license file** (if needed)
   ```bash
   # Create LICENSE file (MIT, Apache, etc.)
   git add LICENSE
   git commit -m "Add MIT license"
   ```

5. **Push documentation**
   ```bash
   git push origin main
   ```

### Deliverables:
- ✅ Complete README with web app info
- ✅ Setup instructions
- ✅ License file (optional)

---

## Day 5: Final Polish & Verification (30 minutes)

### Tasks:
1. **Review repository on GitHub**
   - Check all files are present
   - Verify no sensitive data is exposed
   - Check file structure looks good

2. **Add repository topics/tags**
   - On GitHub, go to repository settings
   - Add topics: `flask`, `pytorch`, `onnx`, `machine-learning`, `potato-disease-detection`, `computer-vision`

3. **Create a release** (optional but recommended)
   - Go to "Releases" → "Create a new release"
   - Tag: `v1.0.0`
   - Title: "Initial Release - Potato Leaf Disease Detection Web App"
   - Description: Brief overview of features

4. **Final commit** (if any last-minute changes)
   ```bash
   git add .
   git commit -m "Final cleanup and documentation updates"
   git push origin main
   ```

5. **Test the repository**
   - Clone it in a new directory to verify everything works
   - Check that README instructions are clear

### Deliverables:
- ✅ Repository fully documented
- ✅ All files verified
- ✅ Release created (optional)
- ✅ Repository ready for sharing

---

## Quick Reference Commands

### Initial Setup
```bash
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Daily Workflow
```bash
# Check status
git status

# Add files
git add <file_or_directory>

# Commit
git commit -m "Descriptive commit message"

# Push to GitHub
git push origin main
```

### If you need to update later
```bash
git add .
git commit -m "Update: description of changes"
git push origin main
```

---

## Important Notes

⚠️ **DO NOT commit:**
- Virtual environment (`.venv/`)
- Model files (`.onnx`, `.pth`, `.h5`) - too large
- Database files (`database.db`)
- Uploaded images in `static/uploads/`
- Personal API keys or secrets

✅ **DO commit:**
- Source code (`.py` files)
- Templates (`.html` files)
- Static assets (CSS, JS)
- Configuration files
- Documentation (README, etc.)
- Requirements file

---

## Estimated Total Time: 3-4 hours over 5 days

Good luck with your GitHub upload! 🚀

