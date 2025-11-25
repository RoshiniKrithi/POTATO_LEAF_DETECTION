# Potato Leaf Detection Web App Guide

## 1. Prerequisites

- Python 3.10+
- Virtual environment (recommended)
- Trained TensorFlow/Keras model saved as `model/potato_disease_model.h5`

## 2. Setup

```bash
cd D:\potato_leaf_detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place your trained `.h5` file inside `model/` using the expected filename.

## 3. Initialize the App

```bash
python app.py
```

The first launch will:

- Create `database.db`
- Build tables (`users`, `predictions`)
- Seed demo user `admin / admin123`
- Load the TensorFlow model

## 4. VS Code Tips

- Open the workspace (`potato_leaf_detection.code-workspace`)
- Run the Flask app using the integrated terminal (`python app.py`)
- Use the "Python: Flask" debug configuration if preferred; set `FLASK_APP=app.py`

## 5. Using the App

1. Navigate to `http://127.0.0.1:5000`
2. Log in with `admin / admin123` (create new users directly in the SQLite DB if needed)
3. Upload an image on the Home page
4. Click **Analyze Leaf** and wait for the spinner to finish
5. Review results and confidence; every prediction is stored automatically
6. Go to **History** to see past predictions, search, or download the CSV

## 6. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | POST | Authenticates user credentials |
| `/predict` | POST | Accepts `image` file, returns prediction JSON |
| `/history` | GET | Renders history page or returns JSON (`?format=json`) |
| `/history/download` | GET | Downloads CSV for the logged-in user |

## 7. Testing Checklist

- **Login**: Submit wrong credentials (expect error), then correct credentials
- **Upload**: Use sample healthy/diseased leaf images
- **Prediction**: Confirm result card updates and entries appear in `database.db`
- **History**: Verify thumbnails, search bar, and CSV download
- **Session**: Use Logout button and ensure pages require authentication

## 8. Notes

- The model expects 224×224 RGB inputs normalized to `[0,1]`. Adjust `IMAGE_SIZE` or preprocessing in `app.py` if your model differs.
- To run without debug mode, set environment variable `FLASK_ENV=production`.
- Store a stronger secret key via `POTATO_APP_SECRET` environment variable before deploying.

