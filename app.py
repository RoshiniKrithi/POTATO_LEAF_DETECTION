from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database.db"
MODEL_PATH = BASE_DIR / "model" / "potato_disease_model.onnx"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMAGE_SIZE = (380, 380)  # EfficientNet-B2 uses 380x380 input

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("POTATO_APP_SECRET", "super-secret-potato-key")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=6)

ort_session: Optional[ort.InferenceSession] = None
model_load_error: Optional[str] = None
_app_ready = False


def init_db() -> None:
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            image_path TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            description TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def ensure_default_user() -> None:
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users;")
    count = cur.fetchone()[0]
    if count == 0:
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?);",
            (
                "admin",
                generate_password_hash("admin123"),
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    conn.close()


def load_trained_model() -> None:
    global ort_session, model_load_error
    if ort_session is not None:
        return
    if not MODEL_PATH.exists():
        model_load_error = f"Model file not found at {MODEL_PATH}. Please export the ONNX model."
        return
    try:
        ort_session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # pragma: no cover
        model_load_error = f"Unable to load ONNX model at {MODEL_PATH}: {exc}"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path: Path) -> np.ndarray:
    """Preprocess image for ONNX model: resize to 380x380, normalize, convert to NCHW format."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)  # Resize to 380x380
    array = np.array(image, dtype=np.float32) / 255.0  # Shape: (H, W, C) = (380, 380, 3)
    # Convert from NHWC to NCHW format: (H, W, C) -> (C, H, W)
    array = np.transpose(array, (2, 0, 1))  # Shape: (3, 380, 380)
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    array = np.expand_dims(array, axis=0)  # Shape: (1, 3, 380, 380)
    return array


def predict_image(image_path: Path) -> Dict[str, Any]:
    if ort_session is None:
        raise RuntimeError(model_load_error or "Model is not loaded.")
    input_tensor = preprocess_image(image_path)
    inputs = {ort_session.get_inputs()[0].name: input_tensor.astype(np.float32)}
    outputs = ort_session.run(None, inputs)
    logits = outputs[0][0]
    label_idx = int(np.argmax(logits))
    confidence = float(softmax(logits)[label_idx])
    label = "Healthy" if label_idx == 0 else "Diseased"
    description = (
        "This leaf appears healthy. Continue regular monitoring."
        if label == "Healthy"
        else "Signs of disease detected. Remove infected leaves and consider treatment."
    )
    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "description": description,
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    e_x = np.exp(logits - np.max(logits))
    return e_x / np.sum(e_x)


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def setup_app() -> None:
    global _app_ready
    if _app_ready:
        return
    init_db()
    ensure_default_user()
    load_trained_model()
    _app_ready = True


@app.before_request
def ensure_setup() -> None:  # pragma: no cover
    setup_app()


@app.route("/")
def root() -> Any:
    if "username" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login() -> Any:
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        conn = sqlite3.connect(DATABASE_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username = ?;", (username,))
        row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row[0], password):
            session["username"] = username
            session.permanent = True
            flash("Welcome back!", "success")
            return redirect(url_for("home"))
        flash("Invalid username or password. Please try again.", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register() -> Any:
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("register.html")
        
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")
        
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template("register.html")
        
        conn = sqlite3.connect(DATABASE_PATH)
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?);",
                (
                    username,
                    generate_password_hash(password),
                    datetime.utcnow().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Please choose a different username.", "danger")
        finally:
            conn.close()
    return render_template("register.html")


@app.route("/logout")
def logout() -> Any:
    session.pop("username", None)
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home() -> Any:
    return render_template("home.html", model_error=model_load_error)


@app.route("/predict", methods=["POST"])
@login_required
def predict() -> Any:
    if "image" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a PNG or JPG image."}), 400

    filename = secure_filename(f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}")
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    try:
        result = predict_image(save_path)
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500

    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    cur.execute(
        """
        INSERT INTO predictions (username, image_path, prediction, confidence, description, timestamp)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            session["username"],
            f"static/uploads/{filename}",
            result["label"],
            result["confidence"],
            result["description"],
            timestamp,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()

    payload = {
        "id": row_id,
        "username": session["username"],
        "image_path": f"/static/uploads/{filename}",
        "prediction": result["label"],
        "confidence": result["confidence"],
        "description": result["description"],
        "timestamp": timestamp,
    }
    return jsonify(payload)


def fetch_predictions(username: Optional[str] = None) -> list[Dict[str, Any]]:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if username:
        cur.execute(
            "SELECT * FROM predictions WHERE username = ? ORDER BY timestamp DESC;",
            (username,),
        )
    else:
        cur.execute("SELECT * FROM predictions ORDER BY timestamp DESC;")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


@app.route("/history", methods=["GET"])
@login_required
def history() -> Any:
    predictions = fetch_predictions(session.get("username"))
    wants_json = request.args.get("format") == "json" or request.accept_mimetypes.best == "application/json"
    if wants_json:
        return jsonify(predictions)
    return render_template("history.html", predictions=predictions)


@app.route("/history/download")
@login_required
def download_history() -> Any:
    predictions = fetch_predictions(session.get("username"))
    csv_path = BASE_DIR / f"predictions_{session['username']}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["id", "username", "image_path", "prediction", "confidence", "description", "timestamp"],
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)
    return send_file(csv_path, mimetype="text/csv", as_attachment=True, download_name="prediction_history.csv")


def main() -> None:
    init_db()
    ensure_default_user()
    load_trained_model()
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()


@app.route("/monitor")
@login_required
def monitor() -> Any:
    """Display available checkpoint files and basic metadata for monitoring/troubleshooting."""
    checkpoint_dir = BASE_DIR / "output" / "checkpoints"
    checkpoints: list[Dict[str, Any]] = []
    if checkpoint_dir.exists():
        for p in sorted(checkpoint_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_file():
                stat = p.stat()
                checkpoints.append(
                    {
                        "name": p.name,
                        "size_kb": round(stat.st_size / 1024, 2),
                        "mtime": datetime.utcfromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    }
                )
    return render_template("monitor.html", checkpoints=checkpoints)


@app.route("/monitor/download/<path:filename>")
@login_required
def download_checkpoint(filename: str) -> Any:
    # sanitize the filename and serve from the checkpoints folder
    safe_name = secure_filename(filename)
    target = BASE_DIR / "output" / "checkpoints" / safe_name
    if not target.exists() or not target.is_file():
        flash("Requested file not found.", "warning")
        return redirect(url_for("monitor"))
    return send_file(target, as_attachment=True)

