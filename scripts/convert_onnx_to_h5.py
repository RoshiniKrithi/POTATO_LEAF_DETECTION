from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import tensorflow as tf
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onnx2tf import convert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an ONNX model into a TensorFlow/Keras .h5 model.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX file exported from PyTorch.")
    parser.add_argument("--output", type=Path, default=Path("model") / "potato_disease_model.h5", help="Destination .h5 file.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate SavedModel directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    temp_dir = Path(tempfile.mkdtemp(prefix="onnx2tf_"))
    print(f"Converting ONNX -> TensorFlow SavedModel (tmp: {temp_dir})")
    convert(
        input_onnx_file_path=str(args.onnx),
        output_folder_path=str(temp_dir),
        non_verbose=True,
    )

    print("Loading SavedModel and writing .h5...")
    model = tf.keras.models.load_model(temp_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output, include_optimizer=False)
    print(f"Saved TensorFlow model to: {args.output}")

    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"Intermediate SavedModel retained at {temp_dir}")


if __name__ == "__main__":
    if tf.__version__.startswith("1."):
        sys.exit("TensorFlow 2.x is required.")
    main()

