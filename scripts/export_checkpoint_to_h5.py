from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from pytorch2keras.converter import pytorch_to_keras

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from potato_leaf_detection.models.factory import create_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a PyTorch checkpoint directly into a Keras .h5 file.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="efficientnet_b2")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("model") / "potato_disease_model.h5")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--change-ordering", action="store_true", help="Use NHWC ordering (set if your model expects channels_last).")
    parser.add_argument("--name", type=str, default="potato_leaf_guard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model" not in checkpoint:
        raise KeyError("Checkpoint must contain a 'model' key with state_dict.")

    model, default_image_size = create_model(args.model_name, num_classes=args.num_classes, pretrained=False)
    image_size = args.image_size or default_image_size
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    print(f"Converting checkpoint to Keras (.h5). Image size: {image_size}x{image_size}")
    keras_model = pytorch_to_keras(
        model,
        dummy_input,
        [(3, image_size, image_size)],
        change_ordering=args.change_ordering,
        verbose=True,
        name=args.name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    keras_model.save(args.output)
    print(f"Saved Keras model to: {args.output}")


if __name__ == "__main__":
    main()

