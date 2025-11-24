from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from potato_leaf_detection.models.factory import create_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained PyTorch model checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth checkpoint (expects 'model' key).")
    parser.add_argument("--model-name", type=str, default="efficientnet_b2", help="Model name to instantiate via factory.")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of output classes.")
    parser.add_argument("--output", type=Path, default=Path("model") / "potato_disease_model.onnx", help="Target ONNX file path.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on (cpu/cuda).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--image-size", type=int, default=None, help="Override default square input size.")
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
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {args.output} (input size {image_size}x{image_size})")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
    print("ONNX export complete.")


if __name__ == "__main__":
    main()

