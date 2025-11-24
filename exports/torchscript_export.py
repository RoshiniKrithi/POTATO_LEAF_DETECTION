from __future__ import annotations

import argparse
from pathlib import Path

import torch

from potato_leaf_detection.models.factory import create_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--model-name", type=str, default="efficientnet_b2")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        num_classes = args.num_classes or ckpt.get("num_classes") or len(ckpt.get("idx_to_class", {})) or 2
        state = ckpt["model"]
    else:
        num_classes = args.num_classes or 2
        state = ckpt
    model, image_size = create_model(args.model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state)
    model.eval().to(device)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    ts = torch.jit.trace(model, dummy)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ts.save(args.out)
    print(f"Saved TorchScript to {args.out}")


if __name__ == "__main__":
    main()


