from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from potato_leaf_detection.models.factory import create_model
from potato_leaf_detection.utils.transforms import build_transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--images", type=str, required=True)
    p.add_argument("--out", type=str, default="./output/gradcam")
    p.add_argument("--model-name", type=str, default="efficientnet_b2")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    model.to(device)
    model.eval()

    # Heuristic: last conv layer name for common models
    target_layers = [m for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)][-1:]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device.type == "cuda")

    transforms = build_transforms(image_size=image_size)["val"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.images)
    paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            np_img = np.array(im)
        t = transforms(image=np_img)["image"]
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t.transpose(2, 0, 1)).float() / 255.0
        input_tensor = t.unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        rgb_float = np_img.astype(np.float32) / 255.0
        vis = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        cv2.imwrite(str(out_dir / f"{p.stem}_cam.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()


