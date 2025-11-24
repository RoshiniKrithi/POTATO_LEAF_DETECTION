from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from potato_leaf_detection.models.factory import create_model
from potato_leaf_detection.utils.transforms import build_transforms
from potato_leaf_detection.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path to .pth checkpoint (state dict or full ckpt)")
    p.add_argument("--images", type=str, required=True, help="Folder with images")
    p.add_argument("--out", type=str, default="./preds.csv")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-name", type=str, default="efficientnet_b2")
    p.add_argument("--num-classes", type=int, default=None, help="If absent, try to read from checkpoint meta")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    device = torch.device(args.device)

    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        num_classes = args.num_classes or ckpt.get("num_classes") or len(ckpt.get("idx_to_class", {})) or 2
        idx_to_class = ckpt.get("idx_to_class", {i: str(i) for i in range(num_classes)})
        model, image_size = create_model(args.model_name, num_classes=num_classes, pretrained=False)
        model.load_state_dict(ckpt["model"]) 
    else:
        num_classes = args.num_classes or 2
        idx_to_class = {i: str(i) for i in range(num_classes)}
        model, image_size = create_model(args.model_name, num_classes=num_classes, pretrained=False)
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    transforms = build_transforms(image_size=image_size)["val"]

    image_dir = Path(args.images)
    img_paths: List[Path] = [p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]

    rows: List[List[str]] = []
    with torch.no_grad():
        for p in img_paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
                import numpy as np
                arr = np.array(im)
            t = transforms(image=arr)["image"]
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t.transpose(2, 0, 1)).float() / 255.0
            t = t.unsqueeze(0).to(device)
            probs = F.softmax(model(t), dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_cls = idx_to_class.get(pred_idx, str(pred_idx))
            rows.append([str(p), pred_cls] + [str(float(s)) for s in probs.tolist()])

    header = ["filename", "pred_label"] + [f"score_{idx_to_class[i]}" for i in range(len(idx_to_class))]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

    logger.info(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()


