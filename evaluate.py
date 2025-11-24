from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from potato_leaf_detection.datasets import build_dataloaders, detect_dataset_layout
from potato_leaf_detection.models.factory import create_model
from potato_leaf_detection.utils.transforms import build_transforms
from potato_leaf_detection.utils.metrics import compute_classification_metrics, per_class_report, confusion_matrix_array, try_multiclass_roc_auc
from potato_leaf_detection.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./potato_dataset")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"]) 
    p.add_argument("--out", type=str, default="./output/eval")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--splits-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(out_dir / "eval.log"))
    device = torch.device(args.device)

    layout = detect_dataset_layout(Path(args.data_dir))
    model, image_size = create_model("efficientnet_b2", num_classes=len(layout.class_to_index), pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"]) if "model" in ckpt else model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    transforms = build_transforms(image_size=image_size)
    loaders = build_dataloaders(args.data_dir, transforms, batch_size=args.batch_size, num_workers=args.num_workers, oversample=False, split_csv_dir=args.splits_dir)
    _, _, test_loader, _, _, idx_to_class = loaders
    loader = {"train": loaders[0], "val": loaders[1], "test": test_loader}[args.split]
    if loader is None:
        logger.warning("Requested split not available; falling back to val")
        loader = loaders[1]

    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores_list: List[np.ndarray] = []
    filenames: List[str] = []

    with torch.no_grad():
        for images, targets, paths in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()
            y_true.extend(targets.tolist())
            y_pred.extend(preds)
            y_scores_list.append(probs)
            filenames.extend(paths)

    y_scores = np.concatenate(y_scores_list, axis=0) if y_scores_list else np.zeros((0, len(layout.class_to_index)))
    metrics = compute_classification_metrics(y_true, y_pred)
    report = per_class_report(y_true, y_pred)
    cm = confusion_matrix_array(y_true, y_pred)
    auc = try_multiclass_roc_auc(y_true, y_scores)

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # Save metrics
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
        if auc is not None:
            writer.writerow(["roc_auc_ovr", auc])

    # Save predictions CSV
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["filename", "true_label", "pred_label"] + [f"score_{idx_to_class[i]}" for i in range(len(idx_to_class))]
        writer.writerow(header)
        for i in range(len(y_pred)):
            row = [filenames[i], idx_to_class[y_true[i]], idx_to_class[y_pred[i]]] + list(map(str, y_scores[i].tolist()))
            writer.writerow(row)

    logger.info(f"Eval done. Acc={metrics['accuracy']:.4f} F1={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()


