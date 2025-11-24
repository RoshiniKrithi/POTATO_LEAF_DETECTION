from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path to allow imports
# When running train.py from within the package directory, add parent so Python can find 'potato_leaf_detection'
_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from potato_leaf_detection.datasets import build_dataloaders, detect_dataset_layout, stratified_split_and_save
from potato_leaf_detection.models.factory import create_model
from potato_leaf_detection.utils.transforms import build_transforms
from potato_leaf_detection.utils.metrics import compute_classification_metrics
from potato_leaf_detection.utils.checkpoint import save_checkpoint, load_checkpoint, save_hparams_yaml
from potato_leaf_detection.utils.logging_utils import setup_logger
from potato_leaf_detection.utils.seed import set_seed
from potato_leaf_detection.utils.db import ExperimentDB


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./potato_dataset")
    p.add_argument("--output", type=str, default="./output")
    p.add_argument("--model", type=str, default="efficientnet_b2")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--oversample", action="store_true")
    p.add_argument("--focal-loss", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--splits-dir", type=str, default=None, help="Directory to write/read split CSVs")
    p.add_argument("--db-path", type=str, default=None, help="Path to SQLite DB file for logging metrics (defaults to output/experiments.db)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    ckpt_dir = output_dir / "checkpoints"
    tb_dir = output_dir / "tb"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=str(output_dir / "train.log"))
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    layout = detect_dataset_layout(data_dir)
    logger.info(f"Detected dataset layout: {layout.kind}; classes={len(layout.class_to_index)}")

    # If needed, write stratified splits
    splits_dir = args.splits_dir
    if layout.kind in {"per_class", "flat_csv"}:
        splits_dir = splits_dir or str(output_dir / "splits")
        stratified_split_and_save(data_dir, splits_dir, seed=args.seed)
        logger.info(f"Stratified split CSVs saved to: {splits_dir}")

    # Model
    model, default_image_size = create_model(args.model, num_classes=len(layout.class_to_index), pretrained=True)
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if any(k in name for k in ["classifier", "fc", "head"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    device = torch.device(args.device)
    model.to(device)

    # Data
    transforms = build_transforms(image_size=default_image_size)
    train_loader, val_loader, test_loader, class_weights, class_counts, idx_to_class = build_dataloaders(
        data_dir, transforms, batch_size=args.batch_size, num_workers=args.num_workers, oversample=args.oversample, split_csv_dir=splits_dir
    )
    logger.info(f"Class counts: {class_counts}")

    # Loss
    if args.focal_loss:
        criterion = FocalLoss()
    else:
        if class_weights:
            weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=args.amp)
    writer = SummaryWriter(log_dir=str(tb_dir))

    # Database logging
    db = None
    experiment_id = None
    last_epoch_completed = start_epoch - 1
    db_path = Path(args.db_path) if args.db_path else (output_dir / "experiments.db")
    try:
        db = ExperimentDB(db_path)
        experiment_id = db.start_experiment(
            args=vars(args),
            num_classes=len(layout.class_to_index),
            idx_to_class=idx_to_class,
            resume_from=args.resume,
        )
        logger.info(f"Logging training metrics to SQLite DB at {db_path}")
    except Exception as exc:
        logger.warning(f"Failed to initialize experiment database at {db_path}: {exc}")
        db = None

    start_epoch = 1
    best_metric = -np.inf
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", best_metric)
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Save hparams
    save_hparams_yaml({
        "args": vars(args),
        "num_classes": len(layout.class_to_index),
        "idx_to_class": idx_to_class,
    }, output_dir)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            images, targets, _ = batch
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(images.size(0))
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(enabled=args.amp):
                    logits = model(images)
                    loss = criterion(logits, targets)
                val_loss += float(loss.item()) * images.size(0)
                preds = logits.argmax(dim=1)
                y_true.extend(targets.tolist())
                y_pred.extend(preds.tolist())
        val_loss /= max(1, len(y_true))
        metrics = compute_classification_metrics(y_true, y_pred)

        # Step scheduler with epoch-level step for CosineAnnealingWarmRestarts
        scheduler.step(epoch + 1)

        # Logging
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", metrics["accuracy"], epoch)
        writer.add_scalar("f1_macro/val", metrics["f1_macro"], epoch)
        logger.info(
            f"Epoch {epoch}/{args.epochs} | TrainLoss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"ValLoss {val_loss:.4f} Acc {metrics['accuracy']:.4f} F1 {metrics['f1_macro']:.4f}"
        )
        if db and experiment_id is not None:
            db.log_epoch(
                experiment_id=experiment_id,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=metrics["accuracy"],
                f1_macro=metrics["f1_macro"],
            )
        last_epoch_completed = epoch

        # Checkpoints
        if epoch % args.save_every == 0:
            ckpt_path = save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_metric": best_metric,
            }, ckpt_dir, f"ckpt_epoch_{epoch}.pth")
            if db and experiment_id is not None:
                db.log_checkpoint(experiment_id, epoch, ckpt_path=str(ckpt_path), is_best=False)

        # Save best by F1 macro
        if metrics["f1_macro"] > best_metric:
            best_metric = metrics["f1_macro"]
            best_path = save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_metric": best_metric,
            }, ckpt_dir, "best_model.pth")
            if db and experiment_id is not None:
                db.log_checkpoint(experiment_id, epoch, ckpt_path=str(best_path), is_best=True)

    writer.close()
    logger.info("Training complete.")
    if db and experiment_id is not None:
        db.complete_experiment(experiment_id, last_epoch_completed, best_metric, status="completed")
        db.close()


if __name__ == "__main__":
    main()


