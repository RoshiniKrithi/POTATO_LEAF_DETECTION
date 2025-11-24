from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from potato_leaf_detection.datasets.leaf_dataset import detect_dataset_layout, _is_image_file
from potato_leaf_detection.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./potato_dataset")
    p.add_argument("--out", type=str, default="./output/eda")
    p.add_argument("--sample", type=int, default=50, help="Sample at most N images for size stats")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    root = Path(args.data_dir)
    layout = detect_dataset_layout(root)
    logger.info(f"Detected layout: {layout.kind}; classes={len(layout.class_to_index)}")

    image_paths: List[Tuple[str, int]] = []
    if layout.kind == "canonical":
        for split in ["train", "val", "test"]:
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            for cls, idx in layout.class_to_index.items():
                cls_dir = split_dir / cls
                if not cls_dir.is_dir():
                    continue
                for p in cls_dir.rglob("*"):
                    if p.is_file() and _is_image_file(p):
                        image_paths.append((str(p), idx))
    elif layout.kind == "flat_csv":
        images_dir = root / "images"
        for p in images_dir.rglob("*"):
            if p.is_file() and _is_image_file(p):
                image_paths.append((str(p), -1))
    else:
        # per_class at root (handle nested structure)
        for cls, idx in layout.class_to_index.items():
            # Try direct path first
            cls_dir = root / cls
            if cls_dir.is_dir() and any(_is_image_file(p) for p in cls_dir.rglob("*")):
                for p in cls_dir.rglob("*"):
                    if p.is_file() and _is_image_file(p):
                        image_paths.append((str(p), idx))
            else:
                # Search for class name in nested directories
                for potential_dir in root.rglob(cls):
                    if potential_dir.is_dir() and any(_is_image_file(p) for p in potential_dir.rglob("*")):
                        for p in potential_dir.rglob("*"):
                            if p.is_file() and _is_image_file(p):
                                image_paths.append((str(p), idx))

    class_counts = Counter([idx for _, idx in image_paths if idx >= 0])
    logger.info(f"Class counts: {dict(class_counts)}")

    sizes: List[Tuple[int, int]] = []
    corrupted: List[str] = []
    sample_paths = [p for p, _ in image_paths][: args.sample]
    for p in sample_paths:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except (UnidentifiedImageError, OSError):
            corrupted.append(p)

    if sizes:
        widths = [w for w, h in sizes]
        heights = [h for w, h in sizes]
        logger.info(
            f"Width min/median/max: {min(widths)} / {int(np.median(widths))} / {max(widths)}; "
            f"Height min/median/max: {min(heights)} / {int(np.median(heights))} / {max(heights)}"
        )

    if corrupted:
        logger.warning(f"Found {len(corrupted)} corrupted/unreadable files. Examples: {corrupted[:5]}")

    # Suggest oversampling for classes below 50% of median
    if class_counts:
        median = np.median(list(class_counts.values()))
        under = [cls for cls, cnt in class_counts.items() if cnt < 0.5 * median]
        if under:
            logger.info(f"Suggestion: consider oversampling or targeted augmentation for classes: {under}")


if __name__ == "__main__":
    main()


