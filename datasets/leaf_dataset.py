import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetLayout:
    kind: str  # canonical|per_class|flat_csv
    has_test: bool
    class_to_index: Dict[str, int]
    splits: Optional[Dict[str, List[Tuple[str, int]]]] = None


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def detect_dataset_layout(root_dir: Path) -> DatasetLayout:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root_dir}")

    # Case 1: canonical train/val(/test)/class/*.jpg
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    test_dir = root_dir / "test"
    if train_dir.is_dir() and val_dir.is_dir():
        classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        class_to_index = {c: i for i, c in enumerate(classes)}
        return DatasetLayout(kind="canonical", has_test=test_dir.is_dir(), class_to_index=class_to_index)

    # Case 3: flat with CSV
    images_dir = root_dir / "images"
    labels_csv = root_dir / "labels.csv"
    if images_dir.is_dir() and labels_csv.is_file():
        # Read labels to infer classes
        classes: List[str] = []
        with labels_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("labels.csv must contain columns: filename,label")
            for row in reader:
                classes.append(row["label"]) \
                    if row.get("label") is not None else None
        unique_classes = sorted(list({c for c in classes if c is not None}))
        class_to_index = {c: i for i, c in enumerate(unique_classes)}
        return DatasetLayout(kind="flat_csv", has_test=False, class_to_index=class_to_index)

    # Case 2: per-class folders at root (handle nested structure)
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir() and any(_is_image_file(p) for p in d.rglob("*"))]
    if class_dirs:
        # Check if we have nested structure (e.g., diseased/Potato___Early_blight/)
        # If any top-level dir contains subdirs with images, use subdirs as classes
        nested_classes = []
        for top_dir in class_dirs:
            subdirs = [sd for sd in top_dir.iterdir() if sd.is_dir() and any(_is_image_file(p) for p in sd.rglob("*"))]
            if subdirs:
                nested_classes.extend([sd.name for sd in subdirs])
        
        if nested_classes:
            # Use nested subdirectories as classes
            classes = sorted(list(set(nested_classes)))
            class_to_index = {c: i for i, c in enumerate(classes)}
            return DatasetLayout(kind="per_class", has_test=False, class_to_index=class_to_index)
        else:
            # Use top-level directories as classes
            classes = sorted([d.name for d in class_dirs])
            class_to_index = {c: i for i, c in enumerate(classes)}
            return DatasetLayout(kind="per_class", has_test=False, class_to_index=class_to_index)

    raise ValueError("Could not detect dataset layout. Expected train/val dirs, per-class dirs, or images+labels.csv.")


class AutoLeafDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        transforms=None,
        class_to_index: Optional[Dict[str, int]] = None,
        split_csv_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.layout = detect_dataset_layout(self.root_dir)
        self.class_to_index = class_to_index or self.layout.class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.samples: List[Tuple[Path, int]] = []

        # If split files exist (from previous stratified split), prefer them
        if split_csv_dir is not None:
            split_csv = Path(split_csv_dir) / f"{split}.csv"
            if split_csv.exists():
                self.samples = self._read_split_csv(split_csv)
                return

        if self.layout.kind == "canonical":
            base = self.root_dir / split
            if not base.is_dir():
                raise FileNotFoundError(f"Missing split directory: {base}")
            for cls in sorted(self.class_to_index.keys()):
                cls_dir = base / cls
                if not cls_dir.is_dir():
                    continue
                for p in cls_dir.rglob("*"):
                    if p.is_file() and _is_image_file(p):
                        self.samples.append((p, self.class_to_index[cls]))

        elif self.layout.kind == "flat_csv":
            images_dir = self.root_dir / "images"
            labels_csv = self.root_dir / "labels.csv"
            rows = self._read_labels_csv(labels_csv)
            # If no explicit split available, stratify on-the-fly (but not persisted)
            if not any((self.root_dir / f"{s}.csv").exists() for s in ["train", "val", "test"]):
                train_rows, val_rows, test_rows = _stratified_rows(rows)
                rows = {"train": train_rows, "val": val_rows, "test": test_rows}[split]
            for filename, label in rows:
                img_path = images_dir / filename
                if img_path.is_file() and _is_image_file(img_path):
                    self.samples.append((img_path, self.class_to_index[label]))

        elif self.layout.kind == "per_class":
            # Expect split CSVs or perform split on-the-fly
            split_csv = self.root_dir / f"{split}.csv"
            if split_csv.exists():
                self.samples = self._read_split_csv(split_csv)
            else:
                # On-the-fly split
                all_samples = []
                for cls in sorted(self.class_to_index.keys()):
                    # Try direct path first
                    cls_dir = self.root_dir / cls
                    if cls_dir.is_dir() and any(_is_image_file(p) for p in cls_dir.rglob("*")):
                        # Direct class folder exists
                        for p in cls_dir.rglob("*"):
                            if p.is_file() and _is_image_file(p):
                                all_samples.append((p, self.class_to_index[cls]))
                    else:
                        # Search for class name in nested directories
                        for potential_dir in self.root_dir.rglob(cls):
                            if potential_dir.is_dir() and any(_is_image_file(p) for p in potential_dir.rglob("*")):
                                for p in potential_dir.rglob("*"):
                                    if p.is_file() and _is_image_file(p):
                                        all_samples.append((p, self.class_to_index[cls]))
                labels = [y for _, y in all_samples]
                indices = list(range(len(all_samples)))
                train_idx, val_idx, test_idx = _stratified_indices(labels)
                idx_map = {"train": train_idx, "val": val_idx, "test": test_idx}
                for i in idx_map[split]:
                    self.samples.append(all_samples[i])
        else:
            raise ValueError(f"Unsupported layout: {self.layout.kind}")

    def _read_labels_csv(self, csv_path: Path) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append((r["filename"], r["label"]))
        return rows

    def _read_split_csv(self, csv_path: Path) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                samples.append((Path(r["filepath"]), int(r["label_index"])))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            image_np = np.array(img)
        if self.transforms is not None:
            image_np = self.transforms(image=image_np)["image"]
        # If using albumentations, we get numpy -> convert to tensor
        if isinstance(image_np, np.ndarray):
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
        else:
            image_tensor = image_np  # already tensor
        return image_tensor, label, str(path)


def _stratified_indices(labels: List[int], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(labels))
    labels_np = np.array(labels)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, labels_np))
    val_portion = val_ratio / (1 - train_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_portion, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_idx, labels_np[temp_idx]))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def _stratified_rows(rows: List[Tuple[str, str]], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    labels = [lbl for _, lbl in rows]
    unique = {c: i for i, c in enumerate(sorted(set(labels)))}
    y = [unique[lbl] for lbl in labels]
    train_idx, val_idx, test_idx = _stratified_indices(y, train_ratio, val_ratio, seed)
    rows_np = np.array(rows, dtype=object)
    return rows_np[train_idx].tolist(), rows_np[val_idx].tolist(), rows_np[test_idx].tolist()


def stratified_split_and_save(root_dir: str | Path, out_dir: str | Path, seed: int = 42) -> Dict[str, Path]:
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = detect_dataset_layout(root_dir)
    splits_written: Dict[str, Path] = {}

    def _write_split_csv(path: Path, samples: List[Tuple[Path, int]]):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label_index"])
            writer.writeheader()
            for fp, y in samples:
                writer.writerow({"filepath": str(fp), "label_index": int(y)})

    if layout.kind == "per_class":
        all_samples: List[Tuple[Path, int]] = []
        for cls, idx in layout.class_to_index.items():
            # Try direct path first
            cls_dir = root_dir / cls
            if cls_dir.is_dir() and any(_is_image_file(p) for p in cls_dir.rglob("*")):
                for p in cls_dir.rglob("*"):
                    if p.is_file() and _is_image_file(p):
                        all_samples.append((p, idx))
            else:
                # Search for class name in nested directories
                for potential_dir in root_dir.rglob(cls):
                    if potential_dir.is_dir() and any(_is_image_file(p) for p in potential_dir.rglob("*")):
                        for p in potential_dir.rglob("*"):
                            if p.is_file() and _is_image_file(p):
                                all_samples.append((p, idx))
        labels = [y for _, y in all_samples]
        train_idx, val_idx, test_idx = _stratified_indices(labels, seed=seed)
        split_map = {
            "train": [all_samples[i] for i in train_idx],
            "val": [all_samples[i] for i in val_idx],
            "test": [all_samples[i] for i in test_idx],
        }
        for name, samples in split_map.items():
            csv_path = out_dir / f"{name}.csv"
            _write_split_csv(csv_path, samples)
            splits_written[name] = csv_path
    elif layout.kind == "flat_csv":
        rows = AutoLeafDataset(root_dir, split="train")._read_labels_csv(root_dir / "labels.csv")
        train_rows, val_rows, test_rows = _stratified_rows(rows, seed=seed)
        # Convert to samples pointing to images dir
        images_dir = root_dir / "images"
        for name, rws in {"train": train_rows, "val": val_rows, "test": test_rows}.items():
            samples = [(images_dir / fn, layout.class_to_index[lbl]) for fn, lbl in rws]
            csv_path = out_dir / f"{name}.csv"
            _write_split_csv(csv_path, samples)
            splits_written[name] = csv_path
    elif layout.kind == "canonical":
        # Nothing to do
        pass
    else:
        raise ValueError(f"Unknown layout: {layout.kind}")

    # Save metadata
    meta = {
        "class_to_index": layout.class_to_index,
        "layout": layout.kind,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return splits_written


def build_dataloaders(
    root_dir: str | Path,
    image_transforms: Dict[str, object],
    batch_size: int = 32,
    num_workers: int = 4,
    oversample: bool = False,
    split_csv_dir: Optional[str | Path] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[int, float], Dict[int, int], Dict[int, str]]:
    train_ds = AutoLeafDataset(root_dir, split="train", transforms=image_transforms.get("train"), split_csv_dir=split_csv_dir)
    val_ds = AutoLeafDataset(root_dir, split="val", transforms=image_transforms.get("val"), split_csv_dir=split_csv_dir)
    test_ds: Optional[AutoLeafDataset] = None
    try:
        test_ds = AutoLeafDataset(root_dir, split="test", transforms=image_transforms.get("val"), split_csv_dir=split_csv_dir)
    except Exception:
        test_ds = None

    class_counts: Dict[int, int] = {}
    for _, y, _ in train_ds:
        class_counts[y] = class_counts.get(y, 0) + 1
    class_weights: Dict[int, float] = {}
    if class_counts:
        total = sum(class_counts.values())
        for k, v in class_counts.items():
            class_weights[k] = total / (len(class_counts) * max(1, v))

    if oversample and class_counts:
        sample_weights = [class_weights[y] for _, y, _ in train_ds]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = None if test_ds is None else DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights, class_counts, train_ds.index_to_class


