from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: str | Path, filename: str) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def save_hparams_yaml(hparams: Dict[str, Any], out_dir: str | Path, filename: str = "experiment.json") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2)
    return path


