from pathlib import Path

import pytest

from potato_leaf_detection.datasets import detect_dataset_layout


def test_detect_layout_handles_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        detect_dataset_layout(tmp_path / "missing")


