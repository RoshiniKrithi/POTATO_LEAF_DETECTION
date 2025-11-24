from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _utcnow() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


class ExperimentDB:
    """
    Lightweight SQLite helper used to log experiment metadata, epoch metrics, and checkpoints.
    """

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                args_json TEXT,
                num_classes INTEGER,
                idx_to_class_json TEXT,
                resume_from TEXT,
                last_epoch INTEGER,
                best_metric REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                train_acc REAL,
                val_loss REAL,
                val_acc REAL,
                f1_macro REAL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                path TEXT NOT NULL,
                is_best INTEGER NOT NULL DEFAULT 0,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def start_experiment(
        self,
        args: Dict[str, Any],
        num_classes: int,
        idx_to_class: Dict[int, str],
        resume_from: Optional[str] = None,
    ) -> int:
        now = _utcnow()
        payload = {
            "args": args,
            "num_classes": num_classes,
            "idx_to_class": idx_to_class,
        }
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO experiments (created_at, updated_at, status, args_json, num_classes,
                                     idx_to_class_json, resume_from, last_epoch, best_metric)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                now,
                now,
                "running",
                json.dumps(payload),
                num_classes,
                json.dumps(idx_to_class),
                resume_from,
                0,
                None,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_epoch(
        self,
        experiment_id: int,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        f1_macro: float,
    ) -> None:
        now = _utcnow()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO epochs (experiment_id, epoch, train_loss, train_acc,
                                val_loss, val_acc, f1_macro, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (experiment_id, epoch, train_loss, train_acc, val_loss, val_acc, f1_macro, now),
        )
        cur.execute(
            "UPDATE experiments SET updated_at = ?, last_epoch = ? WHERE id = ?;",
            (now, epoch, experiment_id),
        )
        self.conn.commit()

    def log_checkpoint(self, experiment_id: int, epoch: int, ckpt_path: str, is_best: bool) -> None:
        now = _utcnow()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO checkpoints (experiment_id, epoch, path, is_best, recorded_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (experiment_id, epoch, ckpt_path, 1 if is_best else 0, now),
        )
        cur.execute(
            "UPDATE experiments SET updated_at = ? WHERE id = ?;",
            (now, experiment_id),
        )
        self.conn.commit()

    def complete_experiment(self, experiment_id: int, last_epoch: int, best_metric: float, status: str = "completed") -> None:
        now = _utcnow()
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE experiments
            SET status = ?, updated_at = ?, last_epoch = ?, best_metric = ?
            WHERE id = ?;
            """,
            (status, now, last_epoch, best_metric, experiment_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

