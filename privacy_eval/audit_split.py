#!/usr/bin/env python3
"""Check that user-day trajectories or users do not cross split partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def group_hash(user_id: str, day: str, grouping: str) -> str:
    values = [user_id, day] if grouping == "user-day" else [user_id]
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_split(db: sqlite3.Connection, path: Path, name: str, grouping: str,
               user_column: str, day_column: str) -> tuple[int, int]:
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{path.name} has no header")
        required = {user_column} | ({day_column} if grouping == "user-day" else set())
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            user_id = (row.get(user_column) or "").strip()
            day = (row.get(day_column) or "").strip() if grouping == "user-day" else ""
            if not user_id or (grouping == "user-day" and not day):
                raise ValueError(f"{path.name} contains an empty group key")
            db.execute("INSERT OR IGNORE INTO groups VALUES (?, ?)",
                       (group_hash(user_id, day, grouping), name))
            rows += 1
    db.commit()
    count = db.execute("SELECT COUNT(*) FROM groups WHERE partition_name = ?", (name,)).fetchone()[0]
    return rows, int(count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--grouping", choices=("user-day", "user"), default="user-day")
    parser.add_argument("--user-column", default="c_number")
    parser.add_argument("--day-column", default="c_day")
    parser.add_argument("--output", type=Path, help="Optional aggregate JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_path, holdout_path = args.train.expanduser().resolve(), args.holdout.expanduser().resolve()
    for path in (train_path, holdout_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with tempfile.TemporaryDirectory(prefix="trajectory-audit-") as temp_dir:
        with sqlite3.connect(Path(temp_dir) / "audit.sqlite") as db:
            db.execute(
                "CREATE TABLE groups (key TEXT NOT NULL, partition_name TEXT NOT NULL, "
                "PRIMARY KEY (key, partition_name))"
            )
            train_rows, train_groups = read_split(
                db, train_path, "train", args.grouping, args.user_column, args.day_column
            )
            holdout_rows, holdout_groups = read_split(
                db, holdout_path, "holdout", args.grouping, args.user_column, args.day_column
            )
            overlap = int(db.execute(
                "SELECT COUNT(*) FROM groups a JOIN groups b ON a.key = b.key "
                "WHERE a.partition_name = 'train' AND b.partition_name = 'holdout'"
            ).fetchone()[0])

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grouping": args.grouping,
        "train_rows": train_rows,
        "train_groups": train_groups,
        "holdout_rows": holdout_rows,
        "holdout_groups": holdout_groups,
        "overlapping_groups": overlap,
        "passed": overlap == 0 and train_groups > 0 and holdout_groups > 0,
        "identifiers_included": False,
    }
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
