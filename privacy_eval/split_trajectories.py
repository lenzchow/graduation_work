#!/usr/bin/env python3
"""Split a trajectory CSV without dividing a selected group across partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def digest(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def partition(values: list[str], seed: int, ratio: float) -> str:
    payload = json.dumps([seed, *values], ensure_ascii=False, separators=(",", ":"))
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return "train" if value < int(ratio * (2**64)) else "holdout"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--user-column", default="c_number")
    parser.add_argument("--day-column", default="c_day")
    parser.add_argument("--grouping", choices=("user-day", "user"), default="user-day")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": output_dir / "train.csv",
        "holdout": output_dir / "holdout.csv",
        "mapping": output_dir / "membership_map.csv",
        "manifest": output_dir / "split_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite to replace generated files only")
    for path in existing:
        path.unlink()

    temps = {name: path.with_suffix(path.suffix + ".tmp") for name, path in outputs.items()}
    try:
        with tempfile.TemporaryDirectory(prefix="trajectory-split-") as temp_dir:
            with sqlite3.connect(Path(temp_dir) / "groups.sqlite") as db:
                db.execute(
                    "CREATE TABLE groups (key TEXT PRIMARY KEY, user_id TEXT, day TEXT, "
                    "partition_name TEXT, rows INTEGER NOT NULL)"
                )
                with source.open(encoding="utf-8-sig", newline="") as file:
                    reader = csv.DictReader(file)
                    if not reader.fieldnames:
                        raise ValueError("Input CSV has no header")
                    required = {args.user_column}
                    if args.grouping == "user-day":
                        required.add(args.day_column)
                    missing = required.difference(reader.fieldnames)
                    if missing:
                        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
                    fields = reader.fieldnames
                    for row in reader:
                        user_id = (row.get(args.user_column) or "").strip()
                        day = (row.get(args.day_column) or "").strip() if args.grouping == "user-day" else ""
                        if not user_id or (args.grouping == "user-day" and not day):
                            raise ValueError("A row has an empty trajectory grouping key")
                        values = [user_id, day] if args.grouping == "user-day" else [user_id]
                        key = digest(values)
                        name = partition(values, args.seed, args.train_ratio)
                        db.execute("INSERT OR IGNORE INTO groups VALUES (?, ?, ?, ?, 0)",
                                   (key, user_id, day, name))
                        db.execute("UPDATE groups SET rows = rows + 1 WHERE key = ?", (key,))
                db.commit()

                stats = {
                    name: {"groups": int(groups), "rows": int(rows)}
                    for name, groups, rows in db.execute(
                        "SELECT partition_name, COUNT(*), SUM(rows) FROM groups GROUP BY partition_name"
                    )
                }
                for name in ("train", "holdout"):
                    stats.setdefault(name, {"groups": 0, "rows": 0})
                if not stats["train"]["groups"] or not stats["holdout"]["groups"]:
                    raise ValueError("One split is empty; change the seed or train ratio")

                with temps["mapping"].open("w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["group_hash", args.user_column, args.day_column, "partition"])
                    writer.writerows(db.execute(
                        "SELECT key, user_id, day, partition_name FROM groups ORDER BY key"
                    ))

            written = {"train": 0, "holdout": 0}
            with source.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                with (
                    temps["train"].open("w", encoding="utf-8", newline="") as train_file,
                    temps["holdout"].open("w", encoding="utf-8", newline="") as holdout_file,
                ):
                    writers = {
                        "train": csv.DictWriter(train_file, fieldnames=fields),
                        "holdout": csv.DictWriter(holdout_file, fieldnames=fields),
                    }
                    for writer in writers.values():
                        writer.writeheader()
                    for row in reader:
                        user_id = (row.get(args.user_column) or "").strip()
                        day = (row.get(args.day_column) or "").strip() if args.grouping == "user-day" else ""
                        values = [user_id, day] if args.grouping == "user-day" else [user_id]
                        name = partition(values, args.seed, args.train_ratio)
                        writers[name].writerow(row)
                        written[name] += 1

            manifest = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_file": source.name,
                "source_sha256": sha256_file(source),
                "grouping": args.grouping,
                "user_column": args.user_column,
                "day_column": args.day_column,
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "counts": stats,
                "rows_written": written,
                "identifiers_included": False,
            }
            temps["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
        for name, temp_path in temps.items():
            os.replace(temp_path, outputs[name])
    finally:
        for temp_path in temps.values():
            temp_path.unlink(missing_ok=True)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Private files written under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
