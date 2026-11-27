#!/usr/bin/env python3
"""Compare generated inter-visit times with train and holdout trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def elapsed_minutes(a: float, b: float) -> float:
    delta = b - a
    scale = 60000.0 if max(abs(a), abs(b)) >= 1e11 else 60.0 if max(abs(a), abs(b)) >= 1e8 else 1.0
    return max(0.0, delta / scale)


def read_real(path: Path) -> list[tuple[int, tuple[float, ...]]]:
    groups: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {"c_number", "c_day", "c_timestamp"}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {sorted(needed)}")
        for row_number, row in enumerate(reader, start=2):
            user, day = (row.get("c_number") or "").strip(), (row.get("c_day") or "").strip()
            raw = (row.get("c_timestamp") or "").strip()
            if not user or not day or not raw:
                continue
            groups[(user, day)].append((float(raw), row_number))

    trajectories = []
    for rows in groups.values():
        rows.sort()
        if len(rows) < 2:
            continue
        timestamps = [x[0] for x in rows]
        gaps = tuple(elapsed_minutes(a, b) for a, b in zip(timestamps, timestamps[1:]))
        trajectories.append((len(rows), gaps))
    return trajectories


def read_generated(directory: Path, model: str) -> list[tuple[int, tuple[float, ...]]]:
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No generated .txt files under {directory}")
    result = []
    for path in files:
        current = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if len(current) > 1:
                        if model == "signaling":
                            gaps = tuple(((b - a) % 24.0) * 60.0 for a, b in zip(current, current[1:]))
                        else:
                            gaps = tuple(max(0.0, b - a) for a, b in zip(current, current[1:]))
                        result.append((len(current), gaps))
                    current = []
                    continue
                columns = next(csv.reader([line], skipinitialspace=True))
                if len(columns) < 5:
                    raise ValueError(f"Expected time column 5 in {path}:{line_number}")
                current.append(float(columns[4]))
        if len(current) > 1:
            if model == "signaling":
                gaps = tuple(((b - a) % 24.0) * 60.0 for a, b in zip(current, current[1:]))
            else:
                gaps = tuple(max(0.0, b - a) for a, b in zip(current, current[1:]))
            result.append((len(current), gaps))
    return result


def gap_mae(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return math.inf
    return statistics.fmean(abs(a - b) for a, b in zip(left, right))


def sample_by_length(items: list[tuple[int, tuple[float, ...]]], limit: int, rng: random.Random):
    buckets: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    for length, gaps in items:
        buckets[length].append(gaps)
    for length, values in buckets.items():
        if len(values) > limit:
            buckets[length] = rng.sample(values, limit)
    return buckets


def nearest_errors(
    generated: list[tuple[int, tuple[float, ...]]],
    references: dict[int, list[tuple[float, ...]]],
    per_length_limit: int,
    rng: random.Random,
) -> list[float]:
    grouped: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    for length, gaps in generated:
        grouped[length].append(gaps)
    errors = []
    for length, values in grouped.items():
        refs = references.get(length, [])
        if not refs:
            continue
        if len(values) > per_length_limit:
            values = rng.sample(values, per_length_limit)
        for gaps in values:
            errors.append(min(gap_mae(gaps, candidate) for candidate in refs))
    return errors


def summarize(errors: list[float], tolerance: float) -> dict:
    if not errors:
        return {"matched_count": 0, "mean_gap_mae_minutes": None, "median_gap_mae_minutes": None,
                "p95_gap_mae_minutes": None, "within_tolerance_rate": None}
    ordered = sorted(errors)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "matched_count": len(errors),
        "mean_gap_mae_minutes": statistics.fmean(errors),
        "median_gap_mae_minutes": statistics.median(errors),
        "p95_gap_mae_minutes": p95,
        "within_tolerance_rate": sum(value <= tolerance for value in errors) / len(errors),
        "near_duplicate_threshold_minutes_per_interval": tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("signaling", "cost"), required=True)
    parser.add_argument("--operator", choices=("operator0", "operator1", "operator3"), required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-minutes", type=float, default=None)
    parser.add_argument("--max-reference-per-length", type=int, default=200)
    parser.add_argument("--max-generated-per-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    split_model = "SignalingTraj" if args.model == "signaling" else "CoST-Diff"
    split_dir = repo_root / "privacy_eval" / "local" / "splits" / split_model / args.operator
    train = read_real(split_dir / "train.csv")
    holdout = read_real(split_dir / "holdout.csv")
    generated = read_generated(args.generated_dir, args.model)
    if not train or not holdout or not generated:
        raise ValueError(f"Empty input: train={len(train)}, holdout={len(holdout)}, generated={len(generated)}")

    tolerance = args.tolerance_minutes
    if tolerance is None:
        tolerance = 60.0 if args.model == "signaling" else 5.0
    rng = random.Random(args.seed)
    train_refs = sample_by_length(train, args.max_reference_per_length, rng)
    holdout_refs = sample_by_length(holdout, args.max_reference_per_length, rng)
    train_errors = nearest_errors(generated, train_refs, args.max_generated_per_length, rng)
    holdout_errors = nearest_errors(generated, holdout_refs, args.max_generated_per_length, rng)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "operator": args.operator,
        "generated_directory": str(args.generated_dir.resolve()),
        "comparison": "minimum mean absolute error between inter-visit interval sequences, matched by emitted trajectory length",
        "time_unit": "minutes",
        "training_reference": summarize(train_errors, tolerance),
        "holdout_reference": summarize(holdout_errors, tolerance),
        "interpretation": (
            "A temporal similarity screen, not a standalone privacy guarantee. Signal outputs encode time "
            "at hour resolution, so its interval comparison is coarse. Generated-to-training similarity "
            "should be interpreted alongside the holdout comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
