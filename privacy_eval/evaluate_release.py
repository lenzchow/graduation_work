#!/usr/bin/env python3
"""Evaluate generated trajectory exposure and a black-box, distance-based MIA probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MODEL_LAYOUT = {
    "signaling": {
        "root": Path("SignalingTraj_exp/SignalingTraj"),
        "split": "SignalingTraj",
        "min_length": 1,
        "max_length": 12000,
        "generated_columns": 5,
    },
    "cost": {
        "root": Path("CoST-Diff_exp/CoST-Diff"),
        "split": "CoST-Diff",
        "min_length": 10,
        "max_length": 60,
        "generated_columns": 6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_LAYOUT, required=True)
    parser.add_argument("--operator", choices=("operator0", "operator1", "operator3"), required=True)
    parser.add_argument(
        "--generated-dir", type=Path, required=True,
        help="One exact generate_data output directory; do not combine separate runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=3000,
                        help="Maximum train and holdout trajectories used for the MIA probe")
    parser.add_argument("--max-generated-for-mia", type=int, default=10000,
                        help="Maximum generated trajectories used for similarity scoring")
    parser.add_argument("--rare-transition-count", type=int, default=3)
    return parser.parse_args()


def read_index_map(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(key): str(value) for key, value in raw.items()}
    if not mapping:
        raise ValueError(f"Empty station-index map: {path}")
    return mapping


def numeric(value: str, field: str, path: Path, line_number: int) -> float:
    try:
        return float(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in {path}:{line_number}: {value!r}") from exc


def read_real_trajectories(
    path: Path,
    station_vocabulary: set[str],
    min_length: int,
    max_length: int,
) -> list[tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    groups: dict[tuple[str, str], list[tuple[float, int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"c_number", "c_day", "c_uli", "c_timestamp"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            station = (row.get("c_uli") or "").strip()
            user = (row.get("c_number") or "").strip()
            day = (row.get("c_day") or "").strip()
            if not station or not user or not day:
                continue
            timestamp = numeric(row.get("c_timestamp", ""), "c_timestamp", path, row_number)
            groups[(user, day)].append((timestamp, row_number, station))

    trajectories = []
    for rows in groups.values():
        rows.sort(key=lambda item: (item[0], item[1]))
        sequence = tuple(item[2] for item in rows)
        if min_length <= len(sequence) <= max_length and set(sequence).issubset(station_vocabulary):
            trajectories.append(sequence)
    return trajectories


def scalar_float(value: str, path: Path, line_number: int) -> float:
    return numeric(value, "generated value", path, line_number)


def read_generated_trajectories(
    directory: Path,
    index_to_station: dict[str, str],
    expected_columns: int,
) -> tuple[list[tuple[str, ...]], int]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No generated .txt files in {directory}")

    generated: list[tuple[str, ...]] = []
    unmapped_rows = 0
    for path in files:
        current: list[str] = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if current:
                        generated.append(tuple(current))
                        current = []
                    continue
                columns = next(csv.reader([line], skipinitialspace=True))
                if len(columns) < expected_columns:
                    raise ValueError(
                        f"Expected at least {expected_columns} columns in {path}:{line_number}, "
                        f"found {len(columns)}"
                    )
                rid_value = scalar_float(columns[3], path, line_number)
                rid = str(int(rid_value))
                station = index_to_station.get(rid)
                if station is None:
                    unmapped_rows += 1
                    raise ValueError(f"Unknown generated station index {rid!r} in {path}:{line_number}")
                # Validate the emitted time and, for CoST-Diff, interval fields.
                scalar_float(columns[4], path, line_number)
                if expected_columns == 6:
                    scalar_float(columns[5], path, line_number)
                current.append(station)
        if current:
            generated.append(tuple(current))

    return generated, unmapped_rows


def sample(items: list, limit: int, rng: random.Random) -> list:
    if len(items) <= limit:
        return items
    return rng.sample(items, limit)


def transition_set(sequence: tuple[str, ...]) -> set[tuple[str, str]]:
    if len(sequence) == 1:
        return {("__single__", sequence[0])}
    return set(zip(sequence, sequence[1:]))


def max_transition_jaccard(
    sequence: tuple[str, ...],
    generated_by_length: dict[int, list[set[tuple[str, str]]]],
) -> float:
    query = transition_set(sequence)
    best = 0.0
    for length in range(max(1, len(sequence) - 1), len(sequence) + 2):
        for candidate in generated_by_length.get(length, []):
            other = candidate
            union = len(query | other)
            score = len(query & other) / union if union else 1.0
            if score > best:
                best = score
                if best == 1.0:
                    return best
    return best


def auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    ordered = sorted([(score, 1) for score in positive] + [(score, 0) for score in negative])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
    }


def main() -> int:
    args = parse_args()
    layout = MODEL_LAYOUT[args.model]
    repo_root = Path(__file__).resolve().parents[1]
    model_root = repo_root / layout["root"]
    split_dir = repo_root / "privacy_eval" / "local" / "splits" / layout["split"] / args.operator
    artifacts = model_root / "data" / args.operator / "extra_file"
    index_to_station = read_index_map(artifacts / "index_uli_dict.json")
    station_vocabulary = set(index_to_station.values())

    train = read_real_trajectories(
        split_dir / "train.csv", station_vocabulary,
        layout["min_length"], layout["max_length"],
    )
    holdout = read_real_trajectories(
        split_dir / "holdout.csv", station_vocabulary,
        layout["min_length"], layout["max_length"],
    )
    generated, unmapped_rows = read_generated_trajectories(
        args.generated_dir, index_to_station, layout["generated_columns"],
    )
    if not train or not holdout or not generated:
        raise ValueError(
            f"Need non-empty train, holdout, and generated trajectories; got "
            f"{len(train)}, {len(holdout)}, {len(generated)}"
        )

    train_counts = Counter(train)
    holdout_counts = Counter(holdout)
    train_set = set(train)
    holdout_set = set(holdout)
    generated_set = set(generated)

    exact_train = sum(sequence in train_set for sequence in generated)
    exact_holdout = sum(sequence in holdout_set for sequence in generated)
    generated_train_unique = len(generated_set & train_set)
    generated_holdout_unique = len(generated_set & holdout_set)

    train_transition_counts = Counter(
        transition for trajectory in train for transition in zip(trajectory, trajectory[1:])
    )
    generated_transitions = [
        transition for trajectory in generated for transition in zip(trajectory, trajectory[1:])
    ]
    unseen_transition_count = sum(
        train_transition_counts[transition] == 0 for transition in generated_transitions
    )
    rare_transition_count = sum(
        train_transition_counts[transition] <= args.rare_transition_count
        for transition in generated_transitions
    )
    rare_exact_train = sum(
        sequence in train_counts and train_counts[sequence] == 1 for sequence in generated
    )

    rng = random.Random(args.seed)
    generated_sample = sample(generated, args.max_generated_for_mia, rng)
    generated_by_length: dict[int, list[set[tuple[str, str]]]] = defaultdict(list)
    for sequence in generated_sample:
        generated_by_length[len(sequence)].append(transition_set(sequence))

    train_candidates = sample(train, args.max_candidates, rng)
    holdout_candidates = sample(holdout, args.max_candidates, rng)
    train_scores = [
        max_transition_jaccard(sequence, generated_by_length)
        for sequence in train_candidates
    ]
    holdout_scores = [
        max_transition_jaccard(sequence, generated_by_length)
        for sequence in holdout_candidates
    ]

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "operator": args.operator,
        "generated_directory": str(args.generated_dir.resolve()),
        "evaluation_unit": "user-day trajectory",
        "counts": {
            "training_trajectories": len(train),
            "holdout_trajectories": len(holdout),
            "generated_trajectories": len(generated),
            "generated_unique_trajectories": len(generated_set),
            "generated_unmapped_rows": unmapped_rows,
        },
        "exact_repetition": {
            "generated_records_matching_training": exact_train,
            "generated_records_matching_holdout": exact_holdout,
            "generated_record_rate_matching_training": exact_train / len(generated),
            "generated_record_rate_matching_holdout": exact_holdout / len(generated),
            "unique_generated_sequences_matching_training": generated_train_unique,
            "unique_generated_sequences_matching_holdout": generated_holdout_unique,
            "generated_sequences_matching_a_singleton_training_sequence": rare_exact_train,
        },
        "transition_exposure": {
            "generated_transitions": len(generated_transitions),
            "transitions_unseen_in_training": unseen_transition_count,
            "unseen_training_transition_rate": (
                unseen_transition_count / len(generated_transitions)
                if generated_transitions else None
            ),
            "transitions_seen_at_most_threshold_times_in_training": rare_transition_count,
            "rare_training_transition_rate": (
                rare_transition_count / len(generated_transitions)
                if generated_transitions else None
            ),
            "rare_transition_threshold": args.rare_transition_count,
        },
        "black_box_distance_mia_probe": {
            "score": "maximum bigram Jaccard similarity to generated trajectories of length L-1, L, or L+1",
            "train_candidates_scored": len(train_scores),
            "holdout_candidates_scored": len(holdout_scores),
            "roc_auc": auc(train_scores, holdout_scores),
            "train_score_summary": distribution(train_scores),
            "holdout_score_summary": distribution(holdout_scores),
            "generated_sampled_for_scoring": len(generated_sample),
            "seed": args.seed,
            "interpretation": (
                "Exploratory black-box membership signal only; this is not a formal privacy guarantee "
                "or a white-box attack."
            ),
        },
        "limitations": [
            "The generated text format contains station-index sequences but no user identity or true source-trajectory label.",
            "Similarity/linkability statistics do not establish that a generated trajectory came from a particular training record.",
            "Results apply only to this split, generated release, station vocabulary, and scoring setup.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
