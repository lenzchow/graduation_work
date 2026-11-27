#!/usr/bin/env python3
"""Measure spatial near-duplicate similarity from generated outputs to train/holdout routes."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from evaluate_release import MODEL_LAYOUT, read_generated_trajectories, read_index_map, read_real_trajectories


def transitions(sequence: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(sequence, sequence[1:]))


def jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a, b = transitions(left), transitions(right)
    if not a and not b:
        return float(left == right)
    return len(a & b) / len(a | b)


def sample_references(items, limit: int, rng: random.Random):
    buckets = defaultdict(list)
    for sequence in items:
        buckets[len(sequence)].append(sequence)
    for length, values in buckets.items():
        if len(values) > limit:
            buckets[length] = rng.sample(values, limit)
    return buckets


def closest_scores(generated, references):
    scores = []
    for sequence in generated:
        candidates = []
        for length in range(max(1, len(sequence) - 1), len(sequence) + 2):
            candidates.extend(references.get(length, []))
        if candidates:
            scores.append(max(jaccard(sequence, candidate) for candidate in candidates))
    return scores


def summarize(scores, threshold):
    if not scores:
        return {
            "matched_outputs": 0,
            "mean_max_transition_jaccard": None,
            "median_max_transition_jaccard": None,
            "p95_max_transition_jaccard": None,
            "fraction_at_or_above_threshold": None,
        }
    ordered = sorted(scores)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "matched_outputs": len(scores),
        "mean_max_transition_jaccard": statistics.fmean(scores),
        "median_max_transition_jaccard": statistics.median(scores),
        "p95_max_transition_jaccard": p95,
        "fraction_at_or_above_threshold": sum(x >= threshold for x in scores) / len(scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_LAYOUT, required=True)
    parser.add_argument("--operator", choices=("operator0", "operator1", "operator3"), required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Exploratory near-duplicate threshold on transition-set Jaccard")
    parser.add_argument("--max-reference-per-length", type=int, default=1000)
    parser.add_argument("--max-generated", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    layout = MODEL_LAYOUT[args.model]
    root = Path(__file__).resolve().parents[1]
    model_root = root / layout["root"]
    split_dir = root / "privacy_eval/local/splits" / layout["split"] / args.operator
    index_to_station = read_index_map(model_root / "data" / args.operator / "extra_file/index_uli_dict.json")
    vocabulary = set(index_to_station.values())

    train = read_real_trajectories(
        split_dir / "train.csv", vocabulary, layout["min_length"], layout["max_length"]
    )
    holdout = read_real_trajectories(
        split_dir / "holdout.csv", vocabulary, layout["min_length"], layout["max_length"]
    )
    generated, _ = read_generated_trajectories(
        args.generated_dir, index_to_station, layout["generated_columns"]
    )
    if not train or not holdout or not generated:
        raise ValueError(f"Empty input: train={len(train)}, holdout={len(holdout)}, generated={len(generated)}")

    rng = random.Random(args.seed)
    train_ref = sample_references(train, args.max_reference_per_length, rng)
    holdout_ref = sample_references(holdout, args.max_reference_per_length, rng)
    if len(generated) > args.max_generated:
        generated = rng.sample(generated, args.max_generated)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "operator": args.operator,
        "generated_directory": str(args.generated_dir.resolve()),
        "similarity": "Jaccard similarity of distinct consecutive station transitions; candidate lengths L-1, L, L+1",
        "near_duplicate_threshold": args.threshold,
        "generated_scored": len(generated),
        "nearest_training_route": summarize(closest_scores(generated, train_ref), args.threshold),
        "nearest_holdout_route": summarize(closest_scores(generated, holdout_ref), args.threshold),
        "interpretation": (
            "Exploratory spatial near-duplicate screening. Transition-set similarity ignores order multiplicity "
            "and is not a privacy guarantee; interpret train and holdout scores together and report the chosen threshold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
