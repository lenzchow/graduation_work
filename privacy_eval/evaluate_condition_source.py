#!/usr/bin/env python3
"""Audit whether CoST-Diff outputs resemble their sampled condition-source trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_generated(path: Path) -> dict[tuple[str, int], tuple[int, ...]]:
    result = {}
    for file_path in sorted(path.glob("*.txt")):
        samples = []
        current = []
        with file_path.open(encoding="utf-8-sig") as stream:
            for line_no, raw in enumerate(stream, 1):
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if current:
                        samples.append(tuple(current))
                        current = []
                    continue
                fields = next(csv.reader([line], skipinitialspace=True))
                if len(fields) < 4:
                    raise ValueError(f"Expected rid in column 4: {file_path}:{line_no}")
                current.append(int(float(fields[3])))
        if current:
            samples.append(tuple(current))
        for index, sequence in enumerate(samples):
            result[(file_path.name, index)] = sequence
    return result


def transition_jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    a = set(zip(left, left[1:]))
    b = set(zip(right, right[1:]))
    if not a and not b:
        return float(left == right)
    return len(a & b) / len(a | b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index-seqs", type=Path, required=True,
                        help="CoST-Diff data/operatorX/gen_all/index_seqs.bin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generated = read_generated(args.generated_dir)
    manifest_rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with args.index_seqs.open("rb") as stream:
        source_buckets = pickle.load(stream)

    rng = random.Random(args.seed)
    exact = []
    source_scores = []
    strict_top1 = []
    top1_or_tied = []
    missing = 0
    bucket_samples = defaultdict(list)

    for row in manifest_rows:
        key = (str(row["generated_file"]), int(row["generated_index"]))
        output_seq = generated.get(key)
        length = int(row["length"])
        source_index = int(row["source_bucket_index"])
        source_list = source_buckets.get(length, [])
        if output_seq is None or source_index < 0 or source_index >= len(source_list):
            missing += 1
            continue

        source_seq = tuple(int(x) for x in source_list[source_index])
        source_score = transition_jaccard(output_seq, source_seq)
        exact_match = output_seq == source_seq

        pool = list(range(len(source_list)))
        if len(pool) > args.max_candidates:
            pool = sorted(set(rng.sample(pool, args.max_candidates)) | {source_index})
        other_scores = [
            transition_jaccard(output_seq, tuple(int(x) for x in source_list[i]))
            for i in pool if i != source_index
        ]
        strict = source_score > max(other_scores) if other_scores else None
        tied = source_score >= max(other_scores) if other_scores else None

        exact.append(exact_match)
        source_scores.append(source_score)
        if strict is not None:
            strict_top1.append(strict)
            top1_or_tied.append(tied)
        bucket_samples[length].append((exact_match, source_score, strict, tied))

    if not exact:
        raise ValueError("No manifest entries matched generated blocks and source sequence buckets")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CoST-Diff sampled condition-source trajectory attribution; not user re-identification",
        "generated_directory": str(args.generated_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "matched_samples": len(exact),
        "unmatched_manifest_rows": missing,
        "exact_source_sequence_match_rate": sum(exact) / len(exact),
        "mean_source_transition_jaccard": statistics.fmean(source_scores),
        "samples_with_at_least_one_decoy": len(strict_top1),
        "source_strict_top1_rate_against_sampled_same_length_decoys": sum(strict_top1) / len(strict_top1) if strict_top1 else None,
        "source_top1_or_tied_rate_against_sampled_same_length_decoys": sum(top1_or_tied) / len(top1_or_tied) if top1_or_tied else None,
        "candidate_pool_limit_including_source": args.max_candidates,
        "by_length": {
            str(length): {
                "count": len(values),
                "exact_match_rate": sum(x[0] for x in values) / len(values),
                "mean_source_transition_jaccard": statistics.fmean(x[1] for x in values),
                "rankable_count": sum(x[2] is not None for x in values),
                "source_strict_top1_rate": (
                    sum(x[2] for x in values if x[2] is not None) /
                    sum(x[2] is not None for x in values)
                    if any(x[2] is not None for x in values) else None
                ),
                "source_top1_or_tied_rate": (
                    sum(x[3] for x in values if x[3] is not None) /
                    sum(x[3] is not None for x in values)
                    if any(x[3] is not None for x in values) else None
                ),
            }
            for length, values in sorted(bucket_samples.items())
        },
        "interpretation": (
            "This measures association with the trajectory sampled to supply conditions. It does not "
            "establish a unique source for arbitrary generated data or identify a person. Top-1 is compared "
            "with sampled same-length decoys; report tied and strict rates separately."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
