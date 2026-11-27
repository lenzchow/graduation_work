#!/usr/bin/env python3
"""Create a deterministic, exact-length-stratified split for trajectory CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_digest(user_id: str, day: str) -> str:
    payload = json.dumps([user_id, day], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_key(seed: int, user_id: str, day: str) -> bytes:
    payload = f"{seed}\0{user_id}\0{day}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw trajectory CSV")
    parser.add_argument("--base-stations", required=True, type=Path,
                        help="Static base-station CSV containing c_uli")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--user-column", default="c_number")
    parser.add_argument("--day-column", default="c_day")
    parser.add_argument("--station-column", default="c_uli")
    parser.add_argument("--latitude-column", default="c_lat")
    parser.add_argument("--longitude-column", default="c_lon")
    parser.add_argument("--bbox", nargs=4, type=float,
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    base_source = args.base_stations.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)
    if not base_source.is_file():
        raise FileNotFoundError(base_source)
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError("Require 1 <= --min-length <= --max-length")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; use --overwrite to replace it")

    # The static station table defines valid locations. It is not learned from
    # the trajectory split; optional geographic filtering is deterministic.
    with base_source.open(encoding="utf-8-sig", newline="") as stream:
        station_reader = csv.DictReader(stream)
        if not station_reader.fieldnames or args.station_column not in station_reader.fieldnames:
            raise ValueError(f"Base-station CSV must contain {args.station_column!r}")
        station_fields = station_reader.fieldnames
        required_geo = {args.latitude_column, args.longitude_column} if args.bbox else set()
        missing_geo = required_geo.difference(station_fields)
        if missing_geo:
            raise ValueError(f"Base-station CSV is missing: {', '.join(sorted(missing_geo))}")

        valid_stations: dict[str, dict[str, str]] = {}
        for row in station_reader:
            station_id = (row.get(args.station_column) or "").strip()
            if not station_id:
                continue
            if args.bbox:
                lat_min, lat_max, lon_min, lon_max = args.bbox
                try:
                    lat = float(row[args.latitude_column])
                    lon = float(row[args.longitude_column])
                except (TypeError, ValueError):
                    continue
                if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                    continue
            if station_id in valid_stations:
                raise ValueError(f"Duplicate station ID in base-station CSV: {station_id}")
            valid_stations[station_id] = row

    if not valid_stations:
        raise ValueError("No valid base stations remain after filtering")

    # Count each user-day trajectory after the same station-membership and
    # optional bounding-box filters used to prepare model inputs.
    lengths: Counter[tuple[str, str]] = Counter()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {args.user_column, args.day_column, args.station_column}
        if not reader.fieldnames:
            raise ValueError("Trajectory CSV has no header")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Trajectory CSV is missing: {', '.join(sorted(missing))}")
        trajectory_fields = reader.fieldnames
        for row in reader:
            user_id = (row.get(args.user_column) or "").strip()
            day = (row.get(args.day_column) or "").strip()
            station_id = (row.get(args.station_column) or "").strip()
            if not user_id or not day:
                raise ValueError("Trajectory CSV contains an empty user/day key")
            if station_id in valid_stations:
                lengths[(user_id, day)] += 1

    eligible = {
        key: length for key, length in lengths.items()
        if args.min_length <= length <= args.max_length
    }
    if len(eligible) < 2:
        raise ValueError("Fewer than two eligible trajectories remain after filtering")

    # Split independently within every exact-length stratum. Singleton strata
    # stay in training because they cannot be divided between both partitions.
    strata: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for key, length in eligible.items():
        strata[length].append(key)

    assignments: dict[tuple[str, str], str] = {}
    histogram: dict[int, dict[str, int]] = {}
    for length in sorted(strata):
        groups = sorted(strata[length], key=lambda key: stable_key(args.seed, *key))
        total = len(groups)
        if total == 1:
            train_count = 1
        else:
            train_count = math.floor(total * args.train_ratio + 0.5)
            train_count = max(1, min(total - 1, train_count))
        for index, key in enumerate(groups):
            assignments[key] = "train" if index < train_count else "holdout"
        histogram[length] = {
            "train": train_count,
            "holdout": total - train_count,
            "total": total,
        }

    if not any(part == "train" for part in assignments.values()):
        raise ValueError("Training split is empty")
    if not any(part == "holdout" for part in assignments.values()):
        raise ValueError(
            "Holdout split is empty; the eligible length strata may all be singletons"
        )

    outputs = {
        "train": "train.csv",
        "holdout": "holdout.csv",
        "train_stations": "base_station_train.csv",
        "holdout_stations": "base_station_holdout.csv",
        "histogram": "length_distribution.csv",
        "membership": "membership_map.csv",
        "manifest": "split_manifest.json",
    }
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        station_sets = {"train": set(), "holdout": set()}
        written_rows = {"train": 0, "holdout": 0}
        with (
            (temp_dir / outputs["train"]).open("w", encoding="utf-8", newline="") as train_stream,
            (temp_dir / outputs["holdout"]).open("w", encoding="utf-8", newline="") as holdout_stream,
        ):
            writers = {
                "train": csv.DictWriter(train_stream, fieldnames=trajectory_fields),
                "holdout": csv.DictWriter(holdout_stream, fieldnames=trajectory_fields),
            }
            for writer in writers.values():
                writer.writeheader()

            with source.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    station_id = (row.get(args.station_column) or "").strip()
                    if station_id not in valid_stations:
                        continue
                    key = (
                        (row.get(args.user_column) or "").strip(),
                        (row.get(args.day_column) or "").strip(),
                    )
                    partition = assignments.get(key)
                    if partition is None:
                        continue
                    writers[partition].writerow(row)
                    written_rows[partition] += 1
                    station_sets[partition].add(station_id)

        expected_rows = {
            part: sum(length for key, length in eligible.items()
                      if assignments[key] == part)
            for part in ("train", "holdout")
        }
        if written_rows != expected_rows:
            raise RuntimeError(
                f"Row-count validation failed: wrote {written_rows}, expected {expected_rows}"
            )

        for part, filename in (
            ("train", outputs["train_stations"]),
            ("holdout", outputs["holdout_stations"]),
        ):
            with (temp_dir / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=station_fields)
                writer.writeheader()
                for station_id in sorted(station_sets[part]):
                    writer.writerow(valid_stations[station_id])

        with (temp_dir / outputs["histogram"]).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["trajectory_length", "train_count", "holdout_count", "total_count"])
            for length, counts in sorted(histogram.items()):
                writer.writerow([length, counts["train"], counts["holdout"], counts["total"]])

        with (temp_dir / outputs["membership"]).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["group_hash", args.user_column, args.day_column, "length", "partition"])
            for key in sorted(assignments):
                writer.writerow([
                    group_digest(*key), key[0], key[1], eligible[key], assignments[key]
                ])

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_file": source.name,
            "source_sha256": file_sha256(source),
            "base_station_file": base_source.name,
            "base_station_sha256": file_sha256(base_source),
            "grouping": [args.user_column, args.day_column],
            "length_stratification": "exact post-station-filter trajectory length",
            "train_ratio_target": args.train_ratio,
            "seed": args.seed,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "bbox": args.bbox,
            "trajectory_counts": {
                "train": sum(1 for part in assignments.values() if part == "train"),
                "holdout": sum(1 for part in assignments.values() if part == "holdout"),
            },
            "row_counts": written_rows,
            "station_counts": {
                "train": len(station_sets["train"]),
                "holdout": len(station_sets["holdout"]),
                "holdout_only": len(station_sets["holdout"] - station_sets["train"]),
            },
            "length_histogram": {
                str(length): counts for length, counts in sorted(histogram.items())
            },
            "singleton_lengths_assigned_to_train": [
                length for length, counts in sorted(histogram.items())
                if counts["total"] == 1
            ],
            "identifiers_in_manifest": False,
        }
        (temp_dir / outputs["manifest"]).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        # Verify partition disjointness and that every eligible trajectory was assigned once.
        if len(assignments) != sum(manifest["trajectory_counts"].values()):
            raise RuntimeError("Trajectory assignment validation failed")
        if station_sets["train"] & station_sets["holdout"] and not station_sets["train"]:
            raise RuntimeError("Unexpected station-set validation failure")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Split files written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
