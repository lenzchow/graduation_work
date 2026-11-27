#!/usr/bin/env python3
"""Audit graph edges and embedding dimensions against a saved training split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_index(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(station): int(index) for station, index in raw.items()}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("uli_index_dict.json values must be contiguous zero-based indexes")
    return mapping


def training_edges(path: Path, station_to_index: dict[str, int]) -> set[tuple[int, int]]:
    groups = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"c_number", "c_day", "c_uli", "c_timestamp"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for line, row in enumerate(reader, 2):
            station = (row.get("c_uli") or "").strip()
            if station not in station_to_index:
                continue
            groups[(row["c_number"], row["c_day"])].append(
                (float(row["c_timestamp"]), line, station_to_index[station])
            )
    edges = set()
    for rows in groups.values():
        rows.sort()
        seq = [item[2] for item in rows]
        edges.update(zip(seq, seq[1:]))
    return edges


def graph_edges(path: Path, expected_nodes: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    edges = set()
    rows = 0
    cols = None
    with path.open(encoding="utf-8-sig") as stream:
        for row_index, line in enumerate(stream):
            values = [float(x) for x in next(csv.reader([line]))]
            if cols is None:
                cols = len(values)
            if len(values) != cols:
                raise ValueError(f"Ragged graph matrix at row {row_index}")
            for col_index, value in enumerate(values):
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite graph value at ({row_index}, {col_index})")
                if value != 0:
                    edges.add((row_index, col_index))
            rows += 1
    shape = (rows, cols or 0)
    if shape != (expected_nodes, expected_nodes):
        raise ValueError(f"Graph shape {shape} does not match station vocabulary {(expected_nodes, expected_nodes)}")
    return edges, shape


def embedding_shape(path: Path) -> tuple[int, int, int]:
    with path.open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"Invalid embedding header in {path}")
        nodes, dim = map(int, header)
        count = 0
        seen = set()
        for line_no, line in enumerate(stream, 2):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != dim + 1:
                raise ValueError(f"Embedding row dimension mismatch at {path}:{line_no}")
            node = int(fields[0])
            if node in seen:
                raise ValueError(f"Duplicate embedding index {node} in {path}")
            seen.add(node)
            count += 1
    return nodes, dim, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True, help="SignalingTraj model root")
    parser.add_argument("--operator", choices=("operator0", "operator1", "operator3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, default=None,
                        help="Defaults to data/operatorX/graph/road_embed_vgae.txt")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    split = repo_root / "privacy_eval/local/splits/SignalingTraj" / args.operator / "train.csv"
    data = args.model_root / "data" / args.operator
    station_map_path = data / "extra_file/uli_index_dict.json"
    graph_path = data / "graph/graph_A.csv"
    embedding_path = args.embedding or data / "graph/road_embed_vgae.txt"
    for path in (split, station_map_path, graph_path, embedding_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    station_to_index = read_index(station_map_path)
    train_edges = training_edges(split, station_to_index)
    stored_edges, shape = graph_edges(graph_path, len(station_to_index))
    non_train_edges = sorted(edge for edge in stored_edges if edge[0] != edge[1] and edge not in train_edges)
    emb_nodes, emb_dim, emb_rows = embedding_shape(embedding_path)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "SignalingTraj",
        "operator": args.operator,
        "training_split_sha256": sha256(split),
        "station_map_sha256": sha256(station_map_path),
        "graph_sha256": sha256(graph_path),
        "embedding_sha256": sha256(embedding_path),
        "station_count": len(station_to_index),
        "training_transition_count": len(train_edges),
        "graph_shape": list(shape),
        "graph_nonzero_edge_count_including_self_loops": len(stored_edges),
        "graph_non_self_edges_not_found_in_training_transitions": len(non_train_edges),
        "graph_non_train_edge_examples": [list(edge) for edge in non_train_edges[:20]],
        "graph_train_transition_subset_check": len(non_train_edges) == 0,
        "embedding_header_nodes": emb_nodes,
        "embedding_dimension": emb_dim,
        "embedding_rows": emb_rows,
        "embedding_shape_matches_station_map": emb_nodes == len(station_to_index) and emb_rows == emb_nodes,
        "embedding_training_provenance_confirmed": False,
        "provenance_note": (
            "The saved files have no signed/source manifest. This audit checks graph edges against the "
            "training split and embedding dimensions, but cannot prove from legacy files alone that the "
            "embedding was fit only on training transitions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
