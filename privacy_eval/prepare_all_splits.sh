#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLITTER="$ROOT/privacy_eval/split_trajectories.py"
AUDITOR="$ROOT/privacy_eval/audit_split.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

command -v "$PYTHON_BIN" >/dev/null || {
    echo "Python executable not found: $PYTHON_BIN (set PYTHON_BIN if needed)." >&2
    exit 1
}

split_files_exist() {
    local output="$1"
    local file
    for file in train.csv holdout.csv base_station_train.csv base_station_holdout.csv \
                length_distribution.csv membership_map.csv split_manifest.json; do
        [[ -s "$output/$file" ]] || return 1
    done
}

audit_is_passed() {
    "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    audit = json.load(stream)
if (
    audit.get("passed") is True
    and audit.get("overlapping_groups") == 0
    and audit.get("train_groups", 0) > 0
    and audit.get("holdout_groups", 0) > 0
):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

run_audit() {
    local output="$1"
    "$PYTHON_BIN" "$AUDITOR" \
        --train "$output/train.csv" \
        --holdout "$output/holdout.csv" \
        --grouping user-day \
        --user-column c_number \
        --day-column c_day \
        --output "$output/split_audit.json"
}

prepare_model() {
    local model="$1"
    local min_length="$2"
    local max_length="$3"
    local bbox_mode="$4"
    local raw="$ROOT/$model/preprocess_data/original_data"

    for operator in 0 1 3; do
        local output="$ROOT/privacy_eval/local/splits/$model/operator$operator"
        local audit="$output/split_audit.json"

        if split_files_exist "$output"; then
            if ! audit_is_passed "$audit"; then
                echo "Auditing existing $model/operator$operator split..."
                if ! run_audit "$output"; then
                    echo "Existing split audit failed; regenerating $model/operator$operator."
                fi
            fi
            if audit_is_passed "$audit"; then
                echo "Reusing valid split: $model/operator$operator"
                continue
            fi
        fi

        echo "Creating split: $model/operator$operator"
        local common_args=(
            --input "$raw/traj_$operator.csv"
            --base-stations "$raw/base_station_$operator.csv"
            --output-dir "$output"
            --user-column c_number
            --day-column c_day
            --station-column c_uli
            --min-length "$min_length"
            --max-length "$max_length"
            --train-ratio 0.8
            --seed 42
            --overwrite
        )

        if [[ "$bbox_mode" == "cost" ]]; then
            "$PYTHON_BIN" "$SPLITTER" "${common_args[@]}" \
                --latitude-column c_lat --longitude-column c_lon \
                --bbox 22.705 22.890 108.2 108.46
        else
            "$PYTHON_BIN" "$SPLITTER" "${common_args[@]}"
        fi

        run_audit "$output"
    done
}

prepare_model "SignalingTraj_exp/SignalingTraj" 50 200 signaling
prepare_model "CoST-Diff_exp/CoST-Diff" 10 60 cost

echo "All six splits are present and audited."
