#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLITTER="$ROOT/privacy_eval/split_trajectories.py"
AUDITOR="$ROOT/privacy_eval/audit_split.py"

prepare_model() {
    local model="$1"
    local min_length="$2"
    local max_length="$3"
    local bbox_mode="$4"
    local raw="$ROOT/$model/preprocess_data/original_data"

    for operator in 0 1 3; do
        local output="$ROOT/privacy_eval/local/splits/$model/operator$operator"
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
            python "$SPLITTER" "${common_args[@]}" \
                --latitude-column c_lat --longitude-column c_lon \
                --bbox 22.705 22.890 108.2 108.46
        else
            python "$SPLITTER" "${common_args[@]}"
        fi

        python "$AUDITOR" \
            --train "$output/train.csv" \
            --holdout "$output/holdout.csv" \
            --grouping user-day \
            --user-column c_number \
            --day-column c_day \
            --output "$output/split_audit.json"
    done
}

prepare_model "SignalingTraj_exp/SignalingTraj" 50 200 signaling
prepare_model "CoST-Diff_exp/CoST-Diff" 10 60 cost

echo "All six splits and user-day audits completed."
