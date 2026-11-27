#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${MODELS:-signaling cost}"
OPERATORS="${OPERATORS:-operator0 operator1 operator3}"
EMBED_MODE="${EMBED_MODE:-vgae}"
REPORT_DIR="$ROOT/privacy_eval/local/reports"
LOG_DIR="$ROOT/privacy_eval/local/logs"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

model_root() {
    case "$1" in
        signaling) printf '%s\n' "$ROOT/SignalingTraj_exp/SignalingTraj" ;;
        cost) printf '%s\n' "$ROOT/CoST-Diff_exp/CoST-Diff" ;;
        *) printf 'Unknown model: %s\n' "$1" >&2; return 2 ;;
    esac
}

split_model() {
    case "$1" in signaling) printf 'SignalingTraj\n' ;; cost) printf 'CoST-Diff\n' ;; esac
}

preflight_python() {
    local root="$1"
    python - "$root/models/model.py" "$root/multi_main.py" "$root/generate_data.py" <<'PY'
import ast
import pathlib
import sys

for name in sys.argv[1:]:
    path = pathlib.Path(name)
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        print(f"Python syntax preflight failed: {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)
print("Python syntax preflight passed.")
PY
}

latest_generated_dir() {
    local root="$1" operator="$2"
    find "$root/generate_data" -mindepth 1 -maxdepth 1 -type d -name "*$operator*" \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

for model in $MODELS; do
    root="$(model_root "$model")"
    for operator in $OPERATORS; do
        split_dir="$ROOT/privacy_eval/local/splits/$(split_model "$model")/$operator"
        [[ -s "$split_dir/train.csv" && -s "$split_dir/holdout.csv" ]] || {
            echo "Missing split files under $split_dir; this runner does not create or alter splits." >&2
            exit 1
        }

        echo "== $model / $operator: verify source files =="
        preflight_python "$root"

        if [[ "$model" == "signaling" ]]; then
            graph_report="$REPORT_DIR/signaling_graph_${operator}.json"
            python "$ROOT/privacy_eval/audit_signaling_graph.py" \
                --model-root "$root" --operator "$operator" \
                --embedding "$root/data/$operator/graph/road_embed_$EMBED_MODE.txt" \
                --output "$graph_report"
        fi

        run_tag="$(date +%Y%m%d%H%M%S)"
        log="$LOG_DIR/${model}_${operator}_${run_tag}.log"
        echo "== Training $model / $operator =="
        (
            cd "$root"
            python multi_main.py --dataset "$operator" --embed_mode "$EMBED_MODE"
            python generate_data.py --dataset "$operator" --embed_mode "$EMBED_MODE"
        ) 2>&1 | tee "$log"

        generated_dir="$(latest_generated_dir "$root" "$operator")"
        [[ -n "$generated_dir" && -d "$generated_dir" ]] || {
            echo "Could not locate generated output under $root/generate_data" >&2
            exit 1
        }
        report_tag="$(date +%Y%m%d%H%M%S)"
        echo "Evaluating output: $generated_dir"
        python "$ROOT/privacy_eval/evaluate_release.py" \
            --model "$model" --operator "$operator" --generated-dir "$generated_dir" \
            --output "$REPORT_DIR/${model}_${operator}_release_${report_tag}.json"
        python "$ROOT/privacy_eval/evaluate_temporal_exposure.py" \
            --model "$model" --operator "$operator" --generated-dir "$generated_dir" \
            --output "$REPORT_DIR/${model}_${operator}_temporal_${report_tag}.json"

        if [[ "$model" == "cost" ]]; then
            time_tag="${generated_dir##*curTime}"
            manifest="$ROOT/privacy_eval/local/condition_sources/${operator}_${time_tag}.jsonl"
            [[ -s "$manifest" ]] || {
                echo "Missing CoST-Diff condition-source manifest: $manifest" >&2
                echo "This output predates source tracking or generation failed to write its manifest." >&2
                exit 1
            }
            python "$ROOT/privacy_eval/evaluate_condition_source.py" \
                --generated-dir "$generated_dir" --manifest "$manifest" \
                --index-seqs "$root/data/$operator/gen_all/index_seqs.bin" \
                --output "$REPORT_DIR/cost_${operator}_condition_source_${report_tag}.json"
        fi
    done
done

echo "Training, generation, and implemented release checks finished. Reports are in $REPORT_DIR"
