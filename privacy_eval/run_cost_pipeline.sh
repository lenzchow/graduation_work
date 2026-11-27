/Users/zoulinzi/.zprofile:3: no such file or directory: /usr/local/Homebrew/bin/brew
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OPERATORS="${OPERATORS:-operator0 operator1 operator3}"
EMBED_MODE="${EMBED_MODE:-vgae}"

command -v "$PYTHON_BIN" >/dev/null || {
    echo "Python executable not found: $PYTHON_BIN (set PYTHON_BIN if needed)." >&2
    exit 1
}

echo "== 1/4: split both models by exact trajectory length =="
bash "$ROOT/privacy_eval/prepare_all_splits.sh"

echo "== 2/4: build SignalingTraj training inputs, graph, and embeddings =="
"$PYTHON_BIN" "$ROOT/SignalingTraj_exp/SignalingTraj/preprocess_data/preprocess.py"

echo "== 3/4: build CoST-Diff training inputs, graph, and embeddings =="
"$PYTHON_BIN" "$ROOT/CoST-Diff_exp/CoST-Diff/preprocess_data/preprocess.py"

echo "== 4/4: train, generate, and evaluate CoST-Diff =="
MODELS=cost OPERATORS="$OPERATORS" EMBED_MODE="$EMBED_MODE" PYTHON_BIN="$PYTHON_BIN" \
    bash "$ROOT/privacy_eval/run_train_generate_evaluate.sh"

echo "CoST-Diff pipeline completed. Check privacy_eval/local/logs and privacy_eval/local/reports."
