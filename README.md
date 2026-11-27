# Trajectory generation projects

Source code is organized under separate experiment roots:

- `SignalingTraj_exp/SignalingTraj`: SignalingTraj model.
- `CoST-Diff_exp/CoST-Diff`: CoST-Diff model.
- Each experiment root has a `baselines/` directory for baseline implementations.

## Privacy-evaluation data preparation

Keep raw files only on the private server. Each model's `preprocess_data/original_data/` directory is expected to contain:

- `traj_0.csv`, `traj_1.csv`, `traj_3.csv`
- `base_station_0.csv`, `base_station_1.csv`, `base_station_3.csv`

From the repository root, create deterministic 80/20 splits stratified by exact user-day trajectory length, then audit that no `c_number + c_day` group crosses partitions:

```bash
bash privacy_eval/prepare_all_splits.sh
```

The script creates, for each model and operator, `train.csv`, `holdout.csv`, `base_station_train.csv`, `base_station_holdout.csv`, `length_distribution.csv`, `membership_map.csv`, `split_manifest.json`, and `split_audit.json` under `privacy_eval/local/splits/`. These contain private data or identifiers and are ignored by Git.

The splitter uses the static station table to remove unknown station IDs. CoST-Diff also applies its existing geographic bounding box and length range (10–60); SignalingTraj uses lengths 50–200. Length strata containing only one eligible trajectory are assigned to training because they cannot be represented in both partitions. The manifest records these cases.

## Build model artifacts from training data

After the split script completes, rebuild each model's preprocessing artifacts:

```bash
(cd SignalingTraj_exp/SignalingTraj/preprocess_data && python preprocess.py)
(cd CoST-Diff_exp/CoST-Diff/preprocess_data && python preprocess.py)
```

Station vocabularies, transition graphs, embeddings, and sequence files are rebuilt from training records only. The preprocessor reads holdout rows only to count eligible trajectory lengths after applying the training station vocabulary; it does not use holdout sequences, IDs, coordinates, or transitions to build model artifacts. It saves `length_distri_train.npy` and `length_distri_holdout.npy`; the active `length_distri.npy` is set to holdout counts. The generation code uses each bin's integer count directly and skips empty bins, so it generates the requested count for each length. The generation entry points read this active file. This is an evaluation sampling schedule, not a model-learned statistic.

Do not reuse full-data `data/` or `handled_data/` artifacts. The preprocessing scripts clear and rebuild each operator's derived `data/operatorX/` directory.

## Train and generate

Run from each model root, for each operator:

```bash
python multi_main.py --dataset operator0
python generate_data.py --dataset operator0
```

Repeat with `operator1` and `operator3`. The existing entry points read the rebuilt artifacts under `data/operatorX/`.

## Evaluate a generated release

After the split-based preprocessing, training, and generation have completed, list generated output directories and choose the exact run directory for the matching operator:

```bash
find SignalingTraj_exp/SignalingTraj/generate_data -mindepth 1 -maxdepth 1 -type d -print
find CoST-Diff_exp/CoST-Diff/generate_data -mindepth 1 -maxdepth 1 -type d -print
```

Run the release-level checks from the repository root. Replace `<exact-output-path>` with one complete directory path printed above:

```bash
python privacy_eval/evaluate_release.py --model signaling --operator operator0 --generated-dir '<exact-output-path>' --output privacy_eval/local/reports/signaling_operator0.json
python privacy_eval/evaluate_release.py --model cost --operator operator0 --generated-dir '<exact-output-path>' --output privacy_eval/local/reports/cost_operator0.json
```

The report compares generated station sequences with the matching operator's training and holdout user-day trajectories. It includes exact sequence overlap, rare/unseen transition exposure, and an exploratory black-box distance-based membership signal. The integrated runner adds spatial and temporal near-duplicate screening, CoST-Diff condition-source attribution, and a SignalingTraj graph audit. White-box MIA and user re-identification are not implemented; distance and similarity scores are not privacy guarantees. Generated rows must be plain numeric CSV fields.

## Data protection boundary

No raw trajectory or base-station CSV is committed by this repository. Do not add `privacy_eval/local/`, `original_data/`, generated artifacts, or `membership_map.csv` to Git. Keep `membership_map.csv` on the private server; it contains user and day identifiers.


## Run evaluations after preprocessing

When the split files and model artifacts have already been prepared on the server, do not rerun split or preprocessing steps. From the repository root:

```bash
git pull --ff-only origin privacy-risk-evaluation-prep
conda activate st_new
bash privacy_eval/run_train_generate_evaluate.sh
```

The runner reads the prepared splits under `privacy_eval/local/splits/<model-root>/operatorX/`, trains and generates each selected model, then writes private reports and logs under `privacy_eval/local/`. It does not create or alter split files.

Optional selection:

```bash
MODELS="cost" OPERATORS="operator0" EMBED_MODE="vgae" bash privacy_eval/run_train_generate_evaluate.sh
```

Implemented release checks include spatial overlap and rare-transition screening, black-box similarity MIA probing, spatial and temporal near-duplicate screening, CoST-Diff condition-source attribution, and SignalingTraj graph edge/dimension auditing. The current legacy embedding artifacts have no provenance manifest, so their exact training provenance cannot be independently confirmed by the audit.

White-box MIA remains unimplemented because the current training code does not expose a verified per-trajectory score for both members and non-members. Do not substitute a batch-average loss. The runner checks syntax across all selected model Python files and audits saved user-day splits before it begins training; it does not alter split or preprocessing outputs. Evaluation findings are limited to the data and attack setup; they do not establish a formal privacy guarantee.

**Current SignalingTraj blocker:** `SignalingTraj_exp/SignalingTraj/models/model.py` in this branch currently contains an incomplete assignment (`self.tim_emb_size =`) and references an undefined `args` plus an uninitialized `time_hour_embedding`. Python syntax preflight therefore stops before training or generation. Use `MODELS="cost"` to run only CoST-Diff while this source is repaired; the privacy evaluation scripts themselves pass static syntax checks. Set `PYTHON_BIN` if the environment uses a different Python command.

Keep raw data, splits, generated files, logs, and reports on the private server. Do not commit them to Git.
