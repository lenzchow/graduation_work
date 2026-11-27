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

The splitter uses the static station table to remove unknown station IDs. CoST-Diff also applies its existing geographic bounding box and length range (10–60); SignalingTraj uses lengths 1–12000. Length strata containing only one eligible trajectory are assigned to training because they cannot be represented in both partitions. The manifest records these cases.

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

The report compares generated station sequences with the matching operator's training and holdout user-day trajectories. It includes exact sequence overlap, rare/unseen transition exposure, and an exploratory black-box distance-based membership signal. This first-stage script does not implement white-box MIA, user attribution, or CoST-Diff condition-source attribution; it does not treat distance scores as a privacy guarantee. Generated rows must be plain numeric CSV fields. The generation writers now emit numeric fields; CoST-Diff stops writing after the terminal zero-wait row to omit padded tails.

## Data protection boundary

No raw trajectory or base-station CSV is committed by this repository. Do not add `privacy_eval/local/`, `original_data/`, generated artifacts, or `membership_map.csv` to Git. Keep `membership_map.csv` on the private server; it contains user and day identifiers.


## Re-run after preprocessing is already complete

The privacy split and model artifacts are already prepared on the server. Do **not** rerun `prepare_all_splits.sh` or either preprocessor unless you intentionally want to replace the current split/artifacts. The combined runner below only checks the saved split, audits the available artifacts, trains, generates, and evaluates:

```bash
cd /workspace/traj_generation/graduation_paper_exp/privacy-risk-evaluation-prep-updated
git pull origin privacy-risk-evaluation-prep
conda activate st_new
bash privacy_eval/run_train_generate_evaluate.sh
```

Optional selections (defaults are both models and all three operators):

```bash
MODELS="signaling" OPERATORS="operator0" EMBED_MODE="vgae" bash privacy_eval/run_train_generate_evaluate.sh
```

Implemented checks in this runner:
- release-level spatial overlap and the existing black-box similarity MIA probe;
- spatial near-duplicate screening by transition-set Jaccard against train and holdout trajectories;\n- temporal nearest-sequence similarity against train and holdout trajectories;
- CoST-Diff condition-source attribution, using a per-generation JSONL manifest;
- SignalingTraj graph edge and embedding-dimension audit.

The condition-source check requires a **new generation run** after pulling this branch; it does not require repeating preprocessing. It reports attribution to the sampled condition-source trajectory, not a user identity. The graph audit checks graph edges against training transitions and checks embedding dimensions. The current legacy embedding files have no provenance manifest, so the audit cannot independently prove which exact transitions were used to fit the embedding.

White-box MIA is not yet implemented as a valid per-trajectory attack. Both model implementations need a verified per-sample scoring interface and a matching holdout-input path before reporting white-box results. Do not interpret batch-average training loss as a per-trajectory membership score. The runner performs a Python syntax preflight before training and stops if the checked model scripts do not parse.

All reports and source manifests are written under ignored `privacy_eval/local/`; keep them on the private server.
