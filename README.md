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

Station vocabularies, transition graphs, embeddings, and sequence files are rebuilt from training records only. The preprocessor reads holdout rows only to count eligible trajectory lengths after applying the training station vocabulary; it does not use holdout sequences, IDs, coordinates, or transitions to build model artifacts. It saves `length_distri_train.npy` and `length_distri_holdout.npy`; the active `length_distri.npy` is set to the holdout counts so the generated sample count and length distribution match the evaluation set. The generation entry points read this active file. This is an evaluation sampling schedule, not a model-learned statistic.

Do not reuse full-data `data/` or `handled_data/` artifacts. The preprocessing scripts clear and rebuild each operator's derived `data/operatorX/` directory.

## Train and generate

Run from each model root, for each operator:

```bash
python multi_main.py --dataset operator0
python generate_data.py --dataset operator0
```

Repeat with `operator1` and `operator3`. The existing entry points read the rebuilt artifacts under `data/operatorX/`.

## Data protection boundary

No raw trajectory or base-station CSV is committed by this repository. Do not add `privacy_eval/local/`, `original_data/`, generated artifacts, or `membership_map.csv` to Git. Keep `membership_map.csv` on the private server; it contains user and day identifiers.
