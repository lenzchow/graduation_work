# CoST-Diff processing record

## Changes

- Added deterministic 80/20 user-day splitting stratified by exact trajectory length, using the existing CoST-Diff geographic bounds and length range.
- Saved train and holdout trajectories, per-partition station tables, length counts, a split manifest, and a local membership map.
- Changed preprocessing to read only training partitions and rebuild station indexes, transition graphs, embeddings, sequence files, and length distribution from training rows.
- Bypassed the old full-data `handled_data` cache in the new main preprocessing path.
- Preserved station IDs as strings when reading CSV files to avoid ID coercion.
- Added a batch preparation script and a user-day overlap audit.

## Server workflow

From the repository root:

```bash
bash privacy_eval/prepare_all_splits.sh
(cd CoST-Diff_exp/CoST-Diff/preprocess_data && python preprocess.py)
```

Then train and generate from the model root:

```bash
cd CoST-Diff_exp/CoST-Diff
python multi_main.py --dataset operator0
python generate_data.py --dataset operator0
```

Repeat the train and generation commands for `operator1` and `operator3`.

## Data boundary

Preprocessing reads only `privacy_eval/local/splits/CoST-Diff/operatorX/train.csv` and `base_station_train.csv`. Holdout files remain untouched for later evaluation. `membership_map.csv` contains identifiers; keep it on the private server and never commit it.

CoST-Diff's training `length_distri.npy` is derived only from training trajectories. The companion `length_distribution.csv` records train and holdout counts by length for evaluation analysis.

## Verification limits

The repository was updated and reviewed through GitHub. This session could not start a local terminal and did not access the school server, so splitting, preprocessing, GPU training, and model generation have not been executed here. Run the commands above on the server and inspect each `split_audit.json` before training.
