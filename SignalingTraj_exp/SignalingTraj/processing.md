# SignalingTraj processing record

## Changes

- Added deterministic 80/20 user-day splitting stratified by exact trajectory length.
- Saved train and holdout trajectories, per-partition station tables, length counts, a split manifest, and a local membership map.
- Changed preprocessing to read only the three operator training partitions, then rebuild station indexes, transition graphs, embeddings, sequence files, and length distribution from those training rows.
- Corrected the generated length histogram so every length present in the processed training data is counted.
- Preserved station IDs as strings when reading CSV files to avoid ID coercion.
- Added a batch preparation script and a user-day overlap audit.

## Server workflow

From the repository root:

```bash
bash privacy_eval/prepare_all_splits.sh
(cd SignalingTraj_exp/SignalingTraj/preprocess_data && python preprocess.py)
```

Then train and generate from the model root:

```bash
cd SignalingTraj_exp/SignalingTraj
python multi_main.py --dataset operator0
python generate_data.py --dataset operator0
```

Repeat the train and generation commands for `operator1` and `operator3`.

## Data boundary

Preprocessing reads only `privacy_eval/local/splits/SignalingTraj/operatorX/train.csv` and `base_station_train.csv`. Holdout files remain untouched for later evaluation. `membership_map.csv` contains identifiers; keep it on the private server and never commit it.

The split is stratified by raw user-day length after valid station-table filtering. The existing SignalingTraj preprocessing applies its own station-frequency filtering afterward, so final model-input lengths may differ from the raw split histogram; use the generated `data/operatorX/gen_all/length_distri.npy` as the actual training-input distribution.

## Verification limits

The repository was updated and reviewed through GitHub. This session could not start a local terminal and did not access the school server, so splitting, preprocessing, GPU training, and model generation have not been executed here. Run the commands above on the server and inspect each `split_audit.json` before training.
