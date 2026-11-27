# SignalingTraj processing record

## Changes

- Added deterministic 80/20 user-day splitting stratified by exact trajectory length.
- Saved train and holdout trajectories, per-partition station tables, length counts, a split manifest, and a local membership map.
- Changed preprocessing to build station indexes, transition graphs, embeddings, sequence files, and training length counts from training rows only.
- Fixed the embedding transition list so it counts only within-trajectory transitions; it no longer invents an edge between consecutive rows in the dataframe.
- Corrected the generated length histogram bounds and preserved station IDs as strings.
- The preprocessor reads holdout rows only to count eligible lengths after applying the training station vocabulary. It stores train and holdout histograms separately and sets the active generation histogram to the holdout counts, matching generated sample counts and lengths to the evaluation set.
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

Training sequences, station vocabularies, transition graphs, and embeddings use training data only. Holdout sequences, identifiers, and transitions are not used to build those model artifacts. Holdout is read only to calculate the aggregate generation length schedule; it does not supply generated locations or conditions. `membership_map.csv` contains identifiers; keep it on the private server and never commit it.

SignalingTraj applies its existing station-frequency filter after splitting, so the effective holdout length histogram is recalculated against the training station vocabulary and the same preprocessing length range. The training histogram is retained as `length_distri_train.npy`; the active `length_distri.npy` matches the effective holdout histogram.

## Verification limits

The repository was updated and reviewed through GitHub. This session could not start a local terminal and did not access the school server, so splitting, preprocessing, GPU training, and model generation have not been executed here. Run the commands above on the server and inspect each `split_audit.json` before training.
