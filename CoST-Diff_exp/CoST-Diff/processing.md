# CoST-Diff processing record

## Changes

- Added deterministic 80/20 user-day splitting stratified by exact trajectory length, using the existing CoST-Diff geographic bounds and length range.
- Saved train and holdout trajectories, per-partition station tables, length counts, a split manifest, and a local membership map.
- Changed preprocessing to build station indexes, transition graphs, embeddings, sequence files, and training length counts from training rows only.
- Bypassed the old full-data `handled_data` cache in the new main preprocessing path.
- Fixed the embedding transition list so it counts only within-trajectory transitions; it no longer invents an edge between consecutive rows in the dataframe.
- Preserved station IDs as strings when reading CSV files.
- The preprocessor reads holdout rows only to count eligible lengths after applying the training station vocabulary. It stores train and holdout histograms separately and sets the active generation histogram to the holdout counts, matching generated sample counts and lengths to the evaluation set.
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

Training sequences, station vocabularies, transition graphs, and embeddings use training data only. Holdout sequences, identifiers, and transitions are not used to build those model artifacts. Holdout is read only to calculate the aggregate generation length schedule; it does not supply generated locations or conditions. `membership_map.csv` contains identifiers; keep it on the private server and never commit it.

CoST-Diff uses lengths 10–60 and its existing bounding box. The active `length_distri.npy` matches the filtered holdout length histogram; `length_distri_train.npy` remains available for comparison. Never reuse full-data `data/` or `handled_data/` artifacts.

## Verification limits

The repository was updated and reviewed through GitHub. This session could not start a local terminal and did not access the school server, so splitting, preprocessing, GPU training, and model generation have not been executed here. Run the commands above on the server and inspect each `split_audit.json` before training.
