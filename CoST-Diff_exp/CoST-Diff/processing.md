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
- Generation now uses exact integer counts from the active length histogram and skips empty length bins.

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


## Release-format and first-stage privacy checks

- The generation writer emits plain numeric CSV fields for each row, so the release parser can read station indices and time values without PyTorch tensor representations.
- The shared `privacy_eval/evaluate_release.py` reads this model's exact generated output directory, maps station indices back to original station IDs, and compares user-day station sequences with the split training and holdout sets.
- The report includes exact sequence overlap, rare/unseen transition exposure, and an exploratory black-box distance-based membership signal. These are preliminary empirical checks, not a formal privacy guarantee.

## Risk-evaluation implementation status

- `models/multi_train.py` records the index of each sampled condition trajectory while generating, in a private JSONL manifest under `privacy_eval/local/condition_sources/`. The manifest maps output blocks to a condition-source sequence index; it contains no user identity mapping, so its result is condition-source trajectory attribution, not user re-identification.
- `privacy_eval/evaluate_condition_source.py` compares generated station sequences with their sampled condition-source sequences and same-length decoys.
- `privacy_eval/evaluate_release.py` checks spatial exact repetition and rare transitions, and reports an exploratory black-box membership signal using train-versus-holdout candidates. The spatial and temporal near-duplicate screens are separate scripts.
- `privacy_eval/evaluate_spatial_near_duplicates.py` and `privacy_eval/evaluate_temporal_exposure.py` screen generated releases for spatial and elapsed-time similarity. These are exposure screens, not standalone privacy guarantees.
- White-box MIA is not implemented. The current model training interface returns batch-aggregated losses rather than validated per-trajectory membership scores; do not substitute a batch mean.
- The integrated runner is `privacy_eval/run_train_generate_evaluate.sh`. It does not rerun the split or preprocessing. CoST-Diff can be selected independently with `MODELS="cost"`; runtime execution still needs to be performed on the server.
