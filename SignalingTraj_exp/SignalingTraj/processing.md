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

SignalingTraj applies its existing station-frequency filter after splitting, so the effective holdout length histogram is recalculated against the training station vocabulary and the same preprocessing length range. The training histogram is retained as `length_distri_train.npy`; the active `length_distri.npy` matches the effective holdout histogram. Generation uses integer counts directly and skips zero-count lengths.

## Verification limits

Local syntax checks and spatial-only contract checks pass. No server GPU training or generation was run during this correction; the existing server process was left untouched. Because the model architecture changed, retrain SignalingTraj before generating new outputs. Existing CoST-Diff runs and checkpoints are unaffected.


## Release-format and first-stage privacy checks

- The generation writer emits spatial station-index rows only; SignalingTraj outputs contain no generated calendar-time values.
- The shared `privacy_eval/evaluate_release.py` reads this model's exact generated output directory, maps station indices back to original station IDs, and compares user-day station sequences with the split training and holdout sets.
- The report includes exact sequence overlap, rare/unseen transition exposure, and an exploratory black-box distance-based membership signal. These are preliminary empirical checks, not a formal privacy guarantee.

## Risk-evaluation implementation status

- `privacy_eval/audit_signaling_graph.py` compares saved graph edges with transitions in `train.csv`, checks graph/embedding dimensions, and records artifact hashes. Because the pre-existing graph and embedding files do not contain a preprocessing provenance manifest, the report cannot prove when or from which rows an old embedding was fitted.
- `privacy_eval/evaluate_release.py` checks spatial exact repetition and rare transitions, and reports an exploratory black-box membership signal using train-versus-holdout candidates.
- `privacy_eval/evaluate_spatial_near_duplicates.py` screens SignalingTraj outputs for spatial near-duplicates. Temporal exposure evaluation applies to CoST-Diff, which generates time information.
- White-box MIA is not implemented. The current training code does not expose a validated per-trajectory score. Do not interpret the black-box similarity score as a white-box result.
- The integrated runner is `privacy_eval/run_train_generate_evaluate.sh`. It does not rerun the split or preprocessing.

## Spatial-only model correction

- Restored the SignalingTraj model interface to consume base-station embedding sequences only. Its diffusion-step embedding remains because it represents the DDPM noise step, not a real trajectory timestamp.
- Removed calendar-hour embedding channels, temporal smoothness loss, timestamp inputs to training, and timestamp outputs from generation. Timestamps remain in raw data preprocessing only to sort each day's station visits into chronological order.
- Generation now writes four columns per visit (area code, latitude, longitude, station index). Release evaluation reads that spatial-only format; the elapsed-time exposure check is skipped for SignalingTraj.
