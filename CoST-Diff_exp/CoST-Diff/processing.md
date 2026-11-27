# CoST-Diff processing record

## What changed

- Moved the original code tree to CoST-Diff_exp/CoST-Diff; model behavior is unchanged.
- Added a deterministic user-day split tool and a separate split audit in privacy_eval/.
- Added a whole-user split option for later user-level MIA.
- Expanded .gitignore for private data, membership maps, caches, checkpoints, and outputs.
- Added a baselines/ placeholder for future baseline code.

## Split commands

Run from the repository root, separately for each operator source file:

    python privacy_eval/split_trajectories.py --input /private/path/traj_1.csv --output-dir /private/path/privacy_eval/operator1 --grouping user-day --train-ratio 0.8 --seed 42
    python privacy_eval/audit_split.py --train /private/path/privacy_eval/operator1/train.csv --holdout /private/path/privacy_eval/operator1/holdout.csv --grouping user-day

Use grouping=user in a separate directory for a user-disjoint split. membership_map.csv contains identifiers and must remain on the private server.

## Processing boundary

These scripts prepare and audit the split; they do not run training. Use an isolated working copy of CoST-Diff and preprocess only train.csv. Do not reuse handled_data caches or artifacts created from the full dataset. Rebuild base-station vocabularies, the transition graph, embeddings, frequency tables, temporal statistics, and length distributions using training data only. Reserve holdout.csv for attack and exposure checks.

## Status and limitation

No raw data is present in the repository. The local terminal process could not start in this session, so no split, preprocessing, training, or runtime verification was performed. Before using the results, run the scripts on the server and verify the audit report.