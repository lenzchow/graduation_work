# Trajectory generation projects

Source code is organized under separate experiment roots:

- SignalingTraj_exp/SignalingTraj: SignalingTraj model.
- CoST-Diff_exp/CoST-Diff: CoST-Diff model.
- Each experiment root has a baselines/ directory for baseline implementations.

The privacy_eval/ directory contains a deterministic split tool and an independent audit. By default it keeps each c_number + c_day trajectory intact. Use grouping=user for a whole-user holdout. The recommended starting point is 80% train, 20% holdout, seed 42.

Run separately for each operator file when the raw files are separated by operator:
python privacy_eval/split_trajectories.py --input /private/path/traj_1.csv --output-dir /private/path/privacy_eval/operator1 --grouping user-day --train-ratio 0.8 --seed 42
python privacy_eval/audit_split.py --train /private/path/privacy_eval/operator1/train.csv --holdout /private/path/privacy_eval/operator1/holdout.csv --grouping user-day

The scripts do not upload or bundle data. Split CSVs and membership maps contain private records and are ignored by Git. Model preprocessing must be run from an isolated workspace using only train.csv; rebuild every trajectory-derived artifact from that partition. Reserve holdout.csv for attack and exposure evaluation.
