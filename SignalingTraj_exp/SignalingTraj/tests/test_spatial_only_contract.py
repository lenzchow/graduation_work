import ast
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "privacy_eval"))
from evaluate_release import read_generated_trajectories

ROOT = Path(__file__).resolve().parents[1]


def source(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


class SpatialOnlyContractTest(unittest.TestCase):
    def test_diffusion_model_uses_station_sequences_only(self):
        tree = source("models/model.py")
        model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Diff_RNTraj")
        methods = {node.name: node for node in model_class.body if isinstance(node, ast.FunctionDef)}
        forward_args = [arg.arg for arg in methods["forward"].args.args]
        generate_args = [arg.arg for arg in methods["generate_data"].args.args]
        self.assertEqual(forward_args, ["self", "spatial_A_trans", "SE", "src_index_seqs"])
        self.assertEqual(generate_args, ["self", "spatial_A_trans", "SE", "batchsize", "length", "pre_dim"])

    def test_denoiser_keeps_diffusion_step_embedding_but_no_calendar_time_channels(self):
        text = (ROOT / "models/diff_module.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn("DiffusionEmbedding", classes)
        self.assertNotIn("embedding_hour_dim", text)

    def test_release_reader_accepts_spatial_only_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "50.txt").write_text(
                "#, trajectory num :0\narea, 30.1, 104.1, 0\narea, 30.2, 104.2, 1\n",
                encoding="utf-8",
            )
            trajectories, _ = read_generated_trajectories(
                directory, {"0": "station-a", "1": "station-b"}, expected_columns=4
            )
        self.assertEqual(trajectories, [("station-a", "station-b")])

    def test_training_and_generation_do_not_load_or_predict_timestamps(self):
        files = ["multi_main.py", "models/multi_train.py", "generate_data.py"]
        combined = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in files)
        for token in ("timestamp_seqs", "time_hour_loss", "embedding_hour_dim", "time_embedding_hour_matrix"):
            self.assertNotIn(token, combined)
        self.assertIn("index_seqs.bin", combined)


if __name__ == "__main__":
    unittest.main()
