import importlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def reload_module():
    sys.modules.pop("submit_training_pipeline", None)
    return importlib.import_module("submit_training_pipeline")


class CommandBuilderTests(unittest.TestCase):
    def test_resolve_shared_root_collapses_worktree_path(self):
        submit_module = reload_module()

        repo_root = submit_module.resolve_shared_root(
            Path("/tmp/EDGE/.worktrees/diffusion")
        )

        self.assertEqual(repo_root, Path("/tmp/EDGE"))

    def test_jukebox_preprocess_defaults_to_one_gpu(self):
        submit_module = reload_module()

        args = submit_module.parse_args(
            [
                "--feature_type",
                "jukebox",
                "--train_name",
                "demo",
            ]
        )
        submit_module.apply_dynamic_defaults(args)

        self.assertEqual(args.preprocess_gpus, 1)

    def test_preprocess_command_includes_requested_feature_extractors(self):
        submit_module = reload_module()
        args = submit_module.parse_args(
            [
                "--feature_type",
                "jukebox",
                "--use_beats",
                "--train_name",
                "demo",
            ]
        )

        command = submit_module.build_preprocess_command(args)

        self.assertIn("python data/create_dataset.py", command)
        self.assertIn("--extract-jukebox", command)
        self.assertIn("--extract-beats", command)
        self.assertNotIn("--extract-baseline", command)

    def test_train_command_uses_generated_estimator_checkpoint_when_needed(self):
        submit_module = reload_module()
        args = submit_module.parse_args(
            [
                "--feature_type",
                "baseline",
                "--use_beats",
                "--train_name",
                "demo",
            ]
        )

        command = submit_module.build_train_command(
            args, beat_estimator_ckpt=Path("slurm/pipelines/run/weights/beat_estimator.pt")
        )

        self.assertIn("python train.py", command)
        self.assertIn("--feature_type baseline", command)
        self.assertIn("--use_beats", command)
        self.assertIn("--beat_estimator_ckpt slurm/pipelines/run/weights/beat_estimator.pt", command)


class SubmissionFlowTests(unittest.TestCase):
    def test_submit_pipeline_chains_jobs_and_writes_progress_friendly_scripts(self):
        submit_module = reload_module()
        calls = []

        def fake_sbatch(cmd):
            calls.append(cmd)
            job_id = 100 + len(calls)
            return str(job_id)

        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            args = submit_module.parse_args(
                [
                    "--feature_type",
                    "baseline",
                    "--use_beats",
                    "--train_name",
                    "demo",
                    "--run_id",
                    "20260416-demo",
                ]
            )

            result = submit_module.submit_pipeline(
                args,
                repo_root=repo_root,
                sbatch_submitter=fake_sbatch,
            )

            run_dir = repo_root / "slurm" / "pipelines" / "20260416-demo"
            preprocess_script = (run_dir / "preprocess.sbatch").read_text()
            estimator_script = (run_dir / "beat_estimator.sbatch").read_text()
            train_script = (run_dir / "train.sbatch").read_text()
            submit_log = (run_dir / "submit.log").read_text()

        self.assertEqual(result.run_dir, run_dir)
        self.assertEqual(result.job_ids["preprocess"], "101")
        self.assertEqual(result.job_ids["beat_estimator"], "102")
        self.assertEqual(result.job_ids["train"], "103")

        self.assertEqual(calls[0][0], "sbatch")
        self.assertIn("--dependency=afterok:101", calls[1])
        self.assertIn("--dependency=afterok:102", calls[2])

        self.assertIn(f"source {repo_root / '.venv311' / 'bin' / 'activate'}", preprocess_script)
        self.assertIn("export PYTHONUNBUFFERED=1", preprocess_script)
        self.assertIn("export TERM=xterm-256color", preprocess_script)
        self.assertIn("stdbuf -oL -eL python data/create_dataset.py --extract-baseline --extract-beats", preprocess_script)

        self.assertIn("stdbuf -oL -eL python train_beat_estimator.py", estimator_script)
        self.assertIn("stdbuf -oL -eL python train.py --feature_type baseline --use_beats", train_script)
        self.assertIn("Monitor with:", submit_log)
        self.assertIn("squeue --jobs 101,102,103", submit_log)
        self.assertNotIn("--gres=gpu", preprocess_script)

    def test_submit_pipeline_requests_gpu_for_jukebox_preprocess(self):
        submit_module = reload_module()
        calls = []

        def fake_sbatch(cmd):
            calls.append(cmd)
            return str(200 + len(calls))

        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            args = submit_module.parse_args(
                [
                    "--feature_type",
                    "jukebox",
                    "--use_beats",
                    "--train_name",
                    "demo",
                    "--run_id",
                    "20260416-jukebox",
                ]
            )

            submit_module.submit_pipeline(
                args,
                repo_root=repo_root,
                sbatch_submitter=fake_sbatch,
            )

            preprocess_script = (
                repo_root
                / "slurm"
                / "pipelines"
                / "20260416-jukebox"
                / "preprocess.sbatch"
            ).read_text()

        self.assertIn("#SBATCH --gres=gpu:1", preprocess_script)

    def test_submit_pipeline_uses_worktree_checkout_with_shared_venv(self):
        submit_module = reload_module()

        calls = []

        def fake_sbatch(cmd):
            calls.append(cmd)
            return str(300 + len(calls))

        with TemporaryDirectory() as tmpdir:
            shared_root = Path(tmpdir) / "EDGE"
            worktree_root = shared_root / ".worktrees" / "diffusion"
            (shared_root / ".venv311" / "bin").mkdir(parents=True)
            worktree_root.mkdir(parents=True)

            args = submit_module.parse_args(
                [
                    "--feature_type",
                    "jukebox",
                    "--use_beats",
                    "--train_name",
                    "demo",
                    "--run_id",
                    "20260418-worktree",
                ]
            )

            submit_module.submit_pipeline(
                args,
                repo_root=worktree_root,
                sbatch_submitter=fake_sbatch,
            )

            preprocess_script = (
                worktree_root
                / "slurm"
                / "pipelines"
                / "20260418-worktree"
                / "preprocess.sbatch"
            ).read_text()

        self.assertIn(f"cd {worktree_root}", preprocess_script)
        self.assertIn(
            f"source {shared_root / '.venv311' / 'bin' / 'activate'}",
            preprocess_script,
        )


if __name__ == "__main__":
    unittest.main()
