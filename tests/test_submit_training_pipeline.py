import importlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock


def reload_module():
    sys.modules.pop("submit_training_pipeline", None)
    return importlib.import_module("submit_training_pipeline")


class CommandBuilderTests(unittest.TestCase):
    def test_edge_baseline_preset_uses_official_style_recipe(self):
        submit_module = reload_module()

        args = submit_module.parse_args(
            [
                "--preset",
                "edge_baseline",
                "--train_name",
                "demo",
            ]
        )
        submit_module.apply_dynamic_defaults(args)

        self.assertEqual(args.learning_rate, 2e-4)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.gradient_accumulation_steps, 4)
        self.assertEqual(args.lambda_acc, 0.0)

    def test_preset_defaults_configure_distance_lbeat_experiment(self):
        submit_module = reload_module()

        args = submit_module.parse_args(
            [
                "--preset",
                "edge_beatdistance_lbeat",
                "--train_name",
                "demo",
            ]
        )
        submit_module.apply_dynamic_defaults(args)

        self.assertEqual(args.feature_type, "jukebox")
        self.assertTrue(args.use_beats)
        self.assertEqual(args.beat_rep, "distance")
        self.assertEqual(args.lambda_beat, 0.5)
        self.assertEqual(args.lambda_acc, 0.0)
        self.assertEqual(args.learning_rate, 2e-4)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.gradient_accumulation_steps, 4)

    def test_explicit_cli_overrides_preset_defaults(self):
        submit_module = reload_module()

        args = submit_module.parse_args(
            [
                "--preset",
                "edge_beatdistance_lbeat",
                "--train_name",
                "demo",
                "--batch_size",
                "16",
                "--learning_rate",
                "0.0003",
            ]
        )
        submit_module.apply_dynamic_defaults(args)

        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.learning_rate, 3e-4)

    def test_parse_args_respects_real_cli_overrides_when_argv_is_implicit(self):
        submit_module = reload_module()

        with mock.patch.object(
            sys,
            "argv",
            [
                "submit_training_pipeline.py",
                "--preset",
                "edge_baseline",
                "--train_name",
                "demo",
                "--epochs",
                "600",
            ],
        ):
            args = submit_module.parse_args()
        submit_module.apply_dynamic_defaults(args)

        self.assertEqual(args.epochs, 600)

    def test_apply_dynamic_defaults_uses_official_style_defaults_for_beat_runs(self):
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
        submit_module.apply_dynamic_defaults(args)

        self.assertEqual(args.learning_rate, 2e-4)
        self.assertEqual(args.lambda_acc, 0.0)
        self.assertEqual(args.gradient_accumulation_steps, 1)

    def test_apply_dynamic_defaults_uses_estimator_stage_defaults(self):
        submit_module = reload_module()

        args = submit_module.parse_args(
            [
                "--train_name",
                "demo",
                "--batch_size",
                "8",
            ]
        )
        submit_module.apply_dynamic_defaults(args)

        self.assertEqual(args.estimator_epochs, 50)
        self.assertEqual(args.estimator_batch_size, 8)
        self.assertEqual(args.estimator_learning_rate, 1e-4)
        self.assertEqual(args.estimator_weight_decay, 0.0)
        self.assertEqual(args.estimator_val_split, 0.1)

    def test_apply_dynamic_defaults_resolves_worker_defaults_from_stage_cpus(self):
        submit_module = reload_module()

        args = submit_module.parse_args(
            [
                "--train_name",
                "demo",
                "--train_cpus",
                "8",
                "--estimator_cpus",
                "8",
            ]
        )
        submit_module.apply_dynamic_defaults(args)

        self.assertEqual(args.train_num_workers, 2)
        self.assertEqual(args.test_num_workers, 2)
        self.assertEqual(args.estimator_num_workers, 6)
        self.assertEqual(args.train_mixed_precision, "bf16")
        self.assertEqual(args.estimator_mixed_precision, "bf16")

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
        submit_module.apply_dynamic_defaults(args)

        command = submit_module.build_train_command(
            args, beat_estimator_ckpt=Path("slurm/pipelines/run/weights/beat_estimator.pt")
        )

        self.assertIn("python train.py", command)
        self.assertIn("--feature_type baseline", command)
        self.assertIn("--use_beats", command)
        self.assertIn("--lambda_acc 0.0", command)
        self.assertIn("--beat_estimator_ckpt slurm/pipelines/run/weights/beat_estimator.pt", command)
        self.assertIn("--gradient_accumulation_steps 1", command)
        self.assertIn("--mixed_precision bf16", command)

    def test_estimator_command_threads_explicit_stage_args(self):
        submit_module = reload_module()
        args = submit_module.parse_args(
            [
                "--train_name",
                "demo",
                "--batch_size",
                "64",
                "--estimator_epochs",
                "12",
                "--estimator_batch_size",
                "24",
                "--estimator_learning_rate",
                "0.0002",
                "--estimator_weight_decay",
                "0.03",
                "--estimator_val_split",
                "0.25",
                "--estimator_num_workers",
                "5",
                "--estimator_mixed_precision",
                "no",
                "--estimator_gpus",
                "0",
            ]
        )
        submit_module.apply_dynamic_defaults(args)

        command = submit_module.build_estimator_command(
            args, Path("slurm/pipelines/run/weights/beat_estimator.pt")
        )

        self.assertIn("python train_beat_estimator.py", command)
        self.assertIn("--epochs 12", command)
        self.assertIn("--batch_size 24", command)
        self.assertIn("--learning_rate 0.0002", command)
        self.assertIn("--weight_decay 0.03", command)
        self.assertIn("--val_split 0.25", command)
        self.assertIn("--num_workers 5", command)
        self.assertIn("--mixed_precision no", command)
        self.assertIn("--device cpu", command)

    def test_build_eval_command_targets_final_checkpoint_and_dataset_test_split(self):
        submit_module = reload_module()
        args = submit_module.parse_args(
            [
                "--preset",
                "edge_beatdistance_lbeat",
                "--train_name",
                "demo",
            ]
        )
        submit_module.apply_dynamic_defaults(args)

        command = submit_module.build_eval_command(
            args,
            checkpoint_path=Path("runs/train/demo/weights/train-2000.pt"),
            motion_dir=Path("slurm/pipelines/run/eval/motions"),
            metrics_path=Path("slurm/pipelines/run/eval/metrics.json"),
            render_dir=Path("slurm/pipelines/run/eval/renders"),
        )

        self.assertIn("python eval/run_dataset_eval.py", command)
        self.assertIn("--checkpoint runs/train/demo/weights/train-2000.pt", command)
        self.assertIn("--data_path data", command)
        self.assertIn("--processed_data_dir data/dataset_backups", command)
        self.assertIn("--use_beats", command)
        self.assertIn("--beat_rep distance", command)
        self.assertIn("--seed 1234", command)
        self.assertIn("--edge_table_path slurm/pipelines/run/eval/edge_table.json", command)
        self.assertIn("--beatit_table_path slurm/pipelines/run/eval/beatit_table.json", command)
        self.assertIn("--paper_report_path slurm/pipelines/run/eval/paper_report.md", command)
        self.assertIn("--pfc_audit_path slurm/pipelines/run/eval/pfc_audit.json", command)

    def test_build_validation_command_targets_active_feature_and_cache_settings(self):
        submit_module = reload_module()
        args = submit_module.parse_args(
            [
                "--preset",
                "edge_beatdistance",
                "--train_name",
                "demo",
                "--data_path",
                "custom_data",
                "--processed_data_dir",
                "custom_cache",
            ]
        )
        submit_module.apply_dynamic_defaults(args)

        command = submit_module.build_validation_command(args)

        self.assertIn("python data/validate_preprocessed_data.py", command)
        self.assertIn("--data_path custom_data", command)
        self.assertIn("--processed_data_dir custom_cache", command)
        self.assertIn("--feature_type jukebox", command)
        self.assertIn("--use_beats", command)
        self.assertIn("--beat_rep distance", command)

    def test_submit_sbatch_reports_slurm_stderr_on_failure(self):
        submit_module = reload_module()
        error = submit_module.subprocess.CalledProcessError(
            returncode=1,
            cmd=["sbatch", "train.sbatch"],
            output="",
            stderr="sbatch: error: Operation not permitted",
        )

        with mock.patch.object(submit_module.subprocess, "run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Operation not permitted"):
                submit_module.submit_sbatch(["sbatch", "train.sbatch"])


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
            validate_script = (run_dir / "validate_preprocessed_data.sbatch").read_text()
            estimator_script = (run_dir / "beat_estimator.sbatch").read_text()
            train_script = (run_dir / "train.sbatch").read_text()
            eval_script = (run_dir / "evaluate.sbatch").read_text()
            submit_log = (run_dir / "submit.log").read_text()

        self.assertEqual(result.run_dir, run_dir)
        self.assertEqual(result.job_ids["preprocess"], "101")
        self.assertEqual(result.job_ids["validate_preprocessed_data"], "102")
        self.assertEqual(result.job_ids["beat_estimator"], "103")
        self.assertEqual(result.job_ids["train"], "104")
        self.assertEqual(result.job_ids["evaluate"], "105")

        self.assertEqual(calls[0][0], "sbatch")
        self.assertIn("--dependency=afterok:101", calls[1])
        self.assertIn("--dependency=afterok:102", calls[2])
        self.assertIn("--dependency=afterok:103", calls[3])
        self.assertIn("--dependency=afterok:104", calls[4])

        self.assertIn(f"source {repo_root / '.venv311' / 'bin' / 'activate'}", preprocess_script)
        self.assertIn("export PYTHONUNBUFFERED=1", preprocess_script)
        self.assertIn("export TERM=xterm-256color", preprocess_script)
        self.assertIn("stdbuf -oL -eL python data/create_dataset.py --extract-baseline --extract-beats", preprocess_script)

        self.assertIn(
            "stdbuf -oL -eL python data/validate_preprocessed_data.py --data_path data --processed_data_dir data/dataset_backups --feature_type baseline --sample_count 64 --use_beats --beat_rep distance",
            validate_script,
        )
        self.assertIn("stdbuf -oL -eL python train_beat_estimator.py", estimator_script)
        self.assertIn("stdbuf -oL -eL python train.py --feature_type baseline --use_beats", train_script)
        self.assertIn("--mixed_precision bf16", estimator_script)
        self.assertIn("--mixed_precision bf16", train_script)
        self.assertIn("--gradient_accumulation_steps 1", train_script)
        self.assertIn("--num_workers 6", estimator_script)
        self.assertIn("--train_num_workers 2", train_script)
        self.assertIn("--test_num_workers 2", train_script)
        self.assertIn("stdbuf -oL -eL python eval/run_dataset_eval.py", eval_script)
        self.assertIn("--edge_table_path", eval_script)
        self.assertIn("--beatit_table_path", eval_script)
        self.assertIn("--paper_report_path", eval_script)
        self.assertIn("--pfc_audit_path", eval_script)
        self.assertIn("Monitor with:", submit_log)
        self.assertIn("squeue --jobs 101,102,103,104,105", submit_log)
        self.assertIn("tail -f", submit_log)
        self.assertNotIn("--gres=gpu", preprocess_script)

    def test_submit_pipeline_runs_validation_even_when_preprocess_is_skipped(self):
        submit_module = reload_module()
        calls = []

        def fake_sbatch(cmd):
            calls.append(cmd)
            return str(400 + len(calls))

        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            args = submit_module.parse_args(
                [
                    "--preset",
                    "edge_baseline",
                    "--train_name",
                    "demo",
                    "--run_id",
                    "20260423-skip-preprocess",
                    "--skip_preprocess",
                    "--skip_eval",
                ]
            )

            result = submit_module.submit_pipeline(
                args,
                repo_root=repo_root,
                sbatch_submitter=fake_sbatch,
            )

            validate_script = (
                repo_root
                / "slurm"
                / "pipelines"
                / "20260423-skip-preprocess"
                / "validate_preprocessed_data.sbatch"
            ).read_text()

        self.assertNotIn("preprocess", result.job_ids)
        self.assertEqual(result.job_ids["validate_preprocessed_data"], "401")
        self.assertEqual(result.job_ids["train"], "402")
        self.assertIn("python data/validate_preprocessed_data.py", validate_script)
        self.assertIn("--dependency=afterok:401", calls[1])

    def test_submit_pipeline_can_start_after_external_dependency(self):
        submit_module = reload_module()
        calls = []

        def fake_sbatch(cmd):
            calls.append(cmd)
            return str(500 + len(calls))

        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            args = submit_module.parse_args(
                [
                    "--preset",
                    "edge_baseline",
                    "--train_name",
                    "demo",
                    "--run_id",
                    "20260423-external-dependency",
                    "--skip_preprocess",
                    "--skip_eval",
                    "--initial_dependency",
                    "999",
                ]
            )

            result = submit_module.submit_pipeline(
                args,
                repo_root=repo_root,
                sbatch_submitter=fake_sbatch,
            )

        self.assertEqual(result.job_ids["validate_preprocessed_data"], "501")
        self.assertEqual(result.job_ids["train"], "502")
        self.assertIn("--dependency=afterok:999", calls[0])
        self.assertIn("--dependency=afterok:501", calls[1])

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


class WrapperScriptTests(unittest.TestCase):
    def test_experiment_wrappers_target_expected_presets(self):
        repo_root = Path(__file__).resolve().parents[1]
        expected = {
            "train_edge_baseline.sh": "edge_baseline",
            "train_edge_beatpulse.sh": "edge_beatpulse",
            "train_edge_beatdistance.sh": "edge_beatdistance",
            "train_edge_beatdistance_lbeat.sh": "edge_beatdistance_lbeat",
        }

        for script_name, preset_name in expected.items():
            script_path = repo_root / "scripts" / script_name
            self.assertTrue(script_path.is_file(), script_name)
            self.assertIn(
                f"--preset {preset_name}",
                script_path.read_text(),
                script_name,
            )

    def test_beat_estimator_wrapper_submits_sbatch(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "train_beat_estimator.sh"

        self.assertTrue(script_path.is_file())
        content = script_path.read_text()
        self.assertIn("sbatch", content)
        self.assertIn("python train_beat_estimator.py", content)


if __name__ == "__main__":
    unittest.main()
