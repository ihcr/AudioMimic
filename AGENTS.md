# Repository Guidelines

This branch should be usable when cloned directly as a normal repo. Do not
depend on the old shared EDGE checkout for code or environment.

## Project Layout
- Main entry points: `train.py`, `test.py`, `EDGE.py`, `args.py`, `train_beat_estimator.py`.
- Model code: `model/`
- Dataset loading and transforms: `dataset/`
- Data prep and audio features: `data/` and `data/audio_extraction/`
- Evaluation: `eval/`
- Slurm runs and logs: `slurm/`
- Tests: `tests/`
- Vendored code: `third_party/` and `SMPL-to-FBX/`

Large checkpoints, datasets, cached features, renders, and Slurm outputs are runtime artifacts, not source files.

## Environment And Compute
- Use the repo-local environment: `source .venv311/bin/activate`.
- In the old local worktree layout, `source ../../.venv311/bin/activate` was only
  a convenience; new docs and scripts should assume a direct branch clone.
- Do not move this repo onto the shared `yukun` Conda env unless explicitly requested.
- Run training, full preprocessing, and long evaluation on compute nodes with `srun` or `sbatch`, not on the login node.
- Also use `srun` for any command expected to run longer than about a minute, including large dataset scans, bulk pickle/NumPy/audio reads, full validation passes, and full test suites that import heavy ML libraries.
- Do not request GPUs for CPU-only analysis jobs. Metric-only evaluation, FK-only scoring, candidate selection across existing motion `.pkl` files, report generation, file copying, JSON/table aggregation, and cache/data validation should be submitted without `--gres=gpu:*` unless the command actually runs model training, diffusion sampling, neural feature extraction, or another CUDA workload.
- Before adding `--gres=gpu:*` to any custom Slurm job, identify the CUDA step in the command. If there is no CUDA step, submit a CPU job.
- Keep Slurm logs and generated run files inside repo-local `slurm/`.

## Common Commands
- `python data/create_dataset.py --extract-baseline --extract-beats`
- `python data/create_dataset.py --extract-baseline --extract-jukebox --extract-beats`
- `python train_beat_estimator.py --motion_dir data/train/motions_sliced --beat_dir data/train/beat_feats --output_path weights/beat_estimator.pt`
- `accelerate launch train.py ...`
- `python test.py --music_dir custom_music --checkpoint checkpoint.pt --no_render`
- `python -m eval.run_dataset_eval --checkpoint <ckpt> ...`
- `python -m eval.run_benchmark_eval --motion_path data/test/motions_sliced --motion_source dataset --wav_dir data/test/wavs_sliced ...`
- `python -m unittest discover -s tests`

Use `python -m eval...` for package-style eval entry points. Running those files directly can break imports.

## Experiment Specs
- For nontrivial research, ablations, training/eval runs, or paper-to-method trials, maintain `docs/experiments/INDEX.md` plus one `docs/experiments/EXP-YYYYMMDD-short-slug.md` spec when the experiment history is clear.
- Before launching, resuming, evaluating, or comparing runs, read the experiment index and the active spec; update status, commands, Slurm job IDs/logs, run directories, checkpoints, metric/render paths, conclusions, and next action as work changes.
- If a run is real but the experiment rationale or comparison set is unclear, record it under `Needs Triage Before Promotion` in the index instead of inventing missing history.

## Diffusion And Pipeline Notes
- Keep beat work aligned with the main repo implementation plan and this worktree's `docs/BEAT_ONLY_DEBUGGING_PLAYBOOK.md`.
- Raw AIST++/FineDance/G1 inputs may live outside the repo because they are too
  large for Git. Pass explicit paths or copy/symlink them under this clone; do
  not require another EDGE checkout for code or environment.
- Reuse a configured Jukebox cache when available. Do not require the old main
  checkout's `.cache/jukemirlib` path.
- Treat `data/dataset_backups/*tensor_dataset*` as versioned training artifacts, not ground truth. If motion preprocessing, normalization, coordinate conventions, feature layout, or cache semantics change, bump the tensor-dataset cache version or delete and rebuild those caches before training again.
- Do not assume a checkpoint is clean just because the raw motion slices look correct. Bad tensor caches can silently poison training while leaving the raw `motions_sliced` and smaller `processed_*` caches looking normal.
- When a bug is traced to stale caches, say clearly which cache layer is affected and whether old checkpoints must be retrained. Re-rendering alone does not fix a model trained from corrupted cached tensors.
- Preprocessing is expected to resume. Valid existing sliced motions, features, and cache files should be reused, while broken partial outputs should be replaced.
- Sliced motion/audio outputs, processed dataset caches, and tensor dataset caches should be written atomically so interrupted jobs do not leave half-written files that look valid later.
- Run the preprocessing validation step before training batches. The validator should check every sliced motion file, not just a small sample.
- FineDance preparation is documented in `docs/DATASET_PREPARATION.md`. The local download used 315-column FineDance motion arrays and the prepared mixed tree is `data/aist_finedance`.
- For FineDance or other supplementary datasets, validate the prepared clip tree, not just the raw motion folder. The required training surface is matching `motions_sliced`, `wavs_sliced`, feature files, beat files, and metadata.
- Full G1 AIST FK-beat preparation is documented in `docs/DATASET_PREPARATION.md`. The current prepared tree is `data/g1_aistpp_full_fkbeats`, with fresh caches under `data/g1_aistpp_full_fkbeats_dataset_backups`.
- For G1 FK-beat training, use the `g1_beatdistance_fkbeats` preset. It uses `motion_format=g1`, `use_beats=True`, `beat_rep=distance`, `lambda_beat=0`, `lambda_acc=0`, the fresh FK-beat tree, and FK-enabled G1 evaluation.
- For cautious G1 beat-loss fine-tuning, use `g1_beatdistance_fkbeats_lbeat`. It fine-tunes from `runs/train/g1_aist_beatdistance_fkbeats/weights/train-2000.pt`, trains a G1-native beat estimator on normalized 38-D robot motion, uses `lambda_beat=0.01`, delayed warmup, and `beat_loss_cap_mode=soft`. The stronger normalized override tested on 2026-05-07 was `lambda_beat=0.2`, `beat_loss_max_fraction=1.0`, `beat_loss_start_epoch=50`, and `beat_loss_warmup_epochs=300`; it improved beat metrics but worsened foot/contact metrics, so do not make it the default without additional guards.
- For G1 lbeat comparisons, check `G1BAS`, `G1RoboPerformBAS`, `G1FKBAS`, `G1FKRoboPerformBAS`, and `G1BeatF1` together with `G1FootSliding`, `G1GroundPenetration`, `RootVelocityMean`, `RootSmoothnessJerkMean`, and `G1Dist`. A beat-score gain alone is not enough to accept a robot checkpoint.
- `G1BAS` and `G1FKBAS` are paper-style music-to-motion BAS. `G1RoboPerformBAS` and `G1FKRoboPerformBAS` are RoboPerform-style motion-to-music BAS with `sigma^2=9`; direct G1 uses motor/joint velocity minima and FK G1 uses FK keypoint velocity minima.
- The G1 FK model is the local Unitree MJCF at `third_party/unitree_g1_description/g1_29dof_rev_1_0.xml`. Do not download robot models during evaluation or training.
- G1 rendering defaults to native MuJoCo mesh video with audio. Use `--g1_render_backend mujoco`, `--g1_root_quat_order xyzw`, and `MUJOCO_GL=egl` on the cluster; reserve `--g1_render_backend stick` for diagnostic stick-figure checks.
- For G1 whole-song qualitative renders, use `--full_music_dir` pointing at the
  current server's raw AIST audio folder. The dataset `test/wavs` files are
  choreography-trimmed, and `wavs_sliced` files are 5-second model windows, so
  either source can make videos much shorter than the raw song.
- The current safer training defaults lower dataloader pressure on large Jukebox-feature runs. If training suddenly fails with shared-memory or worker crashes, check worker count and prefetch settings before blaming the raw data.
- Prefer the chained Slurm launcher for long runs: `submit_training_pipeline.py` or the shell wrappers in `scripts/`.
- Treat `edge_beatdistance_lbeat` as a safe fine-tune preset, not a from-scratch default. It should start from a beat-distance checkpoint unless `--allow_lbeat_from_scratch` is explicitly passed.
- Safe `lbeat` uses weak, delayed, warmed, and capped beat loss. Check the saved training config for the effective beat-loss schedule before comparing runs.
- Make Slurm submission failures loud. If `sbatch` fails, surface the real error instead of continuing as if the launch worked.
- Long-running stages should show progress bars in logs: slicing, feature extraction, beat extraction, beat-estimator training, and main training.

## Strict Evaluation Rules
- Fresh comparisons must be run from checkpoints or raw dataset motions, not from older saved motion folders.
- Generated-motion evaluation now scores the saved joint motion directly from `full_pose`, matching the original EDGE style more closely.
- If a saved motion file contains conflicting motion representations, evaluation should fail instead of silently rebuilding or falling back.
- If cache contents, metadata, preprocessing versions, or motion representations do not match expectations, fail loudly with a rebuild instruction. Do not add silent fallback paths to “make it work.”
- Treat older saved outputs produced before the strict-eval cleanup as untrusted. Regenerate them from checkpoints if you need a fair comparison.
- Safe `lbeat` evaluation screens saved checkpoints first, then writes `eval/lbeat_selection.json` after the full eval of the selected checkpoint. A rejected model is a model-quality result, not an evaluator crash.

## Coding And Testing
- Use 4-space indentation, `snake_case` for functions, variables, flags, and filenames, and `PascalCase` for classes.
- Match the existing argparse and path-handling style.
- Add or update focused `unittest` coverage with behavior changes, especially for preprocessing, Slurm launchers, checkpoint loading, and evaluation.
- Validate with the narrowest real command or test that proves the change.

## Git And Review
- Keep commits small and descriptive.
- Do not commit checkpoints, datasets, cached features, renders, or Slurm outputs unless explicitly requested.
- When reporting model quality changes, include the exact command used and the metric files or run directory that produced the result.

## Current Worktree State
- On this cluster, `sacct` may fail from the login node with a Slurm socket permission error. When that happens, use `squeue`, pipeline log timestamps, and checkpoint/render timestamps as the source of truth.
- The 2026-05-10 RoboPerform BAS rescore is in `slurm/rescore_roboperform_bas_20260510/scores.json`; it completed 15 G1 run summaries with `0` rescore errors.
- The stable AIST G1 FKBeat no-lbeat anchor is `slurm/pipelines/20260504-g1-aist-beatdistance-fkbeats`: `G1BAS=0.3320`, `G1RoboPerformBAS=0.5894`, `G1FKBAS=0.3497`, `G1FKRoboPerformBAS=0.5502`, `G1BeatF1=0.3333`, `G1Dist=6.87`.
- The 2026-05-07 normalized G1 `lbeat` fine-tune `slurm/pipelines/20260507-g1-aist-fkbeats-lbeat-relative-finetune-lam020-cap1` is the best AIST lbeat direction so far but still not the clean default. It improves rhythm (`G1BAS=0.4484`, `G1RoboPerformBAS=0.6211`, `G1FKBAS=0.4673`, `G1FKRoboPerformBAS=0.5978`, `G1BeatF1=0.4372`) but worsens contact (`G1FootSliding=0.7102`, `G1GroundPenetration=0.0979`).
- The 2026-05-06 from-scratch normalized G1 `lbeat` run `slurm/pipelines/20260506-g1-aist-fkbeats-lbeat-relative-scratch-lam020-cap1c` remains rejected: high beat scores (`G1BAS=0.6580`, `G1FKBAS=0.8232`, `G1FKRoboPerformBAS=0.8792`, `G1BeatF1=0.8716`) came with `G1FootSliding=3.0399`, `RootSmoothnessJerkMean=3454.9`, and `G1Dist=56.26`.
- On retargeted FineDance G1, `slurm/pipelines/20260509-finedance-g1-fkbeatdistance-1000-r2` is cleaner than `slurm/pipelines/20260509-finedance-g1-relative-lbeat-scratch-lam010-1000-r2`: FKBeat has `G1RoboPerformBAS=0.5416`, `G1FKRoboPerformBAS=0.5142`, `G1BeatF1=0.2910`, `G1Dist=8.63`; relative lbeat has `G1RoboPerformBAS=0.5023`, `G1FKRoboPerformBAS=0.4940`, `G1BeatF1=0.2961`, `G1Dist=39.53`.
