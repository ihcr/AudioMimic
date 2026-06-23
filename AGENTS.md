# Repository Guidelines

This branch should be usable when cloned directly as a normal repo. Do not
depend on the old shared EDGE checkout for code or environment.

## Project Layout
- Main entry points: `train.py`, `test.py`, `EDGE.py`, and `args.py`.
- Model code: `model/`
- Dataset loading and transforms: `dataset/`
- Data prep and audio features: `data/` and `data/audio_extraction/`
- Evaluation: `eval/`
- Tests: `tests/`
- Runtime outputs: `slurm/`, `renders/`, `runs/`, `wandb/`, `cached_features/`, `setup_logs/`, and `data/`

Large checkpoints, datasets, cached features, renders, and Slurm outputs are runtime artifacts, not source files.

## Environment And Compute
- Use the repo-local environment: `source .venv311/bin/activate`
- Do not move EDGE onto the shared `yukun` Conda env unless explicitly requested.
- On the direct-attached 4090 server, there is no Slurm. Use the local bootstrap
  and long-running `tmux` sessions instead of `srun`/`sbatch`.
- Long-running `tmux` training sessions must show live output when attached.
  Prefer `PYTHONUNBUFFERED=1 ... 2>&1 | tee -a setup_logs/<experiment>.log`
  inside the tmux pane instead of redirecting stdout/stderr away from the pane.

## Common Commands
- `python data/create_dataset.py --extract-baseline --extract-jukebox`
- `scripts/bootstrap_finedance_g1_4090.sh --run-validation`
- `accelerate launch train.py --feature_type jukebox ...`
- `python test.py --music_dir custom_music --checkpoint checkpoint.pt --no_render`
- `python -m unittest discover -s tests`

Use package-style entry points for evaluation code when available. Running files directly can break imports.

## Experiment Specs
- For nontrivial research, ablations, training/eval runs, or paper-to-method trials, maintain `docs/experiments/INDEX.md` plus one `docs/experiments/EXP-YYYYMMDD-short-slug.md` spec.
- Before launching, resuming, evaluating, or comparing runs, read the experiment index and the active spec; update status, commands, Slurm job IDs/logs, run directories, checkpoints, metric/render paths, conclusions, and next action as work changes.
- Include the experiment ID in run, Slurm, render, metric, and checkpoint paths when practical.
- Keep durable guidance here general. Put transient run IDs, failed attempts, and one-off conclusions in experiment specs.
- Keep W&B enabled for training unless the user explicitly asks for offline/disabled mode. Future runs should log epoch-level losses, progress, throughput, ETA, and checkpoint markers so training curves are visible while jobs run.
- For long training runs, run full evaluation at every 500-epoch checkpoint by default. Sample renders do not count as eval. If training is still using the only GPU, record the eval as pending in the experiment spec and run it at the next GPU-safe window.

## Research Knowledge Base
- Use `docs/research/README.md` as the entrypoint for durable method research and long-term architecture blueprints.
- Use `docs/research/modules/` for module-level architecture specs such as latent priors, music-to-latent generation, feasibility critics, streaming roadmaps, RL/control layers, and their ablation matrices.
- Use `docs/papers/INDEX.md` as the project paper database. Add papers there with category, link, local asset status, relevance, reusable idea, and limitation.
- Keep paper-library conclusions durable and general. Put run-specific evidence, checkpoint comparisons, and one-off conclusions in experiment specs.

## Branch And Worktree Boundaries
- Keep `main` close to original EDGE plus local environment fixes.
- Beat-conditioned work belongs on the `diffusion` branch.
- New encoder experiments should use their own clean worktree from the original EDGE base unless the user asks to start from another branch.
- Do not copy beat-specific model, dataset, or evaluation changes into main unless explicitly requested.

## Data And Cache Rules
- AIST++ raw data and sliced data are reusable across worktrees, but generated feature folders should stay out of Git.
- If motion preprocessing, normalization, coordinate conventions, feature layout, or cache semantics change, delete or rebuild the affected processed and tensor caches before training again.
- Do not assume a checkpoint is clean just because raw motion slices look correct. Bad caches can silently poison training.
- When a bug is traced to stale data, say clearly whether the wrong layer is raw data, processed cache, tensor cache, checkpoint, saved motion, renderer, or evaluator.

## Coding And Testing
- Use 4-space indentation, `snake_case` for functions, variables, flags, and filenames, and `PascalCase` for classes.
- Match the existing argparse and path-handling style.
- Add or update focused `unittest` coverage for behavior changes, especially preprocessing, checkpoint loading, rendering, and evaluation.
- Validate with the narrowest real command or test that proves the change.

## Agent Work Ethic
- Prefer explicit errors, warnings, cache invalidation, and rebuild instructions over hidden fallback behavior.
- Do not add error handling, fallbacks, or validation for scenarios that cannot happen. Trust internal code and framework guarantees.
- Do not patch over upstream data or cache corruption with downstream render-only or eval-only fixes unless the user explicitly asks for a temporary workaround.

## Git And Review
- Keep commits small and descriptive.
- Do not commit checkpoints, datasets, cached features, renders, Slurm outputs, local virtualenvs, or scratch audio.
- When reporting model quality changes, include the exact command used and the metric files or run directory that produced the result.
