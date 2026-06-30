# Isambard SSH Migration And Slurm Setup

This document is the migration runbook for moving this branch to Isambard. The
branch must be usable as a normal repo clone; do not depend on the old shared
EDGE checkout or the old `yukun` Conda environment for code.

## Current Migration Snapshot

- Source branch: `codex/wav2clip-stage-20260526`.
- Target compute style: Isambard SSH login plus Slurm jobs.
- Preferred Python entrypoint inside the repo: `.venv311/bin/python`.
- Current Isambard GPU nodes report NVIDIA driver CUDA `12.7`; install PyTorch
  from the CUDA `12.6` wheel index rather than the default PyPI CUDA `13.0`
  wheels.
- Slurm runtime outputs: `slurm/`, `setup_logs/`, `runs/`, `wandb/`, `renders/`,
  `eval/`, and `data/` are runtime artifacts, not source files.
- MuJoCo/G1 render or FK eval paths should export:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

V6b Stage A (`EXP-20260623-finedance-g1-v6b-motion-prior`) is implemented and
validated, but its first local 4090 r01 training run was stopped on 2026-06-23
before checkpoint 100. There is no checkpoint to resume from. After migration to
Isambard, relaunch r01 from scratch unless a later experiment spec says
otherwise.

Before launching, resuming, evaluating, or comparing any run, read:

```text
AGENTS.md
docs/experiments/INDEX.md
docs/experiments/EXP-20260623-finedance-g1-v6b-motion-prior.md
```

## Source Checkout Or Workspace Transfer

Clone the repo onto Isambard storage, or rsync the whole working tree from the
old machine. Pick the real project path for the account; the paths below are
examples.

```bash
export M2D_ROOT=/lus/lfs1aip2/projects/u6ed/$USER/Musics2Dance
git clone --branch codex/wav2clip-stage-20260526 git@github.com:lbtwyk/Musics2Dance.git "$M2D_ROOT"
cd "$M2D_ROOT"
```

If transferring the existing workspace directly, it is fine to include large
runtime folders such as `data/`, `runs/`, `eval/`, `renders/`, `wandb/`, and
`setup_logs/`. They can be useful for continuity and comparison. The important
boundary is that these remain runtime artifacts and should not be committed to
Git.

```bash
rsync -a --info=progress2 \
  --exclude .venv311 \
  /home/tianhup/Desktop/Musics2Dance/ \
  user@isambard:/lus/lfs1aip2/projects/u6ed/$USER/Musics2Dance/
```

Do not rsync `.venv311`. Rebuild the Python environment on Isambard so binary
wheels match the cluster image.

## Environment Setup

Use the repo-local `.venv311` environment. Do not move this branch onto the old
shared `yukun` Conda env unless the user explicitly asks.

If `.venv311` does not exist on Isambard, create it with the available Python
3.11 module or executable, then install dependencies. The exact module names can
vary by Isambard image, so record the working module commands in `setup_logs/`.

```bash
cd "$M2D_ROOT"
python3.11 -m venv .venv311
.venv311/bin/python -m pip install --upgrade pip 'setuptools<81' wheel
.venv311/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
.venv311/bin/python -m pip install --no-build-isolation --use-deprecated=legacy-resolver -r requirements-new-server.txt
.venv311/bin/python -m pip install -r requirements-g1-fk.txt
```

If the cluster image changes, re-check the GPU-node driver with a short Slurm
job before changing the PyTorch wheel index. Do not infer the correct wheel from
the login node alone.

The pinned `jukebox`/`jukemirlib` dependencies are old enough that isolated
builds can fail with `ModuleNotFoundError: No module named 'pkg_resources'`.
`scripts/setup_new_server.sh` therefore installs `requirements-new-server.txt`
with `--no-build-isolation --use-deprecated=legacy-resolver` by default. Override
`REQUIREMENTS_PIP_ARGS` only if the dependency set has been modernized. Keep
`setuptools<81`: newer setuptools releases remove the legacy `pkg_resources`
API these pinned packages still import, and current aarch64 PyTorch wheels also
require `setuptools<82`.

Validate the interpreter:

```bash
.venv311/bin/python --version
.venv311/bin/python -m pip list | head
```

Validate CUDA from a short Slurm GPU job, not from the login node. Expected
current output should include:

```text
torch 2.12.1+cu126
cuda_available True
cuda_version 12.6
device0 NVIDIA GH200 120GB
```

For this branch, the repeatable setup command on Isambard after runtime data has
already been transferred is:

```bash
PYTHONUNBUFFERED=1 scripts/setup_new_server.sh --skip-data --torch-index-url https://download.pytorch.org/whl/cu126 2>&1 | tee setup_logs/isambard_env_setup_$(date -u +%Y%m%d).log
```

If the environment was first created without the wheel index and PyPI installed
CUDA `13.0` packages, torch may import on the login node but fail on GPU nodes
with an old-driver warning. Replace the wheels explicitly:

```bash
.venv311/bin/python -m pip install --force-reinstall torch==2.12.1+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Then remove any remaining non-`cu12` NVIDIA runtime packages shown by:

```bash
.venv311/bin/python -m pip list --format=freeze | grep -E '^(nvidia|cuda|torch|torchaudio)'
```

## Runtime Data Transfer

For this migration, transferring large runtime artifacts is acceptable. If the
full workspace transfer above is used, this section is just a checklist for the
paths that matter most. If the source was cloned from Git instead, rsync these
runtime folders from the old machine.

Minimum V6b Stage A runtime inputs:

```text
data/finedance_g1_fkbeats/
third_party/unitree_g1_description/g1_29dof_rev_1_0.xml
```

Recommended V6b cache to avoid rebuilding:

```text
data/finedance_g1_v6b_motion_prior_dataset_backups/
```

Minimum focused rsync commands from the old machine:

```bash
rsync -a --info=progress2 \
  data/finedance_g1_fkbeats/ \
  user@isambard:/lus/lfs1aip2/projects/u6ed/$USER/Musics2Dance/data/finedance_g1_fkbeats/

rsync -a --info=progress2 \
  data/finedance_g1_v6b_motion_prior_dataset_backups/ \
  user@isambard:/lus/lfs1aip2/projects/u6ed/$USER/Musics2Dance/data/finedance_g1_v6b_motion_prior_dataset_backups/
```

For broader historical comparison work, transferring `runs/`, `eval/`,
`renders/`, `wandb/`, and `setup_logs/` is useful. They are still ignored
runtime artifacts; use experiment specs to decide which checkpoints and metrics
are authoritative after the transfer.

Do not rsync `.venv311`. Rebuild it on Isambard so binary wheels match the
cluster image.

If the transferred tree was originally built with symlinks, check for absolute
links back to the old 4090 path:

```bash
find data/finedance_g1_fkbeats -type l -lname '/home/tianhup/*' | head
```

For the current migration, `data/finedance_g1_fkbeats/{train,test}/wavs_sliced`
and `baseline_feats` should be repo-local directory symlinks to
`data/finedance_aistpp/{train,test}/...`, not per-file symlinks to the old
machine. The fixed layout is:

```text
data/finedance_g1_fkbeats/train/wavs_sliced -> ../../finedance_aistpp/train/wavs_sliced
data/finedance_g1_fkbeats/test/wavs_sliced -> ../../finedance_aistpp/test/wavs_sliced
data/finedance_g1_fkbeats/train/baseline_feats -> ../../finedance_aistpp/train/baseline_feats
data/finedance_g1_fkbeats/test/baseline_feats -> ../../finedance_aistpp/test/baseline_feats
```

## Hugging Face Policy

Use this existing HF repo for compact bootstrap artifacts:

```text
wyksdsg/edge-g1-beatdistance
```

HF should hold only compact, reusable artifacts:

- `data/finedance/`: raw FineDance source data needed for local preprocessing.
- `data/finedance-g1-retargeted/`: retargeted G1 sequence motions.
- selected checkpoints under `runs/train/.../weights/*.pt`.
- `hf_manifest.json`.

Do not upload sliced feature folders, tensor caches, renders, W&B runs, or
experiment scratch logs to HF by default. They contain many small files and are
runtime artifacts.

On the old machine, upload or repair compact HF contents with:

```bash
cd /path/to/old/Musics2Dance
.venv311/bin/python scripts/upload_wav2clip_artifacts_to_hf.py \
  --repo-id wyksdsg/edge-g1-beatdistance \
  --repo-type model \
  --source-root /path/to/old/Musics2Dance \
  --diffusion-root /path/to/old/Musics2Dance \
  --prune-large-feature-paths
```

Do not store `HF_TOKEN` in committed files.

## Slurm Sanity Checks

Run cheap checks on a login node only if they do not touch the GPU or heavy data
paths:

```bash
cd "$M2D_ROOT"
.venv311/bin/python -m py_compile train_g1_motion_prior.py dataset/g1_motion_prior_dataset.py model/g1_motion_prior.py eval/run_g1_motion_prior_eval.py
```

Run unit tests through Slurm so imports, file permissions, and node-local
runtime behavior match training:

```bash
mkdir -p slurm/EXP-20260623-finedance-g1-v6b-motion-prior
cat > slurm/EXP-20260623-finedance-g1-v6b-motion-prior/unit_tests.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=m2d_v6b_tests
#SBATCH --output=slurm/EXP-20260623-finedance-g1-v6b-motion-prior/unit_tests_%j.out
#SBATCH --partition=workq
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "$M2D_ROOT"
export PYTHONUNBUFFERED=1
.venv311/bin/python -m unittest tests.test_g1_motion_prior
EOF
sbatch --export=ALL,M2D_ROOT="$M2D_ROOT" slurm/EXP-20260623-finedance-g1-v6b-motion-prior/unit_tests.sbatch
```

Use `squeue -u "$USER"` to watch jobs and `tail -f slurm/.../*.out` to inspect
logs.

## V6b Smoke Job

Submit a one-epoch smoke job before the full r01 launch:

```bash
mkdir -p slurm/EXP-20260623-finedance-g1-v6b-motion-prior
cat > slurm/EXP-20260623-finedance-g1-v6b-motion-prior/smoke.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=m2d_v6b_smoke
#SBATCH --output=slurm/EXP-20260623-finedance-g1-v6b-motion-prior/smoke_%j.out
#SBATCH --partition=workq
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
cd "$M2D_ROOT"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
.venv311/bin/python -m train_g1_motion_prior \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_v6b_motion_prior_dataset_backups \
  --exp_name EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128_smoke \
  --motion_format g1_yaw_delta \
  --prior_type ae \
  --latent_dim 128 \
  --epochs 1 \
  --cache_limit_per_split 64 \
  --data_len 64 \
  --eval_data_len 16 \
  --eval_max_clips 16 \
  --full_eval_interval 0 \
  --wandb_mode disabled
EOF
sbatch --export=ALL,M2D_ROOT="$M2D_ROOT" slurm/EXP-20260623-finedance-g1-v6b-motion-prior/smoke.sbatch
```

## Relaunch V6b R01 On Isambard

Use the repo launcher so the generated `.sbatch` file is preserved in the Slurm
run folder and the Slurm output plus `tee` log are both retained:

```bash
cd "$M2D_ROOT"
PARTITION=workq \
TIME_LIMIT=24:00:00 \
CPUS_PER_TASK=8 \
MEMORY=64G \
GPUS=1 \
WANDB_MODE=online \
scripts/slurm_train_g1_motion_prior.sh
```

If Isambard requires an account directive:

```bash
ACCOUNT=<account_name> scripts/slurm_train_g1_motion_prior.sh
```

The launcher writes:

```text
slurm/EXP-20260623-finedance-g1-v6b-motion-prior/r01_ae_s2_latent128/train_r01_ae_s2_latent128.sbatch
setup_logs/EXP-20260623-finedance-g1-v6b-motion-prior/train_r01_ae_s2_latent128.log
runs/train/EXP-20260623-finedance-g1-v6b-motion-prior_r01_ae_s2_latent128/
```

Checkpoint 100 is the first sanity checkpoint. Checkpoint 500 is the first full
acceptance gate and should run full reconstruction eval automatically.

## Local 4090 Note

The old local 4090 workflow used `tmux` and live `tee` logs because it had no
Slurm. That is useful only when reproducing the stopped local run evidence. For
Isambard migration and relaunch, prefer the Slurm commands above.
