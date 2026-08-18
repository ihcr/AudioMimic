# Pure Commit Forcing d16 inference pipeline

This directory is the complete model-weight package for the selected default
pipeline from `EXP-20260804-v6f-x-pure-commit-forcing-final`.

The model is motion-only and unconditional with respect to music. It performs
strict-causal G1 motion continuation using the following pipeline:

```text
K64 motion history
  -> q0 generator (512-way structural tokens)
  -> d16 residual diffusion generator (NFE sampler)
  -> d16 hybrid codec decode
  -> G1 yaw-delta motion and S66 streaming state
```

Compact notation:

- `K64`: 64-token causal history.
- `H8`: predict eight future tokens per plan.
- `C4`: commit four tokens before replanning.
- `S66`: 66-dimensional physical boundary state.
- `d16`: 16-dimensional continuous residual beside the discrete q0 token.

## Included weights

All three `.pt` files are tracked by Git LFS on `prior-dev`.

| File | Role | Size | SHA256 |
|---|---|---:|---|
| `pure-cf-d16-zero-seed1234-update200000-inference.pt` | Commit Forcing residual generator, trained q0 fork, residual normalization, configuration, and provenance | 929,933,103 | `add4da47b30dcda409bd3d69b4e2acd1e295a223616afd104442e19452e42e9f` |
| `q0-d16-seed1234-update100000.pt` | Frozen q0-100k generator required by the loader's cache/codec/q0 identity check | 1,039,232,553 | `413e319bbe49e69384682c34f18b072bbecd519e1787b6cce1c182fc1dad26a0` |
| `codec-d16-seed1234-train300.pt` | d16 hybrid codec, its 512-entry q0 codebook, motion normalizer, and streaming-state statistics | 155,323,831 | `45285b832ceea4c9977ffac21dca238c693ccbf22fa96cd83e3cf3e0960e799b` |

`pipeline_manifest.json` records the same machine-readable identities.
`SHA256SUMS` verifies the checked-out files.

There is no separate codebook file. The codec checkpoint contains the exact
512-entry q0 codebook used by both encoding and decoding; splitting it out
would create an unnecessary opportunity for codec/codebook mismatch.

The teacher checkpoint is not part of inference. It initialized training but is
never loaded by `eval.run_g1_paper_faithful_dc_eval`.

## Fetch and verify

Install Git LFS before judging file size; without a pull, a checkout contains
small pointer files rather than the model objects.

```bash
git lfs install
git lfs pull --include="models/releases/pure-cf-d16-zero-200k-v1/*.pt"
cd models/releases/pure-cf-d16-zero-200k-v1
sha256sum -c SHA256SUMS
cd ../../../..
```

Expected total model storage is 2,124,489,487 bytes, approximately 1.98 GiB.

## Environment and non-weight prerequisites

- Use the repository-local `.venv311`; follow `docs/NEW_SERVER_SETUP.md` on a
  fresh machine.
- The evaluator needs the prepared FineDance-G1 motion directory, normally
  `data/finedance_g1_fkbeats`. Dataset motion is not a model weight and is not
  included in Git.
- MuJoCo rendering uses the checked-in Unitree description under
  `third_party/unitree_g1_description/`.
- On Isambard, inference must run through Slurm. A login node is not a CUDA
  check. A direct-attached GPU may run the Python entrypoint in a live `tmux`.

The default `k64` mode uses a real 64-token source history to seed motion
continuation. Use `--seed_mode cold` only when deliberately testing the cold
start behavior.

## Direct-attached GPU example

Run from the repository root. Replace the data and output paths with validated
absolute paths.

```bash
REPO_ROOT="$(pwd -P)"
MODEL_DIR="$REPO_ROOT/models/releases/pure-cf-d16-zero-200k-v1"
DATA_PATH="/absolute/path/to/finedance_g1_fkbeats"
OUTPUT_DIR="/absolute/path/to/pure_cf_d16_demo"

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$REPO_ROOT/.venv311/bin/python" \
  -m eval.run_g1_paper_faithful_dc_eval \
  --experiment_id EXP-20260804-v6f-x-pure-commit-forcing-final \
  --generator_checkpoint "$MODEL_DIR/pure-cf-d16-zero-seed1234-update200000-inference.pt" \
  --q0_checkpoint "$MODEL_DIR/q0-d16-seed1234-update100000.pt" \
  --codec_checkpoint "$MODEL_DIR/codec-d16-seed1234-train300.pt" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --split test \
  --max_eval_sequences 1 \
  --duration_seconds 30 \
  --seed_mode k64 \
  --nfe 10 \
  --sampling_seed 1234 \
  --q0_policy sample \
  --temperature 1.0 \
  --render_count 2 \
  --device cuda
```

The generated motion is written under `OUTPUT_DIR/motions/generated/`.
Targets, metrics, manifests, diagnostics, and requested renders are written
beside it.

## Isambard Slurm example

The existing launcher creates the allocation and evaluation matrix and uses the
repository's standard prepared-data location
`data/finedance_g1_fkbeats`. Run this from the repository root. If the prepared
data lives elsewhere, use the direct Python command inside a custom Slurm job
and pass its validated absolute path through `--data_path`.

```bash
REPO_ROOT="$(pwd -P)"
MODEL_DIR="$REPO_ROOT/models/releases/pure-cf-d16-zero-200k-v1"

EXP_ID=EXP-20260804-v6f-x-pure-commit-forcing-final \
RUN_NAME=repo_pure_cf_d16_inference \
GENERATOR_CHECKPOINT="$MODEL_DIR/pure-cf-d16-zero-seed1234-update200000-inference.pt" \
Q0_CHECKPOINT="$MODEL_DIR/q0-d16-seed1234-update100000.pt" \
CODEC_CHECKPOINT_OVERRIDE="$MODEL_DIR/codec-d16-seed1234-train300.pt" \
SUITE=milestone \
MAX_EVAL_SEQUENCES=1 \
TIME_LIMIT=02:00:00 \
MEM_PER_GPU=48G \
scripts/slurm_eval_g1_paper_faithful_dc.sh
```

The launcher prints the Slurm job ID, log path, and follow commands. For a
single custom rollout rather than the milestone matrix, place the direct Python
command above inside an appropriately sized Slurm job.

## Identity and scope

- Method: pure Commit Forcing, zero oracle floor.
- Residual width: d16.
- Generator training seed: 1234.
- Generator update: 200000.
- Deployment contract: `K64_H8_C4_strict_causal`.
- Cache schema: `g1_paper_faithful_dc_v4_latent_width`.
- Source full-training checkpoint SHA256:
  `79500ceebdea2c3d54af2b504728eb32055f690f19a9ea964b7ada482e9ad9ca`.

The published generator preserves its generator and embedded q0 weights
bit-for-bit. Only optimizer, scaler, RNG, and sampler cursor state was removed,
so it supports inference/evaluation but not exact training continuation.

The matching [GitHub Release](https://github.com/lbtwyk/Musics2Dance/releases/tag/pure-cf-d16-zero-200k-v1)
contains the same three model files. The scientific conclusion is deliberately
narrow: d16 is the balanced/default Commit Forcing model, not an
across-the-board quality win over Two-Forward.
