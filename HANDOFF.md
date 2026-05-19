# Handoff

## Goal

Continue the FineDance+G1 non-Jukebox feature experiment from the `wav2clip-stft-beat` branch on the direct-attached 4090 server. The old Slurm account hit GPU/CPU-minute quota, but the new server should run local bash/Python commands rather than `sbatch`.

New-server checkout:

- Clone/fetch this branch directly as a normal repo root; it does not require the
  old EDGE `main` checkout for code or environment.
- Suggested path: `/path/to/EDGE-wav2clip`
- Branch: `wav2clip-stft-beat`
- Repo env: `source .venv311/bin/activate`
- Main spec: `docs/experiments/EXP-20260513-finedance-g1-wav2clip-stft-beat.md`

## Current Progress

Implemented and tested `feature_type=wav2clip_stft_beat` for FineDance+G1:

- raw feature dim: `512 Wav2CLIP + 193 STFT + 1 GaussianBeat = 706`
- fusion variants:
  - `concat_norm`
  - `stream_adapter`
- G1 eval/train wrappers now thread `--feature_fusion`
- `submit_training_pipeline.py` has FineDance+G1 presets for both variants
- `data/audio_extraction/wav2clip_stft_beat_features.py` supports sharded extraction, valid-file skip, and atomic save

Verified before quota block:

- `python -m py_compile ...` passed for touched modules.
- `python -m unittest tests.test_feature_config_and_fusion tests.test_wav2clip_stft_beat_features tests.test_validate_preprocessed_data tests.test_submit_training_pipeline tests.test_phase4_to_6_beat_integration` passed 102 tests.
- G1 synthetic loss smoke was finite for both fusion modes.

Slurm evidence:

- Feature extraction array `4576163_[0-7]`: completed all shards in about 42 minutes.
- Feature cache counts:
  - train `47817/47817`
  - test `3265/3265`
- `stream_adapter`:
  - validate `4576164`: completed
  - train `4576165`: completed 500 epochs in `03:15:16`
  - final checkpoint: `runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt`
  - eval `4576166`: cancelled before running
- `concat_norm`:
  - validate `4576167`: completed
  - train `4576168`: cancelled before running, `Reason=AssocGrpCPUMinutesLimit`
  - eval `4576169`: cancelled before running
- There are no active jobs left for this experiment in `squeue`.

## Runtime Artifacts Policy

Do not upload or migrate the feature folders/tensor caches through HF. The large
feature tree has tens of thousands of files and hit HF API limits. The new plan
is:

- GitHub: code, docs, experiment specs, curated evidence logs.
- HF `wyksdsg/edge-g1-beatdistance`: raw `data/finedance/`, compact
  `data/finedance-g1-retargeted/`, and selected checkpoints only.
- New 4090 server: rebuild `baseline_feats`, `jukebox_feats`, `beat_feats`, and
  `wav2clip_stft_beat_feats` locally with `scripts/bootstrap_finedance_g1_4090.sh`.

The old completed feature/cache sizes were:

- train Wav2CLIP/STFT/GaussianBeat features: `19G`
- test Wav2CLIP/STFT/GaussianBeat features: `1.3G`
- stream-adapter processed/tensor cache: `13G`
- `train-500.pt`: `1.1G`

Optional evidence logs:

```bash
rsync -a --info=progress2 \
  OLD:/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/EXP-20260513-finedance-g1-wav2clip-stft-beat/ \
  NEW:/path/to/EDGE-wav2clip/slurm/EXP-20260513-finedance-g1-wav2clip-stft-beat/

rsync -a --info=progress2 \
  OLD:/projects/u6ed/yukun/EDGE/.worktrees/wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/ \
  NEW:/path/to/EDGE-wav2clip/slurm/pipelines/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/
```

## What Worked

- Sharding feature extraction with a Slurm array was much faster than the first single-process job.
- `stream_adapter` trained cleanly for 500 epochs with finite losses and saved checkpoints every 50 epochs.
- The feature cache completed on the old server, but HF upload of feature folders is no longer the migration path. Rebuild features locally on the 4090.

## What Didn't Work

- The first single-process extraction job `4576095` was too slow and was cancelled after measuring roughly `2.5 clips/sec`.
- The account quota blocked more GPU jobs:
  - fresh 1h, 30m, and 15m GPU test submissions also failed with the same policy error
  - `interactive_qos` was invalid for this user
- No eval metrics exist yet because eval jobs were cancelled to save quota.

## Next Steps

1. Clone/fetch the pushed `wav2clip-stft-beat` branch on the new 4090 server.
2. Run the one-command local bootstrap in `docs/NEW_SERVER_SETUP.md`; it does
   not use Slurm:

```bash
export HF_TOKEN=...
scripts/bootstrap_finedance_g1_4090.sh --run-validation
```

3. If not using that script, recreate `.venv311`, download the compact HF
   artifacts, prepare FineDance/G1 locally, and extract features locally.
4. Validate rebuilt data:

```bash
cd /path/to/EDGE-wav2clip
source .venv311/bin/activate
python data/validate_preprocessed_data.py \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups \
  --feature_type wav2clip_stft_beat \
  --motion_format g1 \
  --feature_cache_mode memmap \
  --feature_cache_dtype float16 \
  --sample_count 64
```

5. Run `concat_norm` first, because it never started. On the 4090 server, run
   training directly or adapt the command in the generated pipeline scripts
   without submitting to Slurm:

```bash
cd /path/to/EDGE-wav2clip
source .venv311/bin/activate
python submit_training_pipeline.py \
  --preset g1_finedance_wav2clip_stft_beat_concat_norm \
  --train_name EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm \
  --run_id EXP-20260513-finedance-g1-wav2clip-stft-beat_r01_concat_norm \
  --skip_preprocess \
  --train_time 04:00:00 \
  --eval_time 02:00:00
```

6. Evaluate `stream_adapter` from the existing checkpoint directly:

```bash
python eval/run_g1_dataset_eval.py \
  --checkpoint runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt \
  --feature_type wav2clip_stft_beat \
  --feature_fusion stream_adapter \
  --data_path data/finedance_g1_fkbeats \
  --processed_data_dir data/finedance_g1_wav2clip_stft_beat_stream_adapter_dataset_backups \
  --render_dir setup_logs/finedance_g1_4090/r02_eval/renders \
  --motion_save_dir setup_logs/finedance_g1_4090/r02_eval/motions \
  --metrics_path setup_logs/finedance_g1_4090/r02_eval/metrics.json \
  --g1_table_path setup_logs/finedance_g1_4090/r02_eval/g1_table.json \
  --motion_audit_path setup_logs/finedance_g1_4090/r02_eval/motion_audit.json \
  --paper_report_path setup_logs/finedance_g1_4090/r02_eval/paper_report.md \
  --seed 1234 \
  --enable_fk_metrics \
  --g1_fk_model_path third_party/unitree_g1_description/g1_29dof_rev_1_0.xml \
  --g1_root_quat_order xyzw
```

7. Compare r01/r02 only after both evals finish; do not interpret the r02 training loss alone as model quality.
