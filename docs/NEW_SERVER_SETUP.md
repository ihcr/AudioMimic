# New Server Setup

The new 4090 server does not use Slurm. Clone `wav2clip-stft-beat`, then run
one local bootstrap script. The script builds `.venv311`, downloads compact
artifacts from Hugging Face, and rebuilds FineDance+G1 prepared data/features on
the 4090 machine.

## Hugging Face Policy

Use this existing HF repo:

```text
wyksdsg/edge-g1-beatdistance
```

Keep HF compact. Do not upload sliced feature folders or tensor caches there;
they contain tens of thousands of small files and can hit API limits. HF should
hold only:

- `data/finedance/`: raw FineDance source data needed for local preprocessing.
- `data/finedance-g1-retargeted/`: retargeted G1 sequence motions.
- selected checkpoints under `runs/train/.../weights/*.pt`.
- `hf_manifest.json`.

Docs, experiment specs, and curated logs live in GitHub.

On the old server, upload/repair the compact HF contents with:

```bash
cd /projects/u6ed/yukun/EDGE/.worktrees/wav2clip
source /projects/u6ed/yukun/EDGE/.venv311/bin/activate
export HF_TOKEN=...
python scripts/upload_wav2clip_artifacts_to_hf.py \
  --repo-id wyksdsg/edge-g1-beatdistance \
  --repo-type model \
  --source-root /projects/u6ed/yukun/EDGE \
  --diffusion-root /projects/u6ed/yukun/EDGE/.worktrees/diffusion \
  --prune-large-feature-paths \
  --progress-seconds 30
```

Add `--private` only when creating a new private HF repo. If an earlier upload
partially pushed feature/caches to HF, keep `--prune-large-feature-paths`.

## One Command On The 4090 Server

From a fresh clone:

```bash
git clone --branch wav2clip-stft-beat git@github.com:lbtwyk/Musics2Dance.git EDGE-wav2clip
cd EDGE-wav2clip
export HF_TOKEN=...
scripts/bootstrap_finedance_g1_4090.sh --run-validation
```

The default bootstrap downloads compact HF artifacts and rebuilds:

- FineDance source clips in `data/finedance_aistpp`.
- FineDance+G1 tree in `data/finedance_g1_fkbeats`.
- Librosa baseline features: `baseline_feats`.
- Jukebox features: `jukebox_feats`.
- G1 FK beat metadata: `beat_feats`.
- Wav2CLIP + STFT + GaussianBeat features: `wav2clip_stft_beat_feats`.

It uses `torch/torchaudio` wheels from `https://download.pytorch.org/whl/cu126`
by default, warms up Jukebox and Wav2CLIP model downloads, and writes logs under
`setup_logs/finedance_g1_4090/`.

The feature extraction is resumable: rerunning the script skips completed
Wav2CLIP/STFT files and overwrites/rebuilds deterministic prepared trees as
needed. Use a smaller Jukebox batch size if the 4090 runs out of memory:

```bash
scripts/bootstrap_finedance_g1_4090.sh \
  --jukebox-batch-size 2 \
  --run-validation
```

Useful variants:

```bash
# Only download compact HF artifacts and install deps.
scripts/bootstrap_finedance_g1_4090.sh --skip-prepare --skip-features

# Rebuild only Librosa baseline and Wav2CLIP/STFT features.
scripts/bootstrap_finedance_g1_4090.sh --features baseline,wav2clip_stft_beat

# Preview what will run.
scripts/bootstrap_finedance_g1_4090.sh --dry-run
```

## Continuation

After bootstrap:

```bash
source .venv311/bin/activate
```

The existing stream-adapter checkpoint should be present at:

```text
runs/train/EXP-20260513-finedance-g1-wav2clip-stft-beat_r02_stream_adapter/weights/train-500.pt
```

The unfinished concat run can be launched locally with the same arguments stored
in the experiment spec, but without Slurm. Keep long run logs under
`setup_logs/` or another ignored runtime directory.
