---
license: other
library_name: pytorch
tags:
- music-to-dance
- diffusion
- humanoid
- unitree-g1
- aist
---

# EDGE G1 BeatDistance Checkpoint

This repository contains a robot-native G1 BeatDistance diffusion checkpoint trained in the EDGE diffusion worktree.

## Checkpoint

- File: `train-2000.pt`
- Source path on the training machine: `runs/train/g1_aist_beatdistance_featurecache/weights/train-2000.pt`
- Code commit: `8323ff7` (`Add G1 FK evaluation and training workflow`)
- GitHub branch: `lbtwyk/Musics2Dance@diffusion`
- AudioMimic mirror branch: `ihcr/AudioMimic@yukun`

## Data and Training Notes

- Motion format: G1 robot-native trajectories
- Dataset tree used for this checkpoint: `data/g1_aistpp_full`
- Train clips: 17,733
- Test clips: 186
- Music features: Jukebox
- Beat mode: distance conditioning
- Auxiliary beat loss: off

A newer FK-beat-label G1 checkpoint is currently training separately and is not this file.

## Evaluation Artifacts

The `eval_fk/` folder, when present, contains FK-enabled evaluation reports for this checkpoint.

## Intended Use

Research use for music-conditioned G1 humanoid motion generation and comparison against the newer FK-beat training pipeline.
