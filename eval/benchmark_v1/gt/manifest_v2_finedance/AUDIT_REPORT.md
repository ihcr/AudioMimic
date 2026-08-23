# GT Benchmark Asset Audit

Generated: `2026-08-23T00:11:48.747557+00:00`

## Result

- AIST++ paired assets: **1408**; status: **pass**.
- AIST++ motion/audio unmatched: 0/0.
- Declared crossmodal test IDs: 20; available paired IDs: 20.
- Processed test-cache IDs: 20; split/cache match: **True**.
- AIST++ split status: **pass**.
- FineDance status: **pass**.
- FineDance raw/G1/audio files: 203/203/207; same-ID paired valid: **203**.
- FineDance cross-genre test assets available: 18/18.

## FineDance pairing

FineDance is valid only when the raw motion, matching WAV, retargeted G1 motion, and label metadata share the same numeric sequence ID. The audit also checks duration agreement and label/G1 frame agreement. The benchmark must not treat metadata counts as usable GT when any paired asset is missing.

## AIST++ split note

The declared `crossmodal_test.txt` IDs exactly match the 20 sequence IDs in the existing processed test cache. This confirms local split/cache consistency; official provenance and the final paper manifest still need to be frozen before final evaluation.

## Next action

1. Review any FineDance IDs listed in `invalid_or_unpaired_ids`.
2. Confirm the cross-genre test IDs against the official split.
3. Run with `--hash-files` once the final roots are frozen.
