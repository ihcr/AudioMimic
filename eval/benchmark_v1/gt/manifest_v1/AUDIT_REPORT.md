# GT Benchmark Asset Audit

Generated: `2026-08-22T23:39:34.807126+00:00`

## Result

- AIST++ paired assets: **1408**; status: **pass**.
- AIST++ motion/audio unmatched: 0/0.
- Declared crossmodal test IDs: 20; available paired IDs: 20.
- Processed test-cache IDs: 20; split/cache match: **True**.
- AIST++ split status: **pass**.
- FineDance status: **blocked_pending_assets**.

## FineDance blocker

The local FineDance directory contains metadata but no G1 motion files. The benchmark is not allowed to treat metadata counts as usable GT. Copy or mount the prepared FineDance-G1 assets, then rerun this script before final evaluation.

## AIST++ split note

The declared `crossmodal_test.txt` IDs exactly match the 20 sequence IDs in the existing processed test cache. This confirms local split/cache consistency; official provenance and the final paper manifest still need to be frozen before final evaluation.

## Next action

1. Provide the FineDance-G1 motion/audio root and rerun the audit.
2. Confirm the AIST++ test IDs against the official split.
3. Run with `--hash-files` once the final roots are frozen.
