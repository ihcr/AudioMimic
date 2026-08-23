# FineDance-G1 Quality Manifest

This is a per-clip diagnostic for the retargeted FineDance-G1 cache. It does not delete or rewrite motion files.

Low-height diagnostic threshold: `0.200 m`.

| split | clips | any negative root | any root below threshold | candidate keep (z >= 0) | candidate keep (z >= threshold) |
|---|---:|---:|---:|---:|---:|
| train | 47817 | 213 | 330 | 47604 | 47487 |
| test | 3265 | 20 | 66 | kept | kept |

## Policy

- The sealed test split is always retained, including unusual retargeting clips; report its full distribution for GT calibration.
- The two train columns are candidate policies for an ablation, not an automatic deletion rule.
- Before training, compare unfiltered training, `z >= 0`, and retargeting correction if the correction is available.
- Any reported generator score must state which training policy was used and must use the same test manifest.

## Files

- `quality_manifest.json`: complete per-clip records and summaries.
- `quality_manifest.csv`: flat table for analysis and plotting.
