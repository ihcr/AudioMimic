# FineDance Windowed Source/G1 Music Metrics

Scope: **all**, paired sequences: **203**.
Fixed windows are non-overlapping complete windows; an incomplete tail is omitted.
The detector, tolerance, FPS and audio clock are identical to the formal retargeting audit.

| window | stage | n windows | BAS median | Event F1 median | Impact corr median | Tempo error median | Phase error median |
|---|---|---:|---:|---:|---:|---:|---:|
| 5s | source | 5405 | 0.0495 | 0.3529 | 0.0542 | 26.9083 | 0.2786 |
| 5s | g1_target | 5405 | 0.2048 | 0.7000 | 0.1305 | 26.9531 | 0.2781 |
| 16s | source | 1624 | 0.1286 | 0.5000 | 0.0365 | 22.2656 | 0.2597 |
| 16s | g1_target | 1624 | 0.2190 | 0.6866 | 0.0713 | 23.0469 | 0.2594 |
| full | source | 203 | 0.1201 | 0.4771 | 0.0282 | 20.9054 | 0.2523 |
| full | g1_target | 203 | 0.2267 | 0.6829 | 0.0392 | 21.0077 | 0.2504 |

Interpretation: these are **music-motion correspondence** metrics, not a complete dance-quality
score. Compare source vs G1 within each window first. If the source is already low at 5 s and
16 s, the difference is not caused by long-sequence averaging alone. If source is reasonable but
G1 drops, the dominant issue is retargeting. Full-sequence values are reported separately because
they mix local rhythm with phrase and transition structure. Overall dance quality must additionally
use energy, jerk, static/repetition, contact, root stability and human evaluation.
