# AIST++ Music-Motion Pairing Audit

This is a benchmark integrity check, not a model score. Same-ID audio is
compared with a cyclic wrong-song control and all other AIST++ test songs.

- Test pairs: **20**
- Paired-vs-wrong best-lag correlation margin: **0.0381**
- Paired-vs-wrong event-F1 margin: **0.0133**
- Paired correlation top-1 rate among wrong songs: **0.3000**

Use the paired distribution as the positive reference and the wrong-song
distribution as a negative control. This calibration is separate from GMR
and generator evaluation.
