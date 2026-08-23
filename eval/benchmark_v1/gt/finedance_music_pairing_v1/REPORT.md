# FineDance Music-Motion Pairing Audit

This is a benchmark integrity check, not a model score. Each official
cross-genre test motion is compared with its same-ID WAV and a deterministic
cyclic and all-other-song wrong-song controls. A positive margin means the paired WAV is more
compatible under that diagnostic; it is not proof of perfect beat alignment.

- Test pairs: **18**
- Paired zero-lag correlation: **0.1098**
- Wrong-song zero-lag correlation: **0.0376**
- Zero-lag margin: **0.0722**
- Paired best-lag correlation: **0.1655**
- Wrong-song best-lag correlation: **0.0919**
- Best-lag margin: **0.0736**
- Paired event F1: **0.7127**
- Wrong-song event F1: **0.6857**
- Event-F1 margin: **0.0270**
- All-wrong-song mean best-correlation margin: **0.0771**
- Paired correlation mean rank percentile among all wrong songs: **0.8366**
- Paired song retrieval top-1 rate: **0.2222**

Interpretation: use positive paired-vs-wrong margins as evidence that the
music and motion pairing contains signal. Use per-sequence results and a
stronger permutation test before making a paper-level musicality claim.
