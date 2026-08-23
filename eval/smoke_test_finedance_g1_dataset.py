"""Load a small FineDance-G1 batch through the real EDGE dataset adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dance_dataset import AISTPPDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/finedance_g1_fkbeats"))
    parser.add_argument("--backup-path", type=Path, default=Path("/tmp/finedance_g1_smoke_cache"))
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = AISTPPDataset(
        data_path=str(args.data_path),
        backup_path=str(args.backup_path),
        train=True,
        feature_type="baseline",
        use_beats=True,
        beat_rep="distance",
        motion_format="g1",
    )
    test = AISTPPDataset(
        data_path=str(args.data_path),
        backup_path=str(args.backup_path),
        train=False,
        feature_type="baseline",
        normalizer=train.normalizer,
        use_beats=True,
        beat_rep="distance",
        motion_format="g1",
    )
    motion, condition, feature_paths, wav_paths = next(
        iter(DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0))
    )
    assert motion.shape[1:] == (150, 38)
    assert condition["music"].shape[1:] == (150, 35)
    assert condition["beat"].shape[1:] == (150,)
    assert condition["beat_target"].shape[1:] == (150,)
    assert condition["audio_mask"].shape[1:] == (150,)
    assert (condition["beat"][0] == train.data["audio_dist"][0]).all()
    assert (condition["beat_target"][0] == train.data["motion_dist"][0].float()).all()
    print(f"train_len={len(train)} test_len={len(test)}")
    print(f"motion_shape={tuple(motion.shape)}")
    print(f"music_shape={tuple(condition['music'].shape)}")
    print(f"beat_shape={tuple(condition['beat'].shape)}")
    print(f"first_feature={feature_paths[0]}")
    print(f"first_wav={wav_paths[0]}")
    print("FineDance-G1 dataset smoke test passed")


if __name__ == "__main__":
    main()
