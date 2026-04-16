import argparse
from pathlib import Path

from filter_split_data import *
from slice import *

DATA_DIR = Path(__file__).resolve().parent
baseline_extract = None
jukebox_extract = None
beat_extract = None


def _resolve_data_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return DATA_DIR / path


def _load_baseline_extract():
    global baseline_extract
    if baseline_extract is None:
        from audio_extraction.baseline_features import extract_folder as extract

        baseline_extract = extract
    return baseline_extract


def _load_jukebox_extract():
    global jukebox_extract
    if jukebox_extract is None:
        from audio_extraction.jukebox_features import extract_folder as extract

        jukebox_extract = extract
    return jukebox_extract


def _load_beat_extract():
    global beat_extract
    if beat_extract is None:
        from audio_extraction.beat_features import extract_folder as extract

        beat_extract = extract
    return beat_extract


def create_dataset(opt):
    dataset_folder = _resolve_data_path(opt.dataset_folder)
    train_motion_dir = DATA_DIR / "train" / "motions"
    train_wav_dir = DATA_DIR / "train" / "wavs"
    test_motion_dir = DATA_DIR / "test" / "motions"
    test_wav_dir = DATA_DIR / "test" / "wavs"

    # split the data according to the splits files
    print("Creating train / test split")
    split_data(str(dataset_folder), output_root=DATA_DIR)
    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_aistpp(str(train_motion_dir), str(train_wav_dir), opt.stride, opt.length)
    print("Slicing test data")
    slice_aistpp(str(test_motion_dir), str(test_wav_dir), opt.stride, opt.length)
    # process dataset to extract audio features
    if opt.extract_baseline:
        print("Extracting baseline features")
        _load_baseline_extract()(
            str(DATA_DIR / "train" / "wavs_sliced"),
            str(DATA_DIR / "train" / "baseline_feats"),
        )
        _load_baseline_extract()(
            str(DATA_DIR / "test" / "wavs_sliced"),
            str(DATA_DIR / "test" / "baseline_feats"),
        )
    if opt.extract_jukebox:
        print("Extracting jukebox features")
        _load_jukebox_extract()(
            str(DATA_DIR / "train" / "wavs_sliced"),
            str(DATA_DIR / "train" / "jukebox_feats"),
        )
        _load_jukebox_extract()(
            str(DATA_DIR / "test" / "wavs_sliced"),
            str(DATA_DIR / "test" / "jukebox_feats"),
        )
    if opt.extract_beats:
        print("Extracting beat metadata")
        _load_beat_extract()(
            str(DATA_DIR / "train" / "motions_sliced"),
            str(DATA_DIR / "train" / "wavs_sliced"),
            str(DATA_DIR / "train" / "beat_feats"),
        )
        _load_beat_extract()(
            str(DATA_DIR / "test" / "motions_sliced"),
            str(DATA_DIR / "test" / "wavs_sliced"),
            str(DATA_DIR / "test" / "beat_feats"),
        )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    parser.add_argument("--extract-beats", action="store_true")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
