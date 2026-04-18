import pickle
import shutil
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent
SPLITS_DIR = DATA_DIR / "splits"


def fileToList(path):
    out = Path(path).read_text(encoding="utf-8").splitlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out


filter_list = set(fileToList(SPLITS_DIR / "ignore_list.txt"))
train_list = set(fileToList(SPLITS_DIR / "crossmodal_train.txt"))
test_list = set(fileToList(SPLITS_DIR / "crossmodal_test.txt"))


def split_data(dataset_path, output_root=DATA_DIR):
    dataset_path = Path(dataset_path)
    output_root = Path(output_root)
    if not dataset_path.is_absolute():
        dataset_path = output_root / dataset_path

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_path}")

    # train - test split
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        split_root = output_root / split_name
        (split_root / "motions").mkdir(parents=True, exist_ok=True)
        (split_root / "wavs").mkdir(parents=True, exist_ok=True)
        ordered_sequences = sorted(split_list)
        for sequence in tqdm(
            ordered_sequences,
            total=len(ordered_sequences),
            desc=f"Split {split_name}",
            unit="seq",
        ):
            if sequence in filter_list:
                continue
            motion = dataset_path / "motions" / f"{sequence}.pkl"
            wav = dataset_path / "wavs" / f"{sequence}.wav"
            if not motion.is_file():
                raise FileNotFoundError(f"Missing motion file for {sequence}: {motion}")
            if not wav.is_file():
                raise FileNotFoundError(f"Missing wav file for {sequence}: {wav}")
            with open(motion, "rb") as handle:
                motion_data = pickle.load(handle)
            trans = motion_data["smpl_trans"]
            pose = motion_data["smpl_poses"]
            scale = motion_data["smpl_scaling"]
            out_data = {"pos": trans, "q": pose, "scale": scale}
            with open(split_root / "motions" / f"{sequence}.pkl", "wb") as handle:
                pickle.dump(out_data, handle)
            shutil.copyfile(wav, split_root / "wavs" / f"{sequence}.wav")
