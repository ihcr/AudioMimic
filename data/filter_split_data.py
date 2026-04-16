import pickle
import shutil
from pathlib import Path

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

    # train - test split
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        split_root = output_root / split_name
        (split_root / "motions").mkdir(parents=True, exist_ok=True)
        (split_root / "wavs").mkdir(parents=True, exist_ok=True)
        for sequence in split_list:
            if sequence in filter_list:
                continue
            motion = dataset_path / "motions" / f"{sequence}.pkl"
            wav = dataset_path / "wavs" / f"{sequence}.wav"
            assert motion.is_file()
            assert wav.is_file()
            motion_data = pickle.load(open(motion, "rb"))
            trans = motion_data["smpl_trans"]
            pose = motion_data["smpl_poses"]
            scale = motion_data["smpl_scaling"]
            out_data = {"pos": trans, "q": pose, "scale": scale}
            pickle.dump(out_data, open(split_root / "motions" / f"{sequence}.pkl", "wb"))
            shutil.copyfile(wav, split_root / "wavs" / f"{sequence}.wav")
