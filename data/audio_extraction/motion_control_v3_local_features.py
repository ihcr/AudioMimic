import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from data.audio_extraction.motion_control_v2_features import (
    LOCAL_CACHE_VERSION,
    LOCAL_FEATURE_DIR_NAME,
    LOCAL_METADATA_NAME,
    ROOT_LOCAL_FRAME,
    extract_motion_control_v2_features,
    parse_args as parse_v2_args,
)


def parse_args(argv=None):
    args = parse_v2_args(argv)
    args.coordinate_frame = ROOT_LOCAL_FRAME
    args.feature_dir_name = LOCAL_FEATURE_DIR_NAME
    args.metadata_name = LOCAL_METADATA_NAME
    args.cache_version = LOCAL_CACHE_VERSION
    return args


def extract_motion_control_v3_local_features(args):
    args.coordinate_frame = ROOT_LOCAL_FRAME
    args.feature_dir_name = LOCAL_FEATURE_DIR_NAME
    args.metadata_name = LOCAL_METADATA_NAME
    args.cache_version = LOCAL_CACHE_VERSION
    return extract_motion_control_v2_features(args)


if __name__ == "__main__":
    extract_motion_control_v3_local_features(parse_args())
