WAV2CLIP_STFT_BEAT_FEATURE_TYPE = "wav2clip_stft_beat"
GAUSSIAN_BEAT_FEATURE_TYPE = "gaussian_beat"
BEAT_FEATURES_8D_FEATURE_TYPE = "beat_features_8d"
BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE = "beat_features_8d_motion_beatness"
WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE = "wav2clip_motion_energy_beat"
WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE = "wav2clip_motion_intensity_beatness"
WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE = (
    "wav2clip_local_motion_intensity_beatness"
)
WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE = "wav2clip_body_support_beatness"

WAV2CLIP_DIM = 512
STFT_DIM = 193
GAUSSIAN_BEAT_DIM = 1
BEAT_FEATURES_8D_DIM = 8
MOTION_ENERGY_DIM = 1
MOTION_INTENSITY_DIM = 1
MOTION_BEATNESS_DIM = 1
BODY_INTENSITY_DIM = 1
SUPPORT_BEATNESS_DIM = 1
UPPER_BEATNESS_DIM = 1
SUPPORT_CONTACT_DIM = 2
BEAT_FEATURES_8D_MOTION_BEATNESS_CONTROL_DIM = MOTION_BEATNESS_DIM
WAV2CLIP_MOTION_ENERGY_BEAT_CONTROL_DIM = GAUSSIAN_BEAT_DIM + MOTION_ENERGY_DIM
WAV2CLIP_MOTION_INTENSITY_BEATNESS_CONTROL_DIM = (
    GAUSSIAN_BEAT_DIM + MOTION_INTENSITY_DIM + MOTION_BEATNESS_DIM
)
WAV2CLIP_BODY_SUPPORT_BEATNESS_CONTROL_DIM = (
    GAUSSIAN_BEAT_DIM
    + BODY_INTENSITY_DIM
    + SUPPORT_BEATNESS_DIM
    + UPPER_BEATNESS_DIM
    + SUPPORT_CONTACT_DIM
)
WAV2CLIP_STFT_BEAT_DIMS = (WAV2CLIP_DIM, STFT_DIM, GAUSSIAN_BEAT_DIM)
WAV2CLIP_STFT_BEAT_DIM = sum(WAV2CLIP_STFT_BEAT_DIMS)

FEATURE_DIMS = {
    "baseline": 35,
    "jukebox": 4800,
    GAUSSIAN_BEAT_FEATURE_TYPE: GAUSSIAN_BEAT_DIM,
    BEAT_FEATURES_8D_FEATURE_TYPE: BEAT_FEATURES_8D_DIM,
    BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE: BEAT_FEATURES_8D_DIM,
    WAV2CLIP_STFT_BEAT_FEATURE_TYPE: WAV2CLIP_STFT_BEAT_DIM,
    WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE: WAV2CLIP_DIM,
    WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE: WAV2CLIP_DIM,
    WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE: WAV2CLIP_DIM,
    WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE: WAV2CLIP_DIM,
}

FEATURE_FUSIONS = ("linear", "concat_norm", "stream_adapter")


def get_cond_feature_dim(feature_type):
    try:
        return FEATURE_DIMS[feature_type]
    except KeyError as exc:
        supported = ", ".join(sorted(FEATURE_DIMS))
        raise ValueError(f"Unsupported feature_type {feature_type!r}; choose one of: {supported}") from exc


def validate_feature_fusion(feature_type, feature_fusion):
    if feature_fusion not in FEATURE_FUSIONS:
        supported = ", ".join(FEATURE_FUSIONS)
        raise ValueError(f"Unsupported feature_fusion {feature_fusion!r}; choose one of: {supported}")
    if feature_type == WAV2CLIP_STFT_BEAT_FEATURE_TYPE:
        if feature_fusion == "linear":
            raise ValueError(
                "wav2clip_stft_beat requires --feature_fusion concat_norm or stream_adapter"
            )
    elif feature_type in (
        BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE,
        WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
        WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
    ):
        if feature_fusion != "linear":
            raise ValueError(
                f"{feature_type} uses structured encoders and requires "
                "--feature_fusion linear"
            )
    elif feature_fusion != "linear":
        raise ValueError(
            f"--feature_fusion {feature_fusion} is only supported with "
            f"--feature_type {WAV2CLIP_STFT_BEAT_FEATURE_TYPE}"
        )
