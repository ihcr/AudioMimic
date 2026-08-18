from dataset.g1_motion_prior_dataset import G1MotionPriorDataset


class G1NativeRVQVAEDataset(G1MotionPriorDataset):
    """Tokenizer-only G1 motion dataset with no music/control fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = dict(self.metadata)
        self.metadata.update(
            {
                "condition": "native_g1_rvqvae_reconstruction_only",
                "music_features": [],
                "control_features": [],
                "generator": "none",
            }
        )
