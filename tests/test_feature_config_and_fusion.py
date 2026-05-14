import unittest

import torch
import torch.nn as nn

from feature_config import (
    WAV2CLIP_STFT_BEAT_DIM,
    get_cond_feature_dim,
    validate_feature_fusion,
)
from model.model import DanceDecoder, Wav2ClipStftBeatFusion


class FeatureConfigAndFusionTests(unittest.TestCase):
    def test_feature_dim_and_fusion_validation(self):
        self.assertEqual(get_cond_feature_dim("wav2clip_stft_beat"), 706)
        validate_feature_fusion("wav2clip_stft_beat", "concat_norm")
        validate_feature_fusion("wav2clip_stft_beat", "stream_adapter")
        with self.assertRaises(ValueError):
            validate_feature_fusion("wav2clip_stft_beat", "linear")
        with self.assertRaises(ValueError):
            validate_feature_fusion("jukebox", "concat_norm")

    def test_fusion_modules_emit_latent_dim(self):
        cond = torch.randn(2, 4, WAV2CLIP_STFT_BEAT_DIM)
        for mode in ("concat_norm", "stream_adapter"):
            fusion = Wav2ClipStftBeatFusion(mode, latent_dim=512)
            out = fusion(cond)
            self.assertEqual(out.shape, (2, 4, 512))
            self.assertTrue(torch.isfinite(out).all())

    def test_one_dimensional_beat_stream_is_not_layer_normalized_away(self):
        concat_fusion = Wav2ClipStftBeatFusion("concat_norm", latent_dim=512)
        adapter_fusion = Wav2ClipStftBeatFusion("stream_adapter", latent_dim=512)
        self.assertIsInstance(concat_fusion.stream_norms[-1], nn.Identity)
        self.assertIsInstance(adapter_fusion.adapters[-1][0], nn.Identity)

    def test_dance_decoder_accepts_both_wav2clip_fusion_modes(self):
        x = torch.randn(2, 4, 151)
        cond = torch.randn(2, 4, WAV2CLIP_STFT_BEAT_DIM)
        times = torch.randint(0, 10, (2,))
        for mode in ("concat_norm", "stream_adapter"):
            decoder = DanceDecoder(
                nfeats=151,
                seq_len=4,
                latent_dim=64,
                ff_size=64,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
                cond_feature_dim=WAV2CLIP_STFT_BEAT_DIM,
                cond_fusion=mode,
                use_rotary=False,
            )
            out = decoder(x, cond, times)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
