import unittest

import torch
import torch.nn as nn

from feature_config import (
    BEAT_FEATURES_8D_DIM,
    BEAT_FEATURES_8D_FEATURE_TYPE,
    BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE,
    BODY_INTENSITY_DIM,
    GAUSSIAN_BEAT_DIM,
    SUPPORT_BEATNESS_DIM,
    SUPPORT_CONTACT_DIM,
    UPPER_BEATNESS_DIM,
    WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
    WAV2CLIP_DIM,
    WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
    WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE,
    WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
    WAV2CLIP_STFT_BEAT_DIM,
    get_cond_feature_dim,
    validate_feature_fusion,
)
from model.diffusion import cond_batch_size, slice_cond
from model.model import (
    BeatFeatures8DMotionBeatnessDecoder,
    DanceDecoder,
    Wav2ClipBodySupportBeatnessDecoder,
    Wav2ClipMotionEnergyBeatDecoder,
    Wav2ClipMotionIntensityBeatnessDecoder,
    Wav2ClipStftBeatFusion,
)


class FeatureConfigAndFusionTests(unittest.TestCase):
    def test_feature_dim_and_fusion_validation(self):
        self.assertEqual(get_cond_feature_dim("wav2clip_stft_beat"), 706)
        self.assertEqual(get_cond_feature_dim("gaussian_beat"), GAUSSIAN_BEAT_DIM)
        self.assertEqual(
            get_cond_feature_dim(BEAT_FEATURES_8D_FEATURE_TYPE),
            BEAT_FEATURES_8D_DIM,
        )
        self.assertEqual(
            get_cond_feature_dim(BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE),
            BEAT_FEATURES_8D_DIM,
        )
        self.assertEqual(
            get_cond_feature_dim(WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE),
            WAV2CLIP_DIM,
        )
        self.assertEqual(
            get_cond_feature_dim(WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE),
            WAV2CLIP_DIM,
        )
        self.assertEqual(
            get_cond_feature_dim(WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE),
            WAV2CLIP_DIM,
        )
        self.assertEqual(
            get_cond_feature_dim(WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE),
            WAV2CLIP_DIM,
        )
        validate_feature_fusion("gaussian_beat", "linear")
        validate_feature_fusion(BEAT_FEATURES_8D_FEATURE_TYPE, "linear")
        validate_feature_fusion(BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE, "linear")
        validate_feature_fusion(WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE, "linear")
        validate_feature_fusion(WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE, "linear")
        validate_feature_fusion(
            WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
            "linear",
        )
        validate_feature_fusion(WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE, "linear")
        validate_feature_fusion("wav2clip_stft_beat", "concat_norm")
        validate_feature_fusion("wav2clip_stft_beat", "stream_adapter")
        with self.assertRaises(ValueError):
            validate_feature_fusion("wav2clip_stft_beat", "linear")
        with self.assertRaises(ValueError):
            validate_feature_fusion("jukebox", "concat_norm")
        with self.assertRaises(ValueError):
            validate_feature_fusion(BEAT_FEATURES_8D_FEATURE_TYPE, "concat_norm")
        with self.assertRaises(ValueError):
            validate_feature_fusion(
                BEAT_FEATURES_8D_MOTION_BEATNESS_FEATURE_TYPE,
                "concat_norm",
            )
        with self.assertRaises(ValueError):
            validate_feature_fusion(WAV2CLIP_MOTION_ENERGY_BEAT_FEATURE_TYPE, "stream_adapter")
        with self.assertRaises(ValueError):
            validate_feature_fusion(
                WAV2CLIP_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
                "stream_adapter",
            )
        with self.assertRaises(ValueError):
            validate_feature_fusion(
                WAV2CLIP_LOCAL_MOTION_INTENSITY_BEATNESS_FEATURE_TYPE,
                "stream_adapter",
            )
        with self.assertRaises(ValueError):
            validate_feature_fusion(
                WAV2CLIP_BODY_SUPPORT_BEATNESS_FEATURE_TYPE,
                "stream_adapter",
            )

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

    def test_motion_energy_decoder_accepts_nested_condition(self):
        x = torch.randn(2, 4, 38)
        cond = {
            "semantic": {"wav2clip": torch.randn(2, 4, WAV2CLIP_DIM)},
            "control": {
                "gaussian_beat": torch.rand(2, 4, 1),
                "beat_energy_envelope": torch.rand(2, 4, 1),
            },
        }
        self.assertEqual(cond_batch_size(cond), 2)
        self.assertEqual(slice_cond(cond, slice(0, 1))["semantic"]["wav2clip"].shape, (1, 4, WAV2CLIP_DIM))
        decoder = Wav2ClipMotionEnergyBeatDecoder(
            nfeats=38,
            seq_len=4,
            latent_dim=64,
            ff_size=64,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            cond_feature_dim=WAV2CLIP_DIM,
            cond_fusion="linear",
            use_rotary=False,
        )
        out = decoder(x, cond, torch.randint(0, 10, (2,)))
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        prepared, stats = decoder.prepare_motion_energy_training_condition(
            cond,
            epoch=101,
            teacher_forcing_epochs=100,
            pred_mix_prob=1.0,
        )
        self.assertEqual(prepared["control"]["beat_energy_envelope"].shape, (2, 4, 1))
        self.assertIn("energy_pred_loss", stats)
        self.assertTrue(torch.isfinite(stats["energy_pred_loss"]))

    def test_motion_intensity_beatness_decoder_accepts_nested_condition(self):
        x = torch.randn(2, 4, 38)
        cond = {
            "semantic": {"wav2clip": torch.randn(2, 4, WAV2CLIP_DIM)},
            "control": {
                "gaussian_beat": torch.rand(2, 4, 1),
                "motion_intensity": torch.rand(2, 4, 1),
                "motion_beatness": torch.rand(2, 4, 1),
            },
        }
        decoder = Wav2ClipMotionIntensityBeatnessDecoder(
            nfeats=38,
            seq_len=4,
            latent_dim=64,
            ff_size=64,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            cond_feature_dim=WAV2CLIP_DIM,
            cond_fusion="linear",
            use_rotary=False,
        )
        out = decoder(x, cond, torch.randint(0, 10, (2,)))
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        predictions = decoder.predict_controls(cond)
        self.assertEqual(predictions["motion_intensity"].shape, (2, 4, 1))
        self.assertEqual(predictions["motion_beatness"].shape, (2, 4, 1))
        prepared, stats = decoder.prepare_motion_energy_training_condition(
            cond,
            epoch=101,
            teacher_forcing_epochs=100,
            pred_mix_prob=1.0,
        )
        self.assertEqual(prepared["control"]["motion_intensity"].shape, (2, 4, 1))
        self.assertEqual(prepared["control"]["motion_beatness"].shape, (2, 4, 1))
        self.assertIn("intensity_pred_loss", stats)
        self.assertIn("beatness_pred_loss", stats)
        self.assertTrue(torch.isfinite(stats["intensity_pred_loss"]))
        self.assertTrue(torch.isfinite(stats["beatness_pred_loss"]))

    def test_beat_features_8d_motion_beatness_decoder_accepts_nested_condition(self):
        x = torch.randn(2, 4, 38)
        cond = {
            "semantic": {"beat_features_8d": torch.randn(2, 4, BEAT_FEATURES_8D_DIM)},
            "control": {
                "motion_beatness": torch.rand(2, 4, 1),
            },
        }
        decoder = BeatFeatures8DMotionBeatnessDecoder(
            nfeats=38,
            seq_len=4,
            latent_dim=64,
            ff_size=64,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            cond_feature_dim=BEAT_FEATURES_8D_DIM,
            cond_fusion="linear",
            use_rotary=False,
        )
        out = decoder(x, cond, torch.randint(0, 10, (2,)))
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        predictions = decoder.predict_controls(cond)
        self.assertEqual(predictions["motion_beatness"].shape, (2, 4, 1))
        self.assertNotIn("motion_intensity", predictions)
        prepared, stats = decoder.prepare_motion_energy_training_condition(
            cond,
            epoch=101,
            teacher_forcing_epochs=100,
            pred_mix_prob=1.0,
        )
        self.assertEqual(prepared["control"]["motion_beatness"].shape, (2, 4, 1))
        self.assertIn("beatness_pred_loss", stats)
        self.assertIn("energy_pred_loss", stats)
        self.assertTrue(torch.isfinite(stats["beatness_pred_loss"]))

    def test_body_support_beatness_decoder_accepts_nested_condition(self):
        x = torch.randn(2, 4, 38)
        cond = {
            "semantic": {"wav2clip": torch.randn(2, 4, WAV2CLIP_DIM)},
            "control": {
                "gaussian_beat": torch.rand(2, 4, 1),
                "body_intensity": torch.rand(2, 4, BODY_INTENSITY_DIM),
                "support_beatness": torch.rand(2, 4, SUPPORT_BEATNESS_DIM),
                "upper_beatness": torch.rand(2, 4, UPPER_BEATNESS_DIM),
                "support_contact": torch.rand(2, 4, SUPPORT_CONTACT_DIM),
            },
        }
        decoder = Wav2ClipBodySupportBeatnessDecoder(
            nfeats=38,
            seq_len=4,
            latent_dim=64,
            ff_size=64,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            cond_feature_dim=WAV2CLIP_DIM,
            cond_fusion="linear",
            use_rotary=False,
        )
        out = decoder(x, cond, torch.randint(0, 10, (2,)))
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        predictions = decoder.predict_controls(cond)
        self.assertEqual(predictions["body_intensity"].shape, (2, 4, BODY_INTENSITY_DIM))
        self.assertEqual(predictions["support_beatness"].shape, (2, 4, SUPPORT_BEATNESS_DIM))
        self.assertEqual(predictions["upper_beatness"].shape, (2, 4, UPPER_BEATNESS_DIM))
        self.assertEqual(predictions["support_contact"].shape, (2, 4, SUPPORT_CONTACT_DIM))
        self.assertEqual(predictions["support_contact_logits"].shape, (2, 4, SUPPORT_CONTACT_DIM))
        prepared, stats = decoder.prepare_motion_energy_training_condition(
            cond,
            epoch=101,
            teacher_forcing_epochs=100,
            pred_mix_prob=1.0,
        )
        self.assertEqual(prepared["control"]["body_intensity"].shape, (2, 4, BODY_INTENSITY_DIM))
        self.assertEqual(prepared["control"]["support_beatness"].shape, (2, 4, SUPPORT_BEATNESS_DIM))
        self.assertEqual(prepared["control"]["upper_beatness"].shape, (2, 4, UPPER_BEATNESS_DIM))
        self.assertEqual(prepared["control"]["support_contact"].shape, (2, 4, SUPPORT_CONTACT_DIM))
        self.assertIn("body_intensity_pred_loss", stats)
        self.assertIn("support_beatness_pred_loss", stats)
        self.assertIn("upper_beatness_pred_loss", stats)
        self.assertIn("support_contact_pred_loss", stats)
        self.assertTrue(torch.isfinite(stats["energy_pred_loss"]))


if __name__ == "__main__":
    unittest.main()
