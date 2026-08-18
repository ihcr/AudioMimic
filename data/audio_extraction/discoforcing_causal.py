"""Strict-causal audio conditioning for the DF-VQPAE-Causal-C1 baseline.

The VQ-PAE architecture is an API-compatible local reimplementation of the
conditioning component released with DiscoForcing at commit
bec97cdb99f9ecb216ee826e1edb45b0a49aebf6. Runtime code intentionally does not
depend on the research checkout. See third_party/discoforcing/ for attribution.
"""

from dataclasses import asdict, dataclass, fields
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


DISCOFORCING_UPSTREAM_COMMIT = "bec97cdb99f9ecb216ee826e1edb45b0a49aebf6"
DF_VQPAE_BASELINE_ID = "DF-VQPAE-Causal-C1"
DF_LIBROSA_BASELINE_ID = "DF-Librosa35-Causal-6s"
DF_NULL_BASELINE_ID = "DF-Null-Causal-C1"
MUSIC_HISTORY_SECONDS = 6.0
MUSIC_FEATURE_FPS = 30
MUSIC_FEATURE_TOKENS = 180
MUSIC_CARRIER_DIM = 35
VQPAE_SAMPLE_RATE = 16_000
LIBROSA_SAMPLE_RATE = 15_360
LIBROSA_HOP_LENGTH = 512
VQPAE_MIN_CODEBOOK_USAGE = 0.05
VQPAE_MIN_CODEBOOK_PERPLEXITY = 32.0
SPHERICAL_RVQ_STABILITY_CONTRACT = {
    "codebook_projection": "unit_l2_after_initialization_and_every_adam_step",
    "projection_dimension": "per_entry",
    "projected_latents": "unit_l2_before_lookup_loss_and_straight_through",
    "projected_latent_dimension": "channel_per_time_step",
    "codebook_initialization": "sequential_spherical_kmeans_on_fit_latents",
    "initialization_windows": 16,
    "initialization_samples": 16384,
    "initialization_iterations": 20,
    "resume_norm_atol": 2e-6,
}


def _manifest_digest(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CausalAudioConfig:
    history_seconds: float = MUSIC_HISTORY_SECONDS
    feature_fps: int = MUSIC_FEATURE_FPS
    feature_tokens: int = MUSIC_FEATURE_TOKENS
    carrier_dim: int = MUSIC_CARRIER_DIM
    vqpae_sample_rate: int = VQPAE_SAMPLE_RATE
    librosa_sample_rate: int = LIBROSA_SAMPLE_RATE
    librosa_hop_length: int = LIBROSA_HOP_LENGTH
    variance_floor: float = 1e-6
    boundary_convention: str = "audio_samples_strictly_before_target_start"
    cold_start_padding: str = "left_zero"

    def __post_init__(self):
        if self.history_seconds <= 0:
            raise ValueError("history_seconds must be positive")
        if self.feature_fps <= 0 or self.feature_tokens <= 0:
            raise ValueError("feature cadence and token count must be positive")
        expected = self.history_seconds * self.feature_fps
        if not math.isclose(expected, self.feature_tokens):
            raise ValueError("feature_tokens must equal history_seconds * feature_fps")
        if self.carrier_dim != MUSIC_CARRIER_DIM:
            raise ValueError("DF causal baseline is locked to a 35D carrier")
        if self.librosa_sample_rate != (
            self.feature_fps * self.librosa_hop_length
        ):
            raise ValueError("Librosa sample rate must give exactly 30 Hz")
        if self.variance_floor <= 0:
            raise ValueError("variance_floor must be positive")

    def manifest(self):
        payload = asdict(self)
        payload.update(
            {
                "baseline_id": DF_VQPAE_BASELINE_ID,
                "upstream_commit": DISCOFORCING_UPSTREAM_COMMIT,
            }
        )
        return payload


@dataclass(frozen=True)
class VQPAEConfig:
    input_channels: int = 1
    intermediate_channels: int = 64
    embedding_channels: int = 8
    time_range: int = 96_000
    window_seconds: float = MUSIC_HISTORY_SECONDS
    use_fft_mlp: bool = False
    vq_only: bool = False
    enc_dilation_rates: tuple = (9, 3, 1)
    enc_kernel_sizes: tuple = (7, 3, 3)
    dec_dilation_rates: tuple = (1, 3, 9)
    dec_kernel_sizes: tuple = (3, 3, 7)
    activation: str = "elu"
    combiner: str = "concat"
    n_codebooks: int = 9
    codebook_size: int = 1024
    codebook_dim: int = 8
    quantizer_dropout: float = 0.5
    normalize_projected_latents: bool = False

    def __post_init__(self):
        if self.input_channels != 1:
            raise ValueError("released VQ-PAE baseline is mono")
        if len(self.enc_dilation_rates) != len(self.enc_kernel_sizes):
            raise ValueError("encoder dilation/kernel lengths differ")
        if len(self.dec_dilation_rates) != len(self.dec_kernel_sizes):
            raise ValueError("decoder dilation/kernel lengths differ")
        if self.embedding_channels <= 0 or self.time_range <= 0:
            raise ValueError("embedding_channels and time_range must be positive")
        if self.activation != "elu":
            raise ValueError("formal DF VQ-PAE route is locked to ELU")
        if self.combiner != "concat":
            raise ValueError("formal DF VQ-PAE route is locked to concat")
        if self.n_codebooks <= 0 or self.codebook_size <= 1:
            raise ValueError("invalid RVQ configuration")
        if not 0.0 <= self.quantizer_dropout <= 1.0:
            raise ValueError("quantizer_dropout must be in [0,1]")

    def manifest(self):
        payload = asdict(self)
        payload.update(
            {
                "model_type": "discoforcing_vqpae_local_compatible",
                "upstream_commit": DISCOFORCING_UPSTREAM_COMMIT,
                "feature_variant": "concat_z_vq_z_pae",
            }
        )
        return payload


class TemporalLayerNorm(nn.Module):
    """Upstream LN_v2: normalize each channel over the complete window."""

    def __init__(self, length, epsilon=1e-5):
        super().__init__()
        self.epsilon = float(epsilon)
        self.alpha = nn.Parameter(torch.ones(1, 1, int(length)))
        self.beta = nn.Parameter(torch.zeros(1, 1, int(length)))

    def forward(self, values):
        mean = values.mean(dim=-1, keepdim=True)
        variance = ((values - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized = (values - mean) / (variance + self.epsilon).sqrt()
        return normalized * self.alpha + self.beta


def _wn_conv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class VectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        normalize_projected_latents=False,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.normalize_projected_latents = bool(normalize_projected_latents)
        self.in_proj = _wn_conv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = _wn_conv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, indices):
        return F.embedding(indices, self.codebook.weight).transpose(1, 2)

    def decode_latents(self, latents):
        batch, _, length = latents.shape
        encodings = F.normalize(latents.transpose(1, 2).reshape(-1, self.codebook_dim))
        codebook = F.normalize(self.codebook.weight)
        distances = (
            encodings.square().sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.square().sum(1, keepdim=True).t()
        )
        indices = (-distances).argmax(1).view(batch, length)
        return self.decode_code(indices), indices

    def forward(self, values):
        projected = self.in_proj(values)
        if self.normalize_projected_latents:
            projected = F.normalize(projected, dim=1)
        quantized, indices = self.decode_latents(projected)
        commitment = F.mse_loss(
            projected,
            quantized.detach(),
            reduction="none",
        ).mean((1, 2))
        codebook = F.mse_loss(
            quantized,
            projected.detach(),
            reduction="none",
        ).mean((1, 2))
        straight_through = projected + (quantized - projected).detach()
        return (
            self.out_proj(straight_through),
            commitment,
            codebook,
            indices,
            projected,
        )


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        n_codebooks,
        codebook_size,
        codebook_dim,
        quantizer_dropout,
        normalize_projected_latents=False,
    ):
        super().__init__()
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = [int(codebook_dim)] * self.n_codebooks
        self.quantizer_dropout = float(quantizer_dropout)
        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(
                    input_dim,
                    codebook_size,
                    codebook_dim,
                    normalize_projected_latents=normalize_projected_latents,
                )
                for _ in range(self.n_codebooks)
            ]
        )

    def forward(self, values, n_quantizers=None):
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        per_sample_quantizers = None
        if self.training:
            per_sample_quantizers = torch.full(
                (values.shape[0],),
                self.n_codebooks + 1,
                device=values.device,
                dtype=torch.long,
            )
            dropout_count = int(values.shape[0] * self.quantizer_dropout)
            if dropout_count:
                per_sample_quantizers[:dropout_count] = torch.randint(
                    1,
                    self.n_codebooks + 1,
                    (dropout_count,),
                    device=values.device,
                )

        quantized_sum = torch.zeros_like(values)
        residual = values
        commitment_sum = values.new_zeros(())
        codebook_sum = values.new_zeros(())
        all_indices = []
        all_latents = []
        for index, quantizer in enumerate(self.quantizers):
            if per_sample_quantizers is None and index >= int(n_quantizers):
                break
            result = quantizer(residual)
            quantized, commitment, codebook, indices, projected = result
            if per_sample_quantizers is None:
                mask = values.new_ones(values.shape[0])
            else:
                mask = (index < per_sample_quantizers).to(values.dtype)
            quantized_sum = quantized_sum + quantized * mask[:, None, None]
            residual = residual - quantized
            commitment_sum = commitment_sum + (commitment * mask).mean()
            codebook_sum = codebook_sum + (codebook * mask).mean()
            all_indices.append(indices)
            all_latents.append(projected)
        return (
            quantized_sum,
            torch.stack(all_indices, dim=1),
            torch.cat(all_latents, dim=1),
            commitment_sum,
            codebook_sum,
        )


class MLPCombiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv1d(2 * channels, 2 * channels, 3, dilation=1, padding="same"),
            nn.BatchNorm1d(2 * channels),
            nn.ELU(),
            nn.Conv1d(2 * channels, 2 * channels, 3, dilation=3, padding="same"),
            nn.BatchNorm1d(2 * channels),
            nn.ELU(),
            nn.Conv1d(2 * channels, channels, 3, dilation=5, padding="same"),
            nn.BatchNorm1d(channels),
            nn.ELU(),
        )

    def forward(self, phase, quantized):
        return self.net(torch.cat((phase, quantized), dim=1))


class DiscoVQPAE(nn.Module):
    """Local, state-dict-compatible implementation of released VQ_AE."""

    def __init__(self, config=VQPAEConfig()):
        super().__init__()
        self.config = config
        self.input_channels = config.input_channels
        self.embedding_channels = config.embedding_channels
        self.time_range = config.time_range
        self.window = config.window_seconds
        self.vq_only = config.vq_only

        # VQ_AE inherits these unused modules from the released PAE base class.
        # Keeping them makes official-format state dicts strict-loadable instead
        # of reproducing the upstream loader's broad strict=False behavior.
        inherited_channels = int(config.input_channels / 3)
        inherited_padding = int((config.time_range - 1) / 2)
        self.conv1 = nn.Conv1d(
            config.input_channels,
            inherited_channels,
            config.time_range,
            padding=inherited_padding,
        )
        self.norm1 = TemporalLayerNorm(config.time_range)
        self.conv2 = nn.Conv1d(
            inherited_channels,
            config.embedding_channels,
            config.time_range,
            padding=inherited_padding,
        )
        self.deconv1 = nn.Conv1d(
            config.embedding_channels,
            inherited_channels,
            config.time_range,
            padding=inherited_padding,
        )
        self.denorm1 = TemporalLayerNorm(config.time_range)
        self.deconv2 = nn.Conv1d(
            inherited_channels,
            config.input_channels,
            config.time_range,
            padding=inherited_padding,
        )

        encoder = []
        for index, (dilation, kernel) in enumerate(
            zip(config.enc_dilation_rates, config.enc_kernel_sizes)
        ):
            input_channels = (
                config.input_channels if index == 0 else config.intermediate_channels
            )
            output_channels = (
                config.embedding_channels
                if index == len(config.enc_kernel_sizes) - 1
                else config.intermediate_channels
            )
            encoder.append(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel,
                    padding="same",
                    dilation=dilation,
                )
            )
            if index != len(config.enc_kernel_sizes) - 1:
                encoder.extend((TemporalLayerNorm(config.time_range), nn.ELU()))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for index, (dilation, kernel) in enumerate(
            zip(config.dec_dilation_rates, config.dec_kernel_sizes)
        ):
            input_channels = (
                config.embedding_channels if index == 0 else config.intermediate_channels
            )
            output_channels = (
                config.input_channels
                if index == len(config.dec_kernel_sizes) - 1
                else config.intermediate_channels
            )
            decoder.append(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel,
                    padding="same",
                    dilation=dilation,
                )
            )
            if index != len(config.dec_kernel_sizes) - 1:
                decoder.extend((TemporalLayerNorm(config.time_range), nn.ELU()))
        self.decoder = nn.Sequential(*decoder)

        self.phase_ffn = nn.Sequential(
            nn.LayerNorm(config.embedding_channels),
            nn.Linear(config.embedding_channels, config.embedding_channels * 2),
            nn.ELU(),
            nn.LayerNorm(config.embedding_channels * 2),
            nn.Linear(config.embedding_channels * 2, config.embedding_channels),
            nn.ELU(),
        )
        self.vq_ffn = nn.Sequential(
            nn.LayerNorm(config.embedding_channels),
            nn.Linear(config.embedding_channels, config.embedding_channels * 2),
            nn.ELU(),
            nn.LayerNorm(config.embedding_channels * 2),
            nn.Linear(config.embedding_channels * 2, config.embedding_channels),
            nn.ELU(),
        )
        self.quantizer = ResidualVectorQuantize(
            config.embedding_channels,
            config.n_codebooks,
            config.codebook_size,
            config.codebook_dim,
            config.quantizer_dropout,
            config.normalize_projected_latents,
        )
        self.fc = nn.ModuleList(
            [nn.Linear(config.time_range, 2) for _ in range(config.embedding_channels)]
        )
        self.latent_combinator = MLPCombiner(config.embedding_channels)
        self.register_buffer(
            "tpi",
            torch.tensor([2.0 * math.pi], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "args",
            torch.linspace(
                -config.window_seconds / 2,
                config.window_seconds / 2,
                config.time_range,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "freqs",
            torch.fft.rfftfreq(config.time_range)[1:]
            * config.time_range
            / config.window_seconds,
            persistent=True,
        )

    def encode(self, waveform):
        waveform = waveform.reshape(
            waveform.shape[0],
            self.input_channels,
            self.time_range,
        )
        return self.encoder(waveform)

    def _fft_parameters(self, values):
        transformed = torch.fft.rfft(values, dim=2)
        spectrum = transformed.abs()[:, :, 1:]
        power = spectrum.square()
        denominator = power.sum(dim=2).clamp_min(torch.finfo(power.dtype).tiny)
        frequency = (self.freqs * power).sum(dim=2) / denominator
        amplitude = 2 * denominator.sqrt() / self.time_range
        offset = transformed.real[:, :, 0] / self.time_range
        return frequency, amplitude, offset

    def forward(self, waveform, n_quantizers=None):
        latent = self.encode(waveform)
        z_pae = latent
        z_vq_input = self.vq_ffn(latent.permute(0, 2, 1)).permute(0, 2, 1)
        z_vq, codes, latents, commitment, codebook = self.quantizer(
            z_vq_input,
            n_quantizers,
        )
        z_combined = z_vq
        z_recon = None
        phase_parameters = None
        if not self.vq_only:
            z_phase = self.phase_ffn(latent.permute(0, 2, 1)).permute(0, 2, 1)
            frequency, amplitude, offset = self._fft_parameters(z_phase)
            phase = torch.empty_like(frequency)
            for channel, projection in enumerate(self.fc):
                vector = projection(z_phase[:, channel])
                phase[:, channel] = torch.atan2(vector[:, 1], vector[:, 0]) / self.tpi
            phase = phase.unsqueeze(2)
            frequency = frequency.unsqueeze(2)
            amplitude = amplitude.unsqueeze(2)
            offset = offset.unsqueeze(2)
            phase_parameters = (phase, frequency, amplitude, offset)
            z_recon = amplitude * torch.sin(
                self.tpi * (frequency * self.args + phase)
            ) + offset
            z_combined = self.latent_combinator(z_recon, z_vq)
        reconstructed = self.decoder(z_combined).reshape(
            waveform.shape[0],
            self.input_channels * self.time_range,
        )
        return {
            "audio": reconstructed[..., : self.time_range],
            "z_pae": z_pae,
            "z_vq": z_vq,
            "z_recon": z_recon,
            "phase_params": phase_parameters,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment,
            "vq/codebook_loss": codebook,
        }

    def condition_features(self, waveform, output_tokens=MUSIC_FEATURE_TOKENS, variant="z_pae"):
        output = self(waveform)
        if variant == "z_pae":
            continuous = output["z_pae"]
        elif variant == "z_recon":
            continuous = output["z_recon"]
        else:
            raise ValueError("variant must be z_pae or z_recon")
        if continuous is None:
            raise ValueError(f"{variant} is unavailable for a VQ-only model")
        discrete = F.adaptive_avg_pool1d(output["z_vq"], int(output_tokens))
        continuous = F.adaptive_avg_pool1d(continuous, int(output_tokens))
        return torch.cat((discrete, continuous), dim=1).transpose(1, 2)


def load_vqpae_state_dict_strict(model, checkpoint):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    stripped = {
        key[6:] if key.startswith("model.") else key: value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(stripped, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "VQ-PAE checkpoint/config mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model


def vqpae_model_from_checkpoint(checkpoint):
    """Instantiate the exact extractor semantics recorded by a local checkpoint."""
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("local VQ-PAE checkpoint is missing model_config")
    if model_config.get("normalize_projected_latents") is not True:
        raise ValueError("VQ-PAE checkpoint is not the reviewed spherical-RVQ route")
    if checkpoint.get("rvq_stability") != SPHERICAL_RVQ_STABILITY_CONTRACT:
        raise ValueError("VQ-PAE checkpoint RVQ stability contract mismatch")
    selection = checkpoint.get("selection")
    if not isinstance(selection, dict) or not all(
        selection.get(key) is True
        for key in (
            "selected_for_downstream",
            "eligible",
            "finite",
            "noncollapsed",
        )
    ):
        raise ValueError("VQ-PAE checkpoint is not an eligible selected checkpoint")
    if checkpoint.get("best_epoch") != checkpoint.get("next_epoch"):
        raise ValueError("VQ-PAE selected checkpoint epoch marker mismatch")
    validation = checkpoint.get("validation_metrics")
    if not isinstance(validation, dict) or not math.isfinite(
        float(validation.get("composite", float("nan")))
    ):
        raise ValueError("VQ-PAE selected checkpoint has invalid validation metrics")
    if any(not math.isfinite(float(value)) for value in validation.values()):
        raise ValueError("VQ-PAE selected checkpoint has nonfinite validation metrics")
    for index in range(int(model_config.get("n_codebooks", 0))):
        usage = float(validation.get(f"codebook/{index}/usage", float("nan")))
        perplexity = float(
            validation.get(f"codebook/{index}/perplexity", float("nan"))
        )
        if (
            not math.isfinite(usage)
            or not math.isfinite(perplexity)
            or usage < VQPAE_MIN_CODEBOOK_USAGE
            or perplexity < VQPAE_MIN_CODEBOOK_PERPLEXITY
        ):
            raise ValueError("VQ-PAE checkpoint fails the global non-collapse gate")
    names = {field.name for field in fields(VQPAEConfig)}
    values = {key: value for key, value in model_config.items() if key in names}
    for key in (
        "enc_dilation_rates",
        "enc_kernel_sizes",
        "dec_dilation_rates",
        "dec_kernel_sizes",
    ):
        if key in values:
            values[key] = tuple(values[key])
    model = DiscoVQPAE(VQPAEConfig(**values))
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or any(
        torch.is_tensor(value)
        and value.is_floating_point()
        and not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("VQ-PAE selected checkpoint has nonfinite model state")
    norm_atol = float(SPHERICAL_RVQ_STABILITY_CONTRACT["resume_norm_atol"])
    for index in range(int(model_config.get("n_codebooks", 0))):
        key = f"quantizer.quantizers.{index}.codebook.weight"
        weight = state.get(key, state.get(f"model.{key}"))
        if not torch.is_tensor(weight) or not torch.allclose(
            weight.norm(dim=1),
            torch.ones(weight.shape[0], dtype=weight.dtype, device=weight.device),
            atol=norm_atol,
            rtol=0.0,
        ):
            raise ValueError("VQ-PAE selected checkpoint violates unit codebooks")
    return load_vqpae_state_dict_strict(model, checkpoint)


def _mono_float32(samples):
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1:
        raise ValueError("audio samples must be mono or [samples,channels]")
    if np.issubdtype(samples.dtype, np.integer):
        scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
        samples = samples.astype(np.float32) / scale
    else:
        samples = samples.astype(np.float32, copy=False)
    if not np.isfinite(samples).all():
        raise ValueError("audio contains non-finite samples")
    return samples


def trailing_audio_window(
    samples,
    source_sample_rate,
    target_start_sample,
    target_sample_rate,
    history_seconds=MUSIC_HISTORY_SECONDS,
):
    """Return a fixed trailing window without reading after target_start_sample."""

    samples = _mono_float32(samples)
    source_sample_rate = int(source_sample_rate)
    target_sample_rate = int(target_sample_rate)
    target_start_sample = int(target_start_sample)
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    if target_start_sample < 0 or target_start_sample > samples.shape[0]:
        raise ValueError("target_start_sample is outside the source audio")
    source_length = int(round(float(history_seconds) * source_sample_rate))
    start = target_start_sample - source_length
    available = samples[max(start, 0) : target_start_sample]
    if start < 0:
        available = np.pad(available, (-start, 0))
    if available.shape[0] != source_length:
        raise RuntimeError("failed to construct the exact source-rate history window")

    ratio = Fraction(target_sample_rate, source_sample_rate)
    resampled = scipy.signal.resample_poly(
        available,
        ratio.numerator,
        ratio.denominator,
        padtype="constant",
    ).astype(np.float32, copy=False)
    target_length = int(round(float(history_seconds) * target_sample_rate))
    if resampled.shape[0] < target_length:
        resampled = np.pad(resampled, (target_length - resampled.shape[0], 0))
    elif resampled.shape[0] > target_length:
        resampled = resampled[-target_length:]
    return np.ascontiguousarray(resampled)


class FeatureNormalizer:
    def __init__(self, mean, std, variance_floor=1e-6):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.variance_floor = float(variance_floor)
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ValueError("mean/std must be same-shaped vectors")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("normalizer contains non-finite values")
        if np.any(self.std < self.variance_floor):
            raise ValueError("normalizer std is below the frozen floor")

    @classmethod
    def fit(cls, arrays, variance_floor=1e-6):
        count = 0
        total = None
        total_square = None
        for array in arrays:
            flat = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
            if not np.isfinite(flat).all():
                raise ValueError("cannot fit normalizer on non-finite features")
            if total is None:
                total = flat.sum(axis=0)
                total_square = np.square(flat).sum(axis=0)
            else:
                total += flat.sum(axis=0)
                total_square += np.square(flat).sum(axis=0)
            count += flat.shape[0]
        if not count:
            raise ValueError("cannot fit normalizer without observations")
        mean = total / count
        variance = np.maximum(total_square / count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), float(variance_floor))
        return cls(mean.astype(np.float32), std.astype(np.float32), variance_floor)

    @classmethod
    def from_manifest(cls, payload):
        expected = payload.get("digest")
        if expected is not None:
            unsigned = {
                key: value
                for key, value in payload.items()
                if key == "mean" or key == "std" or key == "variance_floor"
            }
            if _manifest_digest(unsigned) != expected:
                raise ValueError("normalizer manifest digest mismatch")
        return cls(
            payload["mean"],
            payload["std"],
            payload.get("variance_floor", 1e-6),
        )

    def normalize(self, values):
        if torch.is_tensor(values):
            mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
            std = torch.as_tensor(self.std, device=values.device, dtype=values.dtype)
            return (values - mean) / std
        values = np.asarray(values)
        return (values - self.mean) / self.std

    def manifest(self):
        payload = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "variance_floor": self.variance_floor,
        }
        payload["digest"] = _manifest_digest(payload)
        return payload


def build_music_carrier(features, normalizer, variant):
    if variant not in ("vqpae_z_pae", "vqpae_z_recon", "librosa35", "null"):
        raise ValueError("unknown music feature variant")
    if variant == "null":
        if torch.is_tensor(features):
            return features.new_zeros(*features.shape[:-1], MUSIC_CARRIER_DIM)
        return np.zeros((*np.asarray(features).shape[:-1], MUSIC_CARRIER_DIM), np.float32)
    normalized = normalizer.normalize(features)
    expected = 16 if variant.startswith("vqpae") else 35
    if normalized.shape[-1] != expected:
        raise ValueError(f"{variant} requires {expected} feature channels")
    if expected == MUSIC_CARRIER_DIM:
        return normalized
    padding_shape = (*normalized.shape[:-1], MUSIC_CARRIER_DIM - expected)
    if torch.is_tensor(normalized):
        padding = normalized.new_zeros(padding_shape)
        return torch.cat((normalized, padding), dim=-1)
    return np.concatenate(
        (normalized, np.zeros(padding_shape, dtype=normalized.dtype)),
        axis=-1,
    )


def deterministic_source_split(source_ids, validation_count=18, strata=None):
    """Freeze a stable fit/validation split without depending on file order."""

    source_ids = sorted({str(source_id) for source_id in source_ids})
    validation_count = int(validation_count)
    if validation_count <= 0 or validation_count >= len(source_ids):
        raise ValueError("validation_count must leave non-empty fit and validation sets")
    if strata is None:
        strata = {source_id: "__all__" for source_id in source_ids}
        split_version = "df_vqpae_source_hash_v1"
    else:
        strata = {str(key): str(value) for key, value in strata.items()}
        if set(strata) != set(source_ids):
            raise ValueError("strata must assign every source exactly once")
        split_version = "df_vqpae_stratified_source_hash_v1"

    grouped = {}
    for source_id in source_ids:
        grouped.setdefault(strata[source_id], []).append(source_id)
    exact_quotas = {
        label: validation_count * len(ids) / len(source_ids)
        for label, ids in grouped.items()
    }
    quotas = {label: int(math.floor(value)) for label, value in exact_quotas.items()}
    remainder = validation_count - sum(quotas.values())
    remainder_order = sorted(
        grouped,
        key=lambda label: (
            -(exact_quotas[label] - quotas[label]),
            hashlib.sha256(
                f"{DF_VQPAE_BASELINE_ID}:stratum:{label}".encode("utf-8")
            ).hexdigest(),
        ),
    )
    for label in remainder_order[:remainder]:
        quotas[label] += 1

    validation = []
    for label, ids in sorted(grouped.items()):
        ranked = sorted(
            ids,
            key=lambda source_id: hashlib.sha256(
                f"{DF_VQPAE_BASELINE_ID}:{source_id}".encode("utf-8")
            ).hexdigest(),
        )
        validation.extend(ranked[: quotas[label]])
    validation = sorted(validation)
    validation_set = set(validation)
    fit = [source_id for source_id in source_ids if source_id not in validation_set]
    payload = {
        "split_version": split_version,
        "fit": fit,
        "validation": validation,
        "strata": {source_id: strata[source_id] for source_id in source_ids},
    }
    payload["digest"] = _manifest_digest(payload)
    return payload


def c4_planning_boundaries(
    num_source_samples,
    source_sample_rate,
    *,
    motion_fps=30,
    commit_frames=8,
):
    """Return source-sample boundaries for the H8/C4 replanning cadence."""

    num_source_samples = int(num_source_samples)
    source_sample_rate = int(source_sample_rate)
    motion_fps = int(motion_fps)
    commit_frames = int(commit_frames)
    if min(num_source_samples, source_sample_rate, motion_fps, commit_frames) <= 0:
        raise ValueError("boundary inputs must be positive")
    count = int(
        math.floor(
            Fraction(num_source_samples * motion_fps, source_sample_rate)
            / commit_frames
        )
    )
    boundaries = [
        int(
            Fraction(index * commit_frames * source_sample_rate, motion_fps)
        )
        for index in range(count + 1)
    ]
    boundaries = np.asarray(boundaries, dtype=np.int64)
    return boundaries[boundaries <= num_source_samples]


class CausalLibrosa35:
    def __init__(self, config=CausalAudioConfig()):
        self.config = config

    @staticmethod
    def _last_frames(values, count, feature_axis=-1):
        length = values.shape[feature_axis]
        if length < count:
            padding = [(0, 0)] * values.ndim
            padding[feature_axis] = (count - length, 0)
            values = np.pad(values, padding)
        selection = [slice(None)] * values.ndim
        selection[feature_axis] = slice(values.shape[feature_axis] - count, None)
        return values[tuple(selection)]

    def extract_window(self, waveform):
        import librosa

        waveform = _mono_float32(waveform)
        expected = int(self.config.history_seconds * self.config.librosa_sample_rate)
        if waveform.shape != (expected,):
            raise ValueError(f"Librosa window must contain exactly {expected} samples")
        hop = self.config.librosa_hop_length
        sample_rate = self.config.librosa_sample_rate
        count = self.config.feature_tokens
        envelope = librosa.onset.onset_strength(
            y=waveform,
            sr=sample_rate,
            hop_length=hop,
        )
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=20,
            hop_length=hop,
        )
        chroma = librosa.feature.chroma_cens(
            y=waveform,
            sr=sample_rate,
            hop_length=hop,
            n_chroma=12,
        )
        envelope = self._last_frames(envelope, count)
        mfcc = self._last_frames(mfcc, count)
        chroma = self._last_frames(chroma, count)
        peaks = librosa.onset.onset_detect(
            onset_envelope=envelope,
            sr=sample_rate,
            hop_length=hop,
        )
        peak_onehot = np.zeros(count, dtype=np.float32)
        peak_onehot[np.asarray(peaks, dtype=np.int64)] = 1.0
        try:
            _, beats = librosa.beat.beat_track(
                onset_envelope=envelope,
                sr=sample_rate,
                hop_length=hop,
                start_bpm=120.0,
                tightness=100,
            )
        except Exception:
            beats = np.empty(0, dtype=np.int64)
        beat_onehot = np.zeros(count, dtype=np.float32)
        beats = np.asarray(beats, dtype=np.int64)
        beat_onehot[beats[(beats >= 0) & (beats < count)]] = 1.0
        features = np.concatenate(
            (
                envelope[:, None],
                mfcc.T,
                chroma.T,
                peak_onehot[:, None],
                beat_onehot[:, None],
            ),
            axis=1,
        )
        if features.shape != (count, 35) or not np.isfinite(features).all():
            raise RuntimeError("Librosa35 extraction produced an invalid tensor")
        return features.astype(np.float32, copy=False)


class SourceConditionStore:
    """Small source-level condition store with digest-bound manifests."""

    def __init__(self, root):
        self.root = Path(root)

    def source_paths(self, variant, source_id):
        source = self.root / str(variant) / str(source_id)
        return {
            "features": source.with_suffix(".features.npy"),
            "boundaries": source.with_suffix(".boundaries.npy"),
            "metadata": source.with_suffix(".json"),
        }

    def write(self, variant, source_id, boundaries, features, metadata):
        paths = self.source_paths(variant, source_id)
        boundaries = np.asarray(boundaries, dtype=np.int64)
        features = np.asarray(features)
        if boundaries.ndim != 1 or features.shape[0] != boundaries.shape[0]:
            raise ValueError("condition boundaries/features do not align")
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("condition boundaries must be strictly increasing")
        if not np.isfinite(features).all():
            raise ValueError("condition features contain non-finite values")
        paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **metadata,
            "variant": str(variant),
            "source_id": str(source_id),
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "boundary_count": int(boundaries.shape[0]),
            "boundary_first": int(boundaries[0]) if boundaries.size else None,
            "boundary_last": int(boundaries[-1]) if boundaries.size else None,
            "feature_sha256": hashlib.sha256(
                np.ascontiguousarray(features).tobytes()
            ).hexdigest(),
            "boundary_sha256": hashlib.sha256(boundaries.tobytes()).hexdigest(),
        }
        payload["manifest_digest"] = _manifest_digest(payload)
        temporary_paths = {}
        try:
            for key, values in (
                ("features", features),
                ("boundaries", boundaries),
            ):
                descriptor, temporary = tempfile.mkstemp(
                    dir=paths[key].parent,
                    prefix=f".{paths[key].name}.",
                    suffix=".tmp",
                )
                os.close(descriptor)
                temporary_paths[key] = Path(temporary)
                with temporary_paths[key].open("wb") as handle:
                    np.save(handle, values)
            descriptor, temporary = tempfile.mkstemp(
                dir=paths["metadata"].parent,
                prefix=f".{paths['metadata'].name}.",
                suffix=".tmp",
            )
            os.close(descriptor)
            temporary_paths["metadata"] = Path(temporary)
            temporary_paths["metadata"].write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary_paths["features"].replace(paths["features"])
            temporary_paths["boundaries"].replace(paths["boundaries"])
            temporary_paths["metadata"].replace(paths["metadata"])
        finally:
            for temporary in temporary_paths.values():
                temporary.unlink(missing_ok=True)
        return payload

    def read(self, variant, source_id, mmap_mode="r", verify_content=True):
        paths = self.source_paths(variant, source_id)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        expected = metadata.pop("manifest_digest")
        if _manifest_digest(metadata) != expected:
            raise ValueError("condition-store manifest digest mismatch")
        metadata["manifest_digest"] = expected
        boundaries = np.load(paths["boundaries"], mmap_mode=mmap_mode)
        features = np.load(paths["features"], mmap_mode=mmap_mode)
        if features.shape[0] != boundaries.shape[0]:
            raise ValueError("condition-store arrays no longer align")
        if verify_content:
            feature_sha = hashlib.sha256(
                np.ascontiguousarray(features).tobytes()
            ).hexdigest()
            boundary_sha = hashlib.sha256(
                np.ascontiguousarray(boundaries).tobytes()
            ).hexdigest()
            if feature_sha != metadata["feature_sha256"]:
                raise ValueError("condition-store feature digest mismatch")
            if boundary_sha != metadata["boundary_sha256"]:
                raise ValueError("condition-store boundary digest mismatch")
        return boundaries, features, metadata
