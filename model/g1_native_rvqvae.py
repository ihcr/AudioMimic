from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.motion_representation import (
    G1_YAW_DELTA_MOTION_FORMAT,
    motion_repr_dim,
    validate_motion_format,
)
from model.g1_motion_prior import (
    G1MotionPriorLossWeights,
    ResidualConv1dBlock,
    compute_g1_motion_prior_losses,
)
from model.g1_motion_tokenizer import ResidualVectorQuantizer, summarize_codebook_usage


NATIVE_RVQVAE_VARIANTS = (
    "R0_minimal_rvqvae",
    "R1_robot_feas",
    "R2_robot_feas_amp",
    "R3_chunk2_rvqvae",
    "R4_frame_temporal_decoder",
    "R5_rvq8_layers",
    "R6_fsq_baseline",
    "R8_stream_state_h8c8_rvq8",
    "R9_stream_state_h8c4_rvq8",
    "R10_stream_state_h15c8_rvq8",
)


@dataclass(frozen=True)
class G1NativeRVQVAELossWeights:
    motion: float = 1.0
    velocity: float = 0.5
    acceleration: float = 0.1
    fk: float = 0.0
    contact_bce: float = 0.0
    contact_height: float = 0.0
    contact_slide: float = 0.0
    rvq: float = 1.0
    amplitude: float = 0.0
    amplitude_ratio_floor: float = 0.85
    amplitude_ratio_ceiling: float = 1.5


class FiniteScalarQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        fsq_dim=16,
        levels=8,
        commitment_weight=0.25,
    ):
        super().__init__()
        self.dim = int(dim)
        self.fsq_dim = int(fsq_dim)
        self.levels = int(levels)
        self.commitment_weight = float(commitment_weight)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.fsq_dim <= 0:
            raise ValueError("fsq_dim must be positive")
        if self.levels <= 1:
            raise ValueError("levels must be greater than 1")
        self.to_fsq = nn.Linear(self.dim, self.fsq_dim)
        self.from_fsq = nn.Linear(self.fsq_dim, self.dim)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must have shape [B, N, {self.dim}]")
        continuous = torch.tanh(self.to_fsq(tokens))
        scaled = (continuous + 1.0) * 0.5 * float(self.levels - 1)
        indices = torch.round(scaled).clamp(0, self.levels - 1).to(torch.long)
        quantized_scalar = indices.to(continuous.dtype) / float(self.levels - 1) * 2.0 - 1.0
        quantized_scalar_st = continuous + (quantized_scalar - continuous).detach()
        quantized = self.from_fsq(quantized_scalar_st)
        scalar_error = (continuous.detach() - quantized_scalar.detach()).pow(2).mean(dim=(0, 1))
        codebook_loss = F.mse_loss(quantized, tokens.detach())
        commitment_loss = F.mse_loss(continuous, quantized_scalar.detach())
        rvq_loss = codebook_loss + self.commitment_weight * commitment_loss
        return {
            "quantized": quantized,
            "quantized_raw": quantized.detach(),
            "indices": indices,
            "rvq_loss": rvq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "quantization_error_by_layer": scalar_error,
            "final_residual_mse": F.mse_loss(quantized.detach(), tokens.detach()),
        }


class G1NativeRVQVAE(nn.Module):
    def __init__(
        self,
        input_dim=None,
        hidden_dim=256,
        code_dim=256,
        temporal_downsample=2,
        num_codebooks=4,
        codebook_size=512,
        commitment_weight=0.25,
        dropout=0.0,
        motion_format=G1_YAW_DELTA_MOTION_FORMAT,
        quantizer_type="rvq",
        use_attention=False,
        decoder_extra_blocks=0,
        decoder_attention=False,
        fsq_dim=16,
        fsq_levels=8,
        state_conditioning=False,
        state_context_dropout=0.0,
    ):
        super().__init__()
        self.motion_format = validate_motion_format(motion_format)
        self.input_dim = int(input_dim or motion_repr_dim(self.motion_format))
        self.hidden_dim = int(hidden_dim)
        self.code_dim = int(code_dim)
        self.temporal_downsample = int(temporal_downsample)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)
        self.dropout = float(dropout)
        self.quantizer_type = str(quantizer_type)
        self.use_attention = bool(use_attention)
        self.decoder_extra_blocks = int(decoder_extra_blocks)
        self.decoder_attention_enabled = bool(decoder_attention)
        self.fsq_dim = int(fsq_dim)
        self.fsq_levels = int(fsq_levels)
        self.state_conditioning = bool(state_conditioning)
        self.state_context_dim = self.input_dim * 2
        self.state_context_dropout = float(state_context_dropout)
        if self.temporal_downsample not in (2, 4):
            raise ValueError("temporal_downsample must be 2 or 4 for V6f-A native RVQ-VAE")
        if self.quantizer_type not in ("rvq", "fsq"):
            raise ValueError("quantizer_type must be rvq or fsq")
        if self.decoder_extra_blocks < 0:
            raise ValueError("decoder_extra_blocks must be non-negative")

        self.input_proj = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.encoder_pre = nn.Sequential(
            ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout),
            ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout),
        )
        self.downsample = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1)
        self.encoder_post = nn.Sequential(
            ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout),
            ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout),
        )
        if self.use_attention:
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=4,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder_attention = nn.TransformerEncoder(layer, num_layers=1)
        else:
            self.encoder_attention = None
        if self.temporal_downsample == 4:
            self.extra_downsample = nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
                ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout),
            )
        else:
            self.extra_downsample = nn.Identity()
        self.code_proj = nn.Conv1d(self.hidden_dim, self.code_dim, kernel_size=1)
        if self.quantizer_type == "rvq":
            self.quantizer = ResidualVectorQuantizer(
                dim=self.code_dim,
                num_codebooks=self.num_codebooks,
                codebook_size=self.codebook_size,
                commitment_weight=self.commitment_weight,
            )
        else:
            self.quantizer = FiniteScalarQuantizer(
                dim=self.code_dim,
                fsq_dim=self.fsq_dim,
                levels=self.fsq_levels,
                commitment_weight=self.commitment_weight,
            )

        self.decoder_in = nn.Conv1d(self.code_dim, self.hidden_dim, kernel_size=1)
        self.decoder_pre = nn.Sequential(
            ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout),
            ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout),
        )
        if self.decoder_attention_enabled:
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=4,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.decoder_attention = nn.TransformerEncoder(decoder_layer, num_layers=1)
        else:
            self.decoder_attention = None
        if self.temporal_downsample == 4:
            self.extra_upsample = nn.ConvTranspose1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        else:
            self.extra_upsample = nn.Identity()
        self.upsample = nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1)
        self.decoder_post = nn.Sequential(
            ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout),
            ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout),
        )
        self.decoder_extra = nn.Sequential(
            *[
                ResidualConv1dBlock(self.hidden_dim, dropout=self.dropout)
                for _ in range(self.decoder_extra_blocks)
            ]
        )
        if self.state_conditioning:
            self.state_context_proj = nn.Sequential(
                nn.Linear(self.state_context_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.state_context_proj = None
        self.output_proj = nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=1)
        self.contact_head = nn.Conv1d(self.hidden_dim, 2, kernel_size=1)

    def encode(self, motion):
        x = motion.transpose(1, 2)
        h = self.input_proj(x)
        h = self.encoder_pre(h)
        h = self.downsample(h)
        h = self.encoder_post(h)
        h = self.extra_downsample(h)
        if self.encoder_attention is not None:
            h = self.encoder_attention(h.transpose(1, 2)).transpose(1, 2)
        return self.code_proj(h).transpose(1, 2)

    def decode(self, quantized, state_context=None, target_frames=None):
        h = self.decoder_in(quantized.transpose(1, 2))
        h = self.decoder_pre(h)
        if self.decoder_attention is not None:
            h = self.decoder_attention(h.transpose(1, 2)).transpose(1, 2)
        h = self.extra_upsample(h)
        h = self.upsample(h)
        if self.state_context_proj is not None:
            if state_context is None:
                state_context = quantized.new_zeros(quantized.shape[0], self.state_context_dim)
            if state_context.ndim != 2 or state_context.shape[-1] != self.state_context_dim:
                raise ValueError(
                    f"state_context must have shape [B, {self.state_context_dim}], "
                    f"got {tuple(state_context.shape)}"
                )
            state_context = state_context.to(device=h.device, dtype=h.dtype)
            if self.training and self.state_context_dropout > 0.0:
                keep = torch.rand(
                    state_context.shape[0],
                    1,
                    device=state_context.device,
                    dtype=state_context.dtype,
                ) >= self.state_context_dropout
                state_context = state_context * keep
            h = h + self.state_context_proj(state_context).unsqueeze(-1)
        h = self.decoder_post(h)
        h = self.decoder_extra(h)
        recon = self.output_proj(h).transpose(1, 2)
        contact_logits = self.contact_head(h).transpose(1, 2)
        recon, contact_logits = self._match_target_frames(recon, contact_logits, target_frames)
        return recon, contact_logits

    def _match_target_frames(self, recon, contact_logits, target_frames):
        if target_frames is None or recon.shape[1] == int(target_frames):
            return recon, contact_logits
        target_frames = int(target_frames)
        if recon.shape[1] > target_frames:
            return recon[:, :target_frames], contact_logits[:, :target_frames]
        pad_frames = target_frames - recon.shape[1]
        recon = torch.cat([recon, recon[:, -1:].expand(-1, pad_frames, -1)], dim=1)
        contact_logits = torch.cat(
            [contact_logits, contact_logits[:, -1:].expand(-1, pad_frames, -1)],
            dim=1,
        )
        return recon, contact_logits

    def decode_token_indices(self, indices, active_layers=None, target_frames=150):
        if self.quantizer_type != "rvq":
            raise ValueError("decode_token_indices is only supported for RVQ tokenizers")
        quantized = self.quantizer.indices_to_quantized(indices, active_layers=active_layers)
        recon, contact_logits = self.decode(quantized, target_frames=target_frames)
        recon, contact_logits = self._match_target_frames(recon, contact_logits, target_frames)
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.long, device=quantized.device)
        else:
            indices = indices.to(device=quantized.device, dtype=torch.long)
        return {
            "recon": recon,
            "contact_logits": contact_logits,
            "latent": quantized,
            "quantized_raw": quantized,
            "token_indices": indices,
            "active_layers": list(range(self.num_codebooks)) if active_layers is None else list(active_layers),
        }

    def forward(self, motion, state_context=None):
        pre_quant = self.encode(motion)
        quantized = self.quantizer(pre_quant)
        recon, contact_logits = self.decode(
            quantized["quantized"],
            state_context=state_context,
            target_frames=motion.shape[1],
        )
        return {
            "recon": recon,
            "contact_logits": contact_logits,
            "latent": quantized["quantized"],
            "pre_quant": pre_quant,
            "quantized_raw": quantized["quantized_raw"],
            "token_indices": quantized["indices"],
            "rvq_loss": quantized["rvq_loss"],
            "codebook_loss": quantized["codebook_loss"],
            "commitment_loss": quantized["commitment_loss"],
            "quantization_error_by_layer": quantized["quantization_error_by_layer"],
            "final_residual_mse": quantized["final_residual_mse"],
            "mu": None,
            "logvar": None,
        }


def build_g1_native_rvqvae_from_checkpoint(checkpoint, motion_format=None):
    config = checkpoint.get("config", {})
    motion_format = motion_format or config.get("motion_format", G1_YAW_DELTA_MOTION_FORMAT)
    model = G1NativeRVQVAE(
        input_dim=motion_repr_dim(motion_format),
        hidden_dim=int(config.get("hidden_dim", 256)),
        code_dim=int(config.get("code_dim", config.get("hidden_dim", 256))),
        temporal_downsample=int(config.get("temporal_downsample", 2)),
        num_codebooks=int(config.get("num_codebooks", 4)),
        codebook_size=int(config.get("codebook_size", 512)),
        commitment_weight=float(config.get("commitment_weight", 0.25)),
        dropout=float(config.get("dropout", 0.0)),
        motion_format=motion_format,
        quantizer_type=config.get("quantizer_type", "rvq"),
        use_attention=bool(config.get("use_attention", False)),
        decoder_extra_blocks=int(config.get("decoder_extra_blocks", 0)),
        decoder_attention=bool(config.get("decoder_attention", False)),
        fsq_dim=int(config.get("fsq_dim", 16)),
        fsq_levels=int(config.get("fsq_levels", 8)),
        state_conditioning=bool(config.get("state_conditioning", False)),
        state_context_dropout=float(config.get("state_context_dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model"])
    return model


def loss_weights_for_objective_family(
    objective_family,
    motion=1.0,
    velocity=0.5,
    acceleration=0.1,
    fk=0.5,
    contact_bce=0.1,
    contact_height=0.2,
    contact_slide=0.1,
    rvq=1.0,
    amplitude=0.2,
    amplitude_ratio_floor=0.85,
    amplitude_ratio_ceiling=1.5,
):
    if objective_family == "R0_minimal_rvqvae":
        return G1NativeRVQVAELossWeights(
            motion=motion,
            velocity=velocity,
            acceleration=acceleration,
            fk=0.0,
            contact_bce=0.0,
            contact_height=0.0,
            contact_slide=0.0,
            rvq=rvq,
            amplitude=0.0,
            amplitude_ratio_floor=amplitude_ratio_floor,
            amplitude_ratio_ceiling=amplitude_ratio_ceiling,
        )
    if objective_family == "R1_robot_feas":
        return G1NativeRVQVAELossWeights(
            motion=motion,
            velocity=velocity,
            acceleration=acceleration,
            fk=fk,
            contact_bce=contact_bce,
            contact_height=contact_height,
            contact_slide=contact_slide,
            rvq=rvq,
            amplitude=0.0,
            amplitude_ratio_floor=amplitude_ratio_floor,
            amplitude_ratio_ceiling=amplitude_ratio_ceiling,
        )
    if objective_family == "R2_robot_feas_amp":
        return G1NativeRVQVAELossWeights(
            motion=motion,
            velocity=velocity,
            acceleration=acceleration,
            fk=fk,
            contact_bce=contact_bce,
            contact_height=contact_height,
            contact_slide=contact_slide,
            rvq=rvq,
            amplitude=amplitude,
            amplitude_ratio_floor=amplitude_ratio_floor,
            amplitude_ratio_ceiling=amplitude_ratio_ceiling,
        )
    if objective_family in (
        "R3_chunk2_rvqvae",
        "R4_frame_temporal_decoder",
        "R5_rvq8_layers",
        "R6_fsq_baseline",
        "R8_stream_state_h8c8_rvq8",
        "R9_stream_state_h8c4_rvq8",
        "R10_stream_state_h15c8_rvq8",
    ):
        return G1NativeRVQVAELossWeights(
            motion=motion,
            velocity=velocity,
            acceleration=acceleration,
            fk=fk,
            contact_bce=contact_bce,
            contact_height=contact_height,
            contact_slide=contact_slide,
            rvq=rvq,
            amplitude=amplitude,
            amplitude_ratio_floor=amplitude_ratio_floor,
            amplitude_ratio_ceiling=amplitude_ratio_ceiling,
        )
    raise ValueError(f"Unknown objective family: {objective_family}")


def _amplitude_guard_loss(recon, target, ratio_floor=0.85, ratio_ceiling=1.5):
    if recon.shape[1] <= 1:
        zero = recon.sum() * 0.0
        return zero, {"amplitude_energy_ratio": zero.detach(), "amplitude_range_ratio": zero.detach()}
    recon_velocity = recon[:, 1:] - recon[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    recon_energy = recon_velocity.pow(2).mean(dim=(1, 2))
    target_energy = target_velocity.pow(2).mean(dim=(1, 2))
    recon_range = (recon.max(dim=1).values - recon.min(dim=1).values).pow(2).mean(dim=1)
    target_range = (target.max(dim=1).values - target.min(dim=1).values).pow(2).mean(dim=1)
    floor = float(ratio_floor)
    ceiling = float(ratio_ceiling)
    eps = 1e-4
    energy_ratio = recon_energy / torch.clamp(target_energy, min=eps)
    range_ratio = recon_range / torch.clamp(target_range, min=eps)
    energy_loss = F.relu(floor - energy_ratio).pow(2).mean()
    range_loss = F.relu(floor - range_ratio).pow(2).mean()
    energy_ceiling_loss = F.relu(energy_ratio - ceiling).pow(2).mean()
    range_ceiling_loss = F.relu(range_ratio - ceiling).pow(2).mean()
    stats = {
        "amplitude_energy_ratio": energy_ratio.mean().detach(),
        "amplitude_range_ratio": range_ratio.mean().detach(),
        "amplitude_energy_upper_violation": F.relu(energy_ratio - ceiling).mean().detach(),
        "amplitude_range_upper_violation": F.relu(range_ratio - ceiling).mean().detach(),
    }
    return energy_loss + range_loss + 0.25 * (energy_ceiling_loss + range_ceiling_loss), stats


def compute_g1_native_rvqvae_losses(
    output,
    target_motion,
    target_contact,
    mean,
    std,
    kinematics=None,
    motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    weights=G1NativeRVQVAELossWeights(),
    ground=None,
    target_fk_keypoints=None,
    target_fk_feet=None,
):
    prior_weights = G1MotionPriorLossWeights(
        motion=weights.motion,
        velocity=weights.velocity,
        acceleration=weights.acceleration,
        fk=weights.fk,
        contact_bce=weights.contact_bce,
        contact_height=weights.contact_height,
        contact_slide=weights.contact_slide,
        kl=0.0,
    )
    base_loss, stats = compute_g1_motion_prior_losses(
        output,
        target_motion,
        target_contact,
        mean,
        std,
        kinematics=kinematics,
        motion_format=motion_format,
        weights=prior_weights,
        ground=ground,
        target_fk_keypoints=target_fk_keypoints,
        target_fk_feet=target_fk_feet,
    )
    amp_loss, amp_stats = _amplitude_guard_loss(
        output["recon"],
        target_motion,
        ratio_floor=weights.amplitude_ratio_floor,
        ratio_ceiling=weights.amplitude_ratio_ceiling,
    )
    total = base_loss + float(weights.rvq) * output["rvq_loss"] + float(weights.amplitude) * amp_loss
    stats = dict(stats)
    stats.update(
        {
            "loss/total": total.detach(),
            "loss/rvq": output["rvq_loss"].detach(),
            "loss/codebook": output["codebook_loss"].detach(),
            "loss/commitment": output["commitment_loss"].detach(),
            "loss/amplitude": amp_loss.detach(),
            "quant/final_residual_mse": output["final_residual_mse"].detach(),
        }
    )
    for layer, value in enumerate(output["quantization_error_by_layer"]):
        stats[f"quant/layer_{layer}_mse"] = value.detach()
    stats.update({f"amplitude/{key}": value for key, value in amp_stats.items()})
    return total, stats


def summarize_native_tokenizer_output(output, codebook_size):
    return {
        "codebook_usage": summarize_codebook_usage(
            output["token_indices"],
            codebook_size=codebook_size,
        ),
        "quantization_error_by_layer": [
            float(value) for value in output["quantization_error_by_layer"].detach().cpu().reshape(-1)
        ],
        "final_residual_mse": float(output["final_residual_mse"].detach().cpu()),
    }
