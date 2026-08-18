from dataclasses import dataclass

import torch
import torch.nn as nn

from dataset.g1_streaming_state import (
    G1_STREAMING_STATE_DIM,
    LEGACY_S66,
    resolve_boundary_state_spec,
)
from dataset.motion_representation import G1_YAW_DELTA_MOTION_FORMAT, motion_repr_dim
from model.g1_motion_prior import ResidualConv1dBlock
from model.g1_motion_tokenizer import ResidualVectorQuantizer, summarize_codebook_usage


STREAMING_ADDITIVE_ROUTE = "A_shared_additive"
STREAMING_MODULATED_ROUTE = "B_deep_modulated"
STREAMING_STATE_ROUTES = (STREAMING_ADDITIVE_ROUTE, STREAMING_MODULATED_ROUTE)


@dataclass(frozen=True)
class G1StreamingRVQVAEConfig:
    route: str = STREAMING_ADDITIVE_ROUTE
    input_dim: int = 34
    state_dim: int = G1_STREAMING_STATE_DIM
    state_layout: str = LEGACY_S66.name
    plan_frames: int = 16
    hidden_dim: int = 256
    code_dim: int = 256
    num_blocks: int = 4
    num_codebooks: int = 8
    codebook_size: int = 512
    commitment_weight: float = 0.25
    dropout: float = 0.0
    use_attention: bool = True
    predict_contact: bool = True

    def asdict(self):
        return {
            "model_type": "g1_streaming_rvqvae",
            "route": self.route,
            "motion_format": G1_YAW_DELTA_MOTION_FORMAT,
            "input_dim": int(self.input_dim),
            "state_dim": int(self.state_dim),
            "state_layout": self.state_layout,
            "plan_frames": int(self.plan_frames),
            "token_horizon": int(self.plan_frames // 2),
            "temporal_downsample": 2,
            "hidden_dim": int(self.hidden_dim),
            "code_dim": int(self.code_dim),
            "num_blocks": int(self.num_blocks),
            "num_codebooks": int(self.num_codebooks),
            "codebook_size": int(self.codebook_size),
            "commitment_weight": float(self.commitment_weight),
            "dropout": float(self.dropout),
            "use_attention": bool(self.use_attention),
            "predict_contact": bool(self.predict_contact),
            "state_dropout": 0.0,
            "state_required": True,
            "intra_plan_attention": "bidirectional",
        }


class StateModulatedResidualConv1dBlock(nn.Module):
    def __init__(self, channels, state_dim, dropout=0.0):
        super().__init__()
        channels = int(channels)
        if channels % 8 != 0:
            raise ValueError("channels must be divisible by 8")
        self.norm1 = nn.GroupNorm(8, channels, affine=False)
        self.norm2 = nn.GroupNorm(8, channels, affine=False)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(float(dropout))
        self.activation = nn.SiLU()
        self.state_affine = nn.Sequential(
            nn.Linear(int(state_dim), channels),
            nn.SiLU(),
            nn.Linear(channels, channels * 4),
        )
        nn.init.zeros_(self.state_affine[-1].weight)
        nn.init.zeros_(self.state_affine[-1].bias)

    def forward(self, x, state):
        scale1, shift1, scale2, shift2 = self.state_affine(state).chunk(4, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + scale1.unsqueeze(-1)) + shift1.unsqueeze(-1)
        h = self.conv1(self.activation(h))
        h = self.norm2(h)
        h = h * (1.0 + scale2.unsqueeze(-1)) + shift2.unsqueeze(-1)
        h = self.conv2(self.dropout(self.activation(h)))
        return x + h


class G1StreamingRVQVAE(nn.Module):
    """Plan-local H8/C4 RVQ-VAE with mandatory physical boundary state."""

    def __init__(
        self,
        route=STREAMING_ADDITIVE_ROUTE,
        input_dim=34,
        state_dim=G1_STREAMING_STATE_DIM,
        state_layout=None,
        plan_frames=16,
        hidden_dim=256,
        code_dim=256,
        num_blocks=4,
        num_codebooks=8,
        codebook_size=512,
        commitment_weight=0.25,
        dropout=0.0,
        use_attention=True,
        predict_contact=True,
    ):
        super().__init__()
        state_spec = resolve_boundary_state_spec(state_layout, state_dim=state_dim)
        self.config = G1StreamingRVQVAEConfig(
            route=str(route),
            input_dim=int(input_dim),
            state_dim=int(state_dim),
            state_layout=state_spec.name,
            plan_frames=int(plan_frames),
            hidden_dim=int(hidden_dim),
            code_dim=int(code_dim),
            num_blocks=int(num_blocks),
            num_codebooks=int(num_codebooks),
            codebook_size=int(codebook_size),
            commitment_weight=float(commitment_weight),
            dropout=float(dropout),
            use_attention=bool(use_attention),
            predict_contact=bool(predict_contact),
        )
        if self.config.route not in STREAMING_STATE_ROUTES:
            raise ValueError(f"route must be one of {STREAMING_STATE_ROUTES}")
        if self.config.input_dim != motion_repr_dim(G1_YAW_DELTA_MOTION_FORMAT):
            raise ValueError("native streaming codec is locked to 34D g1_yaw_delta motion")
        if self.config.plan_frames != 16:
            raise ValueError("native streaming codec is locked to a 16-frame H8 plan")
        if self.config.hidden_dim % 8 != 0:
            raise ValueError("hidden_dim must be divisible by 8")
        if self.config.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.input_proj = nn.Conv1d(self.config.input_dim, self.config.hidden_dim, kernel_size=1)
        self.encoder_blocks = self._make_blocks()
        self.downsample = nn.Conv1d(
            self.config.hidden_dim,
            self.config.hidden_dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.encoder_post_blocks = self._make_blocks()
        self.encoder_attention = self._make_attention() if self.config.use_attention else None
        self.code_proj = nn.Conv1d(self.config.hidden_dim, self.config.code_dim, kernel_size=1)
        self.quantizer = ResidualVectorQuantizer(
            dim=self.config.code_dim,
            num_codebooks=self.config.num_codebooks,
            codebook_size=self.config.codebook_size,
            commitment_weight=self.config.commitment_weight,
        )

        self.decoder_in = nn.Conv1d(self.config.code_dim, self.config.hidden_dim, kernel_size=1)
        self.decoder_blocks = self._make_blocks()
        self.decoder_attention = self._make_attention() if self.config.use_attention else None
        self.upsample = nn.ConvTranspose1d(
            self.config.hidden_dim,
            self.config.hidden_dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.decoder_post_blocks = self._make_blocks()
        self.output_proj = nn.Conv1d(self.config.hidden_dim, self.config.input_dim, kernel_size=1)
        self.contact_head = (
            nn.Conv1d(self.config.hidden_dim, 2, kernel_size=1)
            if self.config.predict_contact
            else None
        )

        if self.config.route == STREAMING_ADDITIVE_ROUTE:
            self.encoder_state_proj = self._make_state_projection()
            self.decoder_state_proj = self._make_state_projection()
        else:
            self.encoder_state_proj = None
            self.decoder_state_proj = None

    @property
    def temporal_downsample(self):
        return 2

    @property
    def num_codebooks(self):
        return self.config.num_codebooks

    @property
    def codebook_size(self):
        return self.config.codebook_size

    def _make_state_projection(self):
        return nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

    def _make_blocks(self):
        block_type = (
            StateModulatedResidualConv1dBlock
            if self.config.route == STREAMING_MODULATED_ROUTE
            else ResidualConv1dBlock
        )
        blocks = []
        for _ in range(self.config.num_blocks):
            if block_type is StateModulatedResidualConv1dBlock:
                blocks.append(
                    block_type(
                        self.config.hidden_dim,
                        self.config.state_dim,
                        dropout=self.config.dropout,
                    )
                )
            else:
                blocks.append(block_type(self.config.hidden_dim, dropout=self.config.dropout))
        return nn.ModuleList(blocks)

    def _make_attention(self):
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=4,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=1)

    def _validate_state(self, state, batch_size, reference):
        if state is None:
            raise ValueError(f"{self.config.state_layout} boundary state is required")
        if state.ndim != 2 or state.shape != (batch_size, self.config.state_dim):
            raise ValueError(
                f"state must have shape {(batch_size, self.config.state_dim)}, got {tuple(state.shape)}"
            )
        if not torch.isfinite(state).all():
            raise ValueError("state contains non-finite values")
        return state.to(device=reference.device, dtype=reference.dtype)

    def _run_blocks(self, blocks, hidden, state):
        for block in blocks:
            if self.config.route == STREAMING_MODULATED_ROUTE:
                hidden = block(hidden, state)
            else:
                hidden = block(hidden)
        return hidden

    def encode(self, motion, state):
        if motion.ndim != 3 or motion.shape[1:] != (self.config.plan_frames, self.config.input_dim):
            raise ValueError(
                f"motion must have shape [B,{self.config.plan_frames},{self.config.input_dim}], "
                f"got {tuple(motion.shape)}"
            )
        state = self._validate_state(state, motion.shape[0], motion)
        hidden = self.input_proj(motion.transpose(1, 2))
        if self.encoder_state_proj is not None:
            hidden = hidden + self.encoder_state_proj(state).unsqueeze(-1)
        hidden = self._run_blocks(self.encoder_blocks, hidden, state)
        hidden = self.downsample(hidden)
        hidden = self._run_blocks(self.encoder_post_blocks, hidden, state)
        if self.encoder_attention is not None:
            hidden = self.encoder_attention(hidden.transpose(1, 2)).transpose(1, 2)
        return self.code_proj(hidden).transpose(1, 2)

    def decode(self, quantized, state, target_frames=None):
        if quantized.ndim != 3 or quantized.shape[-1] != self.config.code_dim:
            raise ValueError(f"quantized must have shape [B,H,{self.config.code_dim}]")
        if quantized.shape[1] != self.config.plan_frames // 2:
            raise ValueError("quantized plan must contain exactly H8 tokens")
        state = self._validate_state(state, quantized.shape[0], quantized)
        hidden = self.decoder_in(quantized.transpose(1, 2))
        if self.decoder_state_proj is not None:
            hidden = hidden + self.decoder_state_proj(state).unsqueeze(-1)
        hidden = self._run_blocks(self.decoder_blocks, hidden, state)
        if self.decoder_attention is not None:
            hidden = self.decoder_attention(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.upsample(hidden)
        hidden = self._run_blocks(self.decoder_post_blocks, hidden, state)
        recon = self.output_proj(hidden).transpose(1, 2)
        contact_logits = (
            self.contact_head(hidden).transpose(1, 2)
            if self.contact_head is not None
            else None
        )
        target_frames = self.config.plan_frames if target_frames is None else int(target_frames)
        if target_frames <= 0 or target_frames > self.config.plan_frames:
            raise ValueError(f"target_frames must be in [1,{self.config.plan_frames}]")
        return (
            recon[:, :target_frames],
            None if contact_logits is None else contact_logits[:, :target_frames],
        )

    def decode_token_indices(self, indices, state, active_layers=None, target_frames=None):
        indices = torch.as_tensor(indices, device=state.device, dtype=torch.long)
        if indices.ndim != 3 or indices.shape[1] != self.config.plan_frames // 2:
            raise ValueError("indices must have shape [B,8,L]")
        quantized = self.quantizer.indices_to_quantized(indices, active_layers=active_layers)
        recon, contact_logits = self.decode(quantized, state=state, target_frames=target_frames)
        return {
            "recon": recon,
            "contact_logits": contact_logits,
            "latent": quantized,
            "quantized_raw": quantized,
            "token_indices": indices,
            "active_layers": list(range(self.config.num_codebooks))
            if active_layers is None
            else list(active_layers),
        }

    def forward(self, motion, state):
        pre_quant = self.encode(motion, state=state)
        quantized = self.quantizer(pre_quant)
        recon, contact_logits = self.decode(quantized["quantized"], state=state)
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

    def manifest(self):
        return self.config.asdict()


def build_g1_streaming_rvqvae_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {})
    if config.get("model_type") != "g1_streaming_rvqvae":
        raise ValueError("checkpoint is not a native streaming RVQ-VAE")
    model = G1StreamingRVQVAE(
        route=config["route"],
        input_dim=int(config.get("input_dim", 34)),
        state_dim=int(config.get("state_dim", G1_STREAMING_STATE_DIM)),
        state_layout=config.get("state_layout", LEGACY_S66.name),
        plan_frames=int(config.get("plan_frames", 16)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        code_dim=int(config.get("code_dim", 256)),
        num_blocks=int(config.get("num_blocks", 4)),
        num_codebooks=int(config.get("num_codebooks", 8)),
        codebook_size=int(config.get("codebook_size", 512)),
        commitment_weight=float(config.get("commitment_weight", 0.25)),
        dropout=float(config.get("dropout", 0.0)),
        use_attention=bool(config.get("use_attention", True)),
        predict_contact=bool(config.get("predict_contact", True)),
    )
    model.load_state_dict(checkpoint["model"])
    return model


def summarize_streaming_tokenizer_output(output, codebook_size):
    return {
        "codebook_usage": summarize_codebook_usage(output["token_indices"], codebook_size=codebook_size),
        "quantization_error_by_layer": [
            float(value) for value in output["quantization_error_by_layer"].detach().cpu().reshape(-1)
        ],
        "final_residual_mse": float(output["final_residual_mse"].detach().cpu()),
    }
