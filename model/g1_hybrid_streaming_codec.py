from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from dataset.g1_streaming_state import (
    G1_STREAMING_STATE_DIM,
    LEGACY_S66,
    resolve_boundary_state_spec,
    transition_features_from_plan,
)
from model.g1_streaming_rvqvae import (
    STREAMING_MODULATED_ROUTE,
    G1StreamingRVQVAE,
)
from model.g1_structural_quantizer import (
    STRUCTURAL_QUANTIZER_BSQ,
    STRUCTURAL_QUANTIZER_FSQ,
    STRUCTURAL_QUANTIZER_VQ,
    STRUCTURAL_QUANTIZERS,
    build_structural_quantizer,
)


HYBRID_CODEC_PLAIN_ROUTE = "C1_singleVQ_contres_plain"
HYBRID_CODEC_COMMIT_SAFE_ROUTE = "C2_commit_safe_singleVQ_contres"
HYBRID_CODEC_CLEAN_REPAIR_ROUTE = "R1_C2_clean_objective_repair"
HYBRID_CODEC_CLEAN_REBUILD_ROUTE = "R2_B1000_clean_singleVQ_rebuild"
HYBRID_CODEC_DETAIL_ADAPTER_ROUTE = "R3_C2_structure_detail_adapter"
HYBRID_CODEC_VOCAB_PRESERVING_ROUTE = "P1_C2_vocab_preserving_residual_adaptation"
HYBRID_CODEC_V6F_Z_CLEAN_ROUTE = "V6f_Z_clean_representation"
HYBRID_CODEC_ROUTES = (
    HYBRID_CODEC_PLAIN_ROUTE,
    HYBRID_CODEC_COMMIT_SAFE_ROUTE,
    HYBRID_CODEC_CLEAN_REPAIR_ROUTE,
    HYBRID_CODEC_CLEAN_REBUILD_ROUTE,
    HYBRID_CODEC_DETAIL_ADAPTER_ROUTE,
    HYBRID_CODEC_VOCAB_PRESERVING_ROUTE,
    HYBRID_CODEC_V6F_Z_CLEAN_ROUTE,
)

CLEAN_OBJECTIVE_ROUTES = (
    HYBRID_CODEC_CLEAN_REPAIR_ROUTE,
    HYBRID_CODEC_CLEAN_REBUILD_ROUTE,
    HYBRID_CODEC_DETAIL_ADAPTER_ROUTE,
)


@dataclass(frozen=True)
class G1HybridStreamingCodecConfig:
    hybrid_route: str = HYBRID_CODEC_PLAIN_ROUTE
    backbone_route: str = STREAMING_MODULATED_ROUTE
    input_dim: int = 34
    state_dim: int = G1_STREAMING_STATE_DIM
    state_layout: str = LEGACY_S66.name
    plan_frames: int = 16
    hidden_dim: int = 256
    code_dim: int = 256
    num_blocks: int = 4
    codebook_size: int = 512
    commitment_weight: float = 0.25
    structural_quantizer: str = STRUCTURAL_QUANTIZER_VQ
    fsq_levels: tuple = (8, 8, 8)
    bsq_bits: int = 9
    bsq_inv_temperature: float = 1.0
    bsq_dataset_entropy_weight: float = 0.1
    bsq_dataset_entropy_gamma: float = 1.0
    bsq_per_sample_entropy_gamma: float = 0.0
    bsq_commitment_beta: float = 0.0
    dropout: float = 0.0
    use_attention: bool = True
    predict_contact: bool = True

    def asdict(self):
        return {
            "model_type": "g1_hybrid_streaming_codec",
            "hybrid_route": self.hybrid_route,
            "route": self.backbone_route,
            "motion_format": "g1_yaw_delta",
            "input_dim": int(self.input_dim),
            "state_dim": int(self.state_dim),
            "state_layout": self.state_layout,
            "plan_frames": int(self.plan_frames),
            "token_horizon": int(self.plan_frames // 2),
            "commit_tokens": 4,
            "temporal_downsample": 2,
            "hidden_dim": int(self.hidden_dim),
            "code_dim": int(self.code_dim),
            "num_blocks": int(self.num_blocks),
            "num_codebooks": 1,
            "codebook_size": int(self.codebook_size),
            "commitment_weight": float(self.commitment_weight),
            "structural_quantizer": self.structural_quantizer,
            "fsq_levels": list(self.fsq_levels),
            "bsq_bits": int(self.bsq_bits),
            "bsq_inv_temperature": float(self.bsq_inv_temperature),
            "bsq_dataset_entropy_weight": float(
                self.bsq_dataset_entropy_weight
            ),
            "bsq_dataset_entropy_gamma": float(
                self.bsq_dataset_entropy_gamma
            ),
            "bsq_per_sample_entropy_gamma": float(
                self.bsq_per_sample_entropy_gamma
            ),
            "bsq_commitment_beta": float(self.bsq_commitment_beta),
            "dropout": float(self.dropout),
            "use_attention": bool(self.use_attention),
            "predict_contact": bool(self.predict_contact),
            "state_required": True,
            "intra_plan_attention": "bidirectional",
            "residual_type": "continuous_quantization_residual",
            "detail_adapter": self.hybrid_route
            == HYBRID_CODEC_DETAIL_ADAPTER_ROUTE,
            "public_interface_frozen": self.hybrid_route
            == HYBRID_CODEC_VOCAB_PRESERVING_ROUTE,
            "vocabulary_preservation": (
                "frozen_C2_soft_assignment_KL"
                if self.hybrid_route == HYBRID_CODEC_VOCAB_PRESERVING_ROUTE
                else "none"
            ),
        }


class G1HybridStreamingCodec(G1StreamingRVQVAE):
    """Single-token structural codec with an explicit continuous residual."""

    def __init__(
        self,
        hybrid_route=HYBRID_CODEC_PLAIN_ROUTE,
        backbone_route=STREAMING_MODULATED_ROUTE,
        input_dim=34,
        state_dim=G1_STREAMING_STATE_DIM,
        state_layout=None,
        plan_frames=16,
        hidden_dim=256,
        code_dim=256,
        num_blocks=4,
        codebook_size=512,
        commitment_weight=0.25,
        structural_quantizer=STRUCTURAL_QUANTIZER_VQ,
        fsq_levels=(8, 8, 8),
        bsq_bits=9,
        bsq_inv_temperature=1.0,
        bsq_dataset_entropy_weight=0.1,
        bsq_dataset_entropy_gamma=1.0,
        bsq_per_sample_entropy_gamma=0.0,
        bsq_commitment_beta=0.0,
        dropout=0.0,
        use_attention=True,
        predict_contact=True,
    ):
        if hybrid_route not in HYBRID_CODEC_ROUTES:
            raise ValueError(f"hybrid_route must be one of {HYBRID_CODEC_ROUTES}")
        if structural_quantizer not in STRUCTURAL_QUANTIZERS:
            raise ValueError(
                f"structural_quantizer must be one of {STRUCTURAL_QUANTIZERS}"
            )
        state_spec = resolve_boundary_state_spec(state_layout, state_dim=state_dim)
        super().__init__(
            route=backbone_route,
            input_dim=input_dim,
            state_dim=state_dim,
            state_layout=state_spec.name,
            plan_frames=plan_frames,
            hidden_dim=hidden_dim,
            code_dim=code_dim,
            num_blocks=num_blocks,
            num_codebooks=1,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            dropout=dropout,
            use_attention=use_attention,
            predict_contact=predict_contact,
        )
        if structural_quantizer != STRUCTURAL_QUANTIZER_VQ:
            self.quantizer = build_structural_quantizer(
                structural_quantizer,
                dim=self.config.code_dim,
                codebook_size=self.config.codebook_size,
                commitment_weight=self.config.commitment_weight,
                fsq_levels=fsq_levels,
                bsq_bits=bsq_bits,
                bsq_inv_temperature=bsq_inv_temperature,
                bsq_dataset_entropy_weight=bsq_dataset_entropy_weight,
                bsq_dataset_entropy_gamma=bsq_dataset_entropy_gamma,
                bsq_per_sample_entropy_gamma=bsq_per_sample_entropy_gamma,
                bsq_commitment_beta=bsq_commitment_beta,
            )
        self.hybrid_config = G1HybridStreamingCodecConfig(
            hybrid_route=str(hybrid_route),
            backbone_route=str(backbone_route),
            input_dim=int(input_dim),
            state_dim=int(state_dim),
            state_layout=state_spec.name,
            plan_frames=int(plan_frames),
            hidden_dim=int(hidden_dim),
            code_dim=int(code_dim),
            num_blocks=int(num_blocks),
            codebook_size=int(codebook_size),
            commitment_weight=float(commitment_weight),
            structural_quantizer=str(structural_quantizer),
            fsq_levels=tuple(int(level) for level in fsq_levels),
            bsq_bits=int(bsq_bits),
            bsq_inv_temperature=float(bsq_inv_temperature),
            bsq_dataset_entropy_weight=float(bsq_dataset_entropy_weight),
            bsq_dataset_entropy_gamma=float(bsq_dataset_entropy_gamma),
            bsq_per_sample_entropy_gamma=float(bsq_per_sample_entropy_gamma),
            bsq_commitment_beta=float(bsq_commitment_beta),
            dropout=float(dropout),
            use_attention=bool(use_attention),
            predict_contact=bool(predict_contact),
        )
        self.detail_residual_projection = None
        self.detail_state_projection = None
        self.detail_upsample = None
        self.detail_output = None
        self.detail_contact = None
        if self.hybrid_route == HYBRID_CODEC_DETAIL_ADAPTER_ROUTE:
            self.detail_residual_projection = nn.Conv1d(
                self.config.code_dim,
                self.config.hidden_dim,
                kernel_size=1,
            )
            self.detail_state_projection = nn.Linear(
                self.config.state_dim,
                self.config.hidden_dim,
            )
            self.detail_upsample = nn.Sequential(
                nn.SiLU(),
                nn.ConvTranspose1d(
                    self.config.hidden_dim,
                    self.config.hidden_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.SiLU(),
            )
            self.detail_output = nn.Conv1d(
                self.config.hidden_dim,
                self.config.input_dim,
                kernel_size=1,
            )
            self.detail_contact = nn.Conv1d(
                self.config.hidden_dim,
                2,
                kernel_size=1,
            )
            nn.init.zeros_(self.detail_output.weight)
            nn.init.zeros_(self.detail_output.bias)
            nn.init.zeros_(self.detail_contact.weight)
            nn.init.zeros_(self.detail_contact.bias)

    @property
    def hybrid_route(self):
        return self.hybrid_config.hybrid_route

    def encode_components(self, motion, state):
        pre_quant = self.encode(motion, state=state)
        quantized = self.quantizer(pre_quant)
        q0_raw = quantized["quantized_raw"]
        if self.hybrid_route == HYBRID_CODEC_VOCAB_PRESERVING_ROUTE:
            # Stage 2 treats the C2 codebook as a frozen public interface. The
            # continuous path receives the only reconstruction gradient; using
            # the quantizer's straight-through value here would count the same
            # encoder gradient twice in q0 + (z - q0).
            q0_latent = q0_raw.detach()
        else:
            q0_latent = quantized["quantized"]
        residual = pre_quant - q0_raw.detach()
        return {
            "pre_quant": pre_quant,
            "q0_ids": quantized["indices"],
            "q0_latent": q0_latent,
            "q0_raw": q0_raw,
            "residual": residual,
            "rvq_loss": quantized["rvq_loss"],
            "codebook_loss": quantized["codebook_loss"],
            "commitment_loss": quantized["commitment_loss"],
            "quantization_error_by_layer": quantized[
                "quantization_error_by_layer"
            ],
            "final_residual_mse": quantized["final_residual_mse"],
            "quantizer_diagnostics": quantized.get(
                "quantizer_diagnostics", {}
            ),
        }

    def encode_clean_components(self, motion, state):
        """Build the single-gradient D+C latent branches used by V6f-Z."""
        pre_quant = self.encode(motion, state=state)
        quantized = self.quantizer(pre_quant)
        q0_raw = quantized["quantized_raw"]
        q0_st = pre_quant + (q0_raw - pre_quant).detach()
        residual = pre_quant - q0_raw.detach()
        return {
            "pre_quant": pre_quant,
            "q0_ids": quantized["indices"],
            "q0_st": q0_st,
            "q0_raw": q0_raw,
            "residual": residual,
            "full_latent": q0_raw.detach() + residual,
            "rvq_loss": quantized["rvq_loss"],
            "codebook_loss": quantized["codebook_loss"],
            "commitment_loss": quantized["commitment_loss"],
        }

    def forward_clean(self, motion, state, robust_residual=None):
        components = self.encode_clean_components(motion, state)
        full_recon, full_contact = self.decode(
            components["full_latent"], state=state
        )
        q0_recon, q0_contact = self.decode(components["q0_st"], state=state)
        output = {
            **components,
            "recon": full_recon,
            "contact_logits": full_contact,
            "q0_recon": q0_recon,
            "q0_contact_logits": q0_contact,
        }
        if robust_residual is not None:
            robust_recon, robust_contact = self.decode(
                components["q0_raw"].detach() + robust_residual,
                state=state,
            )
            output["robust_recon"] = robust_recon
            output["robust_contact_logits"] = robust_contact
        return output

    def decode_components(self, q0_latent, residual, state, target_frames=None):
        if q0_latent.shape != residual.shape:
            raise ValueError("q0_latent and residual must have identical shape")
        if self.hybrid_route == HYBRID_CODEC_DETAIL_ADAPTER_ROUTE:
            structure, structure_contact = self.decode(
                q0_latent,
                state=state,
                target_frames=target_frames,
            )
            detail_hidden = self.detail_residual_projection(
                residual.detach().transpose(1, 2)
            )
            detail_hidden = detail_hidden + self.detail_state_projection(
                state
            ).unsqueeze(-1)
            detail_hidden = self.detail_upsample(detail_hidden)
            detail = self.detail_output(detail_hidden).transpose(1, 2)
            detail_contact = self.detail_contact(detail_hidden).transpose(1, 2)
            frames = self.config.plan_frames if target_frames is None else int(target_frames)
            return (
                structure.detach() + detail[:, :frames],
                structure_contact.detach() + detail_contact[:, :frames],
            )
        return self.decode(
            q0_latent + residual,
            state=state,
            target_frames=target_frames,
        )

    def decode_structure(self, q0_ids, state, target_frames=None):
        q0_ids = torch.as_tensor(q0_ids, device=state.device, dtype=torch.long)
        if q0_ids.ndim == 2:
            q0_ids = q0_ids.unsqueeze(-1)
        if q0_ids.ndim != 3 or q0_ids.shape[-1] != 1:
            raise ValueError("q0_ids must have shape [B,H] or [B,H,1]")
        q0_latent = self.quantizer.lookup(q0_ids)
        return self.decode(q0_latent, state=state, target_frames=target_frames)

    @staticmethod
    def derive_commit_state(decoded_raw_motion, start_state, fps=30.0):
        if decoded_raw_motion.ndim != 3 or decoded_raw_motion.shape[-1] != 34:
            raise ValueError("decoded_raw_motion must have shape [B,T,34]")
        state_spec = resolve_boundary_state_spec(state_dim=start_state.shape[-1])
        if start_state.shape != (decoded_raw_motion.shape[0], state_spec.dim):
            raise ValueError(f"start_state must have shape [B,{state_spec.dim}]")
        transition = transition_features_from_plan(
            start_state,
            decoded_raw_motion,
            fps=fps,
        )
        velocity = transition["velocity"][:, -1]
        acceleration = transition["acceleration"][:, -1]
        final_motion = decoded_raw_motion[:, -1]
        next_state = torch.cat(
            (
                final_motion[:, 2:3],
                velocity[:, :4],
                final_motion[:, 5:34],
                velocity[:, 4:33],
                acceleration[:, :2],
                acceleration[:, 3:4],
            ),
            dim=-1,
        )
        return next_state[:, : state_spec.dim]

    def forward(
        self,
        motion,
        state,
        residual_keep=None,
        residual_noise=None,
        components=None,
        decode_q0=True,
    ):
        if components is None:
            components = self.encode_components(motion, state=state)
        residual = components["residual"]
        if residual_keep is not None:
            residual_keep = torch.as_tensor(
                residual_keep,
                device=residual.device,
                dtype=residual.dtype,
            )
            if residual_keep.ndim == 1:
                residual_keep = residual_keep[:, None, None]
            if residual_keep.shape != (residual.shape[0], 1, 1):
                raise ValueError("residual_keep must have shape [B] or [B,1,1]")
            residual = residual * residual_keep
        if residual_noise is not None:
            residual_noise = torch.as_tensor(
                residual_noise,
                device=residual.device,
                dtype=residual.dtype,
            )
            if residual_noise.shape != residual.shape:
                raise ValueError("residual_noise must match residual shape")
            residual = residual + residual_noise

        recon, contact_logits = self.decode_components(
            components["q0_latent"],
            residual,
            state=state,
        )
        if decode_q0:
            q0_recon, q0_contact_logits = self.decode(
                components["q0_latent"],
                state=state,
            )
        else:
            q0_recon = None
            q0_contact_logits = None
        return {
            "recon": recon,
            "contact_logits": contact_logits,
            "q0_recon": q0_recon,
            "q0_contact_logits": q0_contact_logits,
            "latent": components["q0_latent"] + residual,
            "q0_latent": components["q0_latent"],
            "residual": components["residual"],
            "used_residual": residual,
            "pre_quant": components["pre_quant"],
            "quantized_raw": components["q0_raw"],
            "token_indices": components["q0_ids"],
            "rvq_loss": components["rvq_loss"],
            "codebook_loss": components["codebook_loss"],
            "commitment_loss": components["commitment_loss"],
            "quantization_error_by_layer": components[
                "quantization_error_by_layer"
            ],
            "final_residual_mse": components["final_residual_mse"],
            "quantizer_diagnostics": components["quantizer_diagnostics"],
            "mu": None,
            "logvar": None,
        }

    def manifest(self):
        manifest = self.hybrid_config.asdict()
        manifest["quantizer"] = self.quantizer.quantizer_manifest()
        return manifest

    def freeze_public_interface(self):
        if self.hybrid_route != HYBRID_CODEC_VOCAB_PRESERVING_ROUTE:
            raise ValueError(
                "public-interface freezing is only defined for the "
                "vocabulary-preserving Stage-2 route"
            )
        frozen_prefixes = (
            "quantizer.",
            "decoder_in.",
            "decoder_state_proj.",
            "decoder_blocks.",
            "decoder_attention.",
            "upsample.",
            "decoder_post_blocks.",
            "output_proj.",
            "contact_head.",
        )
        frozen = []
        trainable = []
        for name, parameter in self.named_parameters():
            if name.startswith(frozen_prefixes):
                parameter.requires_grad_(False)
                frozen.append(name)
            else:
                parameter.requires_grad_(True)
                trainable.append(name)
        if not frozen or not trainable:
            raise RuntimeError("failed to partition the Stage-2 public interface")
        return {
            "frozen": sorted(frozen),
            "trainable": sorted(trainable),
        }


def q0_assignment_logits(pre_quant, codebook, temperature):
    if pre_quant.ndim != 3:
        raise ValueError("pre_quant must have shape [B,H,D]")
    if codebook.ndim == 3:
        if codebook.shape[0] != 1:
            raise ValueError("q0 assignment expects a single codebook")
        codebook = codebook[0]
    if codebook.ndim != 2 or codebook.shape[-1] != pre_quant.shape[-1]:
        raise ValueError("codebook must have shape [K,D] matching pre_quant")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    squared_distance = (
        pre_quant.float().square().sum(dim=-1, keepdim=True)
        - 2.0 * torch.matmul(pre_quant.float(), codebook.float().t())
        + codebook.float().square().sum(dim=-1).view(1, 1, -1)
    )
    return -squared_distance / temperature


def q0_assignment_kl(student_pre_quant, teacher_pre_quant, codebook, temperature):
    teacher_logits = q0_assignment_logits(
        teacher_pre_quant.detach(),
        codebook.detach(),
        temperature,
    )
    student_logits = q0_assignment_logits(
        student_pre_quant,
        codebook.detach(),
        temperature,
    )
    teacher_probability = teacher_logits.softmax(dim=-1)
    per_token = torch.nn.functional.kl_div(
        student_logits.log_softmax(dim=-1),
        teacher_probability,
        reduction="none",
    ).sum(dim=-1)
    loss = per_token.mean()
    agreement = student_logits.argmax(dim=-1).eq(
        teacher_logits.argmax(dim=-1)
    ).float().mean()
    teacher_top1 = teacher_probability.max(dim=-1).values
    return loss, {
        "vocab/hard_q0_agreement": agreement.detach(),
        "vocab/teacher_top1_probability_mean": teacher_top1.mean().detach(),
        "vocab/teacher_top1_probability_median": teacher_top1.median().detach(),
        "vocab/kl_per_valid_token": loss.detach(),
    }


def initialize_hybrid_codec_from_b1000(model, checkpoint):
    config = checkpoint.get("config", {})
    if config.get("model_type") != "g1_streaming_rvqvae":
        raise ValueError("warm start must be a native streaming RVQ-VAE checkpoint")
    if int(config.get("num_codebooks", 0)) < 1:
        raise ValueError("warm-start codec does not contain q0")
    source = checkpoint["model"]
    target = model.state_dict()
    loaded = []
    for name, value in target.items():
        if name.startswith("detail_"):
            continue
        if name == "quantizer.codebooks":
            source_value = source[name][:1]
        else:
            if name not in source:
                raise KeyError(f"warm-start checkpoint is missing {name}")
            source_value = source[name]
        if source_value.shape != value.shape:
            raise ValueError(
                f"warm-start shape mismatch for {name}: "
                f"source={tuple(source_value.shape)} target={tuple(value.shape)}"
            )
        target[name] = source_value.detach().clone()
        loaded.append(name)
    model.load_state_dict(target, strict=True)
    return loaded


def initialize_hybrid_codec_from_hybrid_checkpoint(model, checkpoint):
    config = checkpoint.get("config", {})
    if config.get("model_type") != "g1_hybrid_streaming_codec":
        raise ValueError("initialization checkpoint is not a hybrid streaming codec")
    source = checkpoint["model"]
    target = model.state_dict()
    loaded = []
    for name, value in target.items():
        if name.startswith("detail_") and name not in source:
            continue
        if name not in source:
            raise KeyError(f"hybrid initialization checkpoint is missing {name}")
        if source[name].shape != value.shape:
            raise ValueError(
                f"hybrid initialization shape mismatch for {name}: "
                f"source={tuple(source[name].shape)} target={tuple(value.shape)}"
            )
        target[name] = source[name].detach().clone()
        loaded.append(name)
    model.load_state_dict(target, strict=True)
    return loaded


def build_g1_hybrid_streaming_codec_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {})
    if config.get("model_type") != "g1_hybrid_streaming_codec":
        raise ValueError("checkpoint is not a G1 hybrid streaming codec")
    model = G1HybridStreamingCodec(
        hybrid_route=config["hybrid_route"],
        backbone_route=config.get("route", STREAMING_MODULATED_ROUTE),
        input_dim=int(config.get("input_dim", 34)),
        state_dim=int(config.get("state_dim", G1_STREAMING_STATE_DIM)),
        state_layout=config.get("state_layout", LEGACY_S66.name),
        plan_frames=int(config.get("plan_frames", 16)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        code_dim=int(config.get("code_dim", 256)),
        num_blocks=int(config.get("num_blocks", 4)),
        codebook_size=int(config.get("codebook_size", 512)),
        commitment_weight=float(config.get("commitment_weight", 0.25)),
        dropout=float(config.get("dropout", 0.0)),
        use_attention=bool(config.get("use_attention", True)),
        structural_quantizer=config.get(
            "structural_quantizer", STRUCTURAL_QUANTIZER_VQ
        ),
        fsq_levels=tuple(config.get("fsq_levels", (8, 8, 8))),
        bsq_bits=int(config.get("bsq_bits", 9)),
        bsq_inv_temperature=float(config.get("bsq_inv_temperature", 1.0)),
        bsq_dataset_entropy_weight=float(
            config.get("bsq_dataset_entropy_weight", 0.1)
        ),
        bsq_dataset_entropy_gamma=float(
            config.get("bsq_dataset_entropy_gamma", 1.0)
        ),
        bsq_per_sample_entropy_gamma=float(
            config.get("bsq_per_sample_entropy_gamma", 0.0)
        ),
        bsq_commitment_beta=float(config.get("bsq_commitment_beta", 0.0)),
        predict_contact=bool(config.get("predict_contact", True)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
