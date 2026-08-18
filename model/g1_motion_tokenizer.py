import math

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKENIZER_VARIANTS = (
    "T0_frame_rvq",
    "T1_chunk2_rvq",
    "T2_hier_coarse_plus_frame_residual",
)


def _zero_like(reference):
    return reference.sum() * 0.0


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_codebooks=4,
        codebook_size=512,
        commitment_weight=0.25,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_codebooks <= 0:
            raise ValueError("num_codebooks must be positive")
        if self.codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1")
        scale = 1.0 / math.sqrt(self.dim)
        self.codebooks = nn.Parameter(
            torch.empty(self.num_codebooks, self.codebook_size, self.dim).uniform_(
                -scale,
                scale,
            )
        )

    @property
    def quantizer_type(self):
        return "vq"

    def quantizer_manifest(self):
        return {
            "type": self.quantizer_type,
            "vocabulary_size": self.codebook_size,
            "num_token_streams": self.num_codebooks,
            "commitment_weight": self.commitment_weight,
        }

    def lookup(self, indices):
        """Map packed token IDs to latents through the shared quantizer API."""
        indices = torch.as_tensor(
            indices,
            device=self.codebooks.device,
            dtype=torch.long,
        )
        if indices.ndim == 2 and self.num_codebooks == 1:
            indices = indices.unsqueeze(-1)
        return self.indices_to_quantized(indices)

    def hard_diagnostics(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        if indices.ndim == 2 and self.num_codebooks == 1:
            indices = indices.unsqueeze(-1)
        if indices.ndim != 3 or indices.shape[-1] != self.num_codebooks:
            raise ValueError("indices must have shape [B,N,L]")
        diagnostics = {}
        for layer in range(self.num_codebooks):
            counts = torch.bincount(
                indices[..., layer].reshape(-1).cpu(),
                minlength=self.codebook_size,
            ).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            diagnostics[f"layer_{layer}_normalized_entropy"] = entropy / math.log(
                self.codebook_size
            )
        return diagnostics

    def _nearest_indices(self, residual, layer):
        codebook = self.codebooks[int(layer)]
        distance = (
            residual.pow(2).sum(dim=-1, keepdim=True)
            - 2.0 * torch.matmul(residual, codebook.t())
            + codebook.pow(2).sum(dim=-1).view(1, 1, -1)
        )
        return torch.argmin(distance, dim=-1)

    def indices_to_quantized(self, indices, active_layers=None):
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.long, device=self.codebooks.device)
        else:
            indices = indices.to(device=self.codebooks.device, dtype=torch.long)
        if indices.ndim != 3:
            raise ValueError("indices must have shape [B, N, L]")
        if indices.shape[-1] != self.num_codebooks:
            raise ValueError(
                f"indices last dimension must match num_codebooks={self.num_codebooks}, "
                f"got {indices.shape[-1]}"
            )
        if indices.numel():
            min_index = int(indices.min().detach().cpu())
            max_index = int(indices.max().detach().cpu())
            if min_index < 0 or max_index >= self.codebook_size:
                raise ValueError(
                    f"indices must be in [0, {self.codebook_size - 1}], "
                    f"got min={min_index} max={max_index}"
                )
        if active_layers is None:
            active_layers = range(self.num_codebooks)
        active_layers = [int(layer) for layer in active_layers]
        if any(layer < 0 or layer >= self.num_codebooks for layer in active_layers):
            raise ValueError(f"active_layers must be within [0, {self.num_codebooks - 1}]")
        quantized_sum = torch.zeros(
            indices.shape[0],
            indices.shape[1],
            self.dim,
            device=self.codebooks.device,
            dtype=self.codebooks.dtype,
        )
        for layer in active_layers:
            quantized_sum = quantized_sum + F.embedding(indices[:, :, layer], self.codebooks[layer])
        return quantized_sum

    def quantize_with_fixed_prefix(self, tokens, fixed_indices):
        """Run exact RVQ while forcing one or more leading codebook IDs."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must have shape [B, N, {self.dim}]")
        fixed_indices = torch.as_tensor(
            fixed_indices,
            device=tokens.device,
            dtype=torch.long,
        )
        if fixed_indices.ndim != 3 or fixed_indices.shape[:2] != tokens.shape[:2]:
            raise ValueError("fixed_indices must have shape [B,N,P]")
        fixed_levels = int(fixed_indices.shape[-1])
        if fixed_levels <= 0 or fixed_levels > self.num_codebooks:
            raise ValueError(
                f"fixed prefix depth must be in [1,{self.num_codebooks}]"
            )
        if fixed_indices.numel():
            minimum = int(fixed_indices.min().detach().cpu())
            maximum = int(fixed_indices.max().detach().cpu())
            if minimum < 0 or maximum >= self.codebook_size:
                raise ValueError(
                    f"fixed indices must be in [0,{self.codebook_size - 1}]"
                )

        residual = tokens
        quantized_sum = torch.zeros_like(tokens)
        indices = []
        layer_errors = []
        for layer in range(self.num_codebooks):
            if layer < fixed_levels:
                index = fixed_indices[..., layer]
            else:
                index = self._nearest_indices(residual, layer)
            quantized = F.embedding(index, self.codebooks[layer])
            layer_errors.append((residual - quantized).pow(2).mean())
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            indices.append(index)
        return {
            "quantized_raw": quantized_sum,
            "indices": torch.stack(indices, dim=-1),
            "quantization_error_by_layer": torch.stack(layer_errors),
            "final_residual": residual,
            "final_residual_mse": residual.pow(2).mean(),
            "fixed_prefix_levels": fixed_levels,
        }

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must have shape [B, N, {self.dim}]")
        residual = tokens
        quantized_sum = torch.zeros_like(tokens)
        indices = []
        layer_errors = []
        codebook_losses = []
        commitment_losses = []
        for layer in range(self.num_codebooks):
            codebook = self.codebooks[layer]
            index = self._nearest_indices(residual, layer)
            quantized = F.embedding(index, codebook)
            layer_errors.append(F.mse_loss(residual.detach(), quantized.detach()))
            codebook_losses.append(F.mse_loss(quantized, residual.detach()))
            commitment_losses.append(F.mse_loss(residual, quantized.detach()))
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            indices.append(index)

        quantized_st = tokens + (quantized_sum - tokens).detach()
        codebook_loss = torch.stack(codebook_losses).sum()
        commitment_loss = torch.stack(commitment_losses).sum()
        rvq_loss = codebook_loss + self.commitment_weight * commitment_loss
        return {
            "quantized": quantized_st,
            "quantized_raw": quantized_sum,
            "indices": torch.stack(indices, dim=-1),
            "rvq_loss": rvq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "quantization_error_by_layer": torch.stack(layer_errors),
            "final_residual_mse": residual.detach().pow(2).mean(),
        }


class G1MotionTokenizer(nn.Module):
    def __init__(
        self,
        variant="T0_frame_rvq",
        latent_dim=128,
        latent_frames=75,
        num_codebooks=4,
        codebook_size=512,
        commitment_weight=0.25,
        chunk_size=1,
        residual_levels=0,
    ):
        super().__init__()
        if variant not in TOKENIZER_VARIANTS:
            raise ValueError(f"Unknown tokenizer variant: {variant}")
        if variant == "T2_hier_coarse_plus_frame_residual":
            raise NotImplementedError(
                "T2_hier_coarse_plus_frame_residual is intentionally interface-only "
                "for the first V6e-A PR."
            )
        self.variant = variant
        self.latent_dim = int(latent_dim)
        self.latent_frames = int(latent_frames)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)
        self.residual_levels = int(residual_levels)
        if self.latent_dim <= 0 or self.latent_frames <= 0:
            raise ValueError("latent_dim and latent_frames must be positive")
        if self.residual_levels:
            raise ValueError("residual_levels is reserved for the T2 follow-up and must be 0")
        if self.variant == "T0_frame_rvq":
            if int(chunk_size) != 1:
                raise ValueError("T0_frame_rvq requires chunk_size=1")
            self.chunk_size = 1
        elif self.variant == "T1_chunk2_rvq":
            self.chunk_size = int(chunk_size)
            if self.chunk_size <= 1:
                raise ValueError("T1_chunk2_rvq requires chunk_size > 1")
        self.token_dim = self.latent_dim * self.chunk_size
        self.rvq = ResidualVectorQuantizer(
            dim=self.token_dim,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            commitment_weight=self.commitment_weight,
        )

    def _to_tokens(self, latent):
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent must have shape [B, T, {self.latent_dim}]")
        batch, frames, _ = latent.shape
        pad_frames = (self.chunk_size - frames % self.chunk_size) % self.chunk_size
        if pad_frames:
            pad = latent[:, -1:, :].expand(batch, pad_frames, self.latent_dim)
            latent = torch.cat([latent, pad], dim=1)
        tokens = latent.reshape(batch, latent.shape[1] // self.chunk_size, self.token_dim)
        return tokens, int(frames), int(pad_frames)

    def _from_tokens(self, tokens, original_frames):
        latent = tokens.reshape(tokens.shape[0], tokens.shape[1] * self.chunk_size, self.latent_dim)
        return latent[:, :original_frames, :]

    def forward(self, latent):
        tokens, original_frames, pad_frames = self._to_tokens(latent)
        quantized = self.rvq(tokens)
        recon_latent = self._from_tokens(quantized["quantized"], original_frames)
        return {
            "recon_latent": recon_latent,
            "token_indices": quantized["indices"],
            "rvq_loss": quantized["rvq_loss"],
            "codebook_loss": quantized["codebook_loss"],
            "commitment_loss": quantized["commitment_loss"],
            "quantization_error_by_layer": quantized["quantization_error_by_layer"],
            "final_residual_mse": quantized["final_residual_mse"],
            "pad_frames": pad_frames,
            "num_tokens": int(tokens.shape[1]),
            "token_dim": self.token_dim,
        }


def compute_tokenizer_losses(
    output,
    target_latent,
    reconstruction_weight=1.0,
    velocity_weight=0.25,
    latent_smoothness_weight=0.0,
):
    recon = output["recon_latent"]
    reconstruction = F.mse_loss(recon, target_latent)
    velocity = _zero_like(reconstruction)
    if recon.shape[1] > 1:
        velocity = F.mse_loss(
            recon[:, 1:] - recon[:, :-1],
            target_latent[:, 1:] - target_latent[:, :-1],
        )
    smoothness = _zero_like(reconstruction)
    if recon.shape[1] > 2:
        second_diff = recon[:, 2:] - 2.0 * recon[:, 1:-1] + recon[:, :-2]
        smoothness = second_diff.pow(2).mean()
    total = (
        float(reconstruction_weight) * reconstruction
        + float(velocity_weight) * velocity
        + output["rvq_loss"]
        + float(latent_smoothness_weight) * smoothness
    )
    stats = {
        "loss/total": total.detach(),
        "loss/reconstruction": reconstruction.detach(),
        "loss/velocity": velocity.detach(),
        "loss/latent_smoothness": smoothness.detach(),
        "loss/rvq": output["rvq_loss"].detach(),
        "loss/codebook": output["codebook_loss"].detach(),
        "loss/commitment": output["commitment_loss"].detach(),
        "quant/final_residual_mse": output["final_residual_mse"].detach(),
        "quant/pad_frames": torch.as_tensor(float(output["pad_frames"]), device=target_latent.device),
        "quant/num_tokens": torch.as_tensor(float(output["num_tokens"]), device=target_latent.device),
    }
    for layer, value in enumerate(output["quantization_error_by_layer"]):
        stats[f"quant/layer_{layer}_mse"] = value.detach()
    return total, stats


def summarize_codebook_usage(indices, codebook_size, rare_threshold=0.001):
    if torch.is_tensor(indices):
        indices = indices.detach().cpu()
    indices = torch.as_tensor(indices, dtype=torch.long)
    if indices.ndim != 3:
        raise ValueError("indices must have shape [B, N, L]")
    codebook_size = int(codebook_size)
    summaries = []
    for layer in range(indices.shape[-1]):
        layer_indices = indices[:, :, layer].reshape(-1)
        counts = torch.bincount(layer_indices, minlength=codebook_size).to(torch.float64)
        total = float(counts.sum().item())
        probs = counts / max(total, 1.0)
        nonzero = probs[probs > 0]
        entropy = float(-(nonzero * torch.log(nonzero)).sum().item()) if nonzero.numel() else 0.0
        perplexity = float(math.exp(entropy)) if entropy < 80.0 else float("inf")
        active = int((counts > 0).sum().item())
        rare_cutoff = max(1.0, total * float(rare_threshold))
        rare_mask = (counts > 0) & (counts <= rare_cutoff)
        rare_assignments = float(counts[rare_mask].sum().item())
        transition_entropy = 0.0
        unique_transitions = 0
        if indices.shape[1] > 1:
            prev_codes = indices[:, :-1, layer].reshape(-1)
            next_codes = indices[:, 1:, layer].reshape(-1)
            transitions = prev_codes * codebook_size + next_codes
            transition_counts = torch.bincount(
                transitions,
                minlength=codebook_size * codebook_size,
            ).to(torch.float64)
            transition_probs = transition_counts / max(float(transition_counts.sum().item()), 1.0)
            transition_nonzero = transition_probs[transition_probs > 0]
            transition_entropy = (
                float(-(transition_nonzero * torch.log(transition_nonzero)).sum().item())
                if transition_nonzero.numel()
                else 0.0
            )
            unique_transitions = int((transition_counts > 0).sum().item())
        summaries.append(
            {
                "layer": int(layer),
                "assignments": int(total),
                "active_codes": active,
                "dead_codes": int(codebook_size - active),
                "dead_code_ratio": float((codebook_size - active) / max(codebook_size, 1)),
                "perplexity": perplexity,
                "entropy": entropy,
                "rare_threshold_fraction": float(rare_threshold),
                "rare_token_assignment_rate": float(rare_assignments / max(total, 1.0)),
                "transition_entropy": transition_entropy,
                "unique_transitions": unique_transitions,
                "histogram": [int(value) for value in counts.to(torch.int64).tolist()],
            }
        )
    return summaries


def summarize_codebook_usage_flat(indices, codebook_size, rare_threshold=0.001):
    summaries = summarize_codebook_usage(
        indices,
        codebook_size=codebook_size,
        rare_threshold=rare_threshold,
    )
    flat = {}
    for item in summaries:
        layer = item["layer"]
        for key in (
            "active_codes",
            "dead_codes",
            "dead_code_ratio",
            "perplexity",
            "entropy",
            "rare_token_assignment_rate",
            "transition_entropy",
            "unique_transitions",
        ):
            flat[f"codebook/layer_{layer}_{key}"] = item[key]
    return flat
