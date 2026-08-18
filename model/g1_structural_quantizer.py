"""Packed 512-way structural quantizers for the V6f-Q SQA experiment."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.g1_motion_tokenizer import ResidualVectorQuantizer


STRUCTURAL_QUANTIZER_VQ = "vq"
STRUCTURAL_QUANTIZER_FSQ = "fsq"
STRUCTURAL_QUANTIZER_BSQ = "bsq"
STRUCTURAL_QUANTIZERS = (
    STRUCTURAL_QUANTIZER_VQ,
    STRUCTURAL_QUANTIZER_FSQ,
    STRUCTURAL_QUANTIZER_BSQ,
)


def _zero(reference):
    return reference.sum() * 0.0


def _packed_ids(indices, vocabulary_size, device):
    indices = torch.as_tensor(indices, device=device, dtype=torch.long)
    if indices.ndim == 3:
        if indices.shape[-1] != 1:
            raise ValueError("packed structural IDs must have one token stream")
        indices = indices[..., 0]
    if indices.ndim != 2:
        raise ValueError("packed structural IDs must have shape [B,N] or [B,N,1]")
    if indices.numel():
        minimum = int(indices.min().detach().cpu())
        maximum = int(indices.max().detach().cpu())
        if minimum < 0 or maximum >= int(vocabulary_size):
            raise ValueError(
                f"structural IDs must be in [0,{int(vocabulary_size) - 1}], "
                f"got min={minimum} max={maximum}"
            )
    return indices


def _normalized_entropy(values, cardinality):
    counts = torch.bincount(
        values.reshape(-1).detach().cpu(), minlength=cardinality
    ).float()
    probabilities = counts / counts.sum().clamp_min(1.0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy / math.log(cardinality)


class FiniteScalarStructuralQuantizer(nn.Module):
    """FSQ factors packed into one product-cardinality token."""

    def __init__(self, dim, levels=(8, 8, 8)):
        super().__init__()
        self.dim = int(dim)
        self.levels = tuple(int(level) for level in levels)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if not self.levels or any(level <= 1 for level in self.levels):
            raise ValueError("FSQ levels must all be greater than one")
        self.codebook_size = math.prod(self.levels)
        self.num_codebooks = 1
        self.project_in = nn.Linear(self.dim, len(self.levels))
        self.project_out = nn.Linear(len(self.levels), self.dim)
        basis = [1]
        for level in self.levels[:-1]:
            basis.append(basis[-1] * level)
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))
        self.register_buffer(
            "levels_tensor", torch.tensor(self.levels, dtype=torch.long)
        )

    @property
    def quantizer_type(self):
        return STRUCTURAL_QUANTIZER_FSQ

    def quantizer_manifest(self):
        return {
            "type": self.quantizer_type,
            "vocabulary_size": self.codebook_size,
            "num_token_streams": 1,
            "levels": list(self.levels),
            "projection": f"{self.dim}->{len(self.levels)}->{self.dim}",
            "auxiliary_loss": "none",
        }

    def _digits_to_codes(self, digits, dtype):
        levels = self.levels_tensor.to(device=digits.device, dtype=dtype)
        return 2.0 * digits.to(dtype) / (levels - 1.0) - 1.0

    def _digits_to_ids(self, digits):
        return (digits * self.basis.to(digits.device)).sum(dim=-1)

    def _ids_to_digits(self, indices):
        digits = []
        for basis, level in zip(self.basis.tolist(), self.levels):
            digits.append(
                torch.remainder(
                    torch.div(indices, basis, rounding_mode="floor"), level
                )
            )
        return torch.stack(digits, dim=-1)

    def lookup(self, indices):
        indices = _packed_ids(indices, self.codebook_size, self.basis.device)
        codes = self._digits_to_codes(
            self._ids_to_digits(indices), self.project_out.weight.dtype
        )
        return self.project_out(codes)

    def indices_to_quantized(self, indices, active_layers=None):
        active_layers = None if active_layers is None else tuple(active_layers)
        if active_layers not in (None, (), (0,)):
            raise ValueError("FSQ has one packed token stream")
        if active_layers == ():
            indices = _packed_ids(indices, self.codebook_size, self.basis.device)
            return torch.zeros(
                *indices.shape,
                self.dim,
                device=indices.device,
                dtype=self.project_out.weight.dtype,
            )
        return self.lookup(indices)

    def hard_diagnostics(self, indices):
        indices = _packed_ids(indices, self.codebook_size, self.basis.device)
        digits = self._ids_to_digits(indices)
        result = {
            "packed_normalized_entropy": _normalized_entropy(
                indices, self.codebook_size
            )
        }
        for factor, level in enumerate(self.levels):
            result[f"factor_{factor}_normalized_entropy"] = _normalized_entropy(
                digits[..., factor], level
            )
        return result

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must have shape [B,N,{self.dim}]")
        projected = self.project_in(tokens)
        bounded = projected.tanh()
        levels = self.levels_tensor.to(device=tokens.device, dtype=tokens.dtype)
        digits = torch.round(
            (bounded + 1.0) * (levels - 1.0) / 2.0
        ).long()
        codes_raw = self._digits_to_codes(digits, tokens.dtype)
        codes_st = bounded + (codes_raw - bounded).detach()
        quantized_raw = self.project_out(codes_raw)
        quantized = self.project_out(codes_st)
        error = F.mse_loss(tokens.detach(), quantized_raw.detach())
        zero = _zero(tokens)
        packed = self._digits_to_ids(digits)
        return {
            "quantized": quantized,
            "quantized_raw": quantized_raw,
            "indices": packed.unsqueeze(-1),
            "rvq_loss": zero,
            "codebook_loss": zero,
            "commitment_loss": zero,
            "quantization_error_by_layer": error.unsqueeze(0),
            "final_residual_mse": (
                tokens.detach() - quantized_raw.detach()
            ).square().mean(),
            # Full hard-usage diagnostics are computed by evaluation over the
            # complete validation set, not synchronously inside the hot path.
            "quantizer_diagnostics": {},
        }


class BinarySphericalStructuralQuantizer(nn.Module):
    """Nine-bit BSQ with official-style dataset entropy and one packed ID."""

    def __init__(
        self,
        dim,
        bits=9,
        inv_temperature=1.0,
        dataset_entropy_weight=0.1,
        dataset_entropy_gamma=1.0,
        per_sample_entropy_gamma=0.0,
        commitment_beta=0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bits = int(bits)
        self.inv_temperature = float(inv_temperature)
        self.dataset_entropy_weight = float(dataset_entropy_weight)
        self.dataset_entropy_gamma = float(dataset_entropy_gamma)
        self.per_sample_entropy_gamma = float(per_sample_entropy_gamma)
        self.commitment_beta = float(commitment_beta)
        if self.dim <= 0 or self.bits <= 0:
            raise ValueError("dim and bits must be positive")
        if self.bits > 20:
            raise ValueError("BSQ full-code entropy is bounded to at most 20 bits")
        if self.inv_temperature <= 0:
            raise ValueError("inv_temperature must be positive")
        self.codebook_size = 2**self.bits
        self.num_codebooks = 1
        self.project_in = nn.Linear(self.dim, self.bits)
        self.project_out = nn.Linear(self.bits, self.dim)
        self.register_buffer(
            "basis", 2 ** torch.arange(self.bits - 1, -1, -1)
        )
        code_ids = torch.arange(self.codebook_size, dtype=torch.long)
        codebook = self._ids_to_bits(code_ids).float() / math.sqrt(self.bits)
        self.register_buffer("implicit_codebook", codebook, persistent=False)

    @property
    def quantizer_type(self):
        return STRUCTURAL_QUANTIZER_BSQ

    def quantizer_manifest(self):
        return {
            "type": self.quantizer_type,
            "vocabulary_size": self.codebook_size,
            "num_token_streams": 1,
            "bits": self.bits,
            "spherical_scale": 1.0 / math.sqrt(self.bits),
            "inv_temperature": self.inv_temperature,
            "dataset_entropy_weight": self.dataset_entropy_weight,
            "dataset_entropy_gamma": self.dataset_entropy_gamma,
            "per_sample_entropy_gamma": self.per_sample_entropy_gamma,
            "commitment_beta": self.commitment_beta,
            "projection": f"{self.dim}->{self.bits}->{self.dim}",
        }

    def _bits_to_ids(self, bits):
        binary = bits.gt(0).long()
        return (binary * self.basis.to(bits.device)).sum(dim=-1)

    def _ids_to_bits(self, indices):
        indices = indices.unsqueeze(-1)
        binary = torch.remainder(
            torch.div(
                indices,
                self.basis.to(indices.device),
                rounding_mode="floor",
            ),
            2,
        )
        return binary * 2 - 1

    def lookup(self, indices):
        indices = _packed_ids(indices, self.codebook_size, self.basis.device)
        bits = self._ids_to_bits(indices).to(self.project_out.weight.dtype)
        return self.project_out(bits / math.sqrt(self.bits))

    def indices_to_quantized(self, indices, active_layers=None):
        active_layers = None if active_layers is None else tuple(active_layers)
        if active_layers not in (None, (), (0,)):
            raise ValueError("BSQ has one packed token stream")
        if active_layers == ():
            indices = _packed_ids(indices, self.codebook_size, self.basis.device)
            return torch.zeros(
                *indices.shape,
                self.dim,
                device=indices.device,
                dtype=self.project_out.weight.dtype,
            )
        return self.lookup(indices)

    def _soft_entropies(self, projected):
        distance = -2.0 * torch.einsum(
            "...c,kc->...k",
            projected.float(),
            self.implicit_codebook.float(),
        )
        probability = (
            -distance * self.inv_temperature
        ).softmax(dim=-1)
        per_sample = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        average = probability.reshape(-1, self.codebook_size).mean(dim=0)
        dataset = -(average * average.clamp_min(1e-8).log()).sum()
        return per_sample, dataset

    def hard_diagnostics(self, indices):
        indices = _packed_ids(indices, self.codebook_size, self.basis.device)
        bits = self._ids_to_bits(indices)
        result = {
            "packed_normalized_entropy": _normalized_entropy(
                indices, self.codebook_size
            )
        }
        for bit in range(self.bits):
            result[f"bit_{bit}_normalized_entropy"] = _normalized_entropy(
                bits[..., bit].gt(0).long(), 2
            )
        return result

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must have shape [B,N,{self.dim}]")
        projected = self.project_in(tokens)
        hard_bits = torch.where(
            projected > 0,
            torch.ones_like(projected),
            -torch.ones_like(projected),
        )
        bits_st = projected + (hard_bits - projected).detach()
        scale = 1.0 / math.sqrt(self.bits)
        quantized_raw = self.project_out(hard_bits * scale)
        quantized = self.project_out(bits_st * scale)
        indices = self._bits_to_ids(hard_bits)
        per_sample_entropy, dataset_entropy = self._soft_entropies(projected)
        entropy_penalty = (
            self.per_sample_entropy_gamma * per_sample_entropy
            - self.dataset_entropy_gamma * dataset_entropy
        ) / self.inv_temperature
        commitment = self.commitment_beta * (
            quantized_raw.detach() - tokens
        ).square().sum(dim=-1).mean()
        quantizer_loss = (
            commitment + self.dataset_entropy_weight * entropy_penalty
        )
        error = F.mse_loss(tokens.detach(), quantized_raw.detach())
        zero = _zero(tokens)
        return {
            "quantized": quantized,
            "quantized_raw": quantized_raw,
            "indices": indices.unsqueeze(-1),
            "rvq_loss": quantizer_loss,
            "codebook_loss": zero,
            "commitment_loss": commitment,
            "quantization_error_by_layer": error.unsqueeze(0),
            "final_residual_mse": (
                tokens.detach() - quantized_raw.detach()
            ).square().mean(),
            "quantizer_diagnostics": {
                "soft_dataset_entropy": dataset_entropy.detach(),
                "soft_per_sample_entropy": per_sample_entropy.detach(),
                "entropy_penalty": entropy_penalty.detach(),
            },
        }


def build_structural_quantizer(
    quantizer_type,
    dim,
    codebook_size=512,
    commitment_weight=0.25,
    fsq_levels=(8, 8, 8),
    bsq_bits=9,
    bsq_inv_temperature=1.0,
    bsq_dataset_entropy_weight=0.1,
    bsq_dataset_entropy_gamma=1.0,
    bsq_per_sample_entropy_gamma=0.0,
    bsq_commitment_beta=0.0,
):
    quantizer_type = str(quantizer_type)
    if quantizer_type == STRUCTURAL_QUANTIZER_VQ:
        return ResidualVectorQuantizer(
            dim=dim,
            num_codebooks=1,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
        )
    if quantizer_type == STRUCTURAL_QUANTIZER_FSQ:
        quantizer = FiniteScalarStructuralQuantizer(
            dim=dim, levels=fsq_levels
        )
    elif quantizer_type == STRUCTURAL_QUANTIZER_BSQ:
        quantizer = BinarySphericalStructuralQuantizer(
            dim=dim,
            bits=bsq_bits,
            inv_temperature=bsq_inv_temperature,
            dataset_entropy_weight=bsq_dataset_entropy_weight,
            dataset_entropy_gamma=bsq_dataset_entropy_gamma,
            per_sample_entropy_gamma=bsq_per_sample_entropy_gamma,
            commitment_beta=bsq_commitment_beta,
        )
    else:
        raise ValueError(
            f"quantizer_type must be one of {STRUCTURAL_QUANTIZERS}"
        )
    if quantizer.codebook_size != int(codebook_size):
        raise ValueError(
            f"{quantizer_type} configuration implies vocabulary "
            f"{quantizer.codebook_size}, expected {int(codebook_size)}"
        )
    return quantizer
