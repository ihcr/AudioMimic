import math

import torch
import torch.nn as nn

from model.g1_streaming_rvqvae import (
    STREAMING_ADDITIVE_ROUTE,
    STREAMING_MODULATED_ROUTE,
)


RESIDUAL_MODES = ("parallel", "hierarchical")


def sample_token_logits(
    logits,
    *,
    greedy=True,
    temperature=1.0,
    top_k=None,
    generator=None,
):
    if greedy:
        return logits.argmax(dim=-1)
    temperature = max(float(temperature), 1e-6)
    scaled = logits / temperature
    if top_k is not None:
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        top_k = min(top_k, int(scaled.shape[-1]))
        threshold = torch.topk(scaled, top_k, dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    probabilities = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]),
        1,
        generator=generator,
    )
    return sampled.view(probabilities.shape[:-1])


class ResidualTemporalContextEncoder(nn.Module):
    """Issue #8 residual context with an independent trainable S66 projection."""

    def __init__(
        self,
        *,
        route,
        vocab_size,
        num_codebooks,
        history_tokens,
        horizon_tokens,
        state_dim,
        d_model,
        num_layers,
        num_heads,
        ffn_dim,
        dropout,
    ):
        super().__init__()
        if route not in (STREAMING_ADDITIVE_ROUTE, STREAMING_MODULATED_ROUTE):
            raise ValueError("unsupported streaming route")
        self.route = str(route)
        self.vocab_size = int(vocab_size)
        self.num_codebooks = int(num_codebooks)
        self.history_tokens = int(history_tokens)
        self.horizon_tokens = int(horizon_tokens)
        self.state_dim = int(state_dim)
        self.d_model = int(d_model)

        self.history_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.vocab_size + 1, self.d_model)
                for _ in range(self.num_codebooks)
            ]
        )
        self.q0_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = nn.Parameter(
            torch.zeros(self.history_tokens + self.horizon_tokens, self.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.state_projection = nn.Sequential(
            nn.Linear(self.state_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, history_tokens, history_valid, plan_q0, state):
        batch = int(history_tokens.shape[0])
        history_hidden = 0.0
        for level, embedding in enumerate(self.history_embeddings):
            level_tokens = history_tokens[..., level].clone()
            level_tokens[~history_valid] = self.vocab_size
            history_hidden = history_hidden + embedding(level_tokens)
        history_hidden = history_hidden / float(self.num_codebooks)
        plan_hidden = self.q0_embedding(plan_q0)
        hidden = torch.cat((history_hidden, plan_hidden), dim=1)
        hidden = hidden + self.position.unsqueeze(0)

        state_hidden = self.state_projection(state)
        prepended_state = self.route == STREAMING_MODULATED_ROUTE
        if prepended_state:
            hidden = torch.cat((state_hidden.unsqueeze(1), hidden), dim=1)
        else:
            hidden = hidden + state_hidden.unsqueeze(1)

        key_padding = torch.cat(
            (
                ~history_valid,
                torch.zeros(
                    batch,
                    self.horizon_tokens,
                    device=history_tokens.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        if prepended_state:
            key_padding = torch.cat(
                (
                    torch.zeros(
                        batch,
                        1,
                        device=history_tokens.device,
                        dtype=torch.bool,
                    ),
                    key_padding,
                ),
                dim=1,
            )
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding)
        plan_start = self.history_tokens + int(prepended_state)
        return hidden[:, plan_start : plan_start + self.horizon_tokens]


class RVQDepthDecoder(nn.Module):
    """Shared H8 Transformer applied sequentially over RVQ depth."""

    def __init__(
        self,
        *,
        vocab_size,
        residual_levels,
        d_model,
        num_layers=4,
        num_heads=8,
        ffn_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.residual_levels = int(residual_levels)
        self.d_model = int(d_model)
        if self.residual_levels <= 0:
            raise ValueError("residual_levels must be positive")

        self.q0_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.residual_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.vocab_size, self.d_model)
                for _ in range(self.residual_levels)
            ]
        )
        self.level_embedding = nn.Embedding(self.residual_levels, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_heads = nn.ModuleList(
            [
                nn.Linear(self.d_model, self.vocab_size)
                for _ in range(self.residual_levels)
            ]
        )
        self.refinement_gates = nn.Parameter(torch.zeros(self.residual_levels))

    def initialize_output_heads_from_parallel(self, weight, bias):
        expected_weight = (
            self.residual_levels * self.vocab_size,
            self.d_model,
        )
        expected_bias = (self.residual_levels * self.vocab_size,)
        if tuple(weight.shape) != expected_weight or tuple(bias.shape) != expected_bias:
            raise ValueError(
                "parallel residual head has incompatible shape: "
                f"weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
            )
        with torch.no_grad():
            for level, head in enumerate(self.output_heads):
                start = level * self.vocab_size
                end = start + self.vocab_size
                head.weight.copy_(weight[start:end])
                head.bias.copy_(bias[start:end])

    def _conditioned_hidden(self, temporal_context, plan_q0, known_residual, level):
        hidden = temporal_context + self.q0_embedding(plan_q0)
        if level:
            lower = 0.0
            for lower_level in range(level):
                lower = lower + self.residual_embeddings[lower_level](
                    known_residual[..., lower_level]
                )
            hidden = hidden + lower / math.sqrt(float(level))
        level_ids = torch.full(
            plan_q0.shape,
            int(level),
            device=plan_q0.device,
            dtype=torch.long,
        )
        hidden = hidden + self.level_embedding(level_ids)
        refinement = self.decoder(self.input_norm(hidden))
        gate = self.refinement_gates[level].to(dtype=refinement.dtype)
        return temporal_context + gate * refinement

    def logits_for_level(self, temporal_context, plan_q0, known_residual, level):
        level = int(level)
        if level < 0 or level >= self.residual_levels:
            raise ValueError("residual level is outside the configured range")
        if known_residual.shape != (
            temporal_context.shape[0],
            temporal_context.shape[1],
            level,
        ):
            raise ValueError(
                "known_residual must contain exactly the lower RVQ levels"
            )
        hidden = self._conditioned_hidden(
            temporal_context,
            plan_q0,
            known_residual,
            level,
        )
        return self.output_heads[level](hidden)

    def teacher_forced_logits(
        self,
        temporal_context,
        plan_q0,
        target_residual,
    ):
        """Evaluate every teacher-forced RVQ depth in one batched forward."""
        expected = (
            temporal_context.shape[0],
            temporal_context.shape[1],
            self.residual_levels,
        )
        if target_residual.shape != expected:
            raise ValueError(f"target_residual must have shape {expected}")

        batch, horizon, _ = temporal_context.shape
        level_count = self.residual_levels
        base = temporal_context + self.q0_embedding(plan_q0)
        embedded_residual = torch.stack(
            [
                embedding(target_residual[..., level])
                for level, embedding in enumerate(self.residual_embeddings)
            ],
            dim=2,
        )
        cumulative = embedded_residual.cumsum(dim=2)
        zero = embedded_residual.new_zeros(batch, horizon, 1, self.d_model)
        lower_sum = torch.cat((zero, cumulative[:, :, :-1]), dim=2)
        level = torch.arange(
            level_count,
            device=plan_q0.device,
            dtype=torch.long,
        )
        denominator = level.clamp_min(1).sqrt().to(dtype=base.dtype)
        hidden = base.unsqueeze(2) + lower_sum / denominator.view(1, 1, -1, 1)
        hidden = hidden + self.level_embedding(level).view(1, 1, level_count, -1)

        # Depth levels are independent under teacher forcing because every lower
        # token is supplied by the target. Folding depth into the batch dimension
        # preserves the model while replacing seven small Transformer launches
        # with one well-populated launch.
        hidden = hidden.permute(0, 2, 1, 3).reshape(
            batch * level_count,
            horizon,
            self.d_model,
        )
        refinement = self.decoder(self.input_norm(hidden)).reshape(
            batch,
            level_count,
            horizon,
            self.d_model,
        )
        gate = self.refinement_gates.to(dtype=refinement.dtype).view(
            1,
            level_count,
            1,
            1,
        )
        refined = temporal_context.unsqueeze(1) + gate * refinement
        head_weight = torch.stack(
            [head.weight for head in self.output_heads],
            dim=0,
        )
        head_bias = torch.stack(
            [head.bias for head in self.output_heads],
            dim=0,
        )
        logits = torch.einsum(
            "blhd,lvd->blhv",
            refined,
            head_weight,
        ) + head_bias.view(1, level_count, 1, self.vocab_size)
        return logits.permute(0, 2, 1, 3).contiguous()


class HierarchicalResidualGenerator(nn.Module):
    def __init__(
        self,
        *,
        vocab_size,
        residual_levels,
        d_model,
        num_layers=4,
        num_heads=8,
        ffn_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.depth_decoder = RVQDepthDecoder(
            vocab_size=vocab_size,
            residual_levels=residual_levels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    @property
    def residual_levels(self):
        return self.depth_decoder.residual_levels

    def initialize_output_heads_from_parallel(self, weight, bias):
        self.depth_decoder.initialize_output_heads_from_parallel(weight, bias)

    def forward_teacher_forced(
        self,
        temporal_context,
        plan_q0,
        target_residual,
    ):
        logits = self.depth_decoder.teacher_forced_logits(
            temporal_context,
            plan_q0,
            target_residual,
        )
        return {
            "logits_by_level": list(logits.unbind(dim=2)),
            "logits": logits,
        }

    def generate(
        self,
        temporal_context,
        plan_q0,
        *,
        greedy=True,
        temperature=1.0,
        top_k=None,
        generator=None,
        forced_residual=None,
        return_logits=False,
    ):
        if forced_residual is not None:
            expected = (
                temporal_context.shape[0],
                temporal_context.shape[1],
                self.residual_levels,
            )
            if forced_residual.shape != expected:
                raise ValueError(f"forced_residual must have shape {expected}")
        generated = []
        logits_by_level = []
        for level in range(self.residual_levels):
            known = (
                torch.stack(generated, dim=-1)
                if generated
                else plan_q0.new_empty(plan_q0.shape[0], plan_q0.shape[1], 0)
            )
            logits = self.depth_decoder.logits_for_level(
                temporal_context,
                plan_q0,
                known,
                level,
            )
            logits_by_level.append(logits)
            if forced_residual is None:
                token = sample_token_logits(
                    logits,
                    greedy=greedy,
                    temperature=temperature,
                    top_k=top_k,
                    generator=generator,
                )
            else:
                token = forced_residual[..., level]
            generated.append(token)
        tokens = torch.stack(generated, dim=-1)
        if return_logits:
            return {
                "tokens": tokens,
                "logits_by_level": logits_by_level,
                "logits": torch.stack(logits_by_level, dim=2),
            }
        return tokens
