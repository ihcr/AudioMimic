from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.g1_streaming_state import G1_STREAMING_STATE_DIM
from model.g1_streaming_residual_depth import (
    RESIDUAL_MODES,
    HierarchicalResidualGenerator,
    ResidualTemporalContextEncoder,
    sample_token_logits,
)
from model.g1_streaming_rvqvae import (
    STREAMING_ADDITIVE_ROUTE,
    STREAMING_MODULATED_ROUTE,
    STREAMING_STATE_ROUTES,
)


@dataclass(frozen=True)
class G1StreamingTokenGeneratorConfig:
    route: str = STREAMING_ADDITIVE_ROUTE
    vocab_size: int = 512
    num_codebooks: int = 8
    history_tokens: int = 64
    horizon_tokens: int = 8
    state_dim: int = G1_STREAMING_STATE_DIM
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    residual_mode: str = "parallel"
    depth_decoder_layers: int = 4
    q0_frozen: bool = False
    q0_source_checkpoint: str = ""
    q0_source_sha256: str = ""

    @property
    def bos_token_id(self):
        return int(self.vocab_size)

    @property
    def pad_token_id(self):
        return int(self.vocab_size) + 1

    def asdict(self):
        return {
            "model_type": "g1_streaming_token_generator",
            "route": self.route,
            "conditioning": "unconditional_motion_history_plus_S66",
            "vocab_size": int(self.vocab_size),
            "num_codebooks": int(self.num_codebooks),
            "history_tokens": int(self.history_tokens),
            "horizon_tokens": int(self.horizon_tokens),
            "commit_tokens": 4,
            "state_dim": int(self.state_dim),
            "state_layout": "S66_physical",
            "d_model": int(self.d_model),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "ffn_dim": int(self.ffn_dim),
            "dropout": float(self.dropout),
            "q0_generation": "causal_ar_with_kv_cache",
            "residual_mode": self.residual_mode,
            "residual_generation": (
                "none_single_q0"
                if self.num_codebooks == 1
                else (
                    "parallel_q1_to_q7"
                    if self.residual_mode == "parallel"
                    else "hierarchical_q1_to_q7"
                )
            ),
            "q0_frozen": bool(self.q0_frozen),
            "q0_source_checkpoint": self.q0_source_checkpoint,
            "q0_source_sha256": self.q0_source_sha256,
            "rollout_depth_support": [0, 1, 2],
            "depth_decoder_layers": int(self.depth_decoder_layers),
            "depth_decoder_shared": self.residual_mode == "hierarchical",
            "level_specific_embeddings": self.residual_mode == "hierarchical",
            "level_specific_output_heads": self.residual_mode == "hierarchical",
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
        }


class CachedCausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = self.d_model // self.num_heads
        self.qkv = nn.Linear(self.d_model, self.d_model * 3)
        self.output = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(float(dropout))

    def _split_heads(self, value):
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden, key_padding_mask=None, cache=None, return_cache=False):
        batch, query_length, _ = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        if cache is None and key_padding_mask is None:
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
            context = context.transpose(1, 2).contiguous().view(
                batch,
                query_length,
                self.d_model,
            )
            padding = hidden.new_zeros(
                batch,
                query_length,
                dtype=torch.bool,
            )
            new_cache = (key, value, padding) if return_cache else None
            return self.output(context), new_cache
        past_length = 0
        if cache is not None:
            past_key, past_value, past_padding = cache
            past_length = int(past_key.shape[-2])
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
            if key_padding_mask is None:
                key_padding_mask = hidden.new_zeros(batch, query_length, dtype=torch.bool)
            key_padding_mask = torch.cat((past_padding, key_padding_mask), dim=-1)
        elif key_padding_mask is None:
            key_padding_mask = hidden.new_zeros(batch, query_length, dtype=torch.bool)

        key_length = int(key.shape[-2])
        query_positions = torch.arange(query_length, device=hidden.device) + past_length
        key_positions = torch.arange(key_length, device=hidden.device)
        causal_invalid = key_positions[None, :] > query_positions[:, None]
        invalid = causal_invalid.view(1, 1, query_length, key_length) | key_padding_mask[
            :, None, None, :
        ]
        attention_mask = torch.zeros(
            batch,
            1,
            query_length,
            key_length,
            device=hidden.device,
            dtype=hidden.dtype,
        ).masked_fill(invalid, float("-inf"))
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        context = torch.nan_to_num(context)
        context = context.transpose(1, 2).contiguous().view(batch, query_length, self.d_model)
        output = self.output(context)
        new_cache = (key, value, key_padding_mask) if return_cache else None
        return output, new_cache


class CachedCausalTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(int(d_model))
        self.attention = CachedCausalSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(int(d_model))
        self.feedforward = nn.Sequential(
            nn.Linear(int(d_model), int(ffn_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ffn_dim), int(d_model)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, hidden, key_padding_mask=None, cache=None, return_cache=False):
        attended, new_cache = self.attention(
            self.norm1(hidden),
            key_padding_mask=key_padding_mask,
            cache=cache,
            return_cache=return_cache,
        )
        hidden = hidden + attended
        hidden = hidden + self.feedforward(self.norm2(hidden))
        return hidden, new_cache


class G1StreamingTokenGenerator(nn.Module):
    def __init__(
        self,
        route=STREAMING_ADDITIVE_ROUTE,
        vocab_size=512,
        num_codebooks=8,
        history_tokens=64,
        horizon_tokens=8,
        state_dim=G1_STREAMING_STATE_DIM,
        d_model=512,
        num_layers=8,
        num_heads=8,
        ffn_dim=2048,
        dropout=0.1,
        residual_mode="parallel",
        depth_decoder_layers=4,
        q0_frozen=False,
        q0_source_checkpoint="",
        q0_source_sha256="",
    ):
        super().__init__()
        self.config = G1StreamingTokenGeneratorConfig(
            route=str(route),
            vocab_size=int(vocab_size),
            num_codebooks=int(num_codebooks),
            history_tokens=int(history_tokens),
            horizon_tokens=int(horizon_tokens),
            state_dim=int(state_dim),
            d_model=int(d_model),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            ffn_dim=int(ffn_dim),
            dropout=float(dropout),
            residual_mode=str(residual_mode),
            depth_decoder_layers=int(depth_decoder_layers),
            q0_frozen=bool(q0_frozen),
            q0_source_checkpoint=str(q0_source_checkpoint),
            q0_source_sha256=str(q0_source_sha256),
        )
        if self.config.route not in STREAMING_STATE_ROUTES:
            raise ValueError(f"route must be one of {STREAMING_STATE_ROUTES}")
        if self.config.num_codebooks not in (1, 8):
            raise ValueError("streaming generator supports native single-q0 or RVQ-8")
        if self.config.history_tokens != 64 or self.config.horizon_tokens != 8:
            raise ValueError("streaming generator is locked to K64/H8")
        if self.config.state_dim != G1_STREAMING_STATE_DIM:
            raise ValueError("streaming generator is locked to S66_physical")
        if self.config.residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"residual_mode must be one of {RESIDUAL_MODES}")
        if self.config.depth_decoder_layers <= 0:
            raise ValueError("depth_decoder_layers must be positive")

        self.q0_embedding = nn.Embedding(self.config.vocab_size + 2, self.config.d_model)
        self.q0_position = nn.Parameter(
            torch.zeros(self.config.history_tokens + self.config.horizon_tokens, self.config.d_model)
        )
        self.q0_blocks = nn.ModuleList(
            [
                CachedCausalTransformerBlock(
                    self.config.d_model,
                    self.config.num_heads,
                    self.config.ffn_dim,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.q0_norm = nn.LayerNorm(self.config.d_model)
        self.q0_head = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.q0_state_projection = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, self.config.d_model),
        )

        self.residual_context = None
        self.residual_head = None
        self.hierarchical_residual = None
        if self.config.num_codebooks == 8:
            self.residual_context = ResidualTemporalContextEncoder(
                route=self.config.route,
                vocab_size=self.config.vocab_size,
                num_codebooks=self.config.num_codebooks,
                history_tokens=self.config.history_tokens,
                horizon_tokens=self.config.horizon_tokens,
                state_dim=self.config.state_dim,
                d_model=self.config.d_model,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim,
                dropout=self.config.dropout,
            )
        if self.config.num_codebooks == 8 and self.config.residual_mode == "parallel":
            self.residual_head = nn.Linear(
                self.config.d_model,
                (self.config.num_codebooks - 1) * self.config.vocab_size,
            )
            self.hierarchical_residual = None
        elif self.config.num_codebooks == 8:
            self.residual_head = None
            self.hierarchical_residual = HierarchicalResidualGenerator(
                vocab_size=self.config.vocab_size,
                residual_levels=self.config.num_codebooks - 1,
                d_model=self.config.d_model,
                num_layers=self.config.depth_decoder_layers,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim,
                dropout=self.config.dropout,
            )
        self.reset_parameters()
        if self.config.q0_frozen:
            self.freeze_q0()

    @property
    def bos_token_id(self):
        return self.config.bos_token_id

    @property
    def pad_token_id(self):
        return self.config.pad_token_id

    def reset_parameters(self):
        nn.init.normal_(self.q0_position, mean=0.0, std=0.02)

    def _validate_state(self, state, batch_size, reference):
        if state is None:
            raise ValueError("S66_physical boundary state is required")
        if state.ndim != 2 or state.shape != (batch_size, self.config.state_dim):
            raise ValueError(
                f"state must have shape {(batch_size, self.config.state_dim)}, got {tuple(state.shape)}"
            )
        return state.to(device=reference.device, dtype=reference.dtype)

    def _validate_history(self, history_tokens, history_valid):
        expected = (history_tokens.shape[0], self.config.history_tokens, self.config.num_codebooks)
        if history_tokens.ndim != 3 or tuple(history_tokens.shape) != expected:
            raise ValueError(f"history_tokens must have shape [B,64,8], got {tuple(history_tokens.shape)}")
        if history_valid.shape != history_tokens.shape[:2]:
            raise ValueError("history_valid must have shape [B,64]")
        if history_valid.dtype != torch.bool:
            history_valid = history_valid.bool()
        return history_valid

    def _condition_sequence(self, token_hidden, state, state_projection):
        state_hidden = state_projection(state)
        if self.config.route == STREAMING_ADDITIVE_ROUTE:
            return token_hidden + state_hidden.unsqueeze(1), False
        return torch.cat((state_hidden.unsqueeze(1), token_hidden), dim=1), True

    def _q0_inputs(self, history_tokens, history_valid, plan_q0, state):
        batch = history_tokens.shape[0]
        plan_inputs = torch.full(
            (batch, self.config.horizon_tokens),
            self.bos_token_id,
            device=history_tokens.device,
            dtype=torch.long,
        )
        if self.config.horizon_tokens > 1:
            plan_inputs[:, 1:] = plan_q0[:, :-1]
        history_q0 = history_tokens[..., 0].clone()
        history_q0[~history_valid] = self.pad_token_id
        input_ids = torch.cat((history_q0, plan_inputs), dim=1)
        hidden = self.q0_embedding(input_ids) + self.q0_position.unsqueeze(0)
        hidden, prepended_state = self._condition_sequence(
            hidden,
            state,
            self.q0_state_projection,
        )
        key_padding = torch.cat(
            (
                ~history_valid,
                torch.zeros(batch, self.config.horizon_tokens, device=history_tokens.device, dtype=torch.bool),
            ),
            dim=1,
        )
        if prepended_state:
            key_padding = torch.cat(
                (torch.zeros(batch, 1, device=key_padding.device, dtype=torch.bool), key_padding),
                dim=1,
            )
        return hidden, key_padding, prepended_state

    def forward_q0(self, history_tokens, history_valid, plan_q0, state):
        history_valid = self._validate_history(history_tokens, history_valid)
        if plan_q0.shape != (history_tokens.shape[0], self.config.horizon_tokens):
            raise ValueError("plan_q0 must have shape [B,8]")
        state = self._validate_state(state, history_tokens.shape[0], self.q0_position)
        hidden, key_padding, prepended_state = self._q0_inputs(
            history_tokens, history_valid, plan_q0, state
        )
        for block in self.q0_blocks:
            hidden, _ = block(hidden, key_padding_mask=key_padding)
        hidden = self.q0_norm(hidden)
        plan_start = self.config.history_tokens + int(prepended_state)
        return self.q0_head(hidden[:, plan_start : plan_start + self.config.horizon_tokens])

    def _residual_temporal_context(
        self,
        history_tokens,
        history_valid,
        plan_q0,
        state,
    ):
        if self.config.num_codebooks == 1:
            raise ValueError("single-q0 generator has no categorical residual context")
        history_valid = self._validate_history(history_tokens, history_valid)
        expected_q0 = (history_tokens.shape[0], self.config.horizon_tokens)
        if plan_q0.shape != expected_q0:
            raise ValueError(f"plan_q0 must have shape {expected_q0}")
        state = self._validate_state(
            state,
            history_tokens.shape[0],
            self.residual_context.position,
        )
        return self.residual_context(
            history_tokens,
            history_valid,
            plan_q0,
            state,
        )

    def forward_residual(
        self,
        history_tokens,
        history_valid,
        plan_q0,
        state,
        target_residual=None,
    ):
        if self.config.num_codebooks == 1:
            raise ValueError("single-q0 generator has no categorical residual head")
        context = self._residual_temporal_context(
            history_tokens,
            history_valid,
            plan_q0,
            state,
        )
        if self.config.residual_mode == "parallel":
            logits = self.residual_head(context)
            return logits.view(
                history_tokens.shape[0],
                self.config.horizon_tokens,
                self.config.num_codebooks - 1,
                self.config.vocab_size,
            )
        if target_residual is not None:
            return self.hierarchical_residual.forward_teacher_forced(
                context,
                plan_q0,
                target_residual,
            )["logits"]
        return self.hierarchical_residual.generate(
            context,
            plan_q0,
            greedy=True,
            return_logits=True,
        )["logits"]

    def forward_residual_teacher_forced(
        self,
        history_tokens,
        history_valid,
        plan_q0,
        target_residual,
        state,
    ):
        return self.forward_residual(
            history_tokens,
            history_valid,
            plan_q0,
            state,
            target_residual=target_residual,
        )

    def generate_residual_by_level(
        self,
        history_tokens,
        history_valid,
        plan_q0,
        state,
        *,
        greedy=True,
        temperature=1.0,
        top_k=None,
        generator=None,
        forced_residual=None,
        return_logits=False,
    ):
        context = self.encode_residual_context(
            history_tokens,
            history_valid,
            plan_q0,
            state,
        )
        return self.generate_residual_from_context(
            context,
            plan_q0,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
            forced_residual=forced_residual,
            return_logits=return_logits,
        )

    def encode_residual_context(
        self,
        history_tokens,
        history_valid,
        plan_q0,
        state,
    ):
        return self._residual_temporal_context(
            history_tokens,
            history_valid,
            plan_q0,
            state,
        )

    def generate_residual_from_context(
        self,
        context,
        plan_q0,
        *,
        greedy=True,
        temperature=1.0,
        top_k=None,
        generator=None,
        forced_residual=None,
        return_logits=False,
    ):
        if self.config.residual_mode == "parallel":
            logits = self.residual_head(context).view(
                context.shape[0],
                self.config.horizon_tokens,
                self.config.num_codebooks - 1,
                self.config.vocab_size,
            )
            if forced_residual is None:
                tokens = sample_token_logits(
                    logits,
                    greedy=greedy,
                    temperature=temperature,
                    top_k=top_k,
                    generator=generator,
                )
            else:
                expected = (
                    context.shape[0],
                    self.config.horizon_tokens,
                    self.config.num_codebooks - 1,
                )
                if forced_residual.shape != expected:
                    raise ValueError(f"forced_residual must have shape {expected}")
                tokens = forced_residual
            if return_logits:
                return {"tokens": tokens, "logits": logits}
            return tokens
        return self.hierarchical_residual.generate(
            context,
            plan_q0,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
            forced_residual=forced_residual,
            return_logits=return_logits,
        )

    def forward(self, history_tokens, history_valid, plan_tokens, state):
        if plan_tokens.shape != (
            history_tokens.shape[0],
            self.config.horizon_tokens,
            self.config.num_codebooks,
        ):
            raise ValueError(
                f"plan_tokens must have shape [B,8,{self.config.num_codebooks}]"
            )
        output = {
            "q0_logits": self.forward_q0(
                history_tokens,
                history_valid,
                plan_tokens[..., 0],
                state,
            )
        }
        if self.config.num_codebooks == 8:
            output["residual_logits"] = self.forward_residual(
                history_tokens,
                history_valid,
                plan_tokens[..., 0],
                state,
                target_residual=plan_tokens[..., 1:],
            )
        return output

    def _prefill_q0(self, history_tokens, history_valid, state):
        history_valid = self._validate_history(history_tokens, history_valid)
        batch = history_tokens.shape[0]
        history_q0 = history_tokens[..., 0].clone()
        history_q0[~history_valid] = self.pad_token_id
        bos = torch.full((batch, 1), self.bos_token_id, device=history_tokens.device, dtype=torch.long)
        input_ids = torch.cat((history_q0, bos), dim=1)
        position = self.q0_position[: self.config.history_tokens + 1]
        hidden = self.q0_embedding(input_ids) + position.unsqueeze(0)
        state = self._validate_state(state, batch, hidden)
        hidden, prepended_state = self._condition_sequence(
            hidden,
            state,
            self.q0_state_projection,
        )
        key_padding = torch.cat(
            (~history_valid, torch.zeros(batch, 1, device=history_tokens.device, dtype=torch.bool)),
            dim=1,
        )
        if prepended_state:
            key_padding = torch.cat(
                (torch.zeros(batch, 1, device=key_padding.device, dtype=torch.bool), key_padding), dim=1
            )
        caches = []
        for block in self.q0_blocks:
            hidden, cache = block(hidden, key_padding_mask=key_padding, return_cache=True)
            caches.append(cache)
        return self.q0_head(self.q0_norm(hidden[:, -1])), caches, state

    def _cached_q0_step(self, token, position_index, caches, state):
        hidden = self.q0_embedding(token[:, None]) + self.q0_position[position_index][None, None]
        if self.config.route == STREAMING_ADDITIVE_ROUTE:
            hidden = hidden + self.q0_state_projection(state).unsqueeze(1)
        new_caches = []
        key_padding = torch.zeros(token.shape[0], 1, device=token.device, dtype=torch.bool)
        for block, cache in zip(self.q0_blocks, caches):
            hidden, new_cache = block(
                hidden,
                key_padding_mask=key_padding,
                cache=cache,
                return_cache=True,
            )
            new_caches.append(new_cache)
        logits = self.q0_head(self.q0_norm(hidden[:, 0]))
        return logits, new_caches

    @staticmethod
    def _sample(
        logits,
        greedy=True,
        temperature=1.0,
        top_k=None,
        generator=None,
    ):
        return sample_token_logits(
            logits,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )

    @torch.inference_mode()
    def generate_q0(
        self,
        history_tokens,
        history_valid,
        state,
        greedy=True,
        temperature=1.0,
        top_k=None,
        generator=None,
    ):
        self.eval()
        logits, caches, state = self._prefill_q0(history_tokens, history_valid, state)
        generated = []
        for step in range(self.config.horizon_tokens):
            token = self._sample(
                logits,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            generated.append(token)
            if step + 1 < self.config.horizon_tokens:
                logits, caches = self._cached_q0_step(
                    token,
                    self.config.history_tokens + step + 1,
                    caches,
                    state,
                )
        return torch.stack(generated, dim=1)

    @torch.inference_mode()
    def generate(
        self,
        history_tokens,
        history_valid,
        state,
        greedy=True,
        temperature=1.0,
        top_k=None,
        generator=None,
        residual_greedy=None,
        residual_temperature=None,
        residual_top_k=None,
    ):
        residual_greedy = greedy if residual_greedy is None else bool(residual_greedy)
        residual_temperature = (
            temperature
            if residual_temperature is None
            else float(residual_temperature)
        )
        residual_top_k = top_k if residual_top_k is None else residual_top_k
        q0 = self.generate_q0(
            history_tokens,
            history_valid,
            state,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        if self.config.num_codebooks == 1:
            return q0.unsqueeze(-1)
        residual = self.generate_residual_by_level(
            history_tokens,
            history_valid,
            q0,
            state,
            greedy=residual_greedy,
            temperature=residual_temperature,
            top_k=residual_top_k,
            generator=generator,
        )
        return torch.cat((q0.unsqueeze(-1), residual), dim=-1)

    def q0_parameter_names(self):
        prefixes = (
            "q0_embedding.",
            "q0_position",
            "q0_blocks.",
            "q0_norm.",
            "q0_head.",
            "q0_state_projection.",
        )
        return [
            name
            for name, _ in self.named_parameters()
            if name == "q0_position" or name.startswith(prefixes)
        ]

    def residual_parameter_names(self):
        q0_names = set(self.q0_parameter_names())
        return [name for name, _ in self.named_parameters() if name not in q0_names]

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_q0_frozen", self.config.q0_frozen):
            self.q0_embedding.eval()
            self.q0_blocks.eval()
            self.q0_norm.eval()
            self.q0_head.eval()
            self.q0_state_projection.eval()
        return self

    def freeze_q0(self):
        q0_names = set(self.q0_parameter_names())
        for name, parameter in self.named_parameters():
            if name in q0_names:
                parameter.requires_grad_(False)
        self._q0_frozen = True
        self.train(self.training)
        return sorted(q0_names)

    def load_q0_from_checkpoint(self, checkpoint):
        source = _upgrade_legacy_generator_state_dict(checkpoint["model"])
        target = self.state_dict()
        prefixes = (
            "q0_embedding.",
            "q0_position",
            "q0_blocks.",
            "q0_norm.",
            "q0_head.",
            "q0_state_projection.",
        )
        loaded = []
        with torch.no_grad():
            for name in target:
                if not (name == "q0_position" or name.startswith(prefixes)):
                    continue
                if name not in source or source[name].shape != target[name].shape:
                    raise ValueError(f"q0 checkpoint is missing compatible key {name}")
                target[name].copy_(source[name])
                loaded.append(name)
        return loaded

    def load_shared_residual_from_checkpoint(self, checkpoint):
        if self.config.num_codebooks == 1:
            raise ValueError("single-q0 generator has no residual parameters")
        source = _upgrade_legacy_generator_state_dict(checkpoint["model"])
        target = self.state_dict()
        loaded = []
        with torch.no_grad():
            for name in target:
                if not name.startswith("residual_context."):
                    continue
                if name not in source or source[name].shape != target[name].shape:
                    raise ValueError(
                        f"shared residual checkpoint is missing compatible key {name}"
                    )
                target[name].copy_(source[name])
                loaded.append(name)
            if "residual_head.weight" not in source or "residual_head.bias" not in source:
                raise ValueError("shared residual checkpoint is missing the parallel head")
            if self.config.residual_mode == "parallel":
                self.residual_head.weight.copy_(source["residual_head.weight"])
                self.residual_head.bias.copy_(source["residual_head.bias"])
                loaded.extend(("residual_head.weight", "residual_head.bias"))
            else:
                self.hierarchical_residual.initialize_output_heads_from_parallel(
                    source["residual_head.weight"],
                    source["residual_head.bias"],
                )
                loaded.extend(
                    (
                        "hierarchical_residual.depth_decoder.output_heads.*.weight",
                        "hierarchical_residual.depth_decoder.output_heads.*.bias",
                    )
                )
        return loaded

    def manifest(self):
        manifest = self.config.asdict()
        manifest["q0_frozen"] = bool(
            getattr(self, "_q0_frozen", self.config.q0_frozen)
        )
        return manifest


def streaming_generator_cross_entropy(output, targets, optimize_q0=True):
    q0_loss = F.cross_entropy(
        output["q0_logits"].reshape(-1, output["q0_logits"].shape[-1]),
        targets[..., 0].reshape(-1),
    )
    level_losses = [
        F.cross_entropy(
            output["residual_logits"][..., level, :].reshape(
                -1,
                output["residual_logits"].shape[-1],
            ),
            targets[..., level + 1].reshape(-1),
        )
        for level in range(targets.shape[-1] - 1)
    ]
    if not level_losses:
        if not optimize_q0:
            raise ValueError("single-q0 loss cannot disable q0 optimization")
        total = q0_loss
        with torch.no_grad():
            q0_accuracy = output["q0_logits"].argmax(dim=-1).eq(
                targets[..., 0]
            ).float().mean()
        return total, {
            "loss/total": total.detach(),
            "loss/q0": q0_loss.detach(),
            "token/q0_accuracy": q0_accuracy.detach(),
        }
    residual_loss = torch.stack(level_losses).mean()
    total = residual_loss + q0_loss if optimize_q0 else residual_loss
    with torch.no_grad():
        q0_accuracy = output["q0_logits"].argmax(dim=-1).eq(targets[..., 0]).float().mean()
        residual_accuracy = (
            output["residual_logits"].argmax(dim=-1).eq(targets[..., 1:]).float().mean()
        )
    stats = {
        "loss/total": total.detach(),
        "loss/q0": q0_loss.detach(),
        "loss/residual": residual_loss.detach(),
        "loss/residual_unscaled_sum": torch.stack(level_losses).sum().detach(),
        "token/q0_accuracy": q0_accuracy.detach(),
        "token/residual_accuracy": residual_accuracy.detach(),
    }
    with torch.no_grad():
        for level, level_loss in enumerate(level_losses, start=1):
            logits = output["residual_logits"][..., level - 1, :]
            probabilities = torch.softmax(logits.float(), dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
                dim=-1
            )
            predictions = logits.argmax(dim=-1)
            stats[f"loss/CE_q{level}"] = level_loss.detach()
            stats[f"accuracy/q{level}"] = predictions.eq(
                targets[..., level]
            ).float().mean()
            stats[f"entropy/q{level}"] = entropy.mean()
            stats[f"usage/q{level}"] = logits.new_tensor(
                torch.unique(predictions).numel() / float(logits.shape[-1])
            )
    return total, stats


def _upgrade_legacy_generator_state_dict(state_dict):
    upgraded = {}
    for name, value in state_dict.items():
        if name.startswith("state_projection."):
            suffix = name[len("state_projection.") :]
            upgraded[f"q0_state_projection.{suffix}"] = value
            upgraded[f"residual_context.state_projection.{suffix}"] = value
        elif name.startswith("residual_history_embeddings."):
            upgraded[
                "residual_context.history_embeddings."
                + name[len("residual_history_embeddings.") :]
            ] = value
        elif name.startswith("residual_q0_embedding."):
            upgraded[
                "residual_context.q0_embedding."
                + name[len("residual_q0_embedding.") :]
            ] = value
        elif name == "residual_position":
            upgraded["residual_context.position"] = value
        elif name.startswith("residual_encoder."):
            upgraded[
                "residual_context.encoder."
                + name[len("residual_encoder.") :]
            ] = value
        else:
            upgraded[name] = value
    return upgraded


def build_g1_streaming_token_generator_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {})
    if config.get("model_type") != "g1_streaming_token_generator":
        raise ValueError("checkpoint is not a native streaming token generator")
    model = G1StreamingTokenGenerator(
        route=config["route"],
        vocab_size=int(config.get("vocab_size", 512)),
        num_codebooks=int(config.get("num_codebooks", 8)),
        history_tokens=int(config.get("history_tokens", 64)),
        horizon_tokens=int(config.get("horizon_tokens", 8)),
        state_dim=int(config.get("state_dim", G1_STREAMING_STATE_DIM)),
        d_model=int(config.get("d_model", 512)),
        num_layers=int(config.get("num_layers", 8)),
        num_heads=int(config.get("num_heads", 8)),
        ffn_dim=int(config.get("ffn_dim", 2048)),
        dropout=float(config.get("dropout", 0.1)),
        residual_mode=str(config.get("residual_mode", "parallel")),
        depth_decoder_layers=int(config.get("depth_decoder_layers", 4)),
        q0_frozen=bool(config.get("q0_frozen", False)),
        q0_source_checkpoint=str(config.get("q0_source_checkpoint", "")),
        q0_source_sha256=str(config.get("q0_source_sha256", "")),
    )
    model.load_state_dict(
        _upgrade_legacy_generator_state_dict(checkpoint["model"]),
        strict=True,
    )
    return model
