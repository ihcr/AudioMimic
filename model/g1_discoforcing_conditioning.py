"""Capacity-matched DiscoForcing-style music conditioning blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.audio_extraction.discoforcing_causal import (
    MUSIC_CARRIER_DIM,
    MUSIC_FEATURE_TOKENS,
)
from model.g1_streaming_token_generator import CachedCausalTransformerBlock


class MusicConditionProjector(nn.Module):
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.model_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.model_dim, self.model_dim),
        )

    def forward(self, music):
        return self.net(music)


class MusicCrossAttention(nn.Module):
    """Cross-attention with reusable music K/V and zero-output initialization."""

    def __init__(self, model_dim, num_heads, dropout=0.0):
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.head_dim = self.model_dim // self.num_heads
        self.query = nn.Linear(self.model_dim, self.model_dim)
        self.key = nn.Linear(self.model_dim, self.model_dim)
        self.value = nn.Linear(self.model_dim, self.model_dim)
        self.output = nn.Linear(self.model_dim, self.model_dim)
        self.dropout = float(dropout)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _heads(self, values):
        batch, length, _ = values.shape
        return values.view(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def prepare_context(self, context):
        return self._heads(self.key(context)), self._heads(self.value(context))

    def forward(self, hidden, context=None, context_cache=None):
        if (context is None) == (context_cache is None):
            raise ValueError("provide exactly one of context or context_cache")
        if context_cache is None:
            context_cache = self.prepare_context(context)
        key, value = context_cache
        query = self._heads(self.query(hidden))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            hidden.shape[0],
            hidden.shape[1],
            self.model_dim,
        )
        return self.output(attended), context_cache


class MusicConditionedCausalTransformerBlock(CachedCausalTransformerBlock):
    """Preserve parent parameter names and insert music attention before FFN."""

    def __init__(self, model_dim, num_heads, ffn_dim, dropout=0.0):
        super().__init__(model_dim, num_heads, ffn_dim, dropout=dropout)
        self.music_norm = nn.LayerNorm(int(model_dim))
        self.music_attention = MusicCrossAttention(
            model_dim,
            num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        hidden,
        key_padding_mask=None,
        cache=None,
        return_cache=False,
        *,
        music_context,
        music_cache=None,
    ):
        attended, new_cache = self.attention(
            self.norm1(hidden),
            key_padding_mask=key_padding_mask,
            cache=cache,
            return_cache=return_cache,
        )
        hidden = hidden + attended
        music, new_music_cache = self.music_attention(
            self.music_norm(hidden),
            context=music_context if music_cache is None else None,
            context_cache=music_cache,
        )
        hidden = hidden + music
        hidden = hidden + self.feedforward(self.norm2(hidden))
        return hidden, new_cache, new_music_cache


def validate_music_carrier(
    music,
    batch_size,
    *,
    tokens=MUSIC_FEATURE_TOKENS,
    channels=MUSIC_CARRIER_DIM,
):
    if music is None:
        raise ValueError("music-conditioned generator requires a music carrier")
    if music.shape != (int(batch_size), int(tokens), int(channels)):
        raise ValueError(
            f"music must have shape [B,{int(tokens)},{int(channels)}]"
        )
    if not torch.is_floating_point(music):
        raise ValueError("music carrier must be floating point")
    if not torch.isfinite(music).all():
        raise ValueError("music carrier contains non-finite values")
    return music


def apply_music_dropout(music, probability, *, training, generator=None):
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("music dropout probability must be in [0,1]")
    if not training or probability == 0.0:
        keep = torch.ones(music.shape[0], device=music.device, dtype=torch.bool)
        return music, keep
    keep = (
        torch.rand(
            music.shape[0],
            device=music.device,
            generator=generator,
        )
        >= probability
    )
    return music * keep[:, None, None].to(music.dtype), keep


def classifier_free_guidance(conditioned, unconditioned, scale):
    if conditioned.shape != unconditioned.shape:
        raise ValueError("conditioned and unconditioned tensors must align")
    scale = float(scale)
    if scale < 1.0:
        raise ValueError("formal CFG scale must be at least one")
    return unconditioned + (conditioned - unconditioned) * scale


def load_motion_parent_into_conditioned_model(model, parent_state):
    """Strictly load every parent key while allowing only new music keys."""

    if "state_dict" in parent_state:
        parent_state = parent_state["state_dict"]
    incompatible = model.load_state_dict(parent_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"parent has unexpected keys: {incompatible.unexpected_keys}"
        )
    invalid_missing = []
    for key in incompatible.missing_keys:
        if key.startswith("music_projection."):
            continue
        if (
            (key.startswith("blocks.") or key.startswith("temporal_blocks."))
            and (".music_norm." in key or ".music_attention." in key)
        ):
            continue
        invalid_missing.append(key)
    if invalid_missing:
        raise ValueError(
            "parent failed strict pre-existing-parameter load: "
            f"{invalid_missing}"
        )
    return incompatible.missing_keys
