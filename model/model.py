from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
from feature_config import (
    BEAT_FEATURES_8D_DIM,
    BEAT_FEATURES_8D_MOTION_BEATNESS_CONTROL_DIM,
    BODY_INTENSITY_DIM,
    GAUSSIAN_BEAT_DIM,
    MOTION_BEATNESS_DIM,
    MOTION_ENERGY_DIM,
    MOTION_INTENSITY_DIM,
    SUPPORT_BEATNESS_DIM,
    SUPPORT_CONTACT_DIM,
    UPPER_BEATNESS_DIM,
    WAV2CLIP_DIM,
    WAV2CLIP_BODY_SUPPORT_BEATNESS_CONTROL_DIM,
    WAV2CLIP_MOTION_ENERGY_BEAT_CONTROL_DIM,
    WAV2CLIP_MOTION_INTENSITY_BEATNESS_CONTROL_DIM,
    WAV2CLIP_STFT_BEAT_DIMS,
)


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


def _scaled_adapter_dims(latent_dim):
    wav2clip_dim = latent_dim // 2
    stft_dim = (latent_dim * 3) // 8
    beat_dim = latent_dim - wav2clip_dim - stft_dim
    return wav2clip_dim, stft_dim, beat_dim


def _stream_input_norm(stream_dim):
    if stream_dim == 1:
        return nn.Identity()
    return nn.LayerNorm(stream_dim)


class Wav2ClipStftBeatFusion(nn.Module):
    def __init__(self, fusion_mode, latent_dim):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.stream_dims = WAV2CLIP_STFT_BEAT_DIMS
        if fusion_mode == "concat_norm":
            self.stream_norms = nn.ModuleList(
                [_stream_input_norm(stream_dim) for stream_dim in self.stream_dims]
            )
            self.projection = nn.Linear(sum(self.stream_dims), latent_dim)
        elif fusion_mode == "stream_adapter":
            adapter_dims = _scaled_adapter_dims(latent_dim)
            self.adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        _stream_input_norm(stream_dim),
                        nn.Linear(stream_dim, adapter_dim),
                        nn.GELU(),
                        nn.LayerNorm(adapter_dim),
                    )
                    for stream_dim, adapter_dim in zip(self.stream_dims, adapter_dims)
                ]
            )
        else:
            raise ValueError(f"Unsupported wav2clip_stft_beat fusion mode: {fusion_mode}")

    def forward(self, cond_embed):
        streams = torch.split(cond_embed, self.stream_dims, dim=-1)
        if self.fusion_mode == "concat_norm":
            streams = [
                stream_norm(stream)
                for stream_norm, stream in zip(self.stream_norms, streams)
            ]
            return self.projection(torch.cat(streams, dim=-1))
        streams = [
            adapter(stream)
            for adapter, stream in zip(self.adapters, streams)
        ]
        return torch.cat(streams, dim=-1)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        cond_fusion: str = "linear",
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        if cond_fusion == "linear":
            self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        else:
            self.cond_projection = Wav2ClipStftBeatFusion(cond_fusion, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)

    def _apply_condition_dropout(self, cond_tokens, cond_hidden, keep_mask):
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.bool()
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1").to(cond_tokens.dtype)
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1").to(cond_hidden.dtype)
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        null_cond_hidden = self.null_cond_hidden.to(cond_hidden.dtype)
        cond_tokens = cond_tokens * keep_mask_embed + null_cond_embed * (
            1.0 - keep_mask_embed
        )
        cond_hidden = cond_hidden * keep_mask_hidden + null_cond_hidden * (
            1.0 - keep_mask_hidden
        )
        return cond_tokens, cond_hidden

    def _project_condition(self, cond_embed):
        return self.cond_projection(cond_embed)

    def _encode_condition_tokens(self, cond_embed):
        cond_tokens = self._project_condition(cond_embed)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)
        return cond_tokens

    def _encode_condition(self, cond_embed):
        cond_tokens = self._encode_condition_tokens(cond_embed)
        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)
        return cond_tokens, cond_hidden

    def guided_forward(self, x, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def forward(
        self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device

        # project to latent space
        x = self.input_projection(x)
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)

        cond_tokens, cond_hidden = self._encode_condition(cond_embed)
        cond_tokens, cond_hidden = self._apply_condition_dropout(
            cond_tokens, cond_hidden, keep_mask
        )

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        # FiLM conditioning
        t += cond_hidden

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        # Pass through the transformer decoder
        # attending to the conditional embedding
        output = self.seqTransDecoder(x, cond_tokens, t)

        output = self.final_layer(output)
        return output


class MotionEnergyPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        ff_size=1024,
        dropout=0.1,
        activation=F.gelu,
    ):
        super().__init__()
        input_dim = WAV2CLIP_DIM + GAUSSIAN_BEAT_DIM
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(hidden_dim, MOTION_ENERGY_DIM)

    def forward(self, wav2clip, gaussian_beat):
        tokens = torch.cat((wav2clip, gaussian_beat), dim=-1)
        tokens = self.input_projection(tokens)
        tokens = self.encoder(tokens)
        return torch.sigmoid(self.output_projection(tokens))


class MotionControlPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        ff_size=1024,
        dropout=0.1,
        activation=F.gelu,
    ):
        super().__init__()
        input_dim = WAV2CLIP_DIM + GAUSSIAN_BEAT_DIM
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.intensity_head = nn.Linear(hidden_dim, MOTION_INTENSITY_DIM)
        self.beatness_head = nn.Linear(hidden_dim, MOTION_BEATNESS_DIM)

    def forward(self, wav2clip, gaussian_beat):
        tokens = torch.cat((wav2clip, gaussian_beat), dim=-1)
        tokens = self.input_projection(tokens)
        tokens = self.encoder(tokens)
        return {
            "motion_intensity": torch.sigmoid(self.intensity_head(tokens)),
            "motion_beatness": torch.sigmoid(self.beatness_head(tokens)),
        }


class BodySupportControlPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        ff_size=1024,
        dropout=0.1,
        activation=F.gelu,
    ):
        super().__init__()
        input_dim = WAV2CLIP_DIM + GAUSSIAN_BEAT_DIM
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.body_intensity_head = nn.Linear(hidden_dim, BODY_INTENSITY_DIM)
        self.support_beatness_head = nn.Linear(hidden_dim, SUPPORT_BEATNESS_DIM)
        self.upper_beatness_head = nn.Linear(hidden_dim, UPPER_BEATNESS_DIM)
        self.support_contact_head = nn.Linear(hidden_dim, SUPPORT_CONTACT_DIM)

    def forward(self, wav2clip, gaussian_beat):
        tokens = torch.cat((wav2clip, gaussian_beat), dim=-1)
        tokens = self.input_projection(tokens)
        tokens = self.encoder(tokens)
        support_contact_logits = self.support_contact_head(tokens)
        return {
            "body_intensity": torch.sigmoid(self.body_intensity_head(tokens)),
            "support_beatness": torch.sigmoid(self.support_beatness_head(tokens)),
            "upper_beatness": torch.sigmoid(self.upper_beatness_head(tokens)),
            "support_contact_logits": support_contact_logits,
            "support_contact": torch.sigmoid(support_contact_logits),
        }


class BeatFeatures8DMotionBeatnessPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        ff_size=1024,
        dropout=0.1,
        activation=F.gelu,
    ):
        super().__init__()
        self.input_projection = nn.Linear(BEAT_FEATURES_8D_DIM, hidden_dim)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.beatness_head = nn.Linear(hidden_dim, MOTION_BEATNESS_DIM)

    def forward(self, beat_features):
        tokens = self.input_projection(beat_features)
        tokens = self.encoder(tokens)
        return torch.sigmoid(self.beatness_head(tokens))


class BeatFeatures8DMotionBeatnessDecoder(DanceDecoder):
    def __init__(
        self,
        *args,
        semantic_drop_prob: float = 0.10,
        control_drop_prob: float = 0.10,
        control_summary_drop_prob: float = 0.10,
        **kwargs,
    ) -> None:
        seq_len = kwargs.get("seq_len", 150)
        latent_dim = kwargs.get("latent_dim", 512)
        ff_size = kwargs.get("ff_size", 1024)
        num_heads = kwargs.get("num_heads", 8)
        dropout = kwargs.get("dropout", 0.1)
        activation = kwargs.get("activation", F.gelu)
        super().__init__(*args, **kwargs)
        self.semantic_drop_prob = float(semantic_drop_prob)
        self.control_drop_prob = float(control_drop_prob)
        self.control_summary_drop_prob = float(control_summary_drop_prob)
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len * 2, latent_dim))

        self.semantic_projection = nn.Linear(BEAT_FEATURES_8D_DIM, latent_dim)
        self.semantic_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
                for _ in range(2)
            ]
        )

        control_hidden_dim = 256
        self.control_projection = nn.Linear(
            BEAT_FEATURES_8D_MOTION_BEATNESS_CONTROL_DIM,
            control_hidden_dim,
        )
        control_rotary = RotaryEmbedding(dim=control_hidden_dim)
        self.control_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=control_hidden_dim,
                    nhead=4,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=control_rotary,
                )
                for _ in range(2)
            ]
        )
        self.control_output_projection = nn.Linear(control_hidden_dim, latent_dim)

        self.semantic_hidden_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.control_hidden_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.control_summary_projection = nn.Sequential(
            nn.Linear(MOTION_BEATNESS_DIM, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.control_predictor = BeatFeatures8DMotionBeatnessPredictor(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            ff_size=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.null_semantic_tokens = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_semantic_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_control_tokens = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_control_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_control_summary_hidden = nn.Parameter(torch.randn(1, latent_dim))

    @staticmethod
    def _structured_streams(cond_embed):
        if not isinstance(cond_embed, dict):
            raise TypeError("BeatFeatures8DMotionBeatnessDecoder expects a condition dict")
        semantic = cond_embed["semantic"]
        control = cond_embed["control"]
        return semantic["beat_features_8d"], control["motion_beatness"]

    @staticmethod
    def _clone_with_selected_controls(cond_embed, motion_beatness):
        return {
            "semantic": dict(cond_embed["semantic"]),
            "control": {
                **dict(cond_embed["control"]),
                "motion_beatness": motion_beatness,
            },
        }

    def predict_controls(self, cond_embed):
        beat_features, _ = self._structured_streams(cond_embed)
        return {"motion_beatness": self.control_predictor(beat_features)}

    def prepare_motion_energy_training_condition(
        self,
        cond_embed,
        epoch,
        teacher_forcing_epochs=100,
        pred_mix_prob=0.5,
        detach_pred=True,
    ):
        _, gt_beatness = self._structured_streams(cond_embed)
        pred_beatness = self.predict_controls(cond_embed)["motion_beatness"]
        if int(epoch) <= int(teacher_forcing_epochs):
            selected_beatness = gt_beatness
            pred_rate = 0.0
        else:
            mask = prob_mask_like(
                (gt_beatness.shape[0],),
                float(pred_mix_prob),
                device=gt_beatness.device,
            )
            mix_mask = rearrange(mask, "b -> b 1 1").to(gt_beatness.dtype)
            selected_pred_beatness = pred_beatness.detach() if detach_pred else pred_beatness
            selected_beatness = gt_beatness * (1.0 - mix_mask) + selected_pred_beatness * mix_mask
            pred_rate = float(mask.float().mean().detach().cpu())
        beatness_pred_loss = F.mse_loss(pred_beatness, gt_beatness)
        zero = pred_beatness.new_zeros(())
        prepared = self._clone_with_selected_controls(cond_embed, selected_beatness)
        stats = {
            "pred_beatness": pred_beatness,
            "selected_beatness": selected_beatness,
            "gt_beatness": gt_beatness,
            "beatness_pred_loss": beatness_pred_loss,
            "energy_pred_loss": beatness_pred_loss,
            "energy_smoothness_loss": zero,
            "energy_pred_mix_rate": pred_rate,
            "selected_beatness_mean": selected_beatness.detach().mean(),
            "pred_beatness_mean": pred_beatness.detach().mean(),
            "gt_beatness_mean": gt_beatness.detach().mean(),
        }
        return prepared, stats

    def _apply_branch_dropout(self, tokens, hidden, null_tokens, null_hidden, drop_prob):
        if not self.training or drop_prob <= 0:
            return tokens, hidden
        keep_mask = prob_mask_like((tokens.shape[0],), 1.0 - drop_prob, device=tokens.device)
        keep_tokens = rearrange(keep_mask, "b -> b 1 1").to(tokens.dtype)
        keep_hidden = rearrange(keep_mask, "b -> b 1").to(hidden.dtype)
        tokens = tokens * keep_tokens + null_tokens.to(tokens.dtype) * (1.0 - keep_tokens)
        hidden = hidden * keep_hidden + null_hidden.to(hidden.dtype) * (1.0 - keep_hidden)
        return tokens, hidden

    def _apply_control_summary_dropout(self, control_summary_hidden):
        if not self.training or self.control_summary_drop_prob <= 0:
            return control_summary_hidden
        keep_mask = prob_mask_like(
            (control_summary_hidden.shape[0],),
            1.0 - self.control_summary_drop_prob,
            device=control_summary_hidden.device,
        )
        keep_hidden = rearrange(keep_mask, "b -> b 1").to(control_summary_hidden.dtype)
        return control_summary_hidden * keep_hidden + self.null_control_summary_hidden.to(
            control_summary_hidden.dtype
        ) * (1.0 - keep_hidden)

    def _encode_condition(self, cond_embed):
        beat_features, motion_beatness = self._structured_streams(cond_embed)

        semantic_tokens = self.semantic_projection(beat_features)
        semantic_tokens = self.abs_pos_encoding(semantic_tokens)
        semantic_tokens = self.semantic_encoder(semantic_tokens)
        semantic_hidden = self.semantic_hidden_projection(semantic_tokens.mean(dim=-2))
        semantic_tokens, semantic_hidden = self._apply_branch_dropout(
            semantic_tokens,
            semantic_hidden,
            self.null_semantic_tokens,
            self.null_semantic_hidden,
            self.semantic_drop_prob,
        )

        control_tokens = self.control_projection(motion_beatness)
        control_tokens = self.control_encoder(control_tokens)
        control_tokens = self.control_output_projection(control_tokens)
        control_hidden = self.control_hidden_projection(control_tokens.mean(dim=-2))
        control_tokens, control_hidden = self._apply_branch_dropout(
            control_tokens,
            control_hidden,
            self.null_control_tokens,
            self.null_control_hidden,
            self.control_drop_prob,
        )

        control_summary_hidden = self.control_summary_projection(
            motion_beatness.mean(dim=-2)
        )
        control_summary_hidden = self._apply_control_summary_dropout(control_summary_hidden)

        cond_tokens = torch.cat((semantic_tokens, control_tokens), dim=-2)
        cond_hidden = semantic_hidden + control_hidden + control_summary_hidden
        return cond_tokens, cond_hidden


class Wav2ClipMotionEnergyBeatDecoder(DanceDecoder):
    def __init__(
        self,
        *args,
        semantic_drop_prob: float = 0.10,
        control_drop_prob: float = 0.10,
        energy_drop_prob: float = 0.10,
        **kwargs,
    ) -> None:
        seq_len = kwargs.get("seq_len", 150)
        latent_dim = kwargs.get("latent_dim", 512)
        ff_size = kwargs.get("ff_size", 1024)
        num_heads = kwargs.get("num_heads", 8)
        dropout = kwargs.get("dropout", 0.1)
        activation = kwargs.get("activation", F.gelu)
        super().__init__(*args, **kwargs)
        self.semantic_drop_prob = float(semantic_drop_prob)
        self.control_drop_prob = float(control_drop_prob)
        self.energy_drop_prob = float(energy_drop_prob)
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len * 2, latent_dim))

        self.semantic_projection = nn.Linear(WAV2CLIP_DIM, latent_dim)
        self.semantic_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
                for _ in range(2)
            ]
        )

        control_hidden_dim = 256
        self.control_projection = nn.Linear(
            WAV2CLIP_MOTION_ENERGY_BEAT_CONTROL_DIM,
            control_hidden_dim,
        )
        control_rotary = RotaryEmbedding(dim=control_hidden_dim)
        self.control_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=control_hidden_dim,
                    nhead=4,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=control_rotary,
                )
                for _ in range(2)
            ]
        )
        self.control_output_projection = nn.Linear(control_hidden_dim, latent_dim)

        self.semantic_hidden_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.control_hidden_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.energy_hidden_projection = nn.Sequential(
            nn.Linear(MOTION_ENERGY_DIM, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.energy_predictor = MotionEnergyPredictor(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            ff_size=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.null_semantic_tokens = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_semantic_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_control_tokens = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_control_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_energy_hidden = nn.Parameter(torch.randn(1, latent_dim))

    @staticmethod
    def _structured_streams(cond_embed):
        if not isinstance(cond_embed, dict):
            raise TypeError("Wav2ClipMotionEnergyBeatDecoder expects a condition dict")
        semantic = cond_embed["semantic"]
        control = cond_embed["control"]
        wav2clip = semantic["wav2clip"]
        gaussian_beat = control["gaussian_beat"]
        beat_energy = control["beat_energy_envelope"]
        return wav2clip, gaussian_beat, beat_energy

    @staticmethod
    def _clone_with_selected_energy(cond_embed, beat_energy):
        return {
            "semantic": dict(cond_embed["semantic"]),
            "control": {
                **dict(cond_embed["control"]),
                "beat_energy_envelope": beat_energy,
            },
        }

    def predict_energy(self, cond_embed):
        wav2clip, gaussian_beat, _ = self._structured_streams(cond_embed)
        return self.energy_predictor(wav2clip, gaussian_beat)

    def prepare_motion_energy_training_condition(
        self,
        cond_embed,
        epoch,
        teacher_forcing_epochs=100,
        pred_mix_prob=0.5,
        detach_pred=True,
    ):
        _, _, gt_energy = self._structured_streams(cond_embed)
        pred_energy = self.predict_energy(cond_embed)
        if int(epoch) <= int(teacher_forcing_epochs):
            selected_energy = gt_energy
            pred_rate = 0.0
        else:
            mask = prob_mask_like(
                (gt_energy.shape[0],),
                float(pred_mix_prob),
                device=gt_energy.device,
            )
            mix_mask = rearrange(mask, "b -> b 1 1").to(gt_energy.dtype)
            selected_pred = pred_energy.detach() if detach_pred else pred_energy
            selected_energy = gt_energy * (1.0 - mix_mask) + selected_pred * mix_mask
            pred_rate = float(mask.float().mean().detach().cpu())
        energy_pred_loss = F.mse_loss(pred_energy, gt_energy)
        if pred_energy.shape[1] > 1:
            energy_smoothness_loss = F.mse_loss(pred_energy[:, 1:], pred_energy[:, :-1])
        else:
            energy_smoothness_loss = pred_energy.new_zeros(())
        prepared = self._clone_with_selected_energy(cond_embed, selected_energy)
        stats = {
            "pred_energy": pred_energy,
            "selected_energy": selected_energy,
            "gt_energy": gt_energy,
            "energy_pred_loss": energy_pred_loss,
            "energy_smoothness_loss": energy_smoothness_loss,
            "energy_pred_mix_rate": pred_rate,
            "selected_energy_mean": selected_energy.detach().mean(),
            "pred_energy_mean": pred_energy.detach().mean(),
            "gt_energy_mean": gt_energy.detach().mean(),
        }
        return prepared, stats

    def _apply_branch_dropout(self, tokens, hidden, null_tokens, null_hidden, drop_prob):
        if not self.training or drop_prob <= 0:
            return tokens, hidden
        keep_mask = prob_mask_like((tokens.shape[0],), 1.0 - drop_prob, device=tokens.device)
        keep_tokens = rearrange(keep_mask, "b -> b 1 1").to(tokens.dtype)
        keep_hidden = rearrange(keep_mask, "b -> b 1").to(hidden.dtype)
        tokens = tokens * keep_tokens + null_tokens.to(tokens.dtype) * (1.0 - keep_tokens)
        hidden = hidden * keep_hidden + null_hidden.to(hidden.dtype) * (1.0 - keep_hidden)
        return tokens, hidden

    def _apply_energy_dropout(self, energy_hidden):
        if not self.training or self.energy_drop_prob <= 0:
            return energy_hidden
        keep_mask = prob_mask_like(
            (energy_hidden.shape[0],),
            1.0 - self.energy_drop_prob,
            device=energy_hidden.device,
        )
        keep_hidden = rearrange(keep_mask, "b -> b 1").to(energy_hidden.dtype)
        return energy_hidden * keep_hidden + self.null_energy_hidden.to(energy_hidden.dtype) * (
            1.0 - keep_hidden
        )

    def _encode_condition(self, cond_embed):
        wav2clip, gaussian_beat, beat_energy = self._structured_streams(cond_embed)

        semantic_tokens = self.semantic_projection(wav2clip)
        semantic_tokens = self.abs_pos_encoding(semantic_tokens)
        semantic_tokens = self.semantic_encoder(semantic_tokens)
        semantic_hidden = self.semantic_hidden_projection(semantic_tokens.mean(dim=-2))
        semantic_tokens, semantic_hidden = self._apply_branch_dropout(
            semantic_tokens,
            semantic_hidden,
            self.null_semantic_tokens,
            self.null_semantic_hidden,
            self.semantic_drop_prob,
        )

        control_input = torch.cat((gaussian_beat, beat_energy), dim=-1)
        control_tokens = self.control_projection(control_input)
        control_tokens = self.control_encoder(control_tokens)
        control_tokens = self.control_output_projection(control_tokens)
        control_hidden = self.control_hidden_projection(control_tokens.mean(dim=-2))
        control_tokens, control_hidden = self._apply_branch_dropout(
            control_tokens,
            control_hidden,
            self.null_control_tokens,
            self.null_control_hidden,
            self.control_drop_prob,
        )

        energy_hidden = self.energy_hidden_projection(beat_energy.mean(dim=-2))
        energy_hidden = self._apply_energy_dropout(energy_hidden)

        cond_tokens = torch.cat((semantic_tokens, control_tokens), dim=-2)
        cond_hidden = semantic_hidden + control_hidden + energy_hidden
        return cond_tokens, cond_hidden


class Wav2ClipMotionIntensityBeatnessDecoder(DanceDecoder):
    def __init__(
        self,
        *args,
        semantic_drop_prob: float = 0.10,
        control_drop_prob: float = 0.10,
        control_summary_drop_prob: float = 0.10,
        **kwargs,
    ) -> None:
        seq_len = kwargs.get("seq_len", 150)
        latent_dim = kwargs.get("latent_dim", 512)
        ff_size = kwargs.get("ff_size", 1024)
        num_heads = kwargs.get("num_heads", 8)
        dropout = kwargs.get("dropout", 0.1)
        activation = kwargs.get("activation", F.gelu)
        super().__init__(*args, **kwargs)
        self.semantic_drop_prob = float(semantic_drop_prob)
        self.control_drop_prob = float(control_drop_prob)
        self.control_summary_drop_prob = float(control_summary_drop_prob)
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len * 2, latent_dim))

        self.semantic_projection = nn.Linear(WAV2CLIP_DIM, latent_dim)
        self.semantic_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
                for _ in range(2)
            ]
        )

        control_hidden_dim = 256
        self.control_projection = nn.Linear(
            WAV2CLIP_MOTION_INTENSITY_BEATNESS_CONTROL_DIM,
            control_hidden_dim,
        )
        control_rotary = RotaryEmbedding(dim=control_hidden_dim)
        self.control_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=control_hidden_dim,
                    nhead=4,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=control_rotary,
                )
                for _ in range(2)
            ]
        )
        self.control_output_projection = nn.Linear(control_hidden_dim, latent_dim)

        self.semantic_hidden_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.control_hidden_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.control_summary_projection = nn.Sequential(
            nn.Linear(MOTION_INTENSITY_DIM + MOTION_BEATNESS_DIM, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.control_predictor = MotionControlPredictor(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            ff_size=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.null_semantic_tokens = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_semantic_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_control_tokens = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_control_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_control_summary_hidden = nn.Parameter(torch.randn(1, latent_dim))

    @staticmethod
    def _structured_streams(cond_embed):
        if not isinstance(cond_embed, dict):
            raise TypeError("Wav2ClipMotionIntensityBeatnessDecoder expects a condition dict")
        semantic = cond_embed["semantic"]
        control = cond_embed["control"]
        return (
            semantic["wav2clip"],
            control["gaussian_beat"],
            control["motion_intensity"],
            control["motion_beatness"],
        )

    @staticmethod
    def _clone_with_selected_controls(cond_embed, motion_intensity, motion_beatness):
        return {
            "semantic": dict(cond_embed["semantic"]),
            "control": {
                **dict(cond_embed["control"]),
                "motion_intensity": motion_intensity,
                "motion_beatness": motion_beatness,
            },
        }

    def predict_controls(self, cond_embed):
        wav2clip, gaussian_beat, _, _ = self._structured_streams(cond_embed)
        return self.control_predictor(wav2clip, gaussian_beat)

    def prepare_motion_energy_training_condition(
        self,
        cond_embed,
        epoch,
        teacher_forcing_epochs=100,
        pred_mix_prob=0.5,
        detach_pred=True,
    ):
        _, _, gt_intensity, gt_beatness = self._structured_streams(cond_embed)
        predictions = self.predict_controls(cond_embed)
        pred_intensity = predictions["motion_intensity"]
        pred_beatness = predictions["motion_beatness"]
        if int(epoch) <= int(teacher_forcing_epochs):
            selected_intensity = gt_intensity
            selected_beatness = gt_beatness
            pred_rate = 0.0
        else:
            mask = prob_mask_like(
                (gt_intensity.shape[0],),
                float(pred_mix_prob),
                device=gt_intensity.device,
            )
            mix_mask = rearrange(mask, "b -> b 1 1").to(gt_intensity.dtype)
            selected_pred_intensity = pred_intensity.detach() if detach_pred else pred_intensity
            selected_pred_beatness = pred_beatness.detach() if detach_pred else pred_beatness
            selected_intensity = gt_intensity * (1.0 - mix_mask) + selected_pred_intensity * mix_mask
            selected_beatness = gt_beatness * (1.0 - mix_mask) + selected_pred_beatness * mix_mask
            pred_rate = float(mask.float().mean().detach().cpu())
        intensity_pred_loss = F.mse_loss(pred_intensity, gt_intensity)
        beatness_pred_loss = F.mse_loss(pred_beatness, gt_beatness)
        if pred_intensity.shape[1] > 1:
            intensity_smoothness_loss = F.mse_loss(
                pred_intensity[:, 1:],
                pred_intensity[:, :-1],
            )
        else:
            intensity_smoothness_loss = pred_intensity.new_zeros(())
        prepared = self._clone_with_selected_controls(
            cond_embed,
            selected_intensity,
            selected_beatness,
        )
        combined_pred_loss = intensity_pred_loss + beatness_pred_loss
        stats = {
            "pred_intensity": pred_intensity,
            "pred_beatness": pred_beatness,
            "selected_intensity": selected_intensity,
            "selected_beatness": selected_beatness,
            "gt_intensity": gt_intensity,
            "gt_beatness": gt_beatness,
            "intensity_pred_loss": intensity_pred_loss,
            "beatness_pred_loss": beatness_pred_loss,
            "intensity_smoothness_loss": intensity_smoothness_loss,
            "energy_pred_loss": combined_pred_loss,
            "energy_smoothness_loss": intensity_smoothness_loss,
            "energy_pred_mix_rate": pred_rate,
            "selected_energy": selected_intensity,
            "selected_intensity_mean": selected_intensity.detach().mean(),
            "pred_intensity_mean": pred_intensity.detach().mean(),
            "gt_intensity_mean": gt_intensity.detach().mean(),
            "selected_beatness_mean": selected_beatness.detach().mean(),
            "pred_beatness_mean": pred_beatness.detach().mean(),
            "gt_beatness_mean": gt_beatness.detach().mean(),
            "selected_energy_mean": selected_intensity.detach().mean(),
            "pred_energy_mean": pred_intensity.detach().mean(),
            "gt_energy_mean": gt_intensity.detach().mean(),
        }
        return prepared, stats

    def _apply_branch_dropout(self, tokens, hidden, null_tokens, null_hidden, drop_prob):
        if not self.training or drop_prob <= 0:
            return tokens, hidden
        keep_mask = prob_mask_like((tokens.shape[0],), 1.0 - drop_prob, device=tokens.device)
        keep_tokens = rearrange(keep_mask, "b -> b 1 1").to(tokens.dtype)
        keep_hidden = rearrange(keep_mask, "b -> b 1").to(hidden.dtype)
        tokens = tokens * keep_tokens + null_tokens.to(tokens.dtype) * (1.0 - keep_tokens)
        hidden = hidden * keep_hidden + null_hidden.to(hidden.dtype) * (1.0 - keep_hidden)
        return tokens, hidden

    def _apply_control_summary_dropout(self, control_summary_hidden):
        if not self.training or self.control_summary_drop_prob <= 0:
            return control_summary_hidden
        keep_mask = prob_mask_like(
            (control_summary_hidden.shape[0],),
            1.0 - self.control_summary_drop_prob,
            device=control_summary_hidden.device,
        )
        keep_hidden = rearrange(keep_mask, "b -> b 1").to(control_summary_hidden.dtype)
        return control_summary_hidden * keep_hidden + self.null_control_summary_hidden.to(
            control_summary_hidden.dtype
        ) * (1.0 - keep_hidden)

    def _encode_condition(self, cond_embed):
        wav2clip, gaussian_beat, motion_intensity, motion_beatness = self._structured_streams(
            cond_embed
        )

        semantic_tokens = self.semantic_projection(wav2clip)
        semantic_tokens = self.abs_pos_encoding(semantic_tokens)
        semantic_tokens = self.semantic_encoder(semantic_tokens)
        semantic_hidden = self.semantic_hidden_projection(semantic_tokens.mean(dim=-2))
        semantic_tokens, semantic_hidden = self._apply_branch_dropout(
            semantic_tokens,
            semantic_hidden,
            self.null_semantic_tokens,
            self.null_semantic_hidden,
            self.semantic_drop_prob,
        )

        control_input = torch.cat((gaussian_beat, motion_intensity, motion_beatness), dim=-1)
        control_tokens = self.control_projection(control_input)
        control_tokens = self.control_encoder(control_tokens)
        control_tokens = self.control_output_projection(control_tokens)
        control_hidden = self.control_hidden_projection(control_tokens.mean(dim=-2))
        control_tokens, control_hidden = self._apply_branch_dropout(
            control_tokens,
            control_hidden,
            self.null_control_tokens,
            self.null_control_hidden,
            self.control_drop_prob,
        )

        control_summary_input = torch.cat(
            (motion_intensity.mean(dim=-2), motion_beatness.mean(dim=-2)),
            dim=-1,
        )
        control_summary_hidden = self.control_summary_projection(control_summary_input)
        control_summary_hidden = self._apply_control_summary_dropout(control_summary_hidden)

        cond_tokens = torch.cat((semantic_tokens, control_tokens), dim=-2)
        cond_hidden = semantic_hidden + control_hidden + control_summary_hidden
        return cond_tokens, cond_hidden


class Wav2ClipBodySupportBeatnessDecoder(DanceDecoder):
    def __init__(
        self,
        *args,
        semantic_drop_prob: float = 0.10,
        control_drop_prob: float = 0.10,
        control_summary_drop_prob: float = 0.10,
        **kwargs,
    ) -> None:
        seq_len = kwargs.get("seq_len", 150)
        latent_dim = kwargs.get("latent_dim", 512)
        ff_size = kwargs.get("ff_size", 1024)
        num_heads = kwargs.get("num_heads", 8)
        dropout = kwargs.get("dropout", 0.1)
        activation = kwargs.get("activation", F.gelu)
        super().__init__(*args, **kwargs)
        self.semantic_drop_prob = float(semantic_drop_prob)
        self.control_drop_prob = float(control_drop_prob)
        self.control_summary_drop_prob = float(control_summary_drop_prob)
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len * 2, latent_dim))

        self.semantic_projection = nn.Linear(WAV2CLIP_DIM, latent_dim)
        self.semantic_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
                for _ in range(2)
            ]
        )

        control_hidden_dim = 256
        self.control_projection = nn.Linear(
            WAV2CLIP_BODY_SUPPORT_BEATNESS_CONTROL_DIM,
            control_hidden_dim,
        )
        control_rotary = RotaryEmbedding(dim=control_hidden_dim)
        self.control_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=control_hidden_dim,
                    nhead=4,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=control_rotary,
                )
                for _ in range(2)
            ]
        )
        self.control_output_projection = nn.Linear(control_hidden_dim, latent_dim)

        self.semantic_hidden_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.control_hidden_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        summary_dim = (
            BODY_INTENSITY_DIM
            + SUPPORT_BEATNESS_DIM
            + UPPER_BEATNESS_DIM
            + SUPPORT_CONTACT_DIM
        )
        self.control_summary_projection = nn.Sequential(
            nn.Linear(summary_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.control_predictor = BodySupportControlPredictor(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            ff_size=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.null_semantic_tokens = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_semantic_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_control_tokens = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_control_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_control_summary_hidden = nn.Parameter(torch.randn(1, latent_dim))

    @staticmethod
    def _structured_streams(cond_embed):
        if not isinstance(cond_embed, dict):
            raise TypeError("Wav2ClipBodySupportBeatnessDecoder expects a condition dict")
        semantic = cond_embed["semantic"]
        control = cond_embed["control"]
        return (
            semantic["wav2clip"],
            control["gaussian_beat"],
            control["body_intensity"],
            control["support_beatness"],
            control["upper_beatness"],
            control["support_contact"],
        )

    @staticmethod
    def _clone_with_selected_controls(
        cond_embed,
        body_intensity,
        support_beatness,
        upper_beatness,
        support_contact,
    ):
        return {
            "semantic": dict(cond_embed["semantic"]),
            "control": {
                **dict(cond_embed["control"]),
                "body_intensity": body_intensity,
                "support_beatness": support_beatness,
                "upper_beatness": upper_beatness,
                "support_contact": support_contact,
            },
        }

    def predict_controls(self, cond_embed):
        wav2clip, gaussian_beat, _, _, _, _ = self._structured_streams(cond_embed)
        return self.control_predictor(wav2clip, gaussian_beat)

    def prepare_motion_energy_training_condition(
        self,
        cond_embed,
        epoch,
        teacher_forcing_epochs=100,
        pred_mix_prob=0.5,
        detach_pred=True,
    ):
        (
            _,
            _,
            gt_body_intensity,
            gt_support_beatness,
            gt_upper_beatness,
            gt_support_contact,
        ) = self._structured_streams(cond_embed)
        predictions = self.predict_controls(cond_embed)
        pred_body_intensity = predictions["body_intensity"]
        pred_support_beatness = predictions["support_beatness"]
        pred_upper_beatness = predictions["upper_beatness"]
        pred_support_contact = predictions["support_contact"]
        if int(epoch) <= int(teacher_forcing_epochs):
            selected_body_intensity = gt_body_intensity
            selected_support_beatness = gt_support_beatness
            selected_upper_beatness = gt_upper_beatness
            selected_support_contact = gt_support_contact
            pred_rate = 0.0
        else:
            mask = prob_mask_like(
                (gt_body_intensity.shape[0],),
                float(pred_mix_prob),
                device=gt_body_intensity.device,
            )
            mix_mask = rearrange(mask, "b -> b 1 1").to(gt_body_intensity.dtype)
            selected_pred_body = (
                pred_body_intensity.detach() if detach_pred else pred_body_intensity
            )
            selected_pred_support = (
                pred_support_beatness.detach() if detach_pred else pred_support_beatness
            )
            selected_pred_upper = (
                pred_upper_beatness.detach() if detach_pred else pred_upper_beatness
            )
            selected_pred_contact = (
                pred_support_contact.detach() if detach_pred else pred_support_contact
            )
            selected_body_intensity = (
                gt_body_intensity * (1.0 - mix_mask)
                + selected_pred_body * mix_mask
            )
            selected_support_beatness = (
                gt_support_beatness * (1.0 - mix_mask)
                + selected_pred_support * mix_mask
            )
            selected_upper_beatness = (
                gt_upper_beatness * (1.0 - mix_mask)
                + selected_pred_upper * mix_mask
            )
            selected_support_contact = (
                gt_support_contact * (1.0 - mix_mask)
                + selected_pred_contact * mix_mask
            )
            pred_rate = float(mask.float().mean().detach().cpu())
        body_pred_loss = F.mse_loss(pred_body_intensity, gt_body_intensity)
        support_pred_loss = F.mse_loss(pred_support_beatness, gt_support_beatness)
        upper_pred_loss = F.mse_loss(pred_upper_beatness, gt_upper_beatness)
        support_contact_pred_loss = F.binary_cross_entropy_with_logits(
            predictions["support_contact_logits"],
            gt_support_contact,
        )
        if pred_body_intensity.shape[1] > 1:
            body_smoothness_loss = F.mse_loss(
                pred_body_intensity[:, 1:],
                pred_body_intensity[:, :-1],
            )
        else:
            body_smoothness_loss = pred_body_intensity.new_zeros(())
        prepared = self._clone_with_selected_controls(
            cond_embed,
            selected_body_intensity,
            selected_support_beatness,
            selected_upper_beatness,
            selected_support_contact,
        )
        combined_pred_loss = (
            body_pred_loss
            + support_pred_loss
            + upper_pred_loss
            + support_contact_pred_loss
        )
        stats = {
            "pred_body_intensity": pred_body_intensity,
            "pred_support_beatness": pred_support_beatness,
            "pred_upper_beatness": pred_upper_beatness,
            "pred_support_contact": pred_support_contact,
            "selected_body_intensity": selected_body_intensity,
            "selected_support_beatness": selected_support_beatness,
            "selected_upper_beatness": selected_upper_beatness,
            "selected_support_contact": selected_support_contact,
            "gt_body_intensity": gt_body_intensity,
            "gt_support_beatness": gt_support_beatness,
            "gt_upper_beatness": gt_upper_beatness,
            "gt_support_contact": gt_support_contact,
            "intensity_pred_loss": body_pred_loss,
            "beatness_pred_loss": support_pred_loss,
            "body_intensity_pred_loss": body_pred_loss,
            "support_beatness_pred_loss": support_pred_loss,
            "upper_beatness_pred_loss": upper_pred_loss,
            "support_contact_pred_loss": support_contact_pred_loss,
            "intensity_smoothness_loss": body_smoothness_loss,
            "body_intensity_smoothness_loss": body_smoothness_loss,
            "energy_pred_loss": combined_pred_loss,
            "energy_smoothness_loss": body_smoothness_loss,
            "energy_pred_mix_rate": pred_rate,
            "selected_energy": selected_body_intensity,
            "selected_intensity": selected_body_intensity,
            "selected_beatness": selected_support_beatness,
            "selected_intensity_mean": selected_body_intensity.detach().mean(),
            "pred_intensity_mean": pred_body_intensity.detach().mean(),
            "gt_intensity_mean": gt_body_intensity.detach().mean(),
            "selected_beatness_mean": selected_support_beatness.detach().mean(),
            "pred_beatness_mean": pred_support_beatness.detach().mean(),
            "gt_beatness_mean": gt_support_beatness.detach().mean(),
            "selected_energy_mean": selected_body_intensity.detach().mean(),
            "pred_energy_mean": pred_body_intensity.detach().mean(),
            "gt_energy_mean": gt_body_intensity.detach().mean(),
            "body_intensity_mean_gt": gt_body_intensity.detach().mean(),
            "body_intensity_mean_pred": pred_body_intensity.detach().mean(),
            "support_beatness_mean_gt": gt_support_beatness.detach().mean(),
            "support_beatness_mean_pred": pred_support_beatness.detach().mean(),
            "upper_beatness_mean_gt": gt_upper_beatness.detach().mean(),
            "upper_beatness_mean_pred": pred_upper_beatness.detach().mean(),
            "support_contact_mean_gt": gt_support_contact.detach().mean(),
            "support_contact_mean_pred": pred_support_contact.detach().mean(),
        }
        return prepared, stats

    def _apply_branch_dropout(self, tokens, hidden, null_tokens, null_hidden, drop_prob):
        if not self.training or drop_prob <= 0:
            return tokens, hidden
        keep_mask = prob_mask_like((tokens.shape[0],), 1.0 - drop_prob, device=tokens.device)
        keep_tokens = rearrange(keep_mask, "b -> b 1 1").to(tokens.dtype)
        keep_hidden = rearrange(keep_mask, "b -> b 1").to(hidden.dtype)
        tokens = tokens * keep_tokens + null_tokens.to(tokens.dtype) * (1.0 - keep_tokens)
        hidden = hidden * keep_hidden + null_hidden.to(hidden.dtype) * (1.0 - keep_hidden)
        return tokens, hidden

    def _apply_control_summary_dropout(self, control_summary_hidden):
        if not self.training or self.control_summary_drop_prob <= 0:
            return control_summary_hidden
        keep_mask = prob_mask_like(
            (control_summary_hidden.shape[0],),
            1.0 - self.control_summary_drop_prob,
            device=control_summary_hidden.device,
        )
        keep_hidden = rearrange(keep_mask, "b -> b 1").to(control_summary_hidden.dtype)
        return control_summary_hidden * keep_hidden + self.null_control_summary_hidden.to(
            control_summary_hidden.dtype
        ) * (1.0 - keep_hidden)

    def _encode_condition(self, cond_embed):
        (
            wav2clip,
            gaussian_beat,
            body_intensity,
            support_beatness,
            upper_beatness,
            support_contact,
        ) = self._structured_streams(cond_embed)

        semantic_tokens = self.semantic_projection(wav2clip)
        semantic_tokens = self.abs_pos_encoding(semantic_tokens)
        semantic_tokens = self.semantic_encoder(semantic_tokens)
        semantic_hidden = self.semantic_hidden_projection(semantic_tokens.mean(dim=-2))
        semantic_tokens, semantic_hidden = self._apply_branch_dropout(
            semantic_tokens,
            semantic_hidden,
            self.null_semantic_tokens,
            self.null_semantic_hidden,
            self.semantic_drop_prob,
        )

        control_input = torch.cat(
            (
                gaussian_beat,
                body_intensity,
                support_beatness,
                upper_beatness,
                support_contact,
            ),
            dim=-1,
        )
        control_tokens = self.control_projection(control_input)
        control_tokens = self.control_encoder(control_tokens)
        control_tokens = self.control_output_projection(control_tokens)
        control_hidden = self.control_hidden_projection(control_tokens.mean(dim=-2))
        control_tokens, control_hidden = self._apply_branch_dropout(
            control_tokens,
            control_hidden,
            self.null_control_tokens,
            self.null_control_hidden,
            self.control_drop_prob,
        )

        control_summary_input = torch.cat(
            (
                body_intensity.mean(dim=-2),
                support_beatness.mean(dim=-2),
                upper_beatness.mean(dim=-2),
                support_contact.mean(dim=-2),
            ),
            dim=-1,
        )
        control_summary_hidden = self.control_summary_projection(control_summary_input)
        control_summary_hidden = self._apply_control_summary_dropout(control_summary_hidden)

        cond_tokens = torch.cat((semantic_tokens, control_tokens), dim=-2)
        cond_hidden = semantic_hidden + control_hidden + control_summary_hidden
        return cond_tokens, cond_hidden


class BeatEncoder(nn.Module):
    def __init__(
        self,
        beat_rep: str,
        seq_len: int,
        beat_emb_dim: int = 128,
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary: bool = True,
        max_distance_vocab: int = 151,
    ) -> None:
        super().__init__()
        self.beat_rep = beat_rep
        self.max_distance_vocab = max_distance_vocab
        self.rotary = RotaryEmbedding(dim=beat_emb_dim) if use_rotary else None
        self.abs_pos_encoding = (
            nn.Identity()
            if use_rotary
            else PositionalEncoding(beat_emb_dim, dropout, batch_first=True)
        )

        if beat_rep == "distance":
            self.input_projection = nn.Embedding(max_distance_vocab, beat_emb_dim)
        elif beat_rep == "pulse":
            self.input_projection = nn.Linear(1, beat_emb_dim)
        else:
            raise ValueError(f"Unsupported beat representation: {beat_rep}")

        self.encoder = nn.Sequential(
            TransformerEncoderLayer(
                d_model=beat_emb_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                rotary=self.rotary,
            ),
            TransformerEncoderLayer(
                d_model=beat_emb_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                rotary=self.rotary,
            ),
        )
        self.output_projection = nn.Linear(beat_emb_dim, latent_dim)

    def forward(self, beat):
        if self.beat_rep == "distance":
            if beat.ndim == 3 and beat.shape[-1] == 1:
                beat = beat.squeeze(-1)
            beat = beat.long().clamp_(0, self.max_distance_vocab - 1)
            tokens = self.input_projection(beat)
        else:
            if beat.ndim == 2:
                beat = beat.unsqueeze(-1)
            tokens = self.input_projection(beat.float())

        tokens = self.abs_pos_encoding(tokens)
        tokens = self.encoder(tokens)
        return self.output_projection(tokens)


class BeatDanceDecoder(DanceDecoder):
    def __init__(
        self,
        *args,
        beat_rep: str = "distance",
        beat_emb_dim: int = 128,
        max_distance_vocab: int = 151,
        **kwargs,
    ) -> None:
        seq_len = kwargs.get("seq_len", 150)
        latent_dim = kwargs.get("latent_dim", 256)
        ff_size = kwargs.get("ff_size", 1024)
        num_heads = kwargs.get("num_heads", 4)
        dropout = kwargs.get("dropout", 0.1)
        activation = kwargs.get("activation", F.gelu)
        use_rotary = kwargs.get("use_rotary", True)
        super().__init__(*args, **kwargs)
        self.beat_rep = beat_rep
        self.beat_encoder = BeatEncoder(
            beat_rep=beat_rep,
            seq_len=seq_len,
            beat_emb_dim=beat_emb_dim,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            use_rotary=use_rotary,
            max_distance_vocab=max_distance_vocab,
        )
        self.fuse_projection = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def _project_condition(self, cond_embed):
        if not isinstance(cond_embed, dict):
            raise TypeError("BeatDanceDecoder expects a condition dict")
        music_tokens = self.cond_projection(cond_embed["music"])
        beat_tokens = self.beat_encoder(cond_embed["beat"])
        return self.fuse_projection(torch.cat((music_tokens, beat_tokens), dim=-1))
