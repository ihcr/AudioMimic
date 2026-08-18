from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.g1_streaming_state import (
    G1_STREAMING_STATE_DIM,
    LEGACY_S66,
    resolve_boundary_state_spec,
)
from data.audio_extraction.discoforcing_causal import (
    MUSIC_CARRIER_DIM,
    MUSIC_FEATURE_TOKENS,
)
from model.g1_discoforcing_conditioning import (
    MusicConditionProjector,
    MusicConditionedCausalTransformerBlock,
    validate_music_carrier,
)
from model.g1_streaming_token_generator import CachedCausalTransformerBlock


HISTORY_TOKENS = 64
PLAN_TOKENS = 8
COMMIT_TOKENS = 4
TRAINING_ROLLOUT_TOKENS = 64
SELF_FORCING_GRADIENT_TAIL = 21
DIFFUSION_TIMESTEPS = 1000

PAPER_DC_STAGES = (
    "q0_base",
    "teacher",
    "ode_distill",
    "self_forcing_dmd",
    "self_forcing_gan",
    "diffusion_forcing",
    "df_homogeneous_control",
    "two_forward",
    "one_forward_control",
    "two_forward_no_rebasing",
    "two_forward_oracle_state",
    "commit_forcing",
    "y1_cf_joint_control",
    "y2_cf_frozenq0_control",
    "y3_smcf_frozenq0_p50",
    "y4_smcf_frozenq0_full",
)
JOINT_Q0_RESIDUAL_STAGES = (
    "two_forward",
    "one_forward_control",
    "two_forward_no_rebasing",
    "two_forward_oracle_state",
    "commit_forcing",
    "y1_cf_joint_control",
)
Y_ROUTE_STAGES = (
    "y1_cf_joint_control",
    "y2_cf_frozenq0_control",
    "y3_smcf_frozenq0_p50",
    "y4_smcf_frozenq0_full",
)
SMCF_STAGES = (
    "y3_smcf_frozenq0_p50",
    "y4_smcf_frozenq0_full",
)
COMMIT_FORCING_STAGES = ("commit_forcing", *Y_ROUTE_STAGES)
EMBEDDED_Q0_STAGES = (
    *JOINT_Q0_RESIDUAL_STAGES,
    "y2_cf_frozenq0_control",
    "y3_smcf_frozenq0_p50",
    "y4_smcf_frozenq0_full",
)


@dataclass(frozen=True)
class G1PaperFaithfulDCConfig:
    vocab_size: int = 512
    residual_dim: int = 256
    state_dim: int = G1_STREAMING_STATE_DIM
    state_layout: str = LEGACY_S66.name
    history_tokens: int = HISTORY_TOKENS
    max_target_tokens: int = TRAINING_ROLLOUT_TOKENS
    plan_tokens: int = PLAN_TOKENS
    commit_tokens: int = COMMIT_TOKENS
    model_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    diffusion_head_layers: int = 9
    dropout: float = 0.0
    diffusion_timesteps: int = DIFFUSION_TIMESTEPS
    music_conditioning: bool = False
    music_input_dim: int = MUSIC_CARRIER_DIM
    music_tokens: int = MUSIC_FEATURE_TOKENS
    music_dropout: float = 0.1

    def __post_init__(self):
        resolve_boundary_state_spec(self.state_layout, state_dim=self.state_dim)

    def manifest(self):
        payload = asdict(self)
        payload.update(
            {
                "model_type": "g1_paper_faithful_dc_streaming",
                "representation": "hard_q0_plus_continuous_residual",
                "motion_format": "g1_yaw_delta",
                "state_layout": self.state_layout,
                "deployment_contract": "K64_H8_C4_strict_causal",
                "training_rollout_tokens": TRAINING_ROLLOUT_TOKENS,
                "self_forcing_gradient_tail": SELF_FORCING_GRADIENT_TAIL,
                "diffusion_objective": "pred_x0",
                "diffusion_schedule": "cosine",
                "diffusion_forcing_noise": "independent_per_token_uniform",
                "diffusion_forcing_sampler": "pyramid_uncertainty_1_eta_0",
                "music_conditioning": bool(self.music_conditioning),
                "music_baseline": (
                    "DF-VQPAE-Causal-C1" if self.music_conditioning else "none"
                ),
                "music_injection": (
                    "cross_attention_every_q0_and_residual_temporal_block"
                    if self.music_conditioning
                    else "none"
                ),
                "music_history": (
                    "180_tokens_30hz_trailing_6s_zero_lookahead"
                    if self.music_conditioning
                    else "none"
                ),
            }
        )
        return payload


def _validate_history(history_q0, history_residual, history_valid, config):
    batch = history_q0.shape[0]
    if history_q0.shape != (batch, config.history_tokens):
        raise ValueError(
            f"history_q0 must have shape [B,{config.history_tokens}]"
        )
    if history_residual.shape != (
        batch,
        config.history_tokens,
        config.residual_dim,
    ):
        raise ValueError(
            "history_residual must have shape "
            f"[B,{config.history_tokens},{config.residual_dim}]"
        )
    if history_valid.shape != (batch, config.history_tokens):
        raise ValueError(
            f"history_valid must have shape [B,{config.history_tokens}]"
        )
    if history_q0.dtype != torch.long:
        raise ValueError("history_q0 must use torch.long IDs")
    return history_valid.to(device=history_q0.device, dtype=torch.bool)


def _validate_state(state, batch, state_dim):
    if state.shape != (batch, state_dim):
        raise ValueError(f"state must have shape [B,{state_dim}]")
    if not torch.isfinite(state).all():
        raise ValueError("state contains non-finite values")
    return state


def _state_schedule(state, batch, target_length, state_dim):
    if state.ndim == 2:
        current = _validate_state(state, batch, state_dim)
        target = current.unsqueeze(1).expand(-1, int(target_length), -1)
        return current, target
    if state.shape != (batch, int(target_length), state_dim):
        raise ValueError(
            f"state schedule must have shape [B,{target_length},{state_dim}]"
        )
    if not torch.isfinite(state).all():
        raise ValueError("state schedule contains non-finite values")
    return state[:, 0], state


def timestep_embedding(timesteps, dim, max_period=10000):
    if timesteps.ndim not in (1, 2):
        raise ValueError("timesteps must have shape [B] or [B,T]")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    arguments = timesteps.float().unsqueeze(-1) * frequencies
    embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[..., :1])), dim=-1)
    return embedding


class G1PaperQ0Generator(nn.Module):
    def __init__(self, config=G1PaperFaithfulDCConfig()):
        super().__init__()
        self.config = config
        if config.history_tokens != HISTORY_TOKENS:
            raise ValueError("paper q0 generator is locked to K64")
        if config.max_target_tokens < PLAN_TOKENS:
            raise ValueError("max_target_tokens must cover an H8 plan")
        if config.model_dim % config.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if config.music_conditioning and (
            config.music_input_dim,
            config.music_tokens,
        ) != (MUSIC_CARRIER_DIM, MUSIC_FEATURE_TOKENS):
            raise ValueError("DF music conditioning is locked to [180,35]")
        self.bos_token_id = config.vocab_size
        self.pad_token_id = config.vocab_size + 1
        self.embedding = nn.Embedding(config.vocab_size + 2, config.model_dim)
        self.position = nn.Parameter(
            torch.zeros(
                config.history_tokens + config.max_target_tokens,
                config.model_dim,
            )
        )
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        if config.music_conditioning:
            self.music_projection = MusicConditionProjector(
                config.music_input_dim,
                config.model_dim,
            )
            block_type = MusicConditionedCausalTransformerBlock
        else:
            self.music_projection = None
            block_type = CachedCausalTransformerBlock
        self.blocks = nn.ModuleList(
            [
                block_type(
                    config.model_dim,
                    config.num_heads,
                    config.ffn_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.model_dim)
        self.head = nn.Linear(config.model_dim, config.vocab_size)
        nn.init.normal_(self.position, std=0.02)

    def _project_music(self, music, batch):
        if not self.config.music_conditioning:
            if music is not None:
                raise ValueError("motion-only q0 generator does not accept music")
            return None
        music = validate_music_carrier(
            music,
            batch,
            tokens=self.config.music_tokens,
            channels=self.config.music_input_dim,
        )
        return self.music_projection(music)

    def _inputs(self, history_q0, history_valid, target_q0, state, music=None):
        batch, target_length = target_q0.shape
        if target_length <= 0 or target_length > self.config.max_target_tokens:
            raise ValueError("target_q0 length is outside the configured active window")
        dummy_residual = torch.zeros(
            batch,
            self.config.history_tokens,
            self.config.residual_dim,
            device=history_q0.device,
        )
        history_valid = _validate_history(
            history_q0,
            dummy_residual,
            history_valid,
            self.config,
        )
        current_state, target_states = _state_schedule(
            state,
            batch,
            target_length,
            self.config.state_dim,
        )
        history = history_q0.masked_fill(~history_valid, self.pad_token_id)
        bos = torch.full(
            (batch, 1),
            self.bos_token_id,
            device=history_q0.device,
            dtype=torch.long,
        )
        shifted_target = torch.cat((bos, target_q0[:, :-1]), dim=1)
        ids = torch.cat((history, shifted_target), dim=1)
        key_padding = torch.cat(
            (
                ~history_valid,
                torch.zeros(
                    batch,
                    target_length,
                    device=history_q0.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        if bool(history_valid.all()):
            key_padding = None
        hidden = self.embedding(ids)
        hidden = hidden + self.position[: ids.shape[1]].unsqueeze(0)
        state_hidden = torch.cat(
            (
                self.state_projection(current_state).unsqueeze(1).expand(
                    -1,
                    self.config.history_tokens,
                    -1,
                ),
                self.state_projection(target_states),
            ),
            dim=1,
        )
        hidden = hidden + state_hidden
        music_context = self._project_music(music, batch)
        return hidden, key_padding, target_length, music_context

    def forward(self, history_q0, history_valid, target_q0, state, music=None):
        if target_q0.dtype != torch.long or target_q0.ndim != 2:
            raise ValueError("target_q0 must have shape [B,T] with torch.long IDs")
        hidden, key_padding, target_length, music_context = self._inputs(
            history_q0,
            history_valid,
            target_q0,
            state,
            music,
        )
        for block in self.blocks:
            if self.config.music_conditioning:
                hidden, _, _ = block(
                    hidden,
                    key_padding_mask=key_padding,
                    music_context=music_context,
                )
            else:
                hidden, _ = block(hidden, key_padding_mask=key_padding)
        hidden = self.norm(hidden[:, -target_length:])
        return self.head(hidden)

    @staticmethod
    def sample_logits(logits, greedy, temperature, generator=None):
        if greedy:
            return logits.argmax(dim=-1)
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        probability = (logits.float() / float(temperature)).softmax(dim=-1)
        return torch.multinomial(
            probability,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)

    def _prefill(self, history_q0, history_valid, state, target_length, music=None):
        batch = history_q0.shape[0]
        music_context = self._project_music(music, batch)
        dummy_residual = torch.zeros(
            batch,
            self.config.history_tokens,
            self.config.residual_dim,
            device=history_q0.device,
        )
        history_valid = _validate_history(
            history_q0,
            dummy_residual,
            history_valid,
            self.config,
        )
        current_state, target_states = _state_schedule(
            state,
            batch,
            target_length,
            self.config.state_dim,
        )
        history = history_q0.masked_fill(~history_valid, self.pad_token_id)
        bos = torch.full(
            (batch, 1),
            self.bos_token_id,
            device=history_q0.device,
            dtype=torch.long,
        )
        ids = torch.cat((history, bos), dim=1)
        hidden = self.embedding(ids)
        hidden = hidden + self.position[: ids.shape[1]].unsqueeze(0)
        current_state_hidden = self.state_projection(current_state)
        target_state_hidden = self.state_projection(target_states)
        state_hidden = torch.cat(
            (
                current_state_hidden.unsqueeze(1).expand(
                    -1,
                    self.config.history_tokens,
                    -1,
                ),
                target_state_hidden[:, :1],
            ),
            dim=1,
        )
        hidden = hidden + state_hidden
        key_padding = torch.cat(
            (
                ~history_valid,
                torch.zeros(
                    batch,
                    1,
                    device=history_q0.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        if bool(history_valid.all()):
            key_padding = None
        caches = []
        for block in self.blocks:
            if self.config.music_conditioning:
                hidden, cache, music_cache = block(
                    hidden,
                    key_padding_mask=key_padding,
                    return_cache=True,
                    music_context=music_context,
                )
                caches.append((cache, music_cache))
            else:
                hidden, cache = block(
                    hidden,
                    key_padding_mask=key_padding,
                    return_cache=True,
                )
                caches.append(cache)
        logits = self.head(self.norm(hidden[:, -1]))
        return logits, caches, target_state_hidden, music_context

    def _cached_step(
        self,
        token,
        position_index,
        caches,
        state_hidden,
        music_context=None,
    ):
        hidden = self.embedding(token[:, None])
        hidden = hidden + self.position[int(position_index)][None, None]
        hidden = hidden + state_hidden.unsqueeze(1)
        key_padding = torch.zeros(
            token.shape[0],
            1,
            device=token.device,
            dtype=torch.bool,
        )
        new_caches = []
        for block, cache in zip(self.blocks, caches):
            if self.config.music_conditioning:
                self_cache, music_cache = cache
                hidden, new_cache, new_music_cache = block(
                    hidden,
                    key_padding_mask=key_padding,
                    cache=self_cache,
                    return_cache=True,
                    music_context=music_context,
                    music_cache=music_cache,
                )
                new_caches.append((new_cache, new_music_cache))
            else:
                hidden, new_cache = block(
                    hidden,
                    key_padding_mask=key_padding,
                    cache=cache,
                    return_cache=True,
                )
                new_caches.append(new_cache)
        logits = self.head(self.norm(hidden[:, 0]))
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        history_q0,
        history_valid,
        state,
        length=PLAN_TOKENS,
        *,
        music=None,
        greedy=True,
        temperature=1.0,
        generator=None,
    ):
        length = int(length)
        if length <= 0 or length > self.config.max_target_tokens:
            raise ValueError("generation length is outside the active window")
        logits, caches, target_state_hidden, music_context = self._prefill(
            history_q0,
            history_valid,
            state,
            length,
            music,
        )
        generated = []
        for step in range(length):
            generated.append(
                self.sample_logits(
                    logits,
                    greedy=greedy,
                    temperature=temperature,
                    generator=generator,
                )
            )
            if step + 1 < length:
                logits, caches = self._cached_step(
                    generated[-1],
                    self.config.history_tokens + step + 1,
                    caches,
                    target_state_hidden[:, step + 1],
                    music_context,
                )
        return torch.stack(generated, dim=1)

    @torch.no_grad()
    def generate_cfg(
        self,
        history_q0,
        history_valid,
        state,
        *,
        music,
        null_music,
        guidance_scale,
        length=PLAN_TOKENS,
        greedy=True,
        temperature=1.0,
        generator=None,
    ):
        """Autoregressive CFG with shared tokens and separate music K/V caches."""

        from model.g1_discoforcing_conditioning import classifier_free_guidance

        if not self.config.music_conditioning:
            raise ValueError("CFG requires a music-conditioned q0 generator")
        scale = float(guidance_scale)
        if scale == 1.0:
            return self.generate(
                history_q0,
                history_valid,
                state,
                length=length,
                music=music,
                greedy=greedy,
                temperature=temperature,
                generator=generator,
            )
        length = int(length)
        (
            conditioned_logits,
            conditioned_caches,
            state_hidden,
            conditioned_music,
        ) = self._prefill(
            history_q0,
            history_valid,
            state,
            length,
            music,
        )
        (
            unconditioned_logits,
            unconditioned_caches,
            null_state_hidden,
            unconditioned_music,
        ) = self._prefill(
            history_q0,
            history_valid,
            state,
            length,
            null_music,
        )
        if not torch.equal(state_hidden, null_state_hidden):
            raise RuntimeError("conditional and null q0 state schedules differ")
        generated = []
        for index in range(length):
            logits = classifier_free_guidance(
                conditioned_logits,
                unconditioned_logits,
                scale,
            )
            token = self.sample_logits(
                logits,
                greedy,
                temperature,
                generator=generator,
            )
            generated.append(token)
            if index + 1 == length:
                break
            conditioned_logits, conditioned_caches = self._cached_step(
                token,
                self.config.history_tokens + index + 1,
                conditioned_caches,
                state_hidden[:, index + 1],
                conditioned_music,
            )
            unconditioned_logits, unconditioned_caches = self._cached_step(
                token,
                self.config.history_tokens + index + 1,
                unconditioned_caches,
                state_hidden[:, index + 1],
                unconditioned_music,
            )
        return torch.stack(generated, dim=1)


class AdaLNResidualBlock(nn.Module):
    def __init__(self, model_dim, ffn_dim, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, model_dim * 3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, model_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, hidden, condition):
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        modulated = self.norm(hidden) * (1.0 + scale) + shift
        return hidden + gate * self.mlp(modulated)


class G1ResidualDiffusionGenerator(nn.Module):
    def __init__(self, config=G1PaperFaithfulDCConfig()):
        super().__init__()
        self.config = config
        if config.model_dim % config.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if config.music_conditioning and (
            config.music_input_dim,
            config.music_tokens,
        ) != (MUSIC_CARRIER_DIM, MUSIC_FEATURE_TOKENS):
            raise ValueError("DF music conditioning is locked to [180,35]")
        self.q0_embedding = nn.Embedding(config.vocab_size + 1, config.model_dim)
        self.residual_projection = nn.Linear(config.residual_dim, config.model_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        if config.music_conditioning:
            self.music_projection = MusicConditionProjector(
                config.music_input_dim,
                config.model_dim,
            )
            block_type = MusicConditionedCausalTransformerBlock
        else:
            self.music_projection = None
            block_type = CachedCausalTransformerBlock
        self.position = nn.Parameter(
            torch.zeros(
                config.history_tokens + config.max_target_tokens,
                config.model_dim,
            )
        )
        self.temporal_blocks = nn.ModuleList(
            [
                block_type(
                    config.model_dim,
                    config.num_heads,
                    config.ffn_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.temporal_norm = nn.LayerNorm(config.model_dim)
        self.noisy_projection = nn.Linear(config.residual_dim, config.model_dim)
        self.diffusion_head = nn.ModuleList(
            [
                AdaLNResidualBlock(
                    config.model_dim,
                    config.ffn_dim,
                    config.model_dim,
                )
                for _ in range(config.diffusion_head_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.residual_dim)
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _project_music(self, music, batch):
        if not self.config.music_conditioning:
            if music is not None:
                raise ValueError("motion-only residual generator does not accept music")
            return None
        music = validate_music_carrier(
            music,
            batch,
            tokens=self.config.music_tokens,
            channels=self.config.music_input_dim,
        )
        return self.music_projection(music)

    def forward(
        self,
        history_q0,
        history_residual,
        history_valid,
        target_q0,
        noisy_target_residual,
        noise_levels,
        state,
        music=None,
    ):
        history_valid = _validate_history(
            history_q0,
            history_residual,
            history_valid,
            self.config,
        )
        batch, target_length = target_q0.shape
        if target_length <= 0 or target_length > self.config.max_target_tokens:
            raise ValueError("target length is outside the configured active window")
        if target_q0.dtype != torch.long:
            raise ValueError("target_q0 must use torch.long IDs")
        if noisy_target_residual.shape != (
            batch,
            target_length,
            self.config.residual_dim,
        ):
            raise ValueError("noisy_target_residual shape does not match target_q0")
        if noise_levels.shape != (batch, target_length):
            raise ValueError("noise_levels must have shape [B,T]")
        current_state, target_states = _state_schedule(
            state,
            batch,
            target_length,
            self.config.state_dim,
        )
        music_context = self._project_music(music, batch)
        if noise_levels.dtype != torch.long:
            noise_levels = noise_levels.long()
        if int(noise_levels.min()) < 0 or int(noise_levels.max()) >= self.config.diffusion_timesteps:
            raise ValueError("noise_levels are outside the diffusion training range")

        pad_q0 = torch.full_like(history_q0, self.config.vocab_size)
        safe_history_q0 = torch.where(history_valid, history_q0, pad_q0)
        history_hidden = (
            self.q0_embedding(safe_history_q0)
            + self.residual_projection(history_residual)
        )
        target_time = self.time_projection(
            timestep_embedding(noise_levels, self.config.model_dim)
        )
        target_hidden = (
            self.q0_embedding(target_q0)
            + self.residual_projection(noisy_target_residual)
            + target_time
        )
        hidden = torch.cat((history_hidden, target_hidden), dim=1)
        hidden = hidden + self.position[: hidden.shape[1]].unsqueeze(0)
        state_hidden = torch.cat(
            (
                self.state_projection(current_state).unsqueeze(1).expand(
                    -1,
                    self.config.history_tokens,
                    -1,
                ),
                self.state_projection(target_states),
            ),
            dim=1,
        )
        hidden = hidden + state_hidden
        key_padding = torch.cat(
            (
                ~history_valid,
                torch.zeros(
                    batch,
                    target_length,
                    device=history_q0.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        if bool(history_valid.all()):
            key_padding = None
        for block in self.temporal_blocks:
            if self.config.music_conditioning:
                hidden, _, _ = block(
                    hidden,
                    key_padding_mask=key_padding,
                    music_context=music_context,
                )
            else:
                hidden, _ = block(hidden, key_padding_mask=key_padding)
        condition = self.temporal_norm(hidden[:, -target_length:]) + target_time
        prediction = self.noisy_projection(noisy_target_residual)
        for block in self.diffusion_head:
            prediction = block(prediction, condition)
        return self.output(self.output_norm(prediction))


def cosine_beta_schedule(timesteps=DIFFUSION_TIMESTEPS, s=0.008):
    steps = int(timesteps) + 1
    x = torch.linspace(0, int(timesteps), steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / int(timesteps)) + s) / (1 + s) * math.pi * 0.5)
    alpha_bar = (alpha_bar / alpha_bar[0]).square()
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0.0, 0.999).float()


def _extract(values, timesteps, reference):
    extracted = values.gather(0, timesteps.reshape(-1))
    return extracted.reshape(*timesteps.shape, *((1,) * (reference.ndim - timesteps.ndim)))


class PredX0CosineDiffusion(nn.Module):
    def __init__(self, timesteps=DIFFUSION_TIMESTEPS, eta=0.0):
        super().__init__()
        self.timesteps = int(timesteps)
        self.eta = float(eta)
        if self.timesteps <= 1:
            raise ValueError("timesteps must be greater than one")
        if self.eta != 0.0:
            raise ValueError("the formal D+C route is locked to DDIM eta 0")
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alpha_bar)
        self.register_buffer("sqrt_alphas_cumprod", alpha_bar.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            (1.0 - alpha_bar).sqrt(),
        )

    def q_sample(self, x0, noise_levels, noise=None):
        if noise is None:
            noise = torch.randn_like(x0).clamp(-20.0, 20.0)
        return (
            _extract(self.sqrt_alphas_cumprod, noise_levels, x0) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, noise_levels, x0)
            * noise
        )

    def training_loss(self, model_fn, x0, noise_levels, noise=None):
        if noise_levels.shape != x0.shape[:2]:
            raise ValueError("noise_levels must have one value per D+C token")
        if noise is None:
            noise = torch.randn_like(x0).clamp(-20.0, 20.0)
        noisy = self.q_sample(x0, noise_levels, noise)
        prediction = model_fn(noisy, noise_levels)
        if prediction.shape != x0.shape:
            raise ValueError("diffusion model prediction shape does not match x0")
        loss = F.mse_loss(prediction, x0.detach(), reduction="none").mean(dim=-1)
        return prediction, loss

    def ddim_step(self, current, predicted_x0, current_t, next_t):
        if current_t.shape != current.shape[:2] or next_t.shape != current.shape[:2]:
            raise ValueError("DDIM timesteps must have shape [B,T]")
        alpha = _extract(self.alphas_cumprod, current_t.clamp_min(0), current)
        next_alpha_values = torch.where(
            next_t < 0,
            torch.ones_like(next_t, dtype=self.alphas_cumprod.dtype),
            self.alphas_cumprod.gather(0, next_t.clamp_min(0).reshape(-1)).reshape(
                next_t.shape
            ),
        )
        alpha_next = next_alpha_values.reshape(
            *next_t.shape,
            *((1,) * (current.ndim - next_t.ndim)),
        )
        predicted_noise = (
            current - alpha.sqrt() * predicted_x0
        ) / (1.0 - alpha).clamp_min(1e-12).sqrt()
        return (
            alpha_next.sqrt() * predicted_x0
            + (1.0 - alpha_next).clamp_min(0.0).sqrt() * predicted_noise
        )

    def ddim_timesteps(self, nfe, device):
        nfe = int(nfe)
        if nfe <= 0 or nfe > self.timesteps:
            raise ValueError("NFE must be in [1,diffusion_timesteps]")
        values = torch.linspace(
            self.timesteps - 1,
            0,
            steps=nfe,
            device=device,
        ).round().long()
        return torch.cat((values, values.new_tensor([-1])))

    def sample(self, model_fn, shape, nfe, *, noise=None):
        current = (
            torch.randn(shape, device=self.alphas_cumprod.device)
            if noise is None
            else noise
        )
        timeline = self.ddim_timesteps(nfe, current.device)
        batch, length = current.shape[:2]
        for index in range(int(nfe)):
            current_t = torch.full(
                (batch, length),
                int(timeline[index]),
                device=current.device,
                dtype=torch.long,
            )
            next_t = torch.full_like(current_t, int(timeline[index + 1]))
            predicted_x0 = model_fn(current, current_t)
            current = self.ddim_step(current, predicted_x0, current_t, next_t)
        return current


def independent_token_noise_levels(batch, length, timesteps, device, generator=None):
    return torch.randint(
        0,
        int(timesteps),
        (int(batch), int(length)),
        device=device,
        generator=generator,
    )


def homogeneous_sequence_noise_levels(batch, length, timesteps, device, generator=None):
    levels = torch.randint(
        0,
        int(timesteps),
        (int(batch), 1),
        device=device,
        generator=generator,
    )
    return levels.expand(-1, int(length))


def pyramid_scheduling_matrix(horizon, sampling_steps, uncertainty_scale=1.0):
    horizon = int(horizon)
    sampling_steps = int(sampling_steps)
    uncertainty_scale = float(uncertainty_scale)
    if horizon <= 0 or sampling_steps <= 0 or uncertainty_scale < 0:
        raise ValueError("invalid pyramid schedule arguments")
    height = sampling_steps + int((horizon - 1) * uncertainty_scale) + 1
    rows = torch.arange(height).view(-1, 1)
    columns = torch.arange(horizon).view(1, -1)
    matrix = sampling_steps + (columns.float() * uncertainty_scale).long() - rows
    return matrix.clamp(0, sampling_steps).long()


def diffusion_level_to_timestep(level, sampling_steps, diffusion_timesteps):
    level = level.long()
    timestep = (
        level.float()
        / float(sampling_steps)
        * float(diffusion_timesteps - 1)
    ).round().long()
    return torch.where(level <= 0, torch.full_like(timestep, -1), timestep)


def diffusion_forcing_sample(
    diffusion,
    model_fn,
    shape,
    nfe,
    *,
    uncertainty_scale=1.0,
    noise=None,
):
    current = (
        torch.randn(shape, device=diffusion.alphas_cumprod.device)
        if noise is None
        else noise
    )
    schedule = pyramid_scheduling_matrix(shape[1], nfe, uncertainty_scale).to(
        current.device
    )
    batch = int(shape[0])
    for row in range(schedule.shape[0] - 1):
        current_level = schedule[row].unsqueeze(0).expand(batch, -1)
        next_level = schedule[row + 1].unsqueeze(0).expand(batch, -1)
        active = current_level.ne(next_level) & current_level.gt(0)
        if not active.any():
            continue
        current_t = diffusion_level_to_timestep(
            current_level,
            nfe,
            diffusion.timesteps,
        ).clamp_min(0)
        next_t = diffusion_level_to_timestep(
            next_level,
            nfe,
            diffusion.timesteps,
        )
        predicted_x0 = model_fn(current, current_t)
        proposal = diffusion.ddim_step(current, predicted_x0, current_t, next_t)
        current = torch.where(active.unsqueeze(-1), proposal, current)
    return current


def released_gradient_tail_mask(
    total_tokens=TRAINING_ROLLOUT_TOKENS,
    tail_tokens=SELF_FORCING_GRADIENT_TAIL,
    *,
    device=None,
):
    total_tokens = int(total_tokens)
    tail_tokens = int(tail_tokens)
    if total_tokens <= 0 or tail_tokens <= 0 or tail_tokens > total_tokens:
        raise ValueError("gradient tail must be within the rollout")
    mask = torch.zeros(total_tokens, device=device, dtype=torch.bool)
    mask[-tail_tokens:] = True
    return mask


def retain_gradient_where(value, mask):
    if mask.dtype != torch.bool:
        raise TypeError("gradient mask must be boolean")
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = mask.expand_as(value)
    except RuntimeError as error:
        raise ValueError("gradient mask is not broadcastable to value") from error
    return torch.where(mask, value, value.detach())


def stochastic_denoising_exit(
    diffusion,
    model_fn,
    noisy,
    denoising_timesteps,
    exit_index,
    *,
    enable_gradient,
    generator=None,
):
    timeline = torch.as_tensor(
        denoising_timesteps,
        device=noisy.device,
        dtype=torch.long,
    )
    exit_index = int(exit_index)
    if timeline.ndim != 1 or timeline.numel() == 0:
        raise ValueError("denoising_timesteps must be a non-empty list")
    if exit_index < 0 or exit_index >= timeline.numel():
        raise ValueError("exit_index is outside denoising_timesteps")
    current = noisy
    batch, length = current.shape[:2]
    for index, timestep in enumerate(timeline):
        levels = torch.full(
            (batch, length),
            int(timestep),
            device=current.device,
            dtype=torch.long,
        )
        if index == exit_index and enable_gradient:
            predicted_x0 = model_fn(current, levels)
        else:
            with torch.no_grad():
                predicted_x0 = model_fn(current, levels)
        if index == exit_index:
            return predicted_x0
        next_levels = torch.full_like(levels, int(timeline[index + 1]))
        with torch.no_grad():
            noise = torch.randn(
                predicted_x0.shape,
                device=predicted_x0.device,
                dtype=predicted_x0.dtype,
                generator=generator,
            )
            current = diffusion.q_sample(predicted_x0, next_levels, noise)
    raise RuntimeError("stochastic denoising exit did not return")


def self_forcing_rollout(
    *,
    history_q0,
    history_residual,
    history_valid,
    state,
    q0_plan_fn,
    residual_plan_fn,
    state_update_fn,
    rollout_tokens=TRAINING_ROLLOUT_TOKENS,
    plan_tokens=PLAN_TOKENS,
    commit_tokens=COMMIT_TOKENS,
    gradient_tail=SELF_FORCING_GRADIENT_TAIL,
):
    rollout_tokens = int(rollout_tokens)
    if rollout_tokens % int(commit_tokens):
        raise ValueError("rollout_tokens must be divisible by commit_tokens")
    gradient_tail_mask = released_gradient_tail_mask(
        rollout_tokens,
        gradient_tail,
        device=history_q0.device,
    )
    generated_q0 = []
    generated_residual = []
    generated_plan_q0 = []
    generated_plan_residual = []
    plan_states = []
    committed = 0
    while committed < rollout_tokens:
        with torch.no_grad():
            plan_q0 = q0_plan_fn(
                history_q0,
                history_residual,
                history_valid,
                state,
                int(plan_tokens),
            )
        plan_positions = committed + torch.arange(
            int(plan_tokens),
            device=gradient_tail_mask.device,
        )
        plan_gradient_mask = (
            (plan_positions < rollout_tokens)
            & gradient_tail_mask[
                plan_positions.clamp(max=rollout_tokens - 1)
            ]
        )
        enable_gradient = bool(plan_gradient_mask.any())
        plan_states.append(state)
        plan_residual = residual_plan_fn(
            history_q0,
            history_residual,
            history_valid,
            state,
            plan_q0,
            enable_gradient,
        )
        plan_residual = retain_gradient_where(
            plan_residual,
            plan_gradient_mask.view(1, -1),
        )
        commit_q0 = plan_q0[:, :commit_tokens]
        commit_residual = plan_residual[:, :commit_tokens]
        generated_plan_q0.append(plan_q0)
        generated_plan_residual.append(plan_residual)
        generated_q0.append(commit_q0)
        generated_residual.append(commit_residual)
        with torch.no_grad():
            state = state_update_fn(state, plan_q0, plan_residual)
            history_q0 = torch.cat((history_q0, commit_q0), dim=1)[
                :, -history_q0.shape[1] :
            ]
            history_residual = torch.cat(
                (history_residual, commit_residual.detach()),
                dim=1,
            )[:, -history_residual.shape[1] :]
            history_valid = torch.cat(
                (
                    history_valid,
                    torch.ones_like(commit_q0, dtype=torch.bool),
                ),
                dim=1,
            )[:, -history_valid.shape[1] :]
        committed += int(commit_tokens)
    q0 = torch.cat(generated_q0, dim=1)[:, :rollout_tokens]
    residual = torch.cat(generated_residual, dim=1)[:, :rollout_tokens]
    gradient_mask = gradient_tail_mask.unsqueeze(0).expand(q0.shape[0], -1)
    return {
        "q0": q0,
        "residual": residual,
        "gradient_mask": gradient_mask,
        "plan_q0": torch.stack(generated_plan_q0, dim=1),
        "plan_residual": torch.stack(generated_plan_residual, dim=1),
        "plan_states": torch.stack(plan_states, dim=1),
        "final_state": state,
        "history_q0": history_q0,
        "history_residual": history_residual,
        "history_valid": history_valid,
    }


def dmd_surrogate_loss(
    generated_latent,
    real_score_x0,
    fake_score_x0,
    gradient_mask=None,
):
    if not (
        generated_latent.shape
        == real_score_x0.shape
        == fake_score_x0.shape
    ):
        raise ValueError("DMD tensors must have identical shape")
    gradient = fake_score_x0 - real_score_x0
    normalizer = (
        generated_latent.detach() - real_score_x0.detach()
    ).abs().mean(dim=tuple(range(1, generated_latent.ndim)), keepdim=True)
    gradient = torch.nan_to_num(gradient / normalizer.clamp_min(1e-6))
    target = (generated_latent - gradient).detach()
    per_element = 0.5 * (generated_latent - target).double().square()
    if gradient_mask is not None:
        mask = gradient_mask
        while mask.ndim < per_element.ndim:
            mask = mask.unsqueeze(-1)
        selected = per_element.masked_select(mask)
        if selected.numel() == 0:
            raise ValueError("DMD gradient mask selects no elements")
        loss = selected.mean()
    else:
        loss = per_element.mean()
    return loss.to(generated_latent.dtype), {
        "dmd/gradient_abs_mean": gradient.abs().mean().detach(),
        "dmd/normalizer_mean": normalizer.mean().detach(),
    }


def fake_score_denoising_loss(predicted_x0, generated_x0):
    return F.mse_loss(predicted_x0, generated_x0.detach())


class G1LatentSequenceCritic(nn.Module):
    def __init__(self, config=G1PaperFaithfulDCConfig()):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.residual_dim, config.model_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.position = nn.Parameter(
            torch.zeros(config.max_target_tokens, config.model_dim)
        )
        self.blocks = nn.ModuleList(
            [
                CachedCausalTransformerBlock(
                    config.model_dim,
                    config.num_heads,
                    config.ffn_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.diffusion_head_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.model_dim)
        self.head = nn.Linear(config.model_dim, 1)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, noisy_latent, noise_levels, valid=None):
        batch, length, dimension = noisy_latent.shape
        if dimension != self.config.residual_dim:
            raise ValueError("critic latent dimension mismatch")
        if length > self.config.max_target_tokens:
            raise ValueError("critic sequence exceeds active-window capacity")
        if noise_levels.shape != (batch, length):
            raise ValueError("critic noise levels must have shape [B,T]")
        if valid is None:
            valid = torch.ones(
                batch,
                length,
                device=noisy_latent.device,
                dtype=torch.bool,
            )
        hidden = self.input_projection(noisy_latent)
        hidden = hidden + self.time_projection(
            timestep_embedding(noise_levels, self.config.model_dim)
        )
        hidden = hidden + self.position[:length].unsqueeze(0)
        for block in self.blocks:
            hidden, _ = block(hidden, key_padding_mask=~valid)
        token_logits = self.head(self.norm(hidden)).squeeze(-1)
        return (token_logits * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def gan_generator_loss(fake_logits):
    return F.softplus(-fake_logits).mean()


def gan_critic_loss(real_logits, fake_logits):
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def physical_auxiliary_scale(
    main_loss,
    auxiliary_loss,
    parameters=None,
    maximum_ratio=0.10,
    *,
    main_reference=None,
    auxiliary_reference=None,
):
    if main_reference is not None or auxiliary_reference is not None:
        if main_reference is None or auxiliary_reference is None:
            raise ValueError("both gradient-cap references must be provided")
        main_gradients = torch.autograd.grad(
            main_loss,
            main_reference,
            retain_graph=True,
            allow_unused=True,
        )
        auxiliary_gradients = torch.autograd.grad(
            auxiliary_loss,
            auxiliary_reference,
            retain_graph=True,
            allow_unused=True,
        )
    else:
        if parameters is None:
            raise ValueError("physical gradient cap requires parameters or references")
        parameters = [
            parameter for parameter in parameters if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("physical gradient cap requires trainable parameters")
        main_gradients = torch.autograd.grad(
            main_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        auxiliary_gradients = torch.autograd.grad(
            auxiliary_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

    def norm(gradients):
        values = [
            gradient.detach().float().square().sum()
            for gradient in gradients
            if gradient is not None
        ]
        if not values:
            return main_loss.new_zeros(())
        return torch.stack(values).sum().sqrt()

    main_norm = norm(main_gradients)
    auxiliary_norm = norm(auxiliary_gradients)
    scale = (
        float(maximum_ratio)
        * main_norm
        / auxiliary_norm.clamp_min(1e-12)
    ).clamp(max=1.0)
    return scale.detach(), {
        "gradient/main_norm": main_norm.detach(),
        "gradient/physical_aux_norm": auxiliary_norm.detach(),
        "gradient/physical_aux_scale": scale.detach(),
    }


def two_forward_replacement_fraction(step, total_steps):
    step = min(max(int(step), 0), int(total_steps))
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be positive")
    return 0.5 * (1.0 - math.cos(math.pi * step / int(total_steps)))


def complete_commit_replacement_mask(
    batch,
    transitions,
    *,
    fraction,
    device,
    generator=None,
    bernoulli=False,
):
    batch = int(batch)
    transitions = int(transitions)
    if batch <= 0 or transitions <= 0:
        raise ValueError("batch and transitions must be positive")
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("replacement fraction must be in [0,1]")
    if bernoulli:
        return torch.rand(
            batch,
            transitions,
            device=device,
            generator=generator,
        ) < fraction
    replace_count = int(transitions * fraction)
    mask = torch.zeros(
        batch,
        transitions,
        device=device,
        dtype=torch.bool,
    )
    if replace_count:
        random_rank = torch.rand(
            batch,
            transitions,
            device=device,
            generator=generator,
        )
        selected = random_rank.argsort(dim=1)[:, :replace_count]
        mask.scatter_(1, selected, True)
    return mask


def replace_complete_dc_tokens(
    q0,
    residual,
    predicted_q0,
    predicted_residual,
    eligible,
    *,
    fraction,
    generator=None,
):
    if q0.shape != predicted_q0.shape or q0.shape != eligible.shape:
        raise ValueError("q0, predicted_q0, and eligible shapes must match")
    if residual.shape != predicted_residual.shape or residual.shape[:2] != q0.shape:
        raise ValueError("residual replacement shapes do not match q0")
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("replacement fraction must be in [0,1]")
    eligible = eligible.bool()
    random_rank = torch.rand(
        eligible.shape,
        device=q0.device,
        generator=generator,
    ).masked_fill(~eligible, 2.0)
    mask = torch.zeros_like(eligible)
    for batch_index in range(q0.shape[0]):
        count = int(eligible[batch_index].sum())
        replace_count = int(count * fraction)
        if replace_count:
            selected = random_rank[batch_index].argsort()[:replace_count]
            mask[batch_index, selected] = True
    updated_q0 = torch.where(mask, predicted_q0.detach(), q0)
    updated_residual = torch.where(
        mask.unsqueeze(-1),
        predicted_residual.detach(),
        residual,
    )
    return updated_q0, updated_residual, mask


class ExponentialMovingAverage:
    def __init__(self, module, decay=0.99):
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("EMA decay must be in [0,1)")
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, module):
        for name, parameter in module.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, module):
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name])

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state):
        if not math.isclose(float(state["decay"]), self.decay):
            raise ValueError("EMA decay mismatch")
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA parameter set mismatch")
        self.shadow = {
            name: value.detach().clone()
            for name, value in state["shadow"].items()
        }
