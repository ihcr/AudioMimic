import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(buffer, timesteps, shape):
    values = buffer.gather(0, timesteps)
    return values.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat((args.sin(), args.cos()), dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class G1Beat8DLatentDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        beat_dim=8,
        latent_frames=75,
        beat_frames=150,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        ff_size=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.beat_dim = int(beat_dim)
        self.latent_frames = int(latent_frames)
        self.beat_frames = int(beat_frames)
        self.hidden_dim = int(hidden_dim)

        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.beat_projection = nn.Linear(self.beat_dim, self.hidden_dim)
        self.latent_pos = nn.Parameter(torch.randn(1, self.latent_frames, self.hidden_dim) * 0.02)
        self.beat_pos = nn.Parameter(torch.randn(1, self.beat_frames, self.hidden_dim) * 0.02)
        self.null_beat_tokens = nn.Parameter(torch.randn(1, self.beat_frames, self.hidden_dim) * 0.02)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.beat_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(num_heads),
                dim_feedforward=int(ff_size),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=int(num_heads),
                dim_feedforward=int(ff_size),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(num_layers),
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.latent_dim)

    def _encode_beat(self, beat_features, cond_drop_prob=0.0):
        if beat_features.shape[1:] != (self.beat_frames, self.beat_dim):
            raise ValueError(
                f"beat_features expected {(self.beat_frames, self.beat_dim)}, "
                f"got {tuple(beat_features.shape[1:])}"
            )
        tokens = self.beat_projection(beat_features) + self.beat_pos.to(beat_features.dtype)
        tokens = self.beat_encoder(tokens)
        if cond_drop_prob > 0.0:
            keep = (torch.rand(tokens.shape[0], device=tokens.device) >= float(cond_drop_prob))
            keep = keep.reshape(-1, 1, 1).to(tokens.dtype)
            null_tokens = self.null_beat_tokens.to(tokens.dtype)
            tokens = tokens * keep + null_tokens * (1.0 - keep)
        return tokens

    def forward(
        self,
        noisy_latent,
        beat_features,
        timesteps,
        semantic_features=None,
        cond_drop_prob=0.0,
    ):
        if noisy_latent.shape[1:] != (self.latent_frames, self.latent_dim):
            raise ValueError(
                f"noisy_latent expected {(self.latent_frames, self.latent_dim)}, "
                f"got {tuple(noisy_latent.shape[1:])}"
            )
        beat_tokens = self._encode_beat(beat_features, cond_drop_prob=cond_drop_prob)
        time = self.time_mlp(timesteps).unsqueeze(1)
        latent_tokens = (
            self.latent_projection(noisy_latent)
            + self.latent_pos.to(noisy_latent.dtype)
            + time.to(noisy_latent.dtype)
        )
        output = self.decoder(latent_tokens, beat_tokens)
        return self.output_projection(output)

    def guided_forward(
        self,
        noisy_latent,
        beat_features,
        timesteps,
        semantic_features=None,
        guidance_weight=1.0,
    ):
        if float(guidance_weight) == 1.0:
            return self.forward(noisy_latent, beat_features, timesteps, cond_drop_prob=0.0)
        conditioned = self.forward(noisy_latent, beat_features, timesteps, cond_drop_prob=0.0)
        unconditioned = self.forward(noisy_latent, beat_features, timesteps, cond_drop_prob=1.0)
        return unconditioned + (conditioned - unconditioned) * float(guidance_weight)


class G1MusicControlLatentDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        control_dim=8,
        semantic_dim=512,
        latent_frames=75,
        control_frames=150,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        ff_size=1024,
        dropout=0.1,
        use_wav2clip_semantic=False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.control_dim = int(control_dim)
        self.semantic_dim = int(semantic_dim)
        self.latent_frames = int(latent_frames)
        self.control_frames = int(control_frames)
        self.hidden_dim = int(hidden_dim)
        self.use_wav2clip_semantic = bool(use_wav2clip_semantic)

        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.control_projection = nn.Linear(self.control_dim, self.hidden_dim)
        self.latent_pos = nn.Parameter(torch.randn(1, self.latent_frames, self.hidden_dim) * 0.02)
        self.control_pos = nn.Parameter(torch.randn(1, self.control_frames, self.hidden_dim) * 0.02)
        self.null_control_tokens = nn.Parameter(
            torch.randn(1, self.control_frames, self.hidden_dim) * 0.02
        )

        self.semantic_projection = None
        self.semantic_encoder = None
        self.semantic_pos = None
        self.null_semantic_tokens = None
        if self.use_wav2clip_semantic:
            self.semantic_projection = nn.Linear(self.semantic_dim, self.hidden_dim)
            self.semantic_pos = nn.Parameter(
                torch.randn(1, self.control_frames, self.hidden_dim) * 0.02
            )
            self.null_semantic_tokens = nn.Parameter(
                torch.randn(1, self.control_frames, self.hidden_dim) * 0.02
            )

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.control_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(num_heads),
                dim_feedforward=int(ff_size),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        if self.use_wav2clip_semantic:
            self.semantic_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=int(num_heads),
                    dim_feedforward=int(ff_size),
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=2,
            )
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.hidden_dim,
                    nhead=int(num_heads),
                    dim_feedforward=int(ff_size),
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.control_film = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.latent_dim)

    def _drop_tokens(self, tokens, null_tokens, cond_drop_prob):
        if cond_drop_prob <= 0.0:
            return tokens
        keep = (torch.rand(tokens.shape[0], device=tokens.device) >= float(cond_drop_prob))
        keep = keep.reshape(-1, 1, 1).to(tokens.dtype)
        return tokens * keep + null_tokens.to(tokens.dtype) * (1.0 - keep)

    def _encode_control(self, control_features, cond_drop_prob=0.0):
        if control_features.shape[1:] != (self.control_frames, self.control_dim):
            raise ValueError(
                f"control_features expected {(self.control_frames, self.control_dim)}, "
                f"got {tuple(control_features.shape[1:])}"
            )
        tokens = (
            self.control_projection(control_features)
            + self.control_pos.to(control_features.dtype)
        )
        tokens = self.control_encoder(tokens)
        return self._drop_tokens(tokens, self.null_control_tokens, cond_drop_prob)

    def _encode_semantic(self, semantic_features, batch_size, dtype, device, cond_drop_prob=0.0):
        if not self.use_wav2clip_semantic:
            return None
        if semantic_features is None:
            tokens = self.null_semantic_tokens.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
            return tokens
        if semantic_features.shape[1:] != (self.control_frames, self.semantic_dim):
            raise ValueError(
                f"semantic_features expected {(self.control_frames, self.semantic_dim)}, "
                f"got {tuple(semantic_features.shape[1:])}"
            )
        tokens = (
            self.semantic_projection(semantic_features)
            + self.semantic_pos.to(semantic_features.dtype)
        )
        tokens = self.semantic_encoder(tokens)
        return self._drop_tokens(tokens, self.null_semantic_tokens, cond_drop_prob)

    def forward(
        self,
        noisy_latent,
        control_features,
        timesteps,
        semantic_features=None,
        cond_drop_prob=0.0,
    ):
        if noisy_latent.shape[1:] != (self.latent_frames, self.latent_dim):
            raise ValueError(
                f"noisy_latent expected {(self.latent_frames, self.latent_dim)}, "
                f"got {tuple(noisy_latent.shape[1:])}"
            )
        control_tokens = self._encode_control(control_features, cond_drop_prob=cond_drop_prob)
        semantic_tokens = self._encode_semantic(
            semantic_features,
            batch_size=noisy_latent.shape[0],
            dtype=noisy_latent.dtype,
            device=noisy_latent.device,
            cond_drop_prob=cond_drop_prob,
        )
        memory = control_tokens if semantic_tokens is None else torch.cat((semantic_tokens, control_tokens), dim=1)
        control_pool = control_tokens.mean(dim=1)
        time = self.time_mlp(timesteps).unsqueeze(1)
        latent_tokens = (
            self.latent_projection(noisy_latent)
            + self.latent_pos.to(noisy_latent.dtype)
            + time.to(noisy_latent.dtype)
        )
        output = latent_tokens
        for layer, film in zip(self.decoder_layers, self.control_film):
            scale, shift = film(control_pool).chunk(2, dim=-1)
            output = output * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            output = layer(output, memory)
        return self.output_projection(output)

    def guided_forward(
        self,
        noisy_latent,
        control_features,
        timesteps,
        semantic_features=None,
        guidance_weight=1.0,
    ):
        if float(guidance_weight) == 1.0:
            return self.forward(
                noisy_latent,
                control_features,
                timesteps,
                semantic_features=semantic_features,
                cond_drop_prob=0.0,
            )
        conditioned = self.forward(
            noisy_latent,
            control_features,
            timesteps,
            semantic_features=semantic_features,
            cond_drop_prob=0.0,
        )
        unconditioned = self.forward(
            noisy_latent,
            control_features,
            timesteps,
            semantic_features=semantic_features,
            cond_drop_prob=1.0,
        )
        return unconditioned + (conditioned - unconditioned) * float(guidance_weight)


class G1LatentDiffusion(nn.Module):
    def __init__(
        self,
        denoiser,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        cond_drop_prob=0.1,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.cond_drop_prob = float(cond_drop_prob)
        betas = torch.linspace(float(beta_start), float(beta_end), self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start, timesteps, noise):
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def _random_condition(self, features):
        if features is None:
            return None
        if features.shape[0] > 1:
            return features[torch.randperm(features.shape[0], device=features.device)]
        return torch.roll(features, shifts=37, dims=1)

    def _shift_condition(self, features):
        return torch.roll(features, shifts=20, dims=1)

    def _mse_per_batch(self, pred, target):
        return F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)

    def p_losses(
        self,
        x_start,
        control_features,
        semantic_features=None,
        control_rank_weight=0.0,
        control_rank_margin=0.02,
        semantic_rank_weight=0.0,
        semantic_rank_margin=0.01,
    ):
        timesteps = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise)
        pred_noise = self.denoiser(
            noisy,
            control_features,
            timesteps,
            semantic_features=semantic_features,
            cond_drop_prob=self.cond_drop_prob,
        )
        noise_loss = F.mse_loss(pred_noise, noise)
        total_loss = noise_loss
        stats = {
            "loss/total": total_loss.detach(),
            "loss/noise_mse": noise_loss.detach(),
            "loss/loss_noise": noise_loss.detach(),
            "loss_noise": noise_loss.detach(),
            "latent/std": x_start.detach().std(),
            "latent/mean": x_start.detach().mean(),
        }
        real_rank_pred = None
        real_rank_mse = None
        if control_rank_weight > 0.0:
            real_rank_pred = self.denoiser(
                noisy,
                control_features,
                timesteps,
                semantic_features=semantic_features,
                cond_drop_prob=0.0,
            )
            real_rank_mse = self._mse_per_batch(real_rank_pred, noise)
            shifted_pred = self.denoiser(
                noisy,
                self._shift_condition(control_features),
                timesteps,
                semantic_features=semantic_features,
                cond_drop_prob=0.0,
            )
            random_pred = self.denoiser(
                noisy,
                self._random_condition(control_features),
                timesteps,
                semantic_features=semantic_features,
                cond_drop_prob=0.0,
            )
            shifted_mse = self._mse_per_batch(shifted_pred, noise)
            random_mse = self._mse_per_batch(random_pred, noise)
            control_rank_loss = (
                F.relu(float(control_rank_margin) + real_rank_mse - shifted_mse).mean()
                + F.relu(float(control_rank_margin) + real_rank_mse - random_mse).mean()
            )
            total_loss = total_loss + float(control_rank_weight) * control_rank_loss
            stats.update(
                {
                    "loss/control_use": control_rank_loss.detach(),
                    "loss_control_use": control_rank_loss.detach(),
                    "delta_shift_control": (shifted_mse - real_rank_mse).detach().mean(),
                    "delta_random_control": (random_mse - real_rank_mse).detach().mean(),
                }
            )
        else:
            stats.update(
                {
                    "loss/control_use": x_start.new_tensor(0.0),
                    "loss_control_use": x_start.new_tensor(0.0),
                    "delta_shift_control": x_start.new_tensor(0.0),
                    "delta_random_control": x_start.new_tensor(0.0),
                }
            )

        if semantic_features is not None and semantic_rank_weight > 0.0:
            if real_rank_mse is None:
                real_rank_pred = self.denoiser(
                    noisy,
                    control_features,
                    timesteps,
                    semantic_features=semantic_features,
                    cond_drop_prob=0.0,
                )
                real_rank_mse = self._mse_per_batch(real_rank_pred, noise)
            random_semantic_pred = self.denoiser(
                noisy,
                control_features,
                timesteps,
                semantic_features=self._random_condition(semantic_features),
                cond_drop_prob=0.0,
            )
            random_semantic_mse = self._mse_per_batch(random_semantic_pred, noise)
            semantic_rank_loss = F.relu(
                float(semantic_rank_margin) + real_rank_mse - random_semantic_mse
            ).mean()
            total_loss = total_loss + float(semantic_rank_weight) * semantic_rank_loss
            stats.update(
                {
                    "loss/semantic_use": semantic_rank_loss.detach(),
                    "loss_semantic_use": semantic_rank_loss.detach(),
                    "delta_random_semantic": (random_semantic_mse - real_rank_mse).detach().mean(),
                }
            )
        else:
            stats.update(
                {
                    "loss/semantic_use": x_start.new_tensor(0.0),
                    "loss_semantic_use": x_start.new_tensor(0.0),
                    "delta_random_semantic": x_start.new_tensor(0.0),
                }
            )
        stats["loss/total"] = total_loss.detach()
        return total_loss, stats

    @torch.inference_mode()
    def ddim_sample(
        self,
        control_features,
        shape,
        semantic_features=None,
        sampling_steps=50,
        guidance_weight=1.0,
    ):
        device = control_features.device
        x = torch.randn(shape, device=device, dtype=control_features.dtype)
        steps = int(sampling_steps)
        time_points = torch.linspace(self.timesteps - 1, 0, steps + 1, device=device).long()
        for index in range(steps):
            t = time_points[index]
            t_next = time_points[index + 1]
            t_batch = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
            pred_noise = self.denoiser.guided_forward(
                x,
                control_features,
                t_batch,
                semantic_features=semantic_features,
                guidance_weight=guidance_weight,
            )
            alpha = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else x.new_tensor(1.0)
            x_start = (x - torch.sqrt(1.0 - alpha) * pred_noise) / torch.sqrt(alpha)
            x = torch.sqrt(alpha_next) * x_start + torch.sqrt(1.0 - alpha_next) * pred_noise
        return x
