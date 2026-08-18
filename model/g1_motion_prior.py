from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.motion_representation import (
    G1_YAW_DELTA_MOTION_FORMAT,
    decode_g1_motion,
    motion_repr_dim,
    validate_motion_format,
)


@dataclass(frozen=True)
class G1MotionPriorLossWeights:
    motion: float = 1.0
    velocity: float = 0.5
    acceleration: float = 0.1
    fk: float = 0.5
    contact_bce: float = 0.1
    contact_height: float = 0.2
    contact_slide: float = 0.1
    kl: float = 0.0


class ResidualConv1dBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class G1MotionAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=None,
        latent_dim=128,
        hidden_dim=256,
        temporal_downsample=2,
        prior_type="ae",
        dropout=0.0,
        motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    ):
        super().__init__()
        self.motion_format = validate_motion_format(motion_format)
        self.input_dim = int(input_dim or motion_repr_dim(self.motion_format))
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.temporal_downsample = int(temporal_downsample)
        if self.temporal_downsample not in (1, 2):
            raise ValueError("temporal_downsample must be 1 or 2 for V6b-A")
        if prior_type not in ("ae", "vae"):
            raise ValueError("prior_type must be 'ae' or 'vae'")
        self.prior_type = prior_type

        self.input_proj = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.encoder_pre = nn.Sequential(
            ResidualConv1dBlock(self.hidden_dim, dropout=dropout),
            ResidualConv1dBlock(self.hidden_dim, dropout=dropout),
        )
        self.downsample = (
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1)
            if self.temporal_downsample == 2
            else nn.Identity()
        )
        self.encoder_post = nn.Sequential(
            ResidualConv1dBlock(self.hidden_dim, dropout=dropout),
            ResidualConv1dBlock(self.hidden_dim, dropout=dropout),
        )
        if self.prior_type == "vae":
            self.mu_proj = nn.Conv1d(self.hidden_dim, self.latent_dim, kernel_size=1)
            self.logvar_proj = nn.Conv1d(self.hidden_dim, self.latent_dim, kernel_size=1)
        else:
            self.latent_proj = nn.Conv1d(self.hidden_dim, self.latent_dim, kernel_size=1)

        self.decoder_in = nn.Conv1d(self.latent_dim, self.hidden_dim, kernel_size=1)
        self.decoder_pre = nn.Sequential(
            ResidualConv1dBlock(self.hidden_dim, dropout=dropout),
            ResidualConv1dBlock(self.hidden_dim, dropout=dropout),
        )
        self.upsample = (
            nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1)
            if self.temporal_downsample == 2
            else nn.Identity()
        )
        self.decoder_post = nn.Sequential(
            ResidualConv1dBlock(self.hidden_dim, dropout=dropout),
            ResidualConv1dBlock(self.hidden_dim, dropout=dropout),
        )
        self.output_proj = nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=1)
        self.contact_head = nn.Conv1d(self.hidden_dim, 2, kernel_size=1)

    def encode(self, motion, sample=True):
        x = motion.transpose(1, 2)
        h = self.input_proj(x)
        h = self.encoder_pre(h)
        h = self.downsample(h)
        h = self.encoder_post(h)
        if self.prior_type == "vae":
            mu = self.mu_proj(h)
            logvar = self.logvar_proj(h).clamp(min=-12.0, max=8.0)
            if self.training and sample:
                std = torch.exp(0.5 * logvar)
                latent = mu + torch.randn_like(std) * std
            else:
                latent = mu
            return {
                "latent": latent.transpose(1, 2),
                "mu": mu.transpose(1, 2),
                "logvar": logvar.transpose(1, 2),
            }
        latent = self.latent_proj(h)
        return {
            "latent": latent.transpose(1, 2),
            "mu": None,
            "logvar": None,
        }

    def decode(self, latent):
        h = self.decoder_in(latent.transpose(1, 2))
        h = self.decoder_pre(h)
        h = self.upsample(h)
        h = self.decoder_post(h)
        recon = self.output_proj(h).transpose(1, 2)
        contact_logits = self.contact_head(h).transpose(1, 2)
        return recon, contact_logits

    def forward(self, motion, sample=True):
        encoded = self.encode(motion, sample=sample)
        recon, contact_logits = self.decode(encoded["latent"])
        if recon.shape[1] != motion.shape[1]:
            recon = recon[:, : motion.shape[1]]
            contact_logits = contact_logits[:, : motion.shape[1]]
        return {
            "recon": recon,
            "contact_logits": contact_logits,
            "latent": encoded["latent"],
            "mu": encoded["mu"],
            "logvar": encoded["logvar"],
        }


def _zero_like_loss(reference):
    return reference.sum() * 0.0


def _masked_mean_square(value, mask):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    denom = torch.clamp(mask.sum() * value.shape[-1], min=1.0)
    return (value.pow(2) * mask).sum() / denom


def _decode_for_fk(motion, mean, std, motion_format):
    raw = motion * std + mean
    return decode_g1_motion(raw.float(), motion_format=motion_format)


def contact_metrics_from_logits(contact_logits, target_contact):
    pred = (torch.sigmoid(contact_logits) >= 0.5).float()
    target = (target_contact >= 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-8)
    return {
        "contact_precision": precision.detach(),
        "contact_recall": recall.detach(),
        "contact_f1": f1.detach(),
        "contact_mean_gt": target.mean().detach(),
        "contact_mean_pred": pred.mean().detach(),
    }


def compute_g1_motion_prior_losses(
    output,
    target_motion,
    target_contact,
    mean,
    std,
    kinematics=None,
    motion_format=G1_YAW_DELTA_MOTION_FORMAT,
    weights=G1MotionPriorLossWeights(),
    ground=None,
    target_fk_keypoints=None,
    target_fk_feet=None,
):
    motion_format = validate_motion_format(motion_format)
    recon = output["recon"]
    motion_loss = F.mse_loss(recon, target_motion)
    velocity_loss = F.mse_loss(
        recon[:, 1:] - recon[:, :-1],
        target_motion[:, 1:] - target_motion[:, :-1],
    )
    acceleration_loss = F.mse_loss(
        recon[:, 2:] - 2.0 * recon[:, 1:-1] + recon[:, :-2],
        target_motion[:, 2:] - 2.0 * target_motion[:, 1:-1] + target_motion[:, :-2],
    )
    contact_bce = F.binary_cross_entropy_with_logits(
        output["contact_logits"],
        target_contact,
    )
    kl_loss = _zero_like_loss(motion_loss)
    if output.get("mu") is not None and output.get("logvar") is not None:
        mu = output["mu"]
        logvar = output["logvar"]
        kl_loss = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()

    fk_loss = _zero_like_loss(motion_loss)
    contact_height_loss = _zero_like_loss(motion_loss)
    contact_slide_loss = _zero_like_loss(motion_loss)
    fk_mpjpe = _zero_like_loss(motion_loss)
    if kinematics is not None and (
        weights.fk > 0.0 or weights.contact_height > 0.0 or weights.contact_slide > 0.0
    ):
        pred_decoded = _decode_for_fk(recon, mean, std, motion_format)
        pred_fk = kinematics(
            pred_decoded["root_pos"],
            pred_decoded["root_rot"],
            pred_decoded["dof_pos"],
        )
        target_fk = None
        if target_fk_keypoints is None or (ground is None and target_fk_feet is None):
            target_decoded = _decode_for_fk(target_motion, mean, std, motion_format)
            target_fk = kinematics(
                target_decoded["root_pos"],
                target_decoded["root_rot"],
                target_decoded["dof_pos"],
            )
        if target_fk_keypoints is None:
            target_keypoints = target_fk["keypoints"]
        else:
            target_keypoints = target_fk_keypoints.to(
                device=pred_fk["keypoints"].device,
                dtype=pred_fk["keypoints"].dtype,
            )
        keypoint_delta = pred_fk["keypoints"] - target_keypoints
        fk_loss = keypoint_delta.pow(2).mean()
        fk_mpjpe = keypoint_delta.norm(dim=-1).mean()
        pred_feet = pred_fk["feet"]
        if ground is None:
            if target_fk_feet is None:
                target_feet = target_fk["feet"]
            else:
                target_feet = target_fk_feet.to(device=pred_feet.device, dtype=pred_feet.dtype)
            ground = target_feet[..., 2].flatten(1).quantile(0.01, dim=1)
        ground = ground.to(device=pred_feet.device, dtype=pred_feet.dtype).view(-1, 1, 1)
        contact_mask = target_contact.to(device=pred_feet.device, dtype=pred_feet.dtype)
        contact_height_loss = ((pred_feet[..., 2] - ground).pow(2) * contact_mask).sum() / torch.clamp(
            contact_mask.sum(),
            min=1.0,
        )
        if pred_feet.shape[1] > 1:
            slide_mask = contact_mask[:, 1:] * contact_mask[:, :-1]
            pred_foot_velocity = pred_feet[:, 1:, :, :2] - pred_feet[:, :-1, :, :2]
            contact_slide_loss = _masked_mean_square(pred_foot_velocity, slide_mask)

    total = (
        weights.motion * motion_loss
        + weights.velocity * velocity_loss
        + weights.acceleration * acceleration_loss
        + weights.fk * fk_loss
        + weights.contact_bce * contact_bce
        + weights.contact_height * contact_height_loss
        + weights.contact_slide * contact_slide_loss
        + weights.kl * kl_loss
    )
    stats = {
        "loss/total": total.detach(),
        "loss/motion": motion_loss.detach(),
        "loss/velocity": velocity_loss.detach(),
        "loss/acceleration": acceleration_loss.detach(),
        "loss/fk": fk_loss.detach(),
        "loss/fk_mpjpe": fk_mpjpe.detach(),
        "loss/contact_bce": contact_bce.detach(),
        "loss/contact_height": contact_height_loss.detach(),
        "loss/contact_slide": contact_slide_loss.detach(),
        "loss/kl": kl_loss.detach(),
        "latent/mean": output["latent"].detach().mean(),
        "latent/std": output["latent"].detach().std(),
    }
    stats.update(contact_metrics_from_logits(output["contact_logits"], target_contact))
    return total, stats
