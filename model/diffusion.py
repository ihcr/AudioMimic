import copy
import os
import pickle
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from p_tqdm import p_map
from tqdm import tqdm

from dataset.motion_representation import (
    G1_MOTION_FORMAT,
    G1_ROOT_DELTA_MOTION_FORMAT,
    G1_YAW_DELTA_MOTION_FORMAT,
    SMPL_MOTION_FORMAT,
    decode_g1_motion,
    is_g1_motion_format,
    validate_motion_format,
)
from dataset.quaternion import ax_from_6v, quat_slerp
from model.g1_torch_kinematics import G1TorchKinematics
from rotation_transforms import (
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)
from vis import skeleton_render

from .utils import extract, make_beta_schedule


def identity(t, *args, **kwargs):
    return t


def move_cond_to_device(cond, device):
    if torch.is_tensor(cond):
        return cond.to(device)
    if isinstance(cond, dict):
        return {key: move_cond_to_device(value, device) for key, value in cond.items()}
    return cond


def cond_batch_size(cond):
    if torch.is_tensor(cond):
        return cond.shape[0]
    if isinstance(cond, dict):
        for value in cond.values():
            try:
                return cond_batch_size(value)
            except TypeError:
                continue
    raise TypeError("Unsupported condition type")


def cond_device(cond):
    if torch.is_tensor(cond):
        return cond.device
    if isinstance(cond, dict):
        for value in cond.values():
            try:
                return cond_device(value)
            except TypeError:
                continue
    raise TypeError("Unsupported condition type")


def slice_cond(cond, idx):
    if torch.is_tensor(cond):
        return cond[idx]
    if isinstance(cond, dict):
        batch_size = cond_batch_size(cond)
        sliced = {}
        for key, value in cond.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                sliced[key] = value[idx]
            elif isinstance(value, dict):
                sliced[key] = slice_cond(value, idx)
            else:
                sliced[key] = value
        return sliced
    raise TypeError("Unsupported condition type")


def designated_beat_frames_from_track(beat_track, beat_rep):
    if torch.is_tensor(beat_track):
        beat_array = beat_track.detach().cpu().numpy()
    else:
        beat_array = np.asarray(beat_track)

    if beat_rep == "distance":
        beat_array = np.asarray(beat_array, dtype=np.int64).reshape(-1)
        return np.flatnonzero(beat_array == 0).astype(np.int64)
    if beat_rep == "pulse":
        beat_array = np.asarray(beat_array, dtype=np.float32).reshape(-1)
        return np.flatnonzero(beat_array > 0.5).astype(np.int64)
    raise ValueError(f"Unsupported beat representation: {beat_rep}")


def build_saved_motion_metadata(cond, sample_idx, audio_path=None, beat_rep=None):
    metadata = {}
    if audio_path is not None:
        metadata["audio_path"] = audio_path

    if not isinstance(cond, dict) or "beat" not in cond:
        return metadata

    beat_value = cond["beat"]
    if torch.is_tensor(beat_value):
        beat_track = beat_value[sample_idx]
    else:
        beat_track = np.asarray(beat_value)[sample_idx]

    resolved_beat_rep = beat_rep
    if resolved_beat_rep is None:
        beat_shape = beat_track.shape if hasattr(beat_track, "shape") else np.asarray(beat_track).shape
        resolved_beat_rep = "pulse" if len(beat_shape) > 1 and beat_shape[-1] == 1 else "distance"

    metadata["designated_beat_frames"] = designated_beat_frames_from_track(
        beat_track, resolved_beat_rep
    )
    metadata["beat_rep"] = resolved_beat_rep
    return metadata


def build_long_saved_motion_metadata(cond, audio_path=None, beat_rep=None, stride_frames=75):
    metadata = {}
    if audio_path is not None:
        metadata["audio_path"] = audio_path

    if not isinstance(cond, dict) or "beat" not in cond:
        return metadata

    beat_value = cond["beat"]
    resolved_beat_rep = beat_rep
    stitched_beats = []
    num_slices = cond_batch_size(cond)
    for sample_idx in range(num_slices):
        if torch.is_tensor(beat_value):
            beat_track = beat_value[sample_idx]
        else:
            beat_track = np.asarray(beat_value)[sample_idx]
        if resolved_beat_rep is None:
            beat_shape = (
                beat_track.shape
                if hasattr(beat_track, "shape")
                else np.asarray(beat_track).shape
            )
            resolved_beat_rep = (
                "pulse" if len(beat_shape) > 1 and beat_shape[-1] == 1 else "distance"
            )
        slice_beats = designated_beat_frames_from_track(beat_track, resolved_beat_rep)
        stitched_beats.extend((slice_beats + sample_idx * stride_frames).tolist())

    metadata["designated_beat_frames"] = np.unique(
        np.asarray(stitched_beats, dtype=np.int64)
    )
    metadata["beat_rep"] = resolved_beat_rep
    return metadata


def effective_beat_loss_weight(epoch, lambda_beat, start_epoch=0, warmup_epochs=0):
    if lambda_beat <= 0:
        return 0.0
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return float(lambda_beat)
    progress = (epoch - start_epoch + 1) / float(warmup_epochs)
    return float(lambda_beat) * min(max(progress, 0.0), 1.0)


def compute_beat_loss_contribution(
    base_loss,
    beat_loss,
    effective_lambda_beat,
    max_fraction=0.0,
    cap_mode="hard",
):
    raw_contribution = beat_loss * effective_lambda_beat
    if max_fraction <= 0:
        cap = torch.full_like(raw_contribution, float("inf"))
        capped = torch.zeros_like(raw_contribution, dtype=torch.bool)
        return raw_contribution, capped, cap
    if cap_mode not in ("hard", "soft"):
        raise ValueError("beat_loss_cap_mode must be 'hard' or 'soft'")

    cap = base_loss.detach() * float(max_fraction)
    capped = raw_contribution > cap
    if cap_mode == "soft":
        safe_cap = torch.clamp(cap, min=torch.finfo(raw_contribution.dtype).eps)
        contribution = cap * torch.tanh(raw_contribution / safe_cap)
        return contribution, capped, cap

    contribution = torch.where(capped, cap, raw_contribution)
    return contribution, capped, cap


def effective_warmup_scale(epoch, warmup_epochs=0):
    if warmup_epochs <= 0:
        return 1.0
    return min(max(float(epoch) / float(warmup_epochs), 0.0), 1.0)


def effective_linear_ramp(epoch, start_value, final_value=None, start_epoch=0, warmup_epochs=0):
    if final_value is None:
        return float(start_value)
    if epoch <= start_epoch:
        return float(start_value)
    if warmup_epochs <= 0:
        return float(final_value)
    progress = (epoch - start_epoch) / float(warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return float(start_value) + (float(final_value) - float(start_value)) * progress


def compute_capped_aux_contribution(base_loss, raw_contribution, max_fraction=0.0):
    if max_fraction <= 0:
        cap = torch.full_like(raw_contribution, float("inf"))
        capped = torch.zeros_like(raw_contribution, dtype=torch.bool)
        return raw_contribution, capped, cap
    cap = base_loss.detach() * float(max_fraction)
    capped = raw_contribution > cap
    safe_cap = torch.clamp(cap, min=torch.finfo(raw_contribution.dtype).eps)
    contribution = cap * torch.tanh(raw_contribution / safe_cap)
    return contribution, capped, cap


def transform_beat_target_for_estimator(beat_target, beat_spacing, estimator_config):
    target_transform = (estimator_config or {}).get("target_transform", "raw_distance")
    if target_transform == "raw_distance":
        return beat_target.float()
    if target_transform == "relative_distance":
        spacing = torch.clamp(beat_spacing.float(), min=1.0)
        return torch.clamp(beat_target.float() / spacing, min=0.0, max=1.0)
    raise ValueError(f"Unsupported beat estimator target_transform: {target_transform}")


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self._parameter_pairs = None
        self._parameter_pair_model_ids = None

    def update_model_average(self, ma_model, current_model):
        parameter_pair_model_ids = (id(ma_model), id(current_model))
        if self._parameter_pair_model_ids != parameter_pair_model_ids:
            current_parameters = list(current_model.parameters())
            ma_parameters = list(ma_model.parameters())
            if len(current_parameters) != len(ma_parameters):
                raise ValueError(
                    "EMA model parameter count mismatch: "
                    f"current={len(current_parameters)} ema={len(ma_parameters)}"
                )
            self._parameter_pairs = list(zip(current_parameters, ma_parameters))
            self._parameter_pair_model_ids = parameter_pair_model_ids

        for current_params, ma_params in self._parameter_pairs:
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
        beat_estimator=None,
        lambda_acc=0.1,
        lambda_beat=0.5,
        beat_a=10.0,
        beat_c=0.1,
        beat_loss_start_epoch=0,
        beat_loss_warmup_epochs=0,
        beat_loss_max_fraction=0.0,
        beat_loss_cap_mode="hard",
        motion_format=SMPL_MOTION_FORMAT,
        normalizer=None,
        lambda_g1_fk=0.0,
        lambda_g1_fk_vel=0.0,
        lambda_g1_fk_acc=0.0,
        lambda_g1_foot=0.0,
        lambda_g1_kin=1.0,
        g1_kin_loss_warmup_epochs=0,
        g1_kin_loss_max_fraction=0.0,
        g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        g1_root_quat_order="xyzw",
        lambda_g1_root_angular=0.0,
        g1_root_angular_velocity_margin=3.141592653589793,
        g1_root_angular_acceleration_margin=18.84955592153876,
        g1_root_angular_acceleration_weight=0.25,
        g1_root_angular_max_fraction=0.05,
        lambda_motion_energy=0.0,
        lambda_motion_intensity=None,
        lambda_motion_beatness=0.0,
        motion_energy_frame="world",
        motion_beatness_warmup_start_epoch=100,
        motion_beatness_warmup_epochs=400,
        motion_beatness_max_fraction=0.1,
        lambda_energy_pred=0.0,
        energy_teacher_forcing_epochs=100,
        energy_pred_mix_prob=0.5,
        energy_smoothness_weight=0.1,
        lambda_acc_final=None,
        lambda_acc_warmup_start_epoch=0,
        lambda_acc_warmup_epochs=0,
        motion_energy_norm_p05=None,
        motion_energy_norm_p95=None,
        motion_beatness_norm_p05=None,
        motion_beatness_norm_p95=None,
        motion_energy_keypoint_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.motion_format = validate_motion_format(motion_format)
        self.model = model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)

        self.cond_drop_prob = cond_drop_prob
        self.lambda_acc = lambda_acc
        self.lambda_beat = lambda_beat
        self.beat_a = beat_a
        self.beat_c = beat_c
        self.beat_loss_start_epoch = beat_loss_start_epoch
        self.beat_loss_warmup_epochs = beat_loss_warmup_epochs
        self.beat_loss_max_fraction = beat_loss_max_fraction
        if beat_loss_cap_mode not in ("hard", "soft"):
            raise ValueError("beat_loss_cap_mode must be 'hard' or 'soft'")
        self.beat_loss_cap_mode = beat_loss_cap_mode
        self.normalizer = normalizer
        self.lambda_g1_fk = float(lambda_g1_fk)
        self.lambda_g1_fk_vel = float(lambda_g1_fk_vel)
        self.lambda_g1_fk_acc = float(lambda_g1_fk_acc)
        self.lambda_g1_foot = float(lambda_g1_foot)
        self.lambda_g1_kin = float(lambda_g1_kin)
        self.g1_kin_loss_warmup_epochs = int(g1_kin_loss_warmup_epochs)
        self.g1_kin_loss_max_fraction = float(g1_kin_loss_max_fraction)
        self.g1_root_quat_order = g1_root_quat_order
        self.lambda_g1_root_angular = float(lambda_g1_root_angular)
        self.g1_root_angular_velocity_margin = float(g1_root_angular_velocity_margin)
        self.g1_root_angular_acceleration_margin = float(
            g1_root_angular_acceleration_margin
        )
        self.g1_root_angular_acceleration_weight = float(
            g1_root_angular_acceleration_weight
        )
        self.g1_root_angular_max_fraction = float(g1_root_angular_max_fraction)
        self.lambda_motion_energy = float(lambda_motion_energy)
        self.lambda_motion_intensity = (
            self.lambda_motion_energy
            if lambda_motion_intensity is None
            else float(lambda_motion_intensity)
        )
        self.lambda_motion_beatness = float(lambda_motion_beatness)
        if motion_energy_frame not in ("world", "root_local"):
            raise ValueError("motion_energy_frame must be 'world' or 'root_local'.")
        self.motion_energy_frame = motion_energy_frame
        self.motion_beatness_warmup_start_epoch = int(motion_beatness_warmup_start_epoch)
        self.motion_beatness_warmup_epochs = int(motion_beatness_warmup_epochs)
        self.motion_beatness_max_fraction = float(motion_beatness_max_fraction)
        self.lambda_energy_pred = float(lambda_energy_pred)
        self.energy_teacher_forcing_epochs = int(energy_teacher_forcing_epochs)
        self.energy_pred_mix_prob = float(energy_pred_mix_prob)
        self.energy_smoothness_weight = float(energy_smoothness_weight)
        self.lambda_acc_final = None if lambda_acc_final is None else float(lambda_acc_final)
        self.lambda_acc_warmup_start_epoch = int(lambda_acc_warmup_start_epoch)
        self.lambda_acc_warmup_epochs = int(lambda_acc_warmup_epochs)
        self.motion_energy_norm_p05 = (
            None if motion_energy_norm_p05 is None else float(motion_energy_norm_p05)
        )
        self.motion_energy_norm_p95 = (
            None if motion_energy_norm_p95 is None else float(motion_energy_norm_p95)
        )
        self.motion_beatness_norm_p05 = (
            None if motion_beatness_norm_p05 is None else float(motion_beatness_norm_p05)
        )
        self.motion_beatness_norm_p95 = (
            None if motion_beatness_norm_p95 is None else float(motion_beatness_norm_p95)
        )
        g1_weights = (
            self.lambda_g1_fk,
            self.lambda_g1_fk_vel,
            self.lambda_g1_fk_acc,
            self.lambda_g1_foot,
        )
        if any(weight < 0 for weight in g1_weights):
            raise ValueError("G1 robot loss weights must be non-negative.")
        if self.lambda_g1_kin < 0:
            raise ValueError("lambda_g1_kin must be non-negative.")
        if self.g1_kin_loss_warmup_epochs < 0:
            raise ValueError("g1_kin_loss_warmup_epochs must be non-negative.")
        if self.g1_kin_loss_max_fraction < 0:
            raise ValueError("g1_kin_loss_max_fraction must be non-negative.")
        root_angular_weights = (
            self.lambda_g1_root_angular,
            self.g1_root_angular_velocity_margin,
            self.g1_root_angular_acceleration_margin,
            self.g1_root_angular_acceleration_weight,
            self.g1_root_angular_max_fraction,
        )
        if any(weight < 0 for weight in root_angular_weights):
            raise ValueError("G1 root angular loss settings must be non-negative.")
        if self.lambda_motion_energy < 0:
            raise ValueError("lambda_motion_energy must be non-negative.")
        if self.lambda_motion_intensity < 0:
            raise ValueError("lambda_motion_intensity must be non-negative.")
        if self.lambda_motion_beatness < 0:
            raise ValueError("lambda_motion_beatness must be non-negative.")
        if self.motion_beatness_warmup_start_epoch < 0:
            raise ValueError("motion_beatness_warmup_start_epoch must be non-negative.")
        if self.motion_beatness_warmup_epochs < 0:
            raise ValueError("motion_beatness_warmup_epochs must be non-negative.")
        if self.motion_beatness_max_fraction < 0:
            raise ValueError("motion_beatness_max_fraction must be non-negative.")
        if self.lambda_energy_pred < 0:
            raise ValueError("lambda_energy_pred must be non-negative.")
        if self.energy_teacher_forcing_epochs < 0:
            raise ValueError("energy_teacher_forcing_epochs must be non-negative.")
        if not 0.0 <= self.energy_pred_mix_prob <= 1.0:
            raise ValueError("energy_pred_mix_prob must be in [0, 1].")
        if self.energy_smoothness_weight < 0:
            raise ValueError("energy_smoothness_weight must be non-negative.")
        if self.lambda_acc_warmup_start_epoch < 0:
            raise ValueError("lambda_acc_warmup_start_epoch must be non-negative.")
        if self.lambda_acc_warmup_epochs < 0:
            raise ValueError("lambda_acc_warmup_epochs must be non-negative.")
        if (self.motion_energy_norm_p05 is None) != (self.motion_energy_norm_p95 is None):
            raise ValueError("motion_energy_norm_p05 and motion_energy_norm_p95 must be set together.")
        if (
            self.motion_energy_norm_p05 is not None
            and self.motion_energy_norm_p95 <= self.motion_energy_norm_p05
        ):
            raise ValueError("motion_energy_norm_p95 must be greater than motion_energy_norm_p05.")
        if (self.motion_beatness_norm_p05 is None) != (self.motion_beatness_norm_p95 is None):
            raise ValueError("motion_beatness_norm_p05 and motion_beatness_norm_p95 must be set together.")
        if (
            self.motion_beatness_norm_p05 is not None
            and self.motion_beatness_norm_p95 <= self.motion_beatness_norm_p05
        ):
            raise ValueError("motion_beatness_norm_p95 must be greater than motion_beatness_norm_p05.")
        self.use_g1_kinematic_losses = self.lambda_g1_kin > 0 and any(
            weight > 0 for weight in g1_weights
        )
        if self.use_g1_kinematic_losses and not is_g1_motion_format(self.motion_format):
            raise ValueError("G1 robot losses require a G1 motion format.")
        if self.use_g1_kinematic_losses and predict_epsilon:
            raise ValueError("G1 robot losses require predict_epsilon=False.")
        self.use_g1_root_angular_loss = self.lambda_g1_root_angular > 0
        if self.use_g1_root_angular_loss and not is_g1_motion_format(self.motion_format):
            raise ValueError("G1 root angular loss requires a G1 motion format.")
        if self.use_g1_root_angular_loss and predict_epsilon:
            raise ValueError("G1 root angular loss requires predict_epsilon=False.")
        self.g1_kinematics = (
            G1TorchKinematics(g1_fk_model_path, root_quat_order=g1_root_quat_order)
            if self.use_g1_kinematic_losses
            else None
        )
        self.motion_energy_kinematics = (
            self.g1_kinematics
            if self.g1_kinematics is not None
            else G1TorchKinematics(g1_fk_model_path, root_quat_order=g1_root_quat_order)
            if self.lambda_motion_intensity > 0 or self.lambda_motion_beatness > 0
            else None
        )
        self.motion_energy_keypoint_weights = dict(
            motion_energy_keypoint_weights
            if motion_energy_keypoint_weights is not None
            else {
                "left_wrist_yaw_link": 0.35,
                "right_wrist_yaw_link": 0.35,
                "left_ankle_roll_link": 0.10,
                "right_ankle_roll_link": 0.10,
                "torso_link": 0.10,
            }
        )
        if self.motion_energy_kinematics is not None:
            missing = [
                name
                for name in self.motion_energy_keypoint_weights
                if name not in self.motion_energy_kinematics.keypoint_names
            ]
            if missing:
                raise ValueError(f"G1 keypoints missing for motion energy loss: {missing}")
            self.motion_energy_keypoint_indices = [
                self.motion_energy_kinematics.keypoint_names.index(name)
                for name in self.motion_energy_keypoint_weights
            ]
        else:
            self.motion_energy_keypoint_indices = []
        self.current_epoch = 1
        self.last_beat_loss_stats = {}
        self.last_g1_kin_loss_stats = {}
        self.last_motion_energy_stats = {}
        self.beat_estimator = beat_estimator
        self.beat_estimator_config = (
            getattr(beat_estimator, "checkpoint_config_summary", {})
            if beat_estimator is not None
            else {}
        )
        if self.beat_estimator is not None:
            for param in self.beat_estimator.parameters():
                param.requires_grad = False
            self.beat_estimator.eval()

        # make a SMPL instance for FK module
        self.smpl = smpl

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def set_training_epoch(self, epoch):
        self.current_epoch = int(epoch)

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, cond, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, t, weight)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = move_cond_to_device(cond, device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(
            reversed(range(0, start_point)),
            total=start_point,
            desc="Sampling",
            unit="step",
        ):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        
    @torch.no_grad()
    def ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = move_cond_to_device(cond, device)

        x_start = None

        for time, time_next in tqdm(
            time_pairs,
            total=len(time_pairs),
            desc="DDIM sampling",
            unit="step",
        ):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        return x
    
    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
        
        if batch == 1:
            return self.ddim_sample(shape, cond)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = move_cond_to_device(cond, device)
        
        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        for time, time_next, weight in tqdm(
            time_pairs,
            total=len(time_pairs),
            desc="Long DDIM sampling",
            unit="step",
        ):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight=weight, clip_x_start = self.clip_denoised) 

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        return x

    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = move_cond_to_device(cond, device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(
            reversed(range(0, start_point)),
            total=start_point,
            desc="Inpainting",
            unit="step",
        ):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = move_cond_to_device(cond, device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(
            reversed(range(0, start_point)),
            total=start_point,
            desc="Long inpainting",
            unit="step",
        ):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def _decode_g1_samples(self, samples):
        if self.normalizer is None:
            raise ValueError("G1 robot losses require a fitted motion normalizer.")
        return decode_g1_motion(
            self.normalizer.unnormalize(samples.clone()).float(),
            motion_format=self.motion_format,
        )

    def _root_quaternion_wxyz(self, root_rot):
        if self.g1_root_quat_order == "wxyz":
            return root_rot
        if self.g1_root_quat_order == "xyzw":
            return root_rot[..., [3, 0, 1, 2]]
        raise ValueError(f"Unsupported G1 root quaternion order: {self.g1_root_quat_order}")

    def _root_local_keypoints(self, keypoints, decoded):
        root_rotation = quaternion_to_matrix(self._root_quaternion_wxyz(decoded["root_rot"]))
        centered = keypoints - decoded["root_pos"].unsqueeze(-2)
        return torch.matmul(
            root_rotation.transpose(-1, -2).unsqueeze(-3),
            centered.unsqueeze(-1),
        ).squeeze(-1)

    def _decode_g1_for_kinematics(self, samples):
        decoded = self._decode_g1_samples(samples)
        return self.g1_kinematics(
            decoded["root_pos"],
            decoded["root_rot"],
            decoded["dof_pos"],
        ), decoded

    @staticmethod
    def _g1_target_contact_mask(target_feet, height_threshold=0.035, speed_threshold=0.025):
        if target_feet.shape[1] < 2:
            return target_feet.new_zeros(target_feet.shape[0], 0, target_feet.shape[2], 1)
        horizontal_velocity = target_feet[:, 1:, :, :2] - target_feet[:, :-1, :, :2]
        horizontal_speed = torch.linalg.vector_norm(horizontal_velocity, dim=-1)
        min_height = target_feet[..., 2].amin(dim=1, keepdim=True)
        near_ground = target_feet[:, :-1, :, 2] <= (min_height + height_threshold)
        slow_target = horizontal_speed <= speed_threshold
        return (near_ground & slow_target).float().unsqueeze(-1)

    def _g1_robot_losses(self, model_out, target, base_loss):
        zero = torch.zeros((), device=model_out.device, dtype=model_out.dtype)
        empty_stats = {
            "g1_fk_loss": zero,
            "g1_fk_vel_loss": zero,
            "g1_fk_acc_loss": zero,
            "g1_foot_loss": zero,
            "g1_kin_scale": 0.0,
            "g1_kin_contribution": zero,
            "g1_kin_capped": torch.zeros((), dtype=torch.bool, device=model_out.device),
            "g1_kin_cap": torch.full((), float("inf"), device=model_out.device, dtype=model_out.dtype),
        }
        if not self.use_g1_kinematic_losses:
            self.last_g1_kin_loss_stats = empty_stats
            return zero, zero, zero, None, None, None

        model_fk, model_decoded = self._decode_g1_for_kinematics(model_out)
        target_fk, target_decoded = self._decode_g1_for_kinematics(target)
        model_keypoints = model_fk["keypoints"]
        model_feet = model_fk["feet"]
        target_feet = target_fk["feet"].detach()

        g1_fk_vel_loss = zero
        g1_fk_acc_loss = zero
        if (
            self.lambda_g1_fk > 0
            or self.lambda_g1_fk_vel > 0
            or self.lambda_g1_fk_acc > 0
        ):
            target_keypoints = target_fk["keypoints"].detach()
        if self.lambda_g1_fk > 0:
            g1_fk_loss = F.mse_loss(model_keypoints, target_keypoints)
        else:
            g1_fk_loss = zero
        if self.lambda_g1_fk_vel > 0 and model_keypoints.shape[1] > 1:
            model_v = model_keypoints[:, 1:] - model_keypoints[:, :-1]
            target_v = target_keypoints[:, 1:] - target_keypoints[:, :-1]
            g1_fk_vel_loss = F.mse_loss(model_v, target_v)

        if self.lambda_g1_fk_acc > 0 and model_keypoints.shape[1] > 2:
            model_a = model_keypoints[:, 2:] - 2 * model_keypoints[:, 1:-1] + model_keypoints[:, :-2]
            target_a = target_keypoints[:, 2:] - 2 * target_keypoints[:, 1:-1] + target_keypoints[:, :-2]
            g1_fk_acc_loss = F.mse_loss(model_a, target_a)

        g1_foot_loss = zero
        if self.lambda_g1_foot > 0 and model_feet.shape[1] > 1:
            contact_mask = self._g1_target_contact_mask(target_feet)
            model_foot_v = model_feet[:, 1:, :, :2] - model_feet[:, :-1, :, :2]
            masked_velocity = model_foot_v * contact_mask
            denom = torch.clamp(contact_mask.sum() * model_foot_v.shape[-1], min=1.0)
            g1_foot_loss = masked_velocity.pow(2).sum() / denom

        scale = effective_warmup_scale(
            self.current_epoch,
            self.g1_kin_loss_warmup_epochs,
        )
        raw_contribution = self.lambda_g1_kin * scale * (
            self.lambda_g1_fk * g1_fk_loss
            + self.lambda_g1_fk_vel * g1_fk_vel_loss
            + self.lambda_g1_fk_acc * g1_fk_acc_loss
            + self.lambda_g1_foot * g1_foot_loss
        )
        contribution, capped, cap = compute_capped_aux_contribution(
            base_loss=base_loss,
            raw_contribution=raw_contribution,
            max_fraction=self.g1_kin_loss_max_fraction,
        )
        self.last_g1_kin_loss_stats = {
            "g1_fk_loss": g1_fk_loss.detach(),
            "g1_fk_vel_loss": g1_fk_vel_loss.detach(),
            "g1_fk_acc_loss": g1_fk_acc_loss.detach(),
            "g1_foot_loss": g1_foot_loss.detach(),
            "g1_kin_scale": scale,
            "g1_kin_contribution": contribution.detach(),
            "g1_kin_capped": capped.detach(),
            "g1_kin_cap": cap.detach(),
        }
        return (
            contribution,
            g1_fk_loss,
            g1_foot_loss,
            model_keypoints,
            model_decoded,
            target_decoded,
        )

    def _prepare_motion_energy_condition(self, cond):
        zero = torch.zeros((), device=cond_device(cond), dtype=torch.float32)
        empty_stats = {
            "energy_pred_loss": zero,
            "energy_smoothness_loss": zero,
            "energy_pred_contribution": zero,
            "energy_pred_mix_rate": 0.0,
            "selected_energy_mean": zero,
            "pred_energy_mean": zero,
            "gt_energy_mean": zero,
            "intensity_pred_loss": zero,
            "beatness_pred_loss": zero,
            "intensity_smoothness_loss": zero,
            "selected_intensity_mean": zero,
            "pred_intensity_mean": zero,
            "gt_intensity_mean": zero,
            "selected_beatness_mean": zero,
            "pred_beatness_mean": zero,
            "gt_beatness_mean": zero,
            "body_intensity_pred_loss": zero,
            "body_intensity_smoothness_loss": zero,
            "body_intensity_mean_gt": zero,
            "body_intensity_mean_pred": zero,
            "support_beatness_pred_loss": zero,
            "support_beatness_mean_gt": zero,
            "support_beatness_mean_pred": zero,
            "upper_beatness_pred_loss": zero,
            "upper_beatness_mean_gt": zero,
            "upper_beatness_mean_pred": zero,
            "support_contact_pred_loss": zero,
            "support_contact_mean_gt": zero,
            "support_contact_mean_pred": zero,
            "motion_beatness_loss": zero,
            "motion_beatness_contribution": zero,
            "motion_beatness_weight": 0.0,
            "motion_beatness_capped": zero.to(dtype=torch.bool),
        }
        if not hasattr(self.model, "prepare_motion_energy_training_condition"):
            self.last_motion_energy_stats = empty_stats
            return cond, empty_stats
        cond, stats = self.model.prepare_motion_energy_training_condition(
            cond,
            epoch=self.current_epoch,
            teacher_forcing_epochs=self.energy_teacher_forcing_epochs,
            pred_mix_prob=self.energy_pred_mix_prob,
            detach_pred=True,
        )
        pred_loss = stats["energy_pred_loss"]
        smoothness_loss = stats["energy_smoothness_loss"]
        contribution = self.lambda_energy_pred * (
            pred_loss + self.energy_smoothness_weight * smoothness_loss
        )
        self.last_motion_energy_stats = {
            **stats,
            "energy_pred_contribution": contribution,
        }
        return cond, self.last_motion_energy_stats

    def _weighted_motion_energy_from_keypoints(self, keypoints, decoded=None):
        indices = torch.tensor(
            self.motion_energy_keypoint_indices,
            device=keypoints.device,
            dtype=torch.long,
        )
        keypoints = keypoints.index_select(-2, indices)
        if self.motion_energy_frame == "root_local":
            if decoded is None:
                raise ValueError("root_local motion-energy loss requires decoded G1 motion.")
            keypoints = self._root_local_keypoints(keypoints, decoded)
        frame_speed = keypoints.new_zeros(keypoints.shape[:2] + (keypoints.shape[-2],))
        frame_speed[:, 1:] = torch.linalg.vector_norm(
            keypoints[:, 1:] - keypoints[:, :-1],
            dim=-1,
        ) * 30.0
        weights = torch.tensor(
            list(self.motion_energy_keypoint_weights.values()),
            device=keypoints.device,
            dtype=keypoints.dtype,
        )
        return (frame_speed * weights.view(1, 1, -1)).sum(dim=-1)

    def _compute_weighted_motion_energy(self, samples):
        if self.motion_energy_kinematics is None:
            raise ValueError("Motion-energy loss requires G1 kinematics.")
        if self.normalizer is None:
            raise ValueError("Motion-energy loss requires a fitted motion normalizer.")
        decoded = self._decode_g1_samples(samples)
        result = self.motion_energy_kinematics(
            decoded["root_pos"],
            decoded["root_rot"],
            decoded["dof_pos"],
        )
        return self._weighted_motion_energy_from_keypoints(result["keypoints"], decoded)

    def _motion_energy_preservation_loss(
        self,
        model_out,
        motion_energy_stats,
        model_keypoints=None,
        model_decoded=None,
    ):
        zero = torch.zeros((), device=model_out.device, dtype=model_out.dtype)
        if self.lambda_motion_intensity <= 0:
            return zero
        selected_energy = motion_energy_stats.get(
            "selected_intensity",
            motion_energy_stats.get("selected_energy"),
        )
        if selected_energy is None:
            return zero
        if model_keypoints is None:
            pred_speed = self._compute_weighted_motion_energy(model_out)
        else:
            pred_speed = self._weighted_motion_energy_from_keypoints(
                model_keypoints,
                model_decoded,
            )
        if self.motion_energy_norm_p05 is not None:
            pred_speed = torch.clamp(
                (pred_speed - self.motion_energy_norm_p05)
                / (self.motion_energy_norm_p95 - self.motion_energy_norm_p05),
                min=0.0,
                max=1.0,
            )
        pred_global_energy = torch.clamp(pred_speed.mean(dim=1), min=1e-5)
        target_global_energy = torch.clamp(
            selected_energy.squeeze(-1).mean(dim=1).to(pred_global_energy.dtype),
            min=1e-5,
        )
        return F.mse_loss(torch.log(pred_global_energy), torch.log(target_global_energy))

    def _motion_beatness_valley_loss(
        self,
        model_out,
        motion_energy_stats,
        model_keypoints=None,
        model_decoded=None,
    ):
        zero = torch.zeros((), device=model_out.device, dtype=model_out.dtype)
        if self.lambda_motion_beatness <= 0:
            return zero
        selected_beatness = motion_energy_stats.get("selected_beatness")
        if selected_beatness is None:
            return zero
        if model_keypoints is None:
            pred_speed = self._compute_weighted_motion_energy(model_out)
        else:
            pred_speed = self._weighted_motion_energy_from_keypoints(
                model_keypoints,
                model_decoded,
            )
        if self.motion_beatness_norm_p05 is not None:
            shoulder_speed = F.max_pool1d(
                pred_speed.unsqueeze(1),
                kernel_size=13,
                stride=1,
                padding=6,
            ).squeeze(1)
            valley_contrast = torch.clamp(shoulder_speed - pred_speed, min=0.0)
            valley_contrast = torch.clamp(
                (valley_contrast - self.motion_beatness_norm_p05)
                / (self.motion_beatness_norm_p95 - self.motion_beatness_norm_p05),
                min=0.0,
                max=1.0,
            )
        else:
            if self.motion_energy_norm_p05 is not None:
                pred_speed = torch.clamp(
                    (pred_speed - self.motion_energy_norm_p05)
                    / (self.motion_energy_norm_p95 - self.motion_energy_norm_p05),
                    min=0.0,
                    max=1.0,
                )
            shoulder_speed = F.max_pool1d(
                pred_speed.unsqueeze(1),
                kernel_size=13,
                stride=1,
                padding=6,
            ).squeeze(1)
            valley_contrast = torch.clamp(shoulder_speed - pred_speed, min=0.0, max=1.0)
        target = selected_beatness.squeeze(-1).to(valley_contrast.dtype)
        weights = torch.clamp(target, min=0.0)
        denom = torch.clamp(weights.sum(), min=1.0)
        return (weights * (valley_contrast - target).pow(2)).sum() / denom

    def _root_angular_speeds(self, root_rot):
        root_quat = F.normalize(self._root_quaternion_wxyz(root_rot), dim=-1)
        if root_quat.shape[1] < 2:
            return root_quat.new_zeros(root_quat.shape[0], 0)
        dots = torch.sum(root_quat[:, 1:] * root_quat[:, :-1], dim=-1).abs()
        dots = torch.clamp(dots, min=0.0, max=1.0 - 1e-6)
        return 2.0 * torch.acos(dots) * 30.0

    def _g1_root_angular_loss(
        self,
        model_out,
        target,
        base_loss,
        model_decoded=None,
        target_decoded=None,
    ):
        zero = torch.zeros((), device=model_out.device, dtype=model_out.dtype)
        empty_stats = {
            "g1_root_angular_loss": zero,
            "g1_root_angular_velocity_loss": zero,
            "g1_root_angular_acceleration_loss": zero,
            "g1_root_angular_contribution": zero,
            "g1_root_angular_capped": torch.zeros(
                (),
                dtype=torch.bool,
                device=model_out.device,
            ),
            "g1_root_angular_cap": torch.full(
                (),
                float("inf"),
                device=model_out.device,
                dtype=model_out.dtype,
            ),
        }
        if not self.use_g1_root_angular_loss:
            return zero, empty_stats
        if model_decoded is None:
            model_decoded = self._decode_g1_samples(model_out)
        if target_decoded is None:
            target_decoded = self._decode_g1_samples(target)

        model_speed = self._root_angular_speeds(model_decoded["root_rot"])
        target_speed = self._root_angular_speeds(target_decoded["root_rot"]).detach()
        if model_speed.numel() == 0:
            return zero, empty_stats

        velocity_excess = F.relu(
            model_speed - target_speed - self.g1_root_angular_velocity_margin
        )
        velocity_loss = velocity_excess.pow(2).mean()

        acceleration_loss = zero
        if model_speed.shape[1] > 1:
            model_acc = (model_speed[:, 1:] - model_speed[:, :-1]) * 30.0
            target_acc = (target_speed[:, 1:] - target_speed[:, :-1]) * 30.0
            acceleration_excess = F.relu(
                model_acc.abs()
                - target_acc.abs()
                - self.g1_root_angular_acceleration_margin
            )
            acceleration_loss = acceleration_excess.pow(2).mean()

        root_angular_loss = (
            velocity_loss
            + self.g1_root_angular_acceleration_weight * acceleration_loss
        )
        raw_contribution = self.lambda_g1_root_angular * root_angular_loss
        contribution, capped, cap = compute_capped_aux_contribution(
            base_loss=base_loss,
            raw_contribution=raw_contribution,
            max_fraction=self.g1_root_angular_max_fraction,
        )
        stats = {
            "g1_root_angular_loss": root_angular_loss.detach(),
            "g1_root_angular_velocity_loss": velocity_loss.detach(),
            "g1_root_angular_acceleration_loss": acceleration_loss.detach(),
            "g1_root_angular_contribution": contribution.detach(),
            "g1_root_angular_capped": capped.detach(),
            "g1_root_angular_cap": cap.detach(),
        }
        return contribution, stats

    def effective_motion_beatness_weight(self):
        return effective_linear_ramp(
            self.current_epoch,
            0.0,
            final_value=self.lambda_motion_beatness,
            start_epoch=self.motion_beatness_warmup_start_epoch,
            warmup_epochs=self.motion_beatness_warmup_epochs,
        )

    def effective_lambda_acc(self):
        return effective_linear_ramp(
            self.current_epoch,
            self.lambda_acc,
            final_value=self.lambda_acc_final,
            start_epoch=self.lambda_acc_warmup_start_epoch,
            warmup_epochs=self.lambda_acc_warmup_epochs,
        )

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond, motion_energy_stats = self._prepare_motion_energy_condition(cond)

        # reconstruct
        x_recon = self.model(x_noisy, cond, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # full reconstruction loss
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        if is_g1_motion_format(self.motion_format):
            target_v = target[:, 1:] - target[:, :-1]
            model_out_v = model_out[:, 1:] - model_out[:, :-1]
            v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
            v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
            v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

            zero = torch.zeros((), device=x_start.device, dtype=x_start.dtype)
            beat_loss = zero
            if isinstance(cond, dict) and self.beat_estimator is not None:
                pred_beat_dist = self.beat_estimator(model_out)
                raw_gt_beat_dist = cond["beat_target"].to(model_out.device).float()
                beat_spacing = cond["beat_spacing"].to(model_out.device).float()
                gt_beat_dist = transform_beat_target_for_estimator(
                    raw_gt_beat_dist,
                    beat_spacing,
                    self.beat_estimator_config,
                )
                gt_safe = torch.clamp(gt_beat_dist, min=1.0)
                abs_error = torch.abs(gt_beat_dist - pred_beat_dist)
                w_s = 1.0 / (
                    1.0
                    + torch.exp(
                        self.beat_a
                        * (self.beat_c - (abs_error / gt_safe))
                    )
                )
                w_b = torch.exp(
                    -2.0 * raw_gt_beat_dist / torch.clamp(beat_spacing, min=1.0)
                )
                beat_loss = (w_s * w_b * (pred_beat_dist - gt_beat_dist).pow(2)).mean()

            acc_loss = zero
            if model_out.shape[1] > 2:
                target_a = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
                model_a = model_out[:, 2:] - 2 * model_out[:, 1:-1] + model_out[:, :-2]
                acc_loss = F.mse_loss(model_a, target_a)

            base_loss = loss.mean() + v_loss.mean()
            effective_lambda_beat = effective_beat_loss_weight(
                epoch=self.current_epoch,
                lambda_beat=self.lambda_beat,
                start_epoch=self.beat_loss_start_epoch,
                warmup_epochs=self.beat_loss_warmup_epochs,
            )
            beat_contribution, beat_capped, beat_cap = compute_beat_loss_contribution(
                base_loss=base_loss,
                beat_loss=beat_loss,
                effective_lambda_beat=effective_lambda_beat,
                max_fraction=self.beat_loss_max_fraction,
                cap_mode=self.beat_loss_cap_mode,
            )
            self.last_beat_loss_stats = {
                "effective_lambda_beat": effective_lambda_beat,
                "beat_contribution": beat_contribution.detach(),
                "beat_capped": beat_capped.detach(),
                "beat_cap": beat_cap.detach(),
            }
            (
                g1_kin_contribution,
                fk_loss,
                foot_loss,
                model_keypoints,
                model_decoded,
                target_decoded,
            ) = self._g1_robot_losses(model_out, target, base_loss)
            root_angular_contribution, root_angular_stats = self._g1_root_angular_loss(
                model_out,
                target,
                base_loss,
                model_decoded=model_decoded,
                target_decoded=target_decoded,
            )
            self.last_g1_kin_loss_stats = {
                **self.last_g1_kin_loss_stats,
                **root_angular_stats,
            }
            motion_energy_loss = self._motion_energy_preservation_loss(
                model_out,
                motion_energy_stats,
                model_keypoints=model_keypoints,
                model_decoded=model_decoded,
            )
            motion_energy_contribution = self.lambda_motion_intensity * motion_energy_loss
            motion_beatness_loss = self._motion_beatness_valley_loss(
                model_out,
                motion_energy_stats,
                model_keypoints=model_keypoints,
                model_decoded=model_decoded,
            )
            motion_beatness_weight = self.effective_motion_beatness_weight()
            raw_motion_beatness_contribution = motion_beatness_weight * motion_beatness_loss
            (
                motion_beatness_contribution,
                motion_beatness_capped,
                motion_beatness_cap,
            ) = compute_capped_aux_contribution(
                base_loss=base_loss,
                raw_contribution=raw_motion_beatness_contribution,
                max_fraction=self.motion_beatness_max_fraction,
            )
            energy_pred_contribution = motion_energy_stats.get(
                "energy_pred_contribution",
                zero,
            )
            self.last_motion_energy_stats = {
                **motion_energy_stats,
                "motion_energy_loss": motion_energy_loss.detach(),
                "motion_energy_contribution": motion_energy_contribution.detach(),
                "motion_beatness_loss": motion_beatness_loss.detach(),
                "motion_beatness_weight": motion_beatness_weight,
                "motion_beatness_contribution": motion_beatness_contribution.detach(),
                "motion_beatness_capped": motion_beatness_capped.detach(),
                "motion_beatness_cap": motion_beatness_cap.detach(),
            }
            total_loss = (
                base_loss
                + self.effective_lambda_acc() * acc_loss
                + beat_contribution
                + g1_kin_contribution
                + root_angular_contribution
                + motion_energy_contribution
                + motion_beatness_contribution
                + energy_pred_contribution
            )
            return total_loss, (loss.mean(), v_loss.mean(), fk_loss, foot_loss, acc_loss, beat_loss)

        # split off contact from the rest
        model_contact, model_out = torch.split(
            model_out, (4, model_out.shape[2] - 4), dim=2
        )
        target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)

        # velocity loss
        target_v = target[:, 1:] - target[:, :-1]
        model_out_v = model_out[:, 1:] - model_out[:, :-1]
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # FK loss
        b, s, c = model_out.shape
        # unnormalize
        # model_out = self.normalizer.unnormalize(model_out)
        # target = self.normalizer.unnormalize(target)
        # X, Q
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))

        # perform FK
        model_xp = self.smpl.forward(model_q, model_x)
        target_xp = self.smpl.forward(target_q, target_x)

        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

        acc_loss = torch.zeros((), device=x_start.device)
        if model_out.shape[1] > 2:
            target_a = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
            model_a = model_out[:, 2:] - 2 * model_out[:, 1:-1] + model_out[:, :-2]
            target_xpa = target_xp[:, 2:] - 2 * target_xp[:, 1:-1] + target_xp[:, :-2]
            model_xpa = model_xp[:, 2:] - 2 * model_xp[:, 1:-1] + model_xp[:, :-2]
            acc_loss = 0.5 * (
                F.mse_loss(model_a, target_a) + F.mse_loss(model_xpa, target_xpa)
            )

        # foot skate loss
        foot_idx = [7, 8, 10, 11]

        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0
        foot_loss = self.loss_fn(
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

        beat_loss = torch.zeros((), device=x_start.device)
        if isinstance(cond, dict) and self.beat_estimator is not None:
            pred_beat_dist = self.beat_estimator(model_xp)
            raw_gt_beat_dist = cond["beat_target"].to(model_xp.device).float()
            beat_spacing = cond["beat_spacing"].to(model_xp.device).float()
            gt_beat_dist = transform_beat_target_for_estimator(
                raw_gt_beat_dist,
                beat_spacing,
                self.beat_estimator_config,
            )
            gt_safe = torch.clamp(gt_beat_dist, min=1.0)
            abs_error = torch.abs(gt_beat_dist - pred_beat_dist)
            w_s = 1.0 / (
                1.0
                + torch.exp(
                    self.beat_a
                    * (self.beat_c - (abs_error / gt_safe))
                )
            )
            w_b = torch.exp(
                -2.0 * raw_gt_beat_dist / torch.clamp(beat_spacing, min=1.0)
            )
            beat_loss = (w_s * w_b * (pred_beat_dist - gt_beat_dist).pow(2)).mean()

        base_losses = (
            0.636 * loss.mean(),
            2.964 * v_loss.mean(),
            0.646 * fk_loss.mean(),
            10.942 * foot_loss.mean(),
        )
        base_loss = sum(base_losses)
        effective_lambda_beat = effective_beat_loss_weight(
            epoch=self.current_epoch,
            lambda_beat=self.lambda_beat,
            start_epoch=self.beat_loss_start_epoch,
            warmup_epochs=self.beat_loss_warmup_epochs,
        )
        beat_contribution, beat_capped, beat_cap = compute_beat_loss_contribution(
            base_loss=base_loss,
            beat_loss=beat_loss,
            effective_lambda_beat=effective_lambda_beat,
            max_fraction=self.beat_loss_max_fraction,
            cap_mode=self.beat_loss_cap_mode,
        )
        self.last_beat_loss_stats = {
            "effective_lambda_beat": effective_lambda_beat,
            "beat_contribution": beat_contribution.detach(),
            "beat_capped": beat_capped.detach(),
            "beat_cap": beat_cap.detach(),
        }
        total_loss = base_loss
        total_loss = total_loss + self.lambda_acc * acc_loss + beat_contribution
        losses = base_losses + (acc_loss, beat_loss)
        return total_loss, losses

    def loss(self, x, cond, t_override=None):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, x, cond, t_override=None):
        return self.loss(x, cond, t_override)

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x

    def _g1_output_name(self, epoch, num, filename=None, mode="normal"):
        if filename is None:
            stem = f"sample{num}"
        else:
            stem = Path(filename).stem
        if mode == "long":
            parts = stem.split("_")
            if len(parts) > 1:
                stem = "_".join(parts[:-1]) or stem
        return f"{epoch}_{num}_{stem}_g1.pkl"

    def _g1_payload(self, root_pos, root_rot, dof_pos, metadata=None):
        root_pos_np = root_pos.detach().cpu().numpy()
        root_rot_np = root_rot.detach().cpu().numpy()
        dof_pos_np = dof_pos.detach().cpu().numpy()
        payload = {
            "motion_format": G1_MOTION_FORMAT,
            "fps": 30.0,
            "root_pos": root_pos_np,
            "root_rot": root_rot_np,
            "dof_pos": dof_pos_np,
            "pos": root_pos_np,
            "q": np.concatenate((root_rot_np, dof_pos_np), axis=-1),
        }
        if metadata:
            payload.update(metadata)
        return payload

    def _stitch_g1_samples(self, decoded):
        pos = decoded["root_pos"]
        root_rot = decoded["root_rot"]
        dof_pos = decoded["dof_pos"]
        b, s, _ = pos.shape
        if b == 1:
            return pos[0], root_rot[0], dof_pos[0]

        assert s % 2 == 0
        half = s // 2
        fade_out = torch.ones((1, s, 1), device=pos.device, dtype=pos.dtype)
        fade_in = torch.ones((1, s, 1), device=pos.device, dtype=pos.dtype)
        fade_out[:, half:, :] = torch.linspace(1, 0, half, device=pos.device, dtype=pos.dtype)[
            None, :, None
        ]
        fade_in[:, :half, :] = torch.linspace(0, 1, half, device=pos.device, dtype=pos.dtype)[
            None, :, None
        ]

        aligned_pos = pos.clone()
        for index in range(1, b):
            offset = (aligned_pos[index - 1, half:] - aligned_pos[index, :half]).mean(
                dim=0,
                keepdim=True,
            )
            aligned_pos[index] += offset

        blended_pos = aligned_pos.clone()
        blended_dof = dof_pos.clone()
        blended_pos[:-1] *= fade_out
        blended_pos[1:] *= fade_in
        blended_dof[:-1] *= fade_out
        blended_dof[1:] *= fade_in

        full_len = s + half * (b - 1)
        full_pos = torch.zeros((full_len, 3), device=pos.device, dtype=pos.dtype)
        full_dof = torch.zeros((full_len, dof_pos.shape[-1]), device=pos.device, dtype=dof_pos.dtype)
        idx = 0
        for pos_slice, dof_slice in zip(blended_pos, blended_dof):
            full_pos[idx : idx + s] += pos_slice
            full_dof[idx : idx + s] += dof_slice
            idx += half

        slerp_weight = torch.linspace(0, 1, half, device=pos.device, dtype=pos.dtype)[
            None, :, None
        ]
        merged_rot = quat_slerp(
            root_rot[:-1, half:].clone().unsqueeze(2),
            root_rot[1:, :half].clone().unsqueeze(2),
            slerp_weight,
        ).squeeze(2)
        full_rot = torch.zeros((full_len, 4), device=pos.device, dtype=root_rot.dtype)
        full_rot[:half] = root_rot[0, :half]
        idx = half
        for rot_slice in merged_rot:
            full_rot[idx : idx + half] = rot_slice
            idx += half
        full_rot[idx : idx + half] = root_rot[-1, half:]
        return full_pos, full_rot, full_dof

    def _stitch_g1_integrated_root_samples(self, samples):
        b, s, _ = samples.shape
        if b == 1:
            decoded = decode_g1_motion(samples, motion_format=self.motion_format)
            return decoded["root_pos"][0], decoded["root_rot"][0], decoded["dof_pos"][0]

        assert s % 2 == 0
        half = s // 2
        full_samples = torch.cat(
            (samples[0], samples[1:, half:].reshape(-1, samples.shape[-1])),
            dim=0,
        )
        decoded = decode_g1_motion(full_samples.unsqueeze(0), motion_format=self.motion_format)
        return decoded["root_pos"][0], decoded["root_rot"][0], decoded["dof_pos"][0]

    def render_g1_sample(
        self,
        samples,
        cond,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        mode="normal",
        metadata_audio_path=None,
        metadata_stride_frames=75,
        metadata_total_frames=None,
        render=False,
        sound=False,
        g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        g1_root_quat_order="xyzw",
        g1_render_backend="mujoco",
        g1_render_width=960,
        g1_render_height=720,
        g1_mujoco_gl="egl",
    ):
        output_dir = Path(fk_out or render_out)
        output_dir.mkdir(parents=True, exist_ok=True)
        beat_rep = getattr(self.model, "beat_rep", None)
        render_g1_motion = None
        if render:
            from eval.g1_visualization import render_g1_motion

        if mode == "long":
            if self.motion_format in (G1_ROOT_DELTA_MOTION_FORMAT, G1_YAW_DELTA_MOTION_FORMAT):
                root_pos, root_rot, dof_pos = self._stitch_g1_integrated_root_samples(samples)
            else:
                decoded = decode_g1_motion(samples, motion_format=self.motion_format)
                root_pos, root_rot, dof_pos = self._stitch_g1_samples(decoded)
            if metadata_total_frames is not None:
                root_pos = root_pos[:metadata_total_frames]
                root_rot = root_rot[:metadata_total_frames]
                dof_pos = dof_pos[:metadata_total_frames]
            audio_path = metadata_audio_path or (name[0] if name else None)
            metadata = build_long_saved_motion_metadata(
                cond,
                audio_path=audio_path,
                beat_rep=beat_rep,
                stride_frames=metadata_stride_frames,
            )
            outname = self._g1_output_name(
                epoch,
                0,
                filename=name[0] if name else None,
                mode=mode,
            )
            with open(output_dir / outname, "wb") as handle:
                payload = self._g1_payload(
                    root_pos,
                    root_rot,
                    dof_pos,
                    metadata=metadata,
                )
                pickle.dump(payload, handle)
            if render:
                render_name = metadata_audio_path or name
                render_g1_motion(
                    payload,
                    out=render_out,
                    epoch=epoch,
                    num=0,
                    name=render_name,
                    sound=sound,
                    stitch=False if metadata_audio_path else True,
                    model_path=g1_fk_model_path,
                    root_quat_order=g1_root_quat_order,
                    render_backend=g1_render_backend,
                    width=g1_render_width,
                    height=g1_render_height,
                    mujoco_gl=g1_mujoco_gl,
                    output_name=name,
                )
            return

        decoded = decode_g1_motion(samples, motion_format=self.motion_format)
        for num in range(decoded["root_pos"].shape[0]):
            filename = name[num] if name is not None else None
            metadata = build_saved_motion_metadata(
                cond,
                sample_idx=num,
                audio_path=filename,
                beat_rep=beat_rep,
            )
            outname = self._g1_output_name(epoch, num, filename=filename, mode=mode)
            with open(output_dir / outname, "wb") as handle:
                payload = self._g1_payload(
                    decoded["root_pos"][num],
                    decoded["root_rot"][num],
                    decoded["dof_pos"][num],
                    metadata=metadata,
                )
                pickle.dump(payload, handle)
            if render:
                render_g1_motion(
                    payload,
                    out=render_out,
                    epoch=epoch,
                    num=num,
                    name=filename,
                    sound=sound,
                    stitch=False,
                    model_path=g1_fk_model_path,
                    root_quat_order=g1_root_quat_order,
                    render_backend=g1_render_backend,
                    width=g1_render_width,
                    height=g1_render_height,
                    mujoco_gl=g1_mujoco_gl,
                )

    def render_sample(
        self,
        shape,
        cond,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True,
        metadata_audio_path=None,
        metadata_stride_frames=75,
        metadata_total_frames=None,
        g1_fk_model_path="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        g1_root_quat_order="xyzw",
        g1_render_backend="mujoco",
        g1_render_width=960,
        g1_render_height=720,
        g1_mujoco_gl="egl",
    ):
        sound = sound and render
        cond = move_cond_to_device(cond, self.betas.device)
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape

        samples = normalizer.unnormalize(samples)

        if is_g1_motion_format(self.motion_format):
            self.render_g1_sample(
                samples,
                cond,
                epoch,
                render_out,
                fk_out=fk_out,
                name=name,
                mode=mode,
                metadata_audio_path=metadata_audio_path,
                metadata_stride_frames=metadata_stride_frames,
                metadata_total_frames=metadata_total_frames,
                render=render,
                sound=sound,
                g1_fk_model_path=g1_fk_model_path,
                g1_root_quat_order=g1_root_quat_order,
                g1_render_backend=g1_render_backend,
                g1_render_width=g1_render_width,
                g1_render_height=g1_render_height,
                g1_mujoco_gl=g1_mujoco_gl,
            )
            return

        if samples.shape[2] == 151:
            sample_contact, samples = torch.split(
                samples, (4, samples.shape[2] - 4), dim=2
            )
        else:
            sample_contact = None
        # do the FK all at once
        b, s, c = samples.shape
        cond_dev = cond_device(cond)
        pos = samples[:, :, :3].to(cond_dev)  # np.zeros((sample.shape[0], 3))
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        # go 6d to ax
        q = ax_from_6v(q).to(cond_dev)

        if mode == "long":
            b, s, c1, c2 = q.shape
            assert s % 2 == 0
            half = s // 2
            if b > 1:
                # if long mode, stitch position using linear interp

                fade_out = torch.ones((1, s, 1)).to(pos.device)
                fade_in = torch.ones((1, s, 1)).to(pos.device)
                fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                    pos.device
                )
                fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                    pos.device
                )

                pos[:-1] *= fade_out
                pos[1:] *= fade_in

                full_pos = torch.zeros((s + half * (b - 1), 3)).to(pos.device)
                idx = 0
                for pos_slice in pos:
                    full_pos[idx : idx + s] += pos_slice
                    idx += half

                # stitch joint angles with slerp
                slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(pos.device)

                left, right = q[:-1, half:], q[1:, :half]
                # convert to quat
                left, right = (
                    axis_angle_to_quaternion(left),
                    axis_angle_to_quaternion(right),
                )
                merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
                # convert back
                merged = quaternion_to_axis_angle(merged)

                full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(pos.device)
                full_q[:half] += q[0, :half]
                idx = half
                for q_slice in merged:
                    full_q[idx : idx + half] += q_slice
                    idx += half
                full_q[idx : idx + half] += q[-1, half:]

                # unsqueeze for fk
                full_pos = full_pos.unsqueeze(0)
                full_q = full_q.unsqueeze(0)
            else:
                full_pos = pos
                full_q = q
            full_pose = (
                self.smpl.forward(full_q, full_pos).detach().cpu().numpy()
            )  # b, s, 24, 3
            # squeeze the batch dimension away and render
            skeleton_render(
                full_pose[0],
                epoch=f"{epoch}",
                out=render_out,
                name=name,
                sound=sound,
                stitch=True,
                sound_folder=sound_folder,
                render=render
            )
            if fk_out is not None:
                outname = f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                beat_rep = getattr(self.model, "beat_rep", None)
                payload = {
                    "smpl_poses": full_q.squeeze(0).reshape((-1, 72)).cpu().numpy(),
                    "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                    "full_pose": full_pose[0],
                }
                payload.update(
                    build_long_saved_motion_metadata(
                        cond,
                        audio_path=metadata_audio_path,
                        beat_rep=beat_rep,
                        stride_frames=metadata_stride_frames,
                    )
                )
                pickle.dump(payload, open(os.path.join(fk_out, outname), "wb"))
            return

        poses = self.smpl.forward(q, pos).detach().cpu().numpy()
        sample_contact = (
            sample_contact.detach().cpu().numpy()
            if sample_contact is not None
            else None
        )

        def inner(xx):
            num, pose = xx
            filename = name[num] if name is not None else None
            contact = sample_contact[num] if sample_contact is not None else None
            skeleton_render(
                pose,
                epoch=f"e{epoch}_b{num}",
                out=render_out,
                name=filename,
                sound=sound,
                contact=contact,
                render=render,
            )

        p_map(inner, enumerate(poses))

        if fk_out is not None and mode != "long":
            Path(fk_out).mkdir(parents=True, exist_ok=True)
            beat_rep = getattr(self.model, "beat_rep", None)
            for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                path = os.path.normpath(filename)
                pathparts = path.split(os.sep)
                pathparts[-1] = pathparts[-1].replace("npy", "wav")
                # path is like "data/train/features/name"
                pathparts[2] = "wav_sliced"
                outname = f"{epoch}_{num}_{pathparts[-1][:-4]}.pkl"
                payload = {
                    "smpl_poses": qq.reshape((-1, 72)).cpu().numpy(),
                    "smpl_trans": pos_.cpu().numpy(),
                    "full_pose": pose,
                }
                payload.update(
                    build_saved_motion_metadata(
                        cond,
                        sample_idx=num,
                        audio_path=filename,
                        beat_rep=beat_rep,
                    )
                )
                pickle.dump(payload, open(f"{fk_out}/{outname}", "wb"))
