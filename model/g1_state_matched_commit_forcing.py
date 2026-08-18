from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import torch
import torch.nn.functional as F


SMCF_TARGET_POLICIES = ("ordinary", "p50", "full")
TRUST_CONTINUOUS_KEYS = (
    "state_distance",
    "pose_gap",
    "velocity_gap",
    "reconstruction_mse",
    "reconstruction_fk",
    "residual_norm",
)
TRUST_BOOLEAN_KEYS = (
    "finite",
    "causal",
    "contact_equal",
    "normalization_valid",
)
SMCF_TRUST_ARTIFACT_TYPE = "g1_smcf_trust_region"


@dataclass(frozen=True)
class G1SMCFTrustThresholds:
    state_distance: float
    pose_gap: float
    velocity_gap: float
    reconstruction_mse: float
    reconstruction_fk: float
    residual_norm: float

    def __post_init__(self):
        for name in TRUST_CONTINUOUS_KEYS:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} threshold must be finite and non-negative")

    @classmethod
    def from_mapping(cls, values):
        missing = [key for key in TRUST_CONTINUOUS_KEYS if key not in values]
        if missing:
            raise KeyError(f"trust thresholds are missing {missing}")
        return cls(**{key: float(values[key]) for key in TRUST_CONTINUOUS_KEYS})

    def manifest(self):
        return {
            "state_distance": float(self.state_distance),
            "pose_gap": float(self.pose_gap),
            "velocity_gap": float(self.velocity_gap),
            "reconstruction_mse": float(self.reconstruction_mse),
            "reconstruction_fk": float(self.reconstruction_fk),
            "residual_norm": float(self.residual_norm),
        }


@dataclass(frozen=True)
class G1SMCFTrustDecision:
    accepted: torch.Tensor
    checks: Mapping[str, torch.Tensor]
    counters: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class G1SMCFTargetSelection:
    target: torch.Tensor
    ordinary_target: torch.Tensor
    state_matched_target: torch.Tensor
    trust_accepted: torch.Tensor
    dose_selected: torch.Tensor
    use_state_matched: torch.Tensor
    counters: Mapping[str, torch.Tensor]


def smcf_normalizer_digest(
    *,
    state_mean,
    state_std,
    residual_mean,
    residual_std,
    motion_mean,
    motion_std,
):
    digest = hashlib.sha256()
    values = {
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "state_mean": state_mean,
        "state_std": state_std,
    }
    for name in sorted(values):
        value = torch.as_tensor(values[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.numpy().dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_smcf_trust_artifact(
    artifact,
    *,
    parent_sha256,
    codec_sha256,
    cache_manifest_digest,
    normalizer_digest,
    source_digest=None,
):
    if artifact.get("artifact_type") != SMCF_TRUST_ARTIFACT_TYPE:
        raise ValueError("unexpected SMCF trust artifact type")
    if artifact.get("repository_dirty") is not False:
        raise ValueError("SMCF trust artifact requires a clean calibration tree")
    expected = {
        "parent_sha256": parent_sha256,
        "codec_sha256": codec_sha256,
        "cache_manifest_digest": cache_manifest_digest,
        "normalizer_digest": normalizer_digest,
    }
    if source_digest is not None:
        expected["source_digest"] = source_digest
    for key, value in expected.items():
        if not value or artifact.get(key) != value:
            raise ValueError(f"SMCF trust artifact {key} mismatch")
    thresholds = G1SMCFTrustThresholds.from_mapping(artifact["thresholds"])
    calibration = artifact.get("calibration", {})
    acceptance = float(calibration.get("acceptance_fraction", float("nan")))
    observations = int(calibration.get("transition_observations", 0))
    if observations != 92_160:
        raise ValueError("SMCF trust calibration must contain 92,160 observations")
    if not math.isfinite(acceptance) or not 0.10 <= acceptance <= 0.90:
        raise ValueError(
            "SMCF trust calibration acceptance must be within [0.10, 0.90]"
        )
    return thresholds


def smcf_numeric_diagnostics(
    *,
    generated_state_normalized,
    oracle_state_normalized,
    state_matched_residual_raw,
    target_motion,
    reconstruction,
    reconstruction_fk,
    contact_equal,
    causal,
    normalization_valid,
):
    generated = torch.as_tensor(generated_state_normalized)
    oracle = torch.as_tensor(
        oracle_state_normalized,
        device=generated.device,
        dtype=generated.dtype,
    )
    if generated.ndim != 2 or generated.shape != oracle.shape:
        raise ValueError("generated and oracle states must share shape [B,S]")
    if generated.shape[-1] != 66:
        raise ValueError("SMCF diagnostics require S66 states")
    target_motion = torch.as_tensor(
        target_motion,
        device=generated.device,
        dtype=generated.dtype,
    )
    reconstruction = torch.as_tensor(
        reconstruction,
        device=generated.device,
        dtype=generated.dtype,
    )
    state_matched_residual_raw = torch.as_tensor(
        state_matched_residual_raw,
        device=generated.device,
        dtype=generated.dtype,
    )
    if target_motion.shape != reconstruction.shape:
        raise ValueError("target and reconstruction motion shapes differ")
    if target_motion.shape[0] != generated.shape[0]:
        raise ValueError("motion and state batch sizes differ")
    if state_matched_residual_raw.shape[0] != generated.shape[0]:
        raise ValueError("state-matched residual batch size differs")
    state_delta = generated - oracle
    pose_delta = torch.cat(
        (state_delta[:, 0:1], state_delta[:, 5:34]),
        dim=-1,
    )
    velocity_delta = torch.cat(
        (state_delta[:, 1:5], state_delta[:, 34:63]),
        dim=-1,
    )
    reconstruction_fk = torch.as_tensor(
        reconstruction_fk,
        device=generated.device,
        dtype=generated.dtype,
    )
    if reconstruction_fk.shape != (generated.shape[0],):
        raise ValueError("reconstruction_fk must have one value per sample")
    finite = (
        torch.isfinite(generated).all(dim=-1)
        & torch.isfinite(oracle).all(dim=-1)
        & torch.isfinite(state_matched_residual_raw).flatten(1).all(dim=-1)
        & torch.isfinite(target_motion).flatten(1).all(dim=-1)
        & torch.isfinite(reconstruction).flatten(1).all(dim=-1)
        & torch.isfinite(reconstruction_fk)
    )
    diagnostics = {
        "state_distance": state_delta.square().mean(dim=-1).sqrt(),
        "pose_gap": pose_delta.square().mean(dim=-1).sqrt(),
        "velocity_gap": velocity_delta.square().mean(dim=-1).sqrt(),
        "reconstruction_mse": (
            reconstruction - target_motion
        ).square().flatten(1).mean(dim=-1),
        "reconstruction_fk": reconstruction_fk,
        "residual_norm": state_matched_residual_raw.square().flatten(1).mean(
            dim=-1
        ).sqrt(),
        "finite": finite,
        "causal": torch.as_tensor(
            causal,
            device=generated.device,
            dtype=torch.bool,
        ),
        "contact_equal": torch.as_tensor(
            contact_equal,
            device=generated.device,
            dtype=torch.bool,
        ),
        "normalization_valid": torch.as_tensor(
            normalization_valid,
            device=generated.device,
            dtype=torch.bool,
        ),
    }
    for key in TRUST_BOOLEAN_KEYS:
        if diagnostics[key].shape != (generated.shape[0],):
            raise ValueError(f"{key} must have one value per sample")
    return diagnostics


def world_se2_transport_yaw_delta_is_identity(
    local_motion,
    *,
    translation_xy=(0.0, 0.0),
    yaw_radians=0.0,
):
    """Return the unchanged local representation after a rigid world SE(2).

    g1_yaw_delta contains delta-XY and delta-yaw, not absolute world XZ/yaw.
    A rigid transform therefore has no representable action on this tensor.
    The unused transform arguments are retained so callers must state which
    world transform they audited.
    """

    motion = torch.as_tensor(local_motion)
    if motion.ndim < 2 or motion.shape[-1] != 34:
        raise ValueError("g1_yaw_delta motion must end in 34 channels")
    translation = torch.as_tensor(translation_xy)
    if translation.numel() != 2 or not torch.isfinite(translation).all():
        raise ValueError("translation_xy must contain two finite values")
    yaw = torch.as_tensor(yaw_radians)
    if yaw.numel() != 1 or not torch.isfinite(yaw).all():
        raise ValueError("yaw_radians must be finite")
    return motion.clone()


def assert_yaw_delta_se2_invariance(local_motion, transported_local_motion):
    local = torch.as_tensor(local_motion)
    transported = torch.as_tensor(
        transported_local_motion,
        device=local.device,
        dtype=local.dtype,
    )
    if local.shape != transported.shape or not torch.equal(local, transported):
        raise AssertionError(
            "rigid world SE(2) must be an exact identity in g1_yaw_delta"
        )


def q0_latent(codec, q0_ids):
    return F.embedding(q0_ids, codec.quantizer.codebooks[0])


def state_matched_full_latent(
    codec,
    target_motion,
    generated_state_normalized,
):
    if target_motion.ndim != 3 or target_motion.shape[-1] != 34:
        raise ValueError("target_motion must have shape [B,T,34]")
    if generated_state_normalized.ndim != 2:
        raise ValueError("generated_state_normalized must have shape [B,S]")
    if target_motion.shape[0] != generated_state_normalized.shape[0]:
        raise ValueError("target motion and state batch sizes differ")
    components = codec.encode_components(
        target_motion,
        generated_state_normalized,
    )
    full_latent = components.get("pre_quant")
    if full_latent is None:
        raise KeyError("codec.encode_components must return pre_quant")
    if not torch.isfinite(full_latent).all():
        raise FloatingPointError("state-matched encoder output is non-finite")
    return full_latent, components


def normalized_rebased_residual(
    codec,
    full_latent,
    input_q0,
    residual_mean,
    residual_std,
):
    input_latent = q0_latent(codec, input_q0)
    if full_latent.shape != input_latent.shape:
        raise ValueError("full latent and input q0 latent shapes differ")
    raw = full_latent - input_latent
    if torch.any(residual_std <= 0) or not torch.isfinite(residual_std).all():
        raise ValueError("residual_std must be finite and positive")
    normalized = (raw - residual_mean) / residual_std
    if not torch.isfinite(normalized).all():
        raise FloatingPointError("normalized rebased residual is non-finite")
    return normalized, raw


def stable_p50_mask(
    dataset_index,
    transition_index,
    training_seed,
    global_update,
):
    """Return a cross-device deterministic p50 mask without touching RNG state."""

    dataset_index = torch.as_tensor(dataset_index, dtype=torch.int64)
    device = dataset_index.device
    transition_index = torch.as_tensor(
        transition_index,
        device=device,
        dtype=torch.int64,
    )
    dataset_index, transition_index = torch.broadcast_tensors(
        dataset_index,
        transition_index,
    )
    modulus = 2_147_483_647
    value = torch.remainder(
        (dataset_index + 1) * 1_000_003
        + (transition_index + 1) * 9_176
        + (int(training_seed) + 1) * 37
        + (int(global_update) + 1) * 101,
        modulus,
    )
    value = torch.bitwise_xor(value, torch.bitwise_left_shift(value, 13))
    value = torch.bitwise_and(value, modulus)
    value = torch.bitwise_xor(value, torch.bitwise_right_shift(value, 17))
    value = torch.bitwise_xor(value, torch.bitwise_left_shift(value, 5))
    value = torch.bitwise_and(value, modulus)
    return torch.bitwise_and(value, 1).eq(0)


def continuous_trust_checks(
    diagnostics,
    thresholds,
):
    if not isinstance(thresholds, G1SMCFTrustThresholds):
        thresholds = G1SMCFTrustThresholds.from_mapping(thresholds)
    missing = [
        key
        for key in (*TRUST_CONTINUOUS_KEYS, *TRUST_BOOLEAN_KEYS)
        if key not in diagnostics
    ]
    if missing:
        raise KeyError(f"trust diagnostics are missing {missing}")
    reference = torch.as_tensor(diagnostics[TRUST_BOOLEAN_KEYS[0]])
    if reference.ndim != 1:
        raise ValueError("trust diagnostics must be one-dimensional per sample")
    checks = {}
    for key in TRUST_CONTINUOUS_KEYS:
        value = torch.as_tensor(
            diagnostics[key],
            device=reference.device,
        )
        if value.shape != reference.shape:
            raise ValueError(f"{key} diagnostic shape differs from finite")
        checks[key] = torch.isfinite(value) & value.le(
            float(getattr(thresholds, key))
        )
    for key in TRUST_BOOLEAN_KEYS:
        value = torch.as_tensor(
            diagnostics[key],
            device=reference.device,
            dtype=torch.bool,
        )
        if value.shape != reference.shape:
            raise ValueError(f"{key} diagnostic shape differs from finite")
        checks[key] = value
    return checks


def evaluate_hard_trust_region(
    checks,
    *,
    eligible=None,
):
    if not checks:
        raise ValueError("hard trust region requires at least one check")
    normalized = {
        name: torch.as_tensor(value, dtype=torch.bool)
        for name, value in checks.items()
    }
    first = next(iter(normalized.values()))
    if first.ndim != 1:
        raise ValueError("trust checks must be one-dimensional per sample")
    for name, value in normalized.items():
        if value.shape != first.shape:
            raise ValueError(f"trust check {name} has an inconsistent shape")
        normalized[name] = value.to(device=first.device)
    if eligible is None:
        eligible = torch.ones_like(first)
    else:
        eligible = torch.as_tensor(
            eligible,
            device=first.device,
            dtype=torch.bool,
        )
        if eligible.shape != first.shape:
            raise ValueError("eligible mask has an inconsistent shape")
    accepted = eligible.clone()
    for value in normalized.values():
        accepted = accepted & value
    counters = {
        f"rejected/{name}": (eligible & ~value).sum()
        for name, value in normalized.items()
    }
    counters.update(
        {
            "eligible": eligible.sum(),
            "accepted": accepted.sum(),
            "ineligible": (~eligible).sum(),
        }
    )
    return G1SMCFTrustDecision(
        accepted=accepted,
        checks=normalized,
        counters=counters,
    )


def select_smcf_target(
    ordinary_target,
    state_matched_target,
    trust_decision,
    *,
    policy,
    dataset_index,
    transition_index,
    training_seed,
    global_update,
):
    if policy not in SMCF_TARGET_POLICIES:
        raise ValueError(f"unsupported SMCF target policy: {policy}")
    if ordinary_target.shape != state_matched_target.shape:
        raise ValueError("ordinary and state-matched targets must match")
    accepted = torch.as_tensor(
        trust_decision.accepted,
        device=ordinary_target.device,
        dtype=torch.bool,
    )
    if accepted.shape != ordinary_target.shape[:1]:
        raise ValueError("trust decision must have one value per target row")
    if policy == "ordinary":
        dose_selected = torch.zeros_like(accepted)
    elif policy == "full":
        dose_selected = torch.ones_like(accepted)
    else:
        dose_selected = stable_p50_mask(
            torch.as_tensor(dataset_index, device=ordinary_target.device),
            torch.as_tensor(transition_index, device=ordinary_target.device),
            training_seed,
            global_update,
        )
        if dose_selected.shape != accepted.shape:
            raise ValueError("p50 identity tensors must match the target rows")
    use_state_matched = accepted & dose_selected
    broadcast = use_state_matched.view(
        use_state_matched.shape[0],
        *([1] * (ordinary_target.ndim - 1)),
    )
    target = torch.where(
        broadcast,
        state_matched_target,
        ordinary_target,
    )
    counters = dict(trust_decision.counters)
    counters.update(
        {
            "dose_selected": dose_selected.sum(),
            "state_matched": use_state_matched.sum(),
            "ordinary": (~use_state_matched).sum(),
        }
    )
    return G1SMCFTargetSelection(
        target=target,
        ordinary_target=ordinary_target,
        state_matched_target=state_matched_target,
        trust_accepted=accepted,
        dose_selected=dose_selected,
        use_state_matched=use_state_matched,
        counters=counters,
    )
