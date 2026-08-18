import argparse
from contextlib import ExitStack
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.g1_paper_faithful_dc_dataset import (
    CAUSAL_HISTORY_VALID_TOKENS,
    COMMIT_FRAMES,
    MINIMUM_LATENT_WIDTH_CACHE_VERSION,
    MOTION_FRAMES_PER_TOKEN,
    PAPER_DC_CACHE_VERSION,
    STRUCTURAL_QUANTIZER_CACHE_VERSION,
    ROLLOUT_COMMITS,
    G1PaperFaithfulDCDataset,
    _cache_ready,
    _current_cache_provenance,
    paper_dc_cache_paths,
)
from dataset.g1_streaming_state import G1StreamingStateStatistics
from dataset.motion_representation import decode_g1_motion
from model.g1_hybrid_streaming_codec import (
    build_g1_hybrid_streaming_codec_from_checkpoint,
)
from model.g1_paper_faithful_dc_streaming import (
    COMMIT_TOKENS,
    DIFFUSION_TIMESTEPS,
    HISTORY_TOKENS,
    JOINT_Q0_RESIDUAL_STAGES,
    PAPER_DC_STAGES,
    PLAN_TOKENS,
    SELF_FORCING_GRADIENT_TAIL,
    TRAINING_ROLLOUT_TOKENS,
    Y_ROUTE_STAGES,
    AdaLNResidualBlock,
    ExponentialMovingAverage,
    G1LatentSequenceCritic,
    G1PaperFaithfulDCConfig,
    G1PaperQ0Generator,
    G1ResidualDiffusionGenerator,
    PredX0CosineDiffusion,
    dmd_surrogate_loss,
    complete_commit_replacement_mask,
    fake_score_denoising_loss,
    gan_critic_loss,
    gan_generator_loss,
    homogeneous_sequence_noise_levels,
    independent_token_noise_levels,
    physical_auxiliary_scale,
    replace_complete_dc_tokens,
    self_forcing_rollout,
    stochastic_denoising_exit,
    two_forward_replacement_fraction,
)
from model.g1_state_matched_commit_forcing import (
    continuous_trust_checks,
    evaluate_hard_trust_region,
    normalized_rebased_residual,
    select_smcf_target,
    smcf_numeric_diagnostics,
    state_matched_full_latent,
)
from model.g1_torch_kinematics import G1TorchKinematics
from train_g1_native_rvqvae import format_duration, save_checkpoint_atomic


EXPERIMENT_ID = "EXP-20260723-v6f-w-paper-faithful-dc-streaming"
MINIMUM_LATENT_WIDTH_EXPERIMENT_ID = (
    "EXP-20260729-v6f-x-minimum-latent-width"
)
MINIMUM_LATENT_WIDTH_PROFILE = "minimum_latent_width_screening"
MINIMUM_LATENT_WIDTH_EXTENSION_PROFILE = (
    "minimum_latent_width_100k_extension"
)
MINIMUM_LATENT_WIDTH_COMMIT_FORCING_PROFILE = (
    "minimum_latent_width_commit_forcing"
)
MINIMUM_LATENT_WIDTH_FINAL_EXPERIMENT_ID = (
    "EXP-20260804-v6f-x-pure-commit-forcing-final"
)
MINIMUM_LATENT_WIDTH_FINAL_PROFILE = (
    "minimum_latent_width_final_commit_forcing"
)
MINIMUM_LATENT_WIDTH_PROFILES = (
    MINIMUM_LATENT_WIDTH_PROFILE,
    MINIMUM_LATENT_WIDTH_EXTENSION_PROFILE,
    MINIMUM_LATENT_WIDTH_COMMIT_FORCING_PROFILE,
    MINIMUM_LATENT_WIDTH_FINAL_PROFILE,
)
MINIMUM_LATENT_WIDTH_STAGES = ("q0_base", "teacher", "two_forward")
MINIMUM_LATENT_WIDTH_UPDATES = 50_000
MINIMUM_LATENT_WIDTH_SAVE_UPDATES = "10000,25000,50000"
MINIMUM_LATENT_WIDTH_EXTENSION_EXPERIMENT_ID = (
    "EXP-20260802-v6f-x-minimum-latent-width-100k-extension"
)
MINIMUM_LATENT_WIDTH_EXTENSION_UPDATES = 100_000
MINIMUM_LATENT_WIDTH_EXTENSION_SAVE_UPDATES = "75000,100000"
MINIMUM_LATENT_WIDTH_COMMIT_FORCING_EXPERIMENT_ID = (
    "EXP-20260803-v6f-x-commit-forcing-latent-width"
)
MINIMUM_LATENT_WIDTH_COMMIT_FORCING_UPDATES = 100_000
MINIMUM_LATENT_WIDTH_COMMIT_FORCING_SAVE_UPDATES = (
    "10000,25000,50000,75000,100000"
)
MINIMUM_LATENT_WIDTH_FINAL_Q0_SAVE_UPDATES = "75000,100000"
MINIMUM_LATENT_WIDTH_FINAL_TEACHER_SAVE_UPDATES = "50000,100000"
MINIMUM_LATENT_WIDTH_FINAL_CF_UPDATES = 200_000
MINIMUM_LATENT_WIDTH_FINAL_CF_SAVE_UPDATES = (
    "10000,25000,50000,75000,100000,125000,150000,175000,200000"
)
MINIMUM_LATENT_WIDTH_FINAL_CURRICULUM_UPDATES = 100_000
STRUCTURAL_QUANTIZER_EXPERIMENT_ID = (
    "EXP-20260806-v6f-q-commit-state-topological-quantization"
)
STRUCTURAL_QUANTIZER_PROFILE = "structural_quantizer_ablation"
STRUCTURAL_QUANTIZER_STAGES = ("q0_base", "teacher", "commit_forcing")
STRUCTURAL_QUANTIZER_PARENT_UPDATES = 100_000
STRUCTURAL_QUANTIZER_PARENT_SAVE_UPDATES = "50000,100000"
STRUCTURAL_QUANTIZER_CF_UPDATES = 200_000
STRUCTURAL_QUANTIZER_CF_SAVE_UPDATES = (
    "100000,125000,150000,175000,200000"
)
STRUCTURAL_QUANTIZER_CF_CURRICULUM_UPDATES = 100_000
PAPER_NAME_STABLE_INITIALIZATION_VERSION = "paper_name_stable_fresh_v1"
V6F_Z_EXPERIMENT_ID = "EXP-20260807-v6f-z-clean-representation"
STATIC_GENERATED_Q0_STAGES = (
    "teacher",
    "ode_distill",
    "diffusion_forcing",
    "df_homogeneous_control",
)
FORMAL_UPDATES = {
    "q0_base": 100_000,
    "teacher": 100_000,
    "ode_distill": 100_000,
    "self_forcing_dmd": 2_400,
    "self_forcing_gan": 2_400,
    "diffusion_forcing": 100_000,
    "df_homogeneous_control": 100_000,
    "two_forward": 100_000,
    "one_forward_control": 100_000,
    "two_forward_no_rebasing": 100_000,
    "two_forward_oracle_state": 100_000,
    "commit_forcing": 100_000,
}
FORMAL_SAVE_UPDATES = {
    "q0_base": "50000,100000",
    "teacher": "50000,100000",
    "ode_distill": "50000,100000",
    "self_forcing_dmd": "600,1200,2400",
    "self_forcing_gan": "600,1200,2400",
    "diffusion_forcing": "50000,100000",
    "df_homogeneous_control": "50000,100000",
    "two_forward": "50000,100000",
    "one_forward_control": "50000,100000",
    "two_forward_no_rebasing": "50000,100000",
    "two_forward_oracle_state": "50000,100000",
    "commit_forcing": "50000,100000",
}
V6F_Z_FORMAL_UPDATES = {
    "q0_base": 50_000,
    "teacher": 50_000,
    "commit_forcing": 100_000,
}
V6F_Z_FORMAL_SAVE_UPDATES = {
    "q0_base": "25000,50000",
    "teacher": "50000",
    "commit_forcing": "50000,100000",
}
COMMIT_FORCING_CONTINUATION_START = 100_000
COMMIT_FORCING_CONTINUATION_CONTRACTS = {
    150_000: {125_000, 150_000},
    300_000: {175_000, 200_000, 225_000, 250_000, 275_000, 300_000},
    400_000: {325_000, 350_000, 375_000, 400_000},
    500_000: {425_000, 450_000, 475_000, 500_000},
}
COMMIT_FORCING_CONTINUATION_PARENT_UPDATES = {
    150_000: 100_000,
    300_000: 150_000,
    400_000: 300_000,
    500_000: 400_000,
}
COMMIT_FORCING_CONTINUATION_PARENT_RUNS = {
    150_000: "formal_c2_causalv3_commit_forcing_seed2345",
    300_000: (
        "formal_c2_causalv3_commit_forcing_"
        "fullgenerated15_continue100k_150k_seed2345"
    ),
    400_000: (
        "formal_c2_causalv3_commit_forcing_"
        "fullgenerated15_continue150k_300k_seed2345"
    ),
    500_000: (
        "formal_c2_causalv3_commit_forcing_"
        "fullgenerated15_continue300k_400k_seed2345"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default=EXPERIMENT_ID)
    parser.add_argument(
        "--training_profile",
        choices=(
            "legacy",
            *MINIMUM_LATENT_WIDTH_PROFILES,
            STRUCTURAL_QUANTIZER_PROFILE,
        ),
        default="legacy",
    )
    parser.add_argument(
        "--cache_schema_version",
        default=PAPER_DC_CACHE_VERSION,
    )
    parser.add_argument("--stage", choices=PAPER_DC_STAGES, required=True)
    parser.add_argument("--codec_checkpoint", required=True)
    parser.add_argument("--q0_checkpoint", default="")
    parser.add_argument("--teacher_checkpoint", default="")
    parser.add_argument("--student_checkpoint", default="")
    parser.add_argument("--residual_initialization_checkpoint", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--expected_resume_sha256",
        default=os.environ.get("EXPECTED_RESUME_SHA256", ""),
        help=(
            "Frozen SHA256 of --checkpoint. The guarded 100k latent-width "
            "extension requires this and verifies it before deserialization."
        ),
    )
    parser.add_argument("--resume_optimizer", action="store_true")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--data_path", default="data/finedance_g1_fkbeats")
    parser.add_argument(
        "--generator_cache_dir",
        default=f"cached_features/{EXPERIMENT_ID}",
    )
    parser.add_argument(
        "--cache_provenance_experiment_id",
        default="",
        help=(
            "Experiment identity recorded by an immutable reused cache. "
            "Only the guarded minimum-width 100k extension may differ from "
            "--experiment_id."
        ),
    )
    parser.add_argument("--rebuild_generator_cache", action="store_true")
    parser.add_argument("--prepare_cache_only", action="store_true")
    parser.add_argument("--cache_batch_size", default=8, type=int)
    parser.add_argument("--cache_device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cache_data_len", default=0, type=int)
    parser.add_argument("--data_len", default=0, type=int)
    parser.add_argument("--eval_data_len", default=0, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--q0_generation_batch_factor", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--total_updates", default=0, type=int)
    parser.add_argument("--save_updates", default="")
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--max_updates", default=0, type=int)
    parser.add_argument("--profile_run", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--model_dim", default=768, type=int)
    parser.add_argument("--num_layers", default=12, type=int)
    parser.add_argument("--num_heads", default=12, type=int)
    parser.add_argument("--ffn_dim", default=3072, type=int)
    parser.add_argument("--diffusion_head_layers", default=9, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--student_steps", choices=(4, 10), default=10, type=int)
    parser.add_argument("--nfe_teacher", default=50, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--minimum_learning_rate", default=2e-5, type=float)
    parser.add_argument(
        "--constant_learning_rate_from_update",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--commit_forcing_full_generated_from_update",
        default=0,
        type=int,
    )
    parser.add_argument("--critic_learning_rate", default=4e-7, type=float)
    parser.add_argument("--lr_scale", choices=(1.0, 3.0), default=1.0, type=float)
    parser.add_argument(
        "--lr_selection_manifest",
        default="",
        help="Endpoint Self-Forcing LR selector JSON; resolves the frozen LR scale.",
    )
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--warmup_updates", default=10_000, type=int)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--mixed_precision", choices=("no", "bf16", "fp16"), default="bf16")
    parser.add_argument("--ema_decay", default=0.99, type=float)
    parser.add_argument("--ema_start_update", default=200, type=int)
    parser.add_argument("--dmd_fake_updates_per_generator", default=5, type=int)
    parser.add_argument("--physical_gradient_ratio", default=0.10, type=float)
    parser.add_argument(
        "--commit_forcing_generated_ceiling",
        default=1.0,
        type=float,
        help=(
            "Maximum whole-C4 generated-context probability after the "
            "Commit-Forcing curriculum saturates."
        ),
    )
    parser.add_argument(
        "--commit_forcing_curriculum_updates",
        default=0,
        type=int,
        help=(
            "Explicit Commit-Forcing curriculum horizon; zero uses the "
            "training-profile default."
        ),
    )
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument(
        "--g1_fk_model_path",
        default="third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument("--g1_root_quat_order", choices=("wxyz", "xyzw"), default="xyzw")
    parser.add_argument("--wandb_pj_name", default="Musics2Dance")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(int(block_size)), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_sha256(path):
    return sha256(path) if path else ""


def manifest_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def require_existing_generator_cache(
    *,
    codec_checkpoint,
    data_path,
    generator_cache_dir,
    cache_data_len,
    expected_manifest_digest,
):
    provenance, _, _, _ = _current_cache_provenance(
        codec_checkpoint,
        data_path,
        cache_data_len,
    )
    paths = paper_dc_cache_paths(generator_cache_dir, provenance)
    if not _cache_ready(paths, provenance, cache_data_len):
        raise RuntimeError(
            "frozen generator cache is absent for the canonical data path; "
            "refusing an implicit cache rebuild"
        )
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    actual_digest = metadata.get("manifest_digest")
    if actual_digest != expected_manifest_digest:
        raise ValueError("existing generator cache manifest digest mismatch")
    return {
        "cache_dir": str(paths["metadata"].parent),
        "data_path": str(Path(data_path).resolve()),
        "manifest_digest": actual_digest,
        "train_examples": int(metadata["splits"]["train"]["count"]),
        "test_examples": int(metadata["splits"]["test"]["count"]),
    }


def optimizer_parameter_manifest(primary, q0_model, codec, stage):
    manifest = {
        "stage": stage,
        "optimizer_modules": ["residual"],
        "residual": {
            name: int(parameter.numel())
            for name, parameter in primary.named_parameters()
            if parameter.requires_grad
        },
        "q0": {
            name: {
                "numel": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in q0_model.named_parameters()
        },
        "codec": {
            name: {
                "numel": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in codec.named_parameters()
        },
    }
    if stage == "y1_cf_joint_control":
        manifest["optimizer_modules"].append("q0")
    elif stage in Y_ROUTE_STAGES:
        if any(item["requires_grad"] for item in manifest["q0"].values()):
            raise AssertionError(f"{stage} has trainable q0 parameters")
        if any(item["requires_grad"] for item in manifest["codec"].values()):
            raise AssertionError(f"{stage} has trainable codec parameters")
    return manifest


def load_y_parent(path, expected_sha256, config, device):
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Y-route parent checkpoint SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "commit_forcing":
        raise ValueError(
            "Y routes require an accepted Commit-Forcing parent; "
            "Two-Forward substitution is forbidden"
        )
    if "model" not in checkpoint or "q0_model" not in checkpoint:
        raise KeyError("Y-route parent must embed residual and q0 model states")
    if checkpoint.get("config") != config.manifest():
        raise ValueError("Y-route parent model config mismatch")
    residual = G1ResidualDiffusionGenerator(config)
    residual.load_state_dict(checkpoint["model"], strict=True)
    q0 = G1PaperQ0Generator(config)
    q0.load_state_dict(checkpoint["q0_model"], strict=True)
    return residual.to(device), q0.to(device), checkpoint, actual_sha256


def initialize_paper_name_stable(model, seed):
    reset_modules = []
    for name, module in model.named_modules():
        direct_parameters = list(module.named_parameters(recurse=False))
        if not direct_parameters:
            continue
        shape_signature = ",".join(
            f"{parameter_name}:{tuple(parameter.shape)}"
            for parameter_name, parameter in direct_parameters
        )
        payload = (
            f"{PAPER_NAME_STABLE_INITIALIZATION_VERSION}|{int(seed)}|"
            f"{name}|{type(module).__module__}.{type(module).__qualname__}|"
            f"{shape_signature}"
        )
        module_seed = int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:8],
            "big",
        ) % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(module_seed)
            if callable(getattr(module, "reset_parameters", None)):
                module.reset_parameters()
            elif (
                len(direct_parameters) == 1
                and direct_parameters[0][0] == "position"
            ):
                torch.nn.init.normal_(direct_parameters[0][1], std=0.02)
            else:
                raise TypeError(
                    "paper name-stable initialization has no reset rule for "
                    f"{name}:{type(module).__qualname__}"
                )
        reset_modules.append(name)
    for module in model.modules():
        if isinstance(module, AdaLNResidualBlock):
            torch.nn.init.zeros_(module.modulation[-1].weight)
            torch.nn.init.zeros_(module.modulation[-1].bias)
    if isinstance(model, G1ResidualDiffusionGenerator):
        torch.nn.init.zeros_(model.output.weight)
        torch.nn.init.zeros_(model.output.bias)
    return {
        "version": PAPER_NAME_STABLE_INITIALIZATION_VERSION,
        "base_seed": int(seed),
        "reset_modules": reset_modules,
        "state_sha256": model_state_sha256(model),
    }


def parse_update_set(value):
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def prepare_args(args):
    formal_updates = FORMAL_UPDATES
    formal_save_updates = FORMAL_SAVE_UPDATES
    if args.experiment_id == V6F_Z_EXPERIMENT_ID:
        formal_updates = V6F_Z_FORMAL_UPDATES
        formal_save_updates = V6F_Z_FORMAL_SAVE_UPDATES
        if args.stage not in formal_updates:
            raise ValueError(
                "V6f-Z Phase B is limited to q0_base, teacher, and "
                "commit_forcing"
            )
    minimum_width_profile = args.training_profile in MINIMUM_LATENT_WIDTH_PROFILES
    structural_quantizer_profile = (
        args.training_profile == STRUCTURAL_QUANTIZER_PROFILE
    )
    frozen_paired_profile = minimum_width_profile or structural_quantizer_profile
    minimum_width_extension = (
        args.training_profile == MINIMUM_LATENT_WIDTH_EXTENSION_PROFILE
    )
    minimum_width_commit_forcing = (
        args.training_profile == MINIMUM_LATENT_WIDTH_COMMIT_FORCING_PROFILE
    )
    minimum_width_final = (
        args.training_profile == MINIMUM_LATENT_WIDTH_FINAL_PROFILE
    )
    if args.total_updates <= 0:
        if structural_quantizer_profile:
            args.total_updates = (
                STRUCTURAL_QUANTIZER_CF_UPDATES
                if args.stage == "commit_forcing"
                else STRUCTURAL_QUANTIZER_PARENT_UPDATES
            )
        elif minimum_width_final:
            args.total_updates = (
                MINIMUM_LATENT_WIDTH_FINAL_CF_UPDATES
                if args.stage == "commit_forcing"
                else MINIMUM_LATENT_WIDTH_EXTENSION_UPDATES
            )
        elif minimum_width_commit_forcing:
            args.total_updates = MINIMUM_LATENT_WIDTH_COMMIT_FORCING_UPDATES
        elif minimum_width_extension:
            args.total_updates = MINIMUM_LATENT_WIDTH_EXTENSION_UPDATES
        elif minimum_width_profile:
            args.total_updates = MINIMUM_LATENT_WIDTH_UPDATES
        else:
            args.total_updates = formal_updates[args.stage]
    if not args.save_updates:
        if structural_quantizer_profile:
            args.save_updates = (
                STRUCTURAL_QUANTIZER_CF_SAVE_UPDATES
                if args.stage == "commit_forcing"
                else STRUCTURAL_QUANTIZER_PARENT_SAVE_UPDATES
            )
        elif minimum_width_final:
            args.save_updates = {
                "q0_base": MINIMUM_LATENT_WIDTH_FINAL_Q0_SAVE_UPDATES,
                "teacher": MINIMUM_LATENT_WIDTH_FINAL_TEACHER_SAVE_UPDATES,
                "commit_forcing": MINIMUM_LATENT_WIDTH_FINAL_CF_SAVE_UPDATES,
            }[args.stage]
        elif minimum_width_commit_forcing:
            args.save_updates = MINIMUM_LATENT_WIDTH_COMMIT_FORCING_SAVE_UPDATES
        elif minimum_width_extension:
            args.save_updates = MINIMUM_LATENT_WIDTH_EXTENSION_SAVE_UPDATES
        elif minimum_width_profile:
            args.save_updates = MINIMUM_LATENT_WIDTH_SAVE_UPDATES
        else:
            args.save_updates = formal_save_updates[args.stage]
    if not args.exp_name:
        args.exp_name = f"{args.experiment_id}_{args.stage}_seed{args.seed}"
    if args.lr_selection_manifest:
        if not args.stage.startswith("self_forcing_"):
            raise ValueError("LR selection manifests are only valid for Self Forcing")
        selection = json.loads(
            Path(args.lr_selection_manifest).read_text(encoding="utf-8")
        )
        if selection.get("objective") != args.stage:
            raise ValueError("Self-Forcing LR selector objective mismatch")
        if int(selection.get("student_steps", -1)) != args.student_steps:
            raise ValueError("Self-Forcing LR selector student-step mismatch")
        selected_lr_scale = float(selection["selected_lr_scale"])
        if args.lr_scale != 1.0 and not math.isclose(
            args.lr_scale,
            selected_lr_scale,
        ):
            raise ValueError("requested LR scale disagrees with selector")
        args.lr_scale = selected_lr_scale
    formal_capacity = (
        args.model_dim,
        args.num_layers,
        args.num_heads,
        args.ffn_dim,
        args.diffusion_head_layers,
    ) == (768, 12, 12, 3072, 9)
    commit_forcing_continuation = (
        args.stage == "commit_forcing"
        and bool(args.checkpoint)
        and args.resume_optimizer
        and args.total_updates in COMMIT_FORCING_CONTINUATION_CONTRACTS
        and args.constant_learning_rate_from_update
        == COMMIT_FORCING_CONTINUATION_START
        and args.commit_forcing_full_generated_from_update
        == COMMIT_FORCING_CONTINUATION_START
        and parse_update_set(args.save_updates)
        == COMMIT_FORCING_CONTINUATION_CONTRACTS[args.total_updates]
    )
    if (
        args.constant_learning_rate_from_update
        or args.commit_forcing_full_generated_from_update
    ) and not commit_forcing_continuation:
        raise ValueError(
            "Commit Forcing continuation controls require the frozen "
            "100k-to-150k, 150k-to-300k, 300k-to-400k, or "
            "400k-to-500k exact-resume contract"
        )
    if minimum_width_profile:
        if minimum_width_final:
            expected_experiment_id = MINIMUM_LATENT_WIDTH_FINAL_EXPERIMENT_ID
        elif minimum_width_commit_forcing:
            expected_experiment_id = (
                MINIMUM_LATENT_WIDTH_COMMIT_FORCING_EXPERIMENT_ID
            )
        elif minimum_width_extension:
            expected_experiment_id = MINIMUM_LATENT_WIDTH_EXTENSION_EXPERIMENT_ID
        else:
            expected_experiment_id = MINIMUM_LATENT_WIDTH_EXPERIMENT_ID
        if args.experiment_id != expected_experiment_id:
            raise ValueError(
                "minimum-latent-width profile requires its frozen experiment ID"
            )
        if (
            minimum_width_extension
            or minimum_width_commit_forcing
            or minimum_width_final
        ):
            if (
                args.cache_provenance_experiment_id
                != MINIMUM_LATENT_WIDTH_EXPERIMENT_ID
            ):
                raise ValueError(
                    "minimum-latent-width derived profiles require the frozen "
                    "50k cache provenance experiment ID"
                )
        elif args.cache_provenance_experiment_id:
            raise ValueError(
                "screening profile cannot override cache provenance identity"
            )
        if args.cache_schema_version != MINIMUM_LATENT_WIDTH_CACHE_VERSION:
            raise ValueError(
                "minimum-latent-width profile requires the width-safe cache schema"
            )
        if minimum_width_final:
            permitted_stages = ("q0_base", "teacher", "commit_forcing")
        elif minimum_width_commit_forcing:
            permitted_stages = ("commit_forcing",)
        elif minimum_width_extension:
            permitted_stages = ("two_forward",)
        else:
            permitted_stages = MINIMUM_LATENT_WIDTH_STAGES
        if args.stage not in permitted_stages:
            raise ValueError(
                "minimum-latent-width profile does not permit this stage"
            )
        if minimum_width_final:
            if args.stage == "commit_forcing":
                expected_updates = MINIMUM_LATENT_WIDTH_FINAL_CF_UPDATES
                expected_saves = {
                    10_000,
                    25_000,
                    50_000,
                    75_000,
                    100_000,
                    125_000,
                    150_000,
                    175_000,
                    200_000,
                }
            elif args.stage == "teacher":
                expected_updates = MINIMUM_LATENT_WIDTH_EXTENSION_UPDATES
                expected_saves = {50_000, 100_000}
            else:
                expected_updates = MINIMUM_LATENT_WIDTH_EXTENSION_UPDATES
                expected_saves = {75_000, 100_000}
        elif minimum_width_commit_forcing:
            expected_updates = MINIMUM_LATENT_WIDTH_COMMIT_FORCING_UPDATES
            expected_saves = {10_000, 25_000, 50_000, 75_000, 100_000}
        elif minimum_width_extension:
            expected_updates = MINIMUM_LATENT_WIDTH_EXTENSION_UPDATES
            expected_saves = {75_000, 100_000}
        else:
            expected_updates = MINIMUM_LATENT_WIDTH_UPDATES
            expected_saves = {10_000, 25_000, 50_000}
        if args.total_updates != expected_updates:
            raise ValueError(
                "minimum-latent-width profile changed its frozen update budget"
            )
        if parse_update_set(args.save_updates) != expected_saves:
            raise ValueError(
                "minimum-latent-width profile changed its frozen save cadence"
            )
        if minimum_width_extension and not args.checkpoint:
            raise ValueError(
                "minimum-latent-width 100k extension requires an exact-resume checkpoint"
            )
        if minimum_width_extension and (
            len(args.expected_resume_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in args.expected_resume_sha256
            )
        ):
            raise ValueError(
                "minimum-latent-width 100k extension requires the frozen "
                "lowercase resume checkpoint SHA256"
            )
        if minimum_width_final and args.stage == "q0_base":
            if not args.checkpoint:
                raise ValueError(
                    "final minimum-width q0 requires the frozen 50k resume"
                )
            if (
                len(args.expected_resume_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in args.expected_resume_sha256
                )
            ):
                raise ValueError(
                    "final minimum-width q0 resume requires the frozen "
                    "lowercase SHA256"
                )
        if (
            minimum_width_final
            and args.stage in ("teacher", "commit_forcing")
            and args.checkpoint
        ):
            if (
                len(args.expected_resume_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in args.expected_resume_sha256
                )
            ):
                raise ValueError(
                    "final route recovery requires the frozen lowercase SHA256"
                )
        if (
            minimum_width_commit_forcing
            and args.checkpoint
            and (
                len(args.expected_resume_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in args.expected_resume_sha256
                )
            )
        ):
            raise ValueError(
                "minimum-latent-width Commit Forcing resume requires the "
                "lowercase resume checkpoint SHA256"
            )
        if args.seed not in (1234, 2345):
            raise ValueError(
                "minimum-latent-width profile only permits screening seed "
                "1234 or approved replication seed 2345"
            )
        if (
            minimum_width_final
            and args.stage in ("q0_base", "teacher")
            and args.seed != 1234
        ):
            raise ValueError("final parent routes are frozen to seed 1234")
        if not formal_capacity:
            raise ValueError(
                "minimum-latent-width generators require 768/12/12/3072 "
                "plus a nine-layer diffusion head"
            )
        if args.stage == "teacher" and args.residual_initialization_checkpoint:
            raise ValueError(
                "minimum-latent-width teacher must initialize from scratch"
            )
        if minimum_width_final and args.stage == "commit_forcing":
            if args.commit_forcing_curriculum_updates in (0,):
                args.commit_forcing_curriculum_updates = (
                    MINIMUM_LATENT_WIDTH_FINAL_CURRICULUM_UPDATES
                )
            if (
                args.commit_forcing_curriculum_updates
                != MINIMUM_LATENT_WIDTH_FINAL_CURRICULUM_UPDATES
            ):
                raise ValueError("final Commit-Forcing curriculum must be 100k")
            if not any(
                math.isclose(args.commit_forcing_generated_ceiling, value)
                for value in (0.95, 1.0)
            ):
                raise ValueError(
                    "final Commit-Forcing generated ceiling must be 0.95 or 1.0"
                )
        elif (
            not math.isclose(args.commit_forcing_generated_ceiling, 1.0)
            or args.commit_forcing_curriculum_updates != 0
        ):
            raise ValueError(
                "custom Commit-Forcing curriculum is reserved for the final profile"
            )
        if args.stage in ("two_forward", "commit_forcing"):
            if (
                not args.teacher_checkpoint
                or not args.residual_initialization_checkpoint
            ):
                raise ValueError(
                    "minimum-latent-width joint route requires the teacher parent"
                )
            if Path(args.teacher_checkpoint).resolve() != Path(
                args.residual_initialization_checkpoint
            ).resolve():
                raise ValueError(
                    "joint-route teacher and residual initialization must be "
                    "the same frozen parent checkpoint"
                )
    if structural_quantizer_profile:
        if args.experiment_id != STRUCTURAL_QUANTIZER_EXPERIMENT_ID:
            raise ValueError("SQA generator profile requires its frozen experiment ID")
        if args.cache_schema_version != STRUCTURAL_QUANTIZER_CACHE_VERSION:
            raise ValueError("SQA requires the quantizer-safe cache schema")
        if args.cache_provenance_experiment_id:
            raise ValueError("SQA cannot override its cache provenance identity")
        if args.stage not in STRUCTURAL_QUANTIZER_STAGES:
            raise ValueError("SQA only permits q0_base, teacher, and commit_forcing")
        expected_updates = (
            STRUCTURAL_QUANTIZER_CF_UPDATES
            if args.stage == "commit_forcing"
            else STRUCTURAL_QUANTIZER_PARENT_UPDATES
        )
        expected_saves = (
            {100_000, 125_000, 150_000, 175_000, 200_000}
            if args.stage == "commit_forcing"
            else {50_000, 100_000}
        )
        if args.total_updates != expected_updates:
            raise ValueError("SQA changed its frozen update budget")
        if parse_update_set(args.save_updates) != expected_saves:
            raise ValueError("SQA changed its frozen checkpoint cadence")
        if args.seed not in (1234, 2345):
            raise ValueError("SQA permits seed 1234 and approved replication 2345 only")
        if not formal_capacity:
            raise ValueError(
                "SQA generators require 768/12/12/3072 plus a nine-layer head"
            )
        if args.stage == "teacher" and args.residual_initialization_checkpoint:
            raise ValueError("SQA teacher must initialize from scratch")
        if args.stage == "commit_forcing":
            if args.commit_forcing_curriculum_updates == 0:
                args.commit_forcing_curriculum_updates = (
                    STRUCTURAL_QUANTIZER_CF_CURRICULUM_UPDATES
                )
            if (
                args.commit_forcing_curriculum_updates
                != STRUCTURAL_QUANTIZER_CF_CURRICULUM_UPDATES
                or not math.isclose(args.commit_forcing_generated_ceiling, 1.0)
            ):
                raise ValueError(
                    "SQA Commit Forcing is locked to a 100k curriculum and "
                    "zero oracle floor"
                )
            if (
                not args.teacher_checkpoint
                or not args.residual_initialization_checkpoint
                or Path(args.teacher_checkpoint).resolve()
                != Path(args.residual_initialization_checkpoint).resolve()
            ):
                raise ValueError(
                    "SQA Commit Forcing requires one identical frozen teacher parent"
                )
        elif (
            not math.isclose(args.commit_forcing_generated_ceiling, 1.0)
            or args.commit_forcing_curriculum_updates != 0
        ):
            raise ValueError("custom Commit Forcing curriculum is not valid here")
        if args.checkpoint and (
            len(args.expected_resume_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in args.expected_resume_sha256
            )
        ):
            raise ValueError("SQA exact resume requires a frozen lowercase SHA256")
    if not (
        args.dry_run
        or args.profile_run
        or args.prepare_cache_only
        or args.max_updates > 0
    ):
        expected_updates = (
            args.total_updates
            if frozen_paired_profile
            else FORMAL_UPDATES[args.stage]
        )
        if (
            args.total_updates != expected_updates
            and not commit_forcing_continuation
        ):
            raise ValueError("formal stage update budget differs from the frozen spec")
        if not formal_capacity:
            raise ValueError("formal stage must use 768/12/12/3072 plus nine-layer head")
        if args.wandb_mode == "disabled":
            raise ValueError("formal training must keep W&B online or offline")
    if (
        not args.prepare_cache_only
        and args.stage != "q0_base"
        and not args.q0_checkpoint
    ):
        raise ValueError(f"{args.stage} requires --q0_checkpoint")
    if args.stage == "ode_distill" and not args.teacher_checkpoint:
        raise ValueError("ODE distillation requires --teacher_checkpoint")
    if args.stage.startswith("self_forcing_"):
        if not args.teacher_checkpoint or not args.student_checkpoint:
            raise ValueError("Self Forcing requires teacher and ODE-student checkpoints")
        if args.dmd_fake_updates_per_generator != 5:
            raise ValueError("formal DMD is locked to five fake-score updates")
        if not math.isclose(args.ema_decay, 0.99):
            raise ValueError("formal Self Forcing is locked to EMA 0.99")
    if args.stage in JOINT_Q0_RESIDUAL_STAGES:
        if not args.residual_initialization_checkpoint:
            raise ValueError("joint q0/residual routes require residual initialization")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.q0_generation_batch_factor <= 0:
        raise ValueError("q0_generation_batch_factor must be positive")
    if (
        args.q0_generation_batch_factor != 1
        and args.stage not in STATIC_GENERATED_Q0_STAGES
    ):
        raise ValueError(
            "q0 generation batching is only valid for static generated-q0 stages"
        )
    return args


def initialize_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    return rank, world_size, local_rank, device


def set_seed(seed, rank=0):
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def maybe_ddp(module, device, world_size):
    if world_size <= 1:
        return module
    return DistributedDataParallel(
        module,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=False,
    )


def model_config(args, codec):
    return G1PaperFaithfulDCConfig(
        vocab_size=int(codec.config.codebook_size),
        residual_dim=int(codec.config.code_dim),
        state_dim=int(codec.config.state_dim),
        state_layout=str(codec.config.state_layout),
        model_dim=int(args.model_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        ffn_dim=int(args.ffn_dim),
        diffusion_head_layers=int(args.diffusion_head_layers),
        dropout=float(args.dropout),
        diffusion_timesteps=DIFFUSION_TIMESTEPS,
    )


def _checkpoint_model_state(checkpoint):
    for key in ("model", "generator", "student"):
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError("checkpoint has no model/generator/student state")


def _verify_parent_checkpoint(
    checkpoint,
    config,
    *,
    expected_codec_sha,
    expected_cache_manifest,
    expected_q0_source_sha=None,
):
    payload = checkpoint.get("config", {})
    expected_config = {
        name: getattr(config, name)
        for name in config.__dataclass_fields__
    }
    actual_config = {
        name: payload.get(name)
        for name in expected_config
    }
    if actual_config != expected_config:
        raise ValueError("parent generator model config mismatch")
    if checkpoint.get("codec_sha256") != expected_codec_sha:
        raise ValueError("parent generator codec provenance mismatch")
    if checkpoint.get("cache_manifest_digest") != expected_cache_manifest:
        raise ValueError("parent generator cache provenance mismatch")
    if (
        expected_q0_source_sha is not None
        and checkpoint.get("q0_source_sha256") != expected_q0_source_sha
    ):
        raise ValueError("parent residual q0 provenance mismatch")


def load_q0(
    path,
    config,
    device,
    *,
    expected_codec_sha,
    expected_cache_manifest,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "q0_base":
        raise ValueError("q0 checkpoint did not come from the paper q0 stage")
    _verify_parent_checkpoint(
        checkpoint,
        config,
        expected_codec_sha=expected_codec_sha,
        expected_cache_manifest=expected_cache_manifest,
    )
    model = G1PaperQ0Generator(config)
    model.load_state_dict(_checkpoint_model_state(checkpoint), strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def load_residual(
    path,
    config,
    device,
    expected_stages=None,
    *,
    expected_codec_sha,
    expected_cache_manifest,
    expected_q0_source_sha=None,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if expected_stages and checkpoint.get("stage") not in set(expected_stages):
        raise ValueError(
            f"residual checkpoint stage {checkpoint.get('stage')} not in "
            f"{sorted(expected_stages)}"
        )
    _verify_parent_checkpoint(
        checkpoint,
        config,
        expected_codec_sha=expected_codec_sha,
        expected_cache_manifest=expected_cache_manifest,
        expected_q0_source_sha=expected_q0_source_sha,
    )
    model = G1ResidualDiffusionGenerator(config)
    model.load_state_dict(_checkpoint_model_state(checkpoint), strict=True)
    return model.to(device), checkpoint


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def grouped_batches(iterable, factor):
    group = []
    for item in iterable:
        group.append(item)
        if len(group) == int(factor):
            yield group
            group = []
    if group:
        yield group


def validate_generation_batching(
    *,
    batches_per_epoch,
    generation_factor,
    gradient_accumulation_steps,
    checkpoint_updates,
    batch_in_epoch,
):
    if not 0 <= int(batch_in_epoch) < int(batches_per_epoch):
        raise ValueError("checkpoint batch_in_epoch is outside the data epoch")
    if int(generation_factor) % int(gradient_accumulation_steps):
        raise ValueError(
            "q0 generation batch factor must be divisible by gradient accumulation"
        )
    if int(batches_per_epoch) % int(generation_factor):
        raise ValueError(
            "data epoch batch count must be divisible by q0 generation batch factor"
        )
    if int(batch_in_epoch) % int(generation_factor):
        raise ValueError(
            "checkpoint batch_in_epoch must align with q0 generation groups"
        )
    updates_per_group = (
        int(generation_factor) // int(gradient_accumulation_steps)
    )
    if any(int(value) % updates_per_group for value in checkpoint_updates):
        raise ValueError(
            "checkpoint updates must align with q0 generation group boundaries"
        )
    return updates_per_group


def checkpoint_data_cursor(
    sampler_epoch,
    batch_in_epoch,
    batches_per_epoch,
    is_generation_group_end,
):
    if not is_generation_group_end:
        raise RuntimeError(
            "checkpoint update is not a q0 generation group boundary"
        )
    if int(batch_in_epoch) == int(batches_per_epoch):
        return int(sampler_epoch) + 1, 0
    return int(sampler_epoch), int(batch_in_epoch)


@torch.no_grad()
def prefetch_generated_q0(q0_model, plans):
    if not plans:
        raise ValueError("q0 prefetch requires at least one plan")
    target_lengths = {plan["target_q0"].shape[1] for plan in plans}
    if len(target_lengths) != 1:
        raise ValueError("q0 prefetch plans must share one target length")
    batch_sizes = [plan["history_q0"].shape[0] for plan in plans]
    generated = q0_model.generate(
        torch.cat([plan["history_q0"] for plan in plans], dim=0),
        torch.cat([plan["history_valid"] for plan in plans], dim=0),
        torch.cat([plan["state_normalized"] for plan in plans], dim=0),
        length=target_lengths.pop(),
        greedy=False,
        temperature=1.0,
    )
    return list(generated.split(batch_sizes, dim=0))


def normalize_residual(value, mean, std):
    return (value - mean) / std


def denormalize_residual(value, mean, std):
    return value * std + mean


def expand_commit_states(states, state_dim):
    if states.ndim != 3 or states.shape[1:] != (
        ROLLOUT_COMMITS,
        int(state_dim),
    ):
        raise ValueError(
            f"commit_states must have shape [B,{ROLLOUT_COMMITS},{state_dim}]"
        )
    return states.repeat_interleave(COMMIT_TOKENS, dim=1)


def build_plan_batch(batch, statistics, residual_mean, residual_std):
    history_q0 = []
    history_residual = []
    history_valid = []
    for commit in range(ROLLOUT_COMMITS):
        start = commit * COMMIT_TOKENS
        q0 = batch["committed_q0"][
            :, start : start + CAUSAL_HISTORY_VALID_TOKENS
        ]
        residual = batch["committed_residual"][
            :, start : start + CAUSAL_HISTORY_VALID_TOKENS
        ]
        history_q0.append(
            F.pad(q0, (COMMIT_TOKENS, 0))
        )
        history_residual.append(
            F.pad(residual, (0, 0, COMMIT_TOKENS, 0))
        )
        valid = torch.zeros(
            q0.shape[0],
            HISTORY_TOKENS,
            device=q0.device,
            dtype=torch.bool,
        )
        valid[:, COMMIT_TOKENS:] = True
        history_valid.append(valid)
    history_q0 = torch.stack(history_q0, dim=1)
    history_residual = torch.stack(history_residual, dim=1)
    history_valid = torch.stack(history_valid, dim=1)
    batch_size = history_q0.shape[0]
    target_motion = torch.stack(
        [
            batch["target_motion"][
                :,
                commit * COMMIT_FRAMES : commit * COMMIT_FRAMES + 16,
            ]
            for commit in range(ROLLOUT_COMMITS)
        ],
        dim=1,
    )
    state_physical = batch["commit_states"]
    state_dim = state_physical.shape[-1]
    state_normalized = statistics.normalize_state(state_physical)
    return {
        "batch_size": batch_size,
        "history_q0": history_q0.reshape(
            batch_size * ROLLOUT_COMMITS,
            HISTORY_TOKENS,
        ),
        "history_residual": normalize_residual(
            history_residual,
            residual_mean,
            residual_std,
        ).reshape(
            batch_size * ROLLOUT_COMMITS,
            HISTORY_TOKENS,
            residual_mean.shape[-1],
        ),
        "history_valid": history_valid.reshape(
            batch_size * ROLLOUT_COMMITS,
            HISTORY_TOKENS,
        ),
        "target_q0": batch["rollout_plan_q0"].reshape(
            batch_size * ROLLOUT_COMMITS,
            PLAN_TOKENS,
        ),
        "target_residual_raw": batch["rollout_plan_residual"].reshape(
            batch_size * ROLLOUT_COMMITS,
            PLAN_TOKENS,
            residual_mean.shape[-1],
        ),
        "state_physical": state_physical.reshape(
            batch_size * ROLLOUT_COMMITS,
            state_dim,
        ),
        "state_normalized": state_normalized.reshape(
            batch_size * ROLLOUT_COMMITS,
            state_dim,
        ),
        "target_motion": target_motion.reshape(
            batch_size * ROLLOUT_COMMITS,
            16,
            34,
        ),
    }


def build_rollout_batch(batch, statistics, residual_mean, residual_std):
    batch_size = batch["history_q0"].shape[0]
    state_physical = expand_commit_states(
        batch["commit_states"],
        batch["commit_states"].shape[-1],
    )
    return {
        "batch_size": batch_size,
        "history_q0": batch["history_q0"],
        "history_residual": normalize_residual(
            batch["history_residual"],
            residual_mean,
            residual_std,
        ),
        "history_valid": batch["history_valid"],
        "target_q0": batch["target_q0"],
        "target_residual_raw": batch["target_residual"],
        "state_physical": state_physical,
        "state_normalized": statistics.normalize_state(state_physical),
        "target_motion": batch["target_motion"][:, : TRAINING_ROLLOUT_TOKENS * 2],
    }


def prepared_training_batches(
    *,
    loader,
    stage,
    q0_generation_batch_factor,
    q0_model,
    device,
    statistics,
    residual_mean,
    residual_std,
    amp_enabled,
    amp_dtype,
):
    factor = (
        int(q0_generation_batch_factor)
        if stage in STATIC_GENERATED_Q0_STAGES
        else 1
    )
    for raw_group in grouped_batches(loader, factor):
        prepared = []
        for raw_batch in raw_group:
            batch = move_batch(raw_batch, device)
            prepared.append(
                (
                    batch,
                    (
                        build_plan_batch(
                            batch,
                            statistics,
                            residual_mean,
                            residual_std,
                        )
                        if stage in JOINT_Q0_RESIDUAL_STAGES
                        else None
                    ),
                    build_rollout_batch(
                        batch,
                        statistics,
                        residual_mean,
                        residual_std,
                    ),
                )
            )
        if stage in STATIC_GENERATED_Q0_STAGES:
            with autocast(
                device.type,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                generated = prefetch_generated_q0(
                    q0_model,
                    [item[2] for item in prepared],
                )
        else:
            generated = [None] * len(prepared)
        for index, (item, generated_q0) in enumerate(zip(prepared, generated)):
            yield (
                *item,
                generated_q0,
                index + 1 == len(prepared),
            )


def q0_latent(codec, q0_ids):
    return F.embedding(q0_ids, codec.quantizer.codebooks[0])


def corrected_residual_target(
    codec,
    target_q0,
    target_residual_raw,
    input_q0,
    residual_mean,
    residual_std,
):
    if target_q0.shape != input_q0.shape:
        raise ValueError("target_q0 and input_q0 must have matching shapes")
    full_latent = q0_latent(codec, target_q0) + target_residual_raw
    corrected_raw = full_latent - q0_latent(codec, input_q0)
    corrected = normalize_residual(
        corrected_raw,
        residual_mean,
        residual_std,
    )
    return corrected, corrected_raw, full_latent


def residual_target_for_input(
    codec,
    target_q0,
    target_residual_raw,
    input_q0,
    residual_mean,
    residual_std,
    *,
    rebase,
):
    if rebase:
        target, _, _ = corrected_residual_target(
            codec,
            target_q0,
            target_residual_raw,
            input_q0,
            residual_mean,
            residual_std,
        )
        return target
    return normalize_residual(
        target_residual_raw,
        residual_mean,
        residual_std,
    )


@torch.no_grad()
def generated_q0_and_corrected_target(
    q0_model,
    codec,
    plan,
    residual_mean,
    residual_std,
    *,
    greedy=False,
    generated_q0=None,
):
    if generated_q0 is None:
        generated_q0 = q0_model.generate(
            plan["history_q0"],
            plan["history_valid"],
            plan["state_normalized"],
            length=plan["target_q0"].shape[1],
            greedy=greedy,
            temperature=1.0,
        )
    elif generated_q0.shape != plan["target_q0"].shape:
        raise ValueError("prefetched q0 must match the target q0 shape")
    corrected, corrected_raw, full_latent = corrected_residual_target(
        codec,
        plan["target_q0"],
        plan["target_residual_raw"],
        generated_q0,
        residual_mean,
        residual_std,
    )
    return generated_q0, corrected, corrected_raw, full_latent


def decoded_physical_auxiliary(
    codec,
    kinematics,
    q0_ids,
    predicted_residual,
    target_motion,
    state_physical,
    statistics,
    motion_mean,
    motion_std,
    residual_mean,
    residual_std,
):
    predicted_raw_residual = denormalize_residual(
        predicted_residual,
        residual_mean,
        residual_std,
    )
    normalized_state = statistics.normalize_state(state_physical)
    reconstructed, _ = codec.decode_components(
        q0_latent(codec, q0_ids),
        predicted_raw_residual,
        normalized_state,
    )
    motion_loss = F.smooth_l1_loss(reconstructed, target_motion)
    velocity_loss = F.smooth_l1_loss(
        reconstructed[:, 1:] - reconstructed[:, :-1],
        target_motion[:, 1:] - target_motion[:, :-1],
    )
    reconstructed_raw = reconstructed * motion_std + motion_mean
    target_raw = target_motion * motion_std + motion_mean
    reconstructed_state = codec.derive_commit_state(
        reconstructed_raw[:, :COMMIT_FRAMES],
        state_physical,
        fps=statistics.fps,
    )
    target_state = codec.derive_commit_state(
        target_raw[:, :COMMIT_FRAMES],
        state_physical,
        fps=statistics.fps,
    )
    state_loss = F.smooth_l1_loss(
        statistics.normalize_state(reconstructed_state),
        statistics.normalize_state(target_state),
    )
    reconstructed_decoded = decode_g1_motion(
        reconstructed_raw,
        motion_format="g1_yaw_delta",
    )
    target_decoded = decode_g1_motion(
        target_raw,
        motion_format="g1_yaw_delta",
    )
    reconstructed_fk = kinematics(
        reconstructed_decoded["root_pos"],
        reconstructed_decoded["root_rot"],
        reconstructed_decoded["dof_pos"],
    )
    with torch.no_grad():
        target_fk = kinematics(
            target_decoded["root_pos"],
            target_decoded["root_rot"],
            target_decoded["dof_pos"],
        )
        target_feet = target_fk["feet"]
        ground = target_feet[..., 2].amin(dim=(1, 2), keepdim=True)
        target_vertical_speed = torch.zeros_like(target_feet[..., 2])
        target_vertical_speed[:, 1:] = (
            target_feet[:, 1:, :, 2] - target_feet[:, :-1, :, 2]
        ).abs() * float(statistics.fps)
        support = (
            (target_feet[..., 2] <= ground + 0.03)
            & (target_vertical_speed < 0.2)
        ).float()
    fk_loss = F.smooth_l1_loss(
        reconstructed_fk["keypoints"],
        target_fk["keypoints"],
    )
    reconstructed_feet = reconstructed_fk["feet"]
    contact_height = (
        (reconstructed_feet[..., 2] - ground).abs() * support
    ).sum() / support.sum().clamp_min(1.0)
    foot_xy_velocity = torch.zeros_like(reconstructed_feet[..., :2])
    foot_xy_velocity[:, 1:] = (
        reconstructed_feet[:, 1:, :, :2]
        - reconstructed_feet[:, :-1, :, :2]
    ) * float(statistics.fps)
    contact_slide = (
        foot_xy_velocity.norm(dim=-1) * support
    ).sum() / support.sum().clamp_min(1.0)
    total = (
        motion_loss
        + 0.5 * velocity_loss
        + 0.5 * state_loss
        + 0.5 * fk_loss
        + 0.2 * contact_height
        + 0.1 * contact_slide
    )
    return total, {
        "physical/motion": motion_loss.detach(),
        "physical/velocity": velocity_loss.detach(),
        "physical/state": state_loss.detach(),
        "physical/fk": fk_loss.detach(),
        "physical/contact_height": contact_height.detach(),
        "physical/contact_slide": contact_slide.detach(),
    }


def decoded_rollout_physical_auxiliary(
    codec,
    kinematics,
    q0_ids,
    predicted_residual,
    target_motion,
    state_physical,
    statistics,
    motion_mean,
    motion_std,
    residual_mean,
    residual_std,
):
    if q0_ids.shape[1] < PLAN_TOKENS:
        raise ValueError("rollout must contain at least one full H8 plan")
    commit_count = (
        q0_ids.shape[1] - PLAN_TOKENS
    ) // COMMIT_TOKENS + 1
    commit_index = int(
        torch.randint(
            0,
            commit_count,
            (),
            device=q0_ids.device,
        )
    )
    token_start = commit_index * COMMIT_TOKENS
    frame_start = commit_index * COMMIT_FRAMES
    return decoded_physical_auxiliary(
        codec,
        kinematics,
        q0_ids[:, token_start : token_start + PLAN_TOKENS],
        predicted_residual[:, token_start : token_start + PLAN_TOKENS],
        target_motion[:, frame_start : frame_start + 16],
        state_physical[:, token_start],
        statistics,
        motion_mean,
        motion_std,
        residual_mean,
        residual_std,
    )


def q0_stage_loss(model, plan):
    logits = model(
        plan["history_q0"],
        plan["history_valid"],
        plan["target_q0"],
        plan["state_normalized"],
    )
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        plan["target_q0"].reshape(-1),
    )
    accuracy = logits.argmax(dim=-1).eq(plan["target_q0"]).float().mean()
    return loss, {
        "loss/main": loss.detach(),
        "q0/accuracy": accuracy.detach(),
    }


def residual_training_loss(
    *,
    model,
    q0_model,
    codec,
    diffusion,
    plan,
    residual_mean,
    residual_std,
    statistics,
    motion_mean,
    motion_std,
    kinematics,
    noise_mode,
    physical_gradient_ratio,
    prefetched_generated_q0=None,
):
    if q0_model is None and prefetched_generated_q0 is None:
        generated_q0 = plan["target_q0"]
        target = normalize_residual(
            plan["target_residual_raw"],
            residual_mean,
            residual_std,
        )
        target_raw = plan["target_residual_raw"]
        full_latent = (
            q0_latent(codec, plan["target_q0"])
            + plan["target_residual_raw"]
        )
    else:
        generated_q0, target, target_raw, full_latent = (
            generated_q0_and_corrected_target(
                q0_model,
                codec,
                plan,
                residual_mean,
            residual_std,
            greedy=False,
            generated_q0=prefetched_generated_q0,
        )
        )
    if noise_mode == "independent":
        noise_levels = independent_token_noise_levels(
            target.shape[0],
            target.shape[1],
            diffusion.timesteps,
            target.device,
        )
    elif noise_mode == "homogeneous":
        noise_levels = homogeneous_sequence_noise_levels(
            target.shape[0],
            target.shape[1],
            diffusion.timesteps,
            target.device,
        )
    else:
        raise ValueError(f"unknown noise mode {noise_mode}")

    def model_fn(noisy, levels):
        return model(
            plan["history_q0"],
            plan["history_residual"],
            plan["history_valid"],
            generated_q0,
            noisy,
            levels,
            plan["state_normalized"],
        )

    prediction, token_loss = diffusion.training_loss(
        model_fn,
        target,
        noise_levels,
    )
    prediction_raw = denormalize_residual(
        prediction,
        residual_mean,
        residual_std,
    )
    combined_latent = q0_latent(codec, generated_q0) + prediction_raw
    combined_loss = F.mse_loss(combined_latent, full_latent.detach())
    main_loss = token_loss.mean() + combined_loss
    physical_loss, physical_stats = decoded_rollout_physical_auxiliary(
        codec,
        kinematics,
        generated_q0,
        prediction,
        plan["target_motion"],
        plan["state_physical"],
        statistics,
        motion_mean,
        motion_std,
        residual_mean,
        residual_std,
    )
    scale, gradient_stats = physical_auxiliary_scale(
        main_loss,
        physical_loss,
        maximum_ratio=physical_gradient_ratio,
        main_reference=prediction,
        auxiliary_reference=prediction,
    )
    total = main_loss + scale * physical_loss
    stats = {
        "loss/main": main_loss.detach(),
        "loss/diffusion": token_loss.mean().detach(),
        "loss/combined_latent": combined_loss.detach(),
        "loss/physical_unscaled": physical_loss.detach(),
        "loss/total": total.detach(),
        "q0/generated_id_agreement": generated_q0.eq(
            plan["target_q0"]
        ).float().mean().detach(),
        **physical_stats,
        **gradient_stats,
    }
    return total, stats


def ode_distillation_loss(
    *,
    student,
    teacher,
    q0_model,
    codec,
    diffusion,
    plan,
    residual_mean,
    residual_std,
    statistics,
    motion_mean,
    motion_std,
    kinematics,
    student_steps,
    physical_gradient_ratio,
    prefetched_generated_q0=None,
):
    generated_q0, target, _, _ = generated_q0_and_corrected_target(
        q0_model,
        codec,
        plan,
        residual_mean,
        residual_std,
        greedy=False,
        generated_q0=prefetched_generated_q0,
    )
    timeline = diffusion.ddim_timesteps(student_steps, target.device)
    coarse_index = torch.randint(
        0,
        student_steps,
        (target.shape[0], 1),
        device=target.device,
    )
    current_t = timeline[:-1].gather(0, coarse_index.reshape(-1)).reshape(
        target.shape[0],
        1,
    ).expand(-1, target.shape[1])
    next_t = timeline[1:].gather(0, coarse_index.reshape(-1)).reshape(
        target.shape[0],
        1,
    ).expand(-1, target.shape[1])
    noise = torch.randn_like(target).clamp(-20.0, 20.0)
    noisy = diffusion.q_sample(target, current_t, noise)
    with torch.no_grad():
        teacher_x0 = teacher(
            plan["history_q0"],
            plan["history_residual"],
            plan["history_valid"],
            generated_q0,
            noisy,
            current_t,
            plan["state_normalized"],
        )
        teacher_next = diffusion.ddim_step(
            noisy,
            teacher_x0,
            current_t,
            next_t,
        )
    student_x0 = student(
        plan["history_q0"],
        plan["history_residual"],
        plan["history_valid"],
        generated_q0,
        noisy,
        current_t,
        plan["state_normalized"],
    )
    student_next = diffusion.ddim_step(
        noisy,
        student_x0,
        current_t,
        next_t,
    )
    trajectory_loss = F.mse_loss(student_next, teacher_next)
    teacher_loss = F.mse_loss(student_x0, teacher_x0)
    target_loss = F.mse_loss(student_x0, target)
    main_loss = trajectory_loss + teacher_loss + 0.1 * target_loss
    physical_loss, physical_stats = decoded_rollout_physical_auxiliary(
        codec,
        kinematics,
        generated_q0,
        student_x0,
        plan["target_motion"],
        plan["state_physical"],
        statistics,
        motion_mean,
        motion_std,
        residual_mean,
        residual_std,
    )
    scale, gradient_stats = physical_auxiliary_scale(
        main_loss,
        physical_loss,
        maximum_ratio=physical_gradient_ratio,
        main_reference=student_x0,
        auxiliary_reference=student_x0,
    )
    total = main_loss + scale * physical_loss
    return total, {
        "loss/main": main_loss.detach(),
        "loss/ode_trajectory": trajectory_loss.detach(),
        "loss/teacher_x0": teacher_loss.detach(),
        "loss/target_x0": target_loss.detach(),
        "loss/physical_unscaled": physical_loss.detach(),
        "loss/total": total.detach(),
        **physical_stats,
        **gradient_stats,
    }


@torch.no_grad()
def recompute_two_forward_context(
    codec,
    statistics,
    initial_history_q0,
    initial_history_residual,
    initial_history_valid,
    initial_state,
    replaced_plan_q0,
    replaced_plan_residual,
    residual_mean,
    residual_std,
    motion_mean,
    motion_std,
):
    batch_size = initial_history_q0.shape[0]
    history_q0 = initial_history_q0
    history_residual = initial_history_residual
    history_valid = initial_history_valid
    state = initial_state
    contexts_q0 = []
    contexts_residual = []
    contexts_valid = []
    states = []
    for commit in range(ROLLOUT_COMMITS):
        contexts_q0.append(history_q0)
        contexts_residual.append(history_residual)
        contexts_valid.append(history_valid)
        states.append(state)
        plan_q0 = replaced_plan_q0[:, commit]
        plan_residual = replaced_plan_residual[:, commit]
        decoded, _ = codec.decode_components(
            q0_latent(codec, plan_q0),
            denormalize_residual(plan_residual, residual_mean, residual_std),
            statistics.normalize_state(state),
        )
        raw = decoded * motion_std + motion_mean
        state = codec.derive_commit_state(
            raw[:, :COMMIT_FRAMES],
            state,
            fps=statistics.fps,
        )
        history_q0 = torch.cat(
            (history_q0, plan_q0[:, :COMMIT_TOKENS]),
            dim=1,
        )[:, -HISTORY_TOKENS:]
        history_residual = torch.cat(
            (history_residual, plan_residual[:, :COMMIT_TOKENS]),
            dim=1,
        )[:, -HISTORY_TOKENS:]
        history_valid = torch.cat(
            (
                history_valid,
                torch.ones(
                    batch_size,
                    COMMIT_TOKENS,
                    device=history_valid.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )[:, -HISTORY_TOKENS:]
    return {
        "batch_size": batch_size,
        "history_q0": torch.stack(contexts_q0, dim=1).reshape(
            batch_size * ROLLOUT_COMMITS,
            HISTORY_TOKENS,
        ),
        "history_residual": torch.stack(contexts_residual, dim=1).reshape(
            batch_size * ROLLOUT_COMMITS,
            HISTORY_TOKENS,
            residual_mean.shape[-1],
        ),
        "history_valid": torch.stack(contexts_valid, dim=1).reshape(
            batch_size * ROLLOUT_COMMITS,
            HISTORY_TOKENS,
        ),
        "state_physical": torch.stack(states, dim=1).reshape(
            batch_size * ROLLOUT_COMMITS,
            state.shape[-1],
        ),
        "state_normalized": statistics.normalize_state(
            torch.stack(states, dim=1)
        ).reshape(batch_size * ROLLOUT_COMMITS, state.shape[-1]),
    }


def two_forward_loss(
    *,
    q0_model,
    residual_model,
    codec,
    diffusion,
    batch,
    plan,
    residual_mean,
    residual_std,
    statistics,
    motion_mean,
    motion_std,
    global_update,
    total_updates,
    two_forward,
    rebase_residual=True,
    recompute_state=True,
):
    if two_forward:
        with torch.no_grad():
            first_q0_logits = q0_model(
                plan["history_q0"],
                plan["history_valid"],
                plan["target_q0"],
                plan["state_normalized"],
            )
            first_q0 = first_q0_logits.argmax(dim=-1)
            levels = independent_token_noise_levels(
                plan["target_q0"].shape[0],
                PLAN_TOKENS,
                diffusion.timesteps,
                plan["target_q0"].device,
            )
            first_target = residual_target_for_input(
                codec,
                plan["target_q0"],
                plan["target_residual_raw"],
                first_q0,
                residual_mean,
                residual_std,
                rebase=rebase_residual,
            )

            def first_model_fn(noisy, noise_levels):
                return residual_model(
                    plan["history_q0"],
                    plan["history_residual"],
                    plan["history_valid"],
                    first_q0,
                    noisy,
                    noise_levels,
                    plan["state_normalized"],
                )

            first_residual, _ = diffusion.training_loss(
                first_model_fn,
                first_target,
                levels,
            )
            fraction = two_forward_replacement_fraction(
                global_update,
                total_updates,
            )
            eligible = torch.ones_like(plan["target_q0"], dtype=torch.bool)
            replaced_q0, replaced_residual, replacement_mask = (
                replace_complete_dc_tokens(
                    plan["target_q0"],
                    normalize_residual(
                        plan["target_residual_raw"],
                        residual_mean,
                        residual_std,
                    ),
                    first_q0,
                    first_residual,
                    eligible,
                    fraction=fraction,
                )
            )
            replaced_q0 = replaced_q0.view(
                plan["batch_size"],
                ROLLOUT_COMMITS,
                PLAN_TOKENS,
            )
            replaced_residual = replaced_residual.view(
                plan["batch_size"],
                ROLLOUT_COMMITS,
                PLAN_TOKENS,
                residual_mean.shape[-1],
            )
            context = recompute_two_forward_context(
                codec,
                statistics,
                batch["history_q0"],
                normalize_residual(
                    batch["history_residual"],
                    residual_mean,
                    residual_std,
                ),
                batch["history_valid"],
                batch["initial_state"],
                replaced_q0,
                replaced_residual,
                residual_mean,
                residual_std,
                motion_mean,
                motion_std,
            )
            if not recompute_state:
                context["state_physical"] = plan["state_physical"]
                context["state_normalized"] = plan["state_normalized"]
            second_input_q0 = replaced_q0.reshape(
                plan["batch_size"] * ROLLOUT_COMMITS,
                PLAN_TOKENS,
            )
            replacement_fraction = replacement_mask.float().mean()
    else:
        context = {
            key: plan[key]
            for key in (
                "batch_size",
                "history_q0",
                "history_residual",
                "history_valid",
                "state_physical",
                "state_normalized",
            )
        }
        second_input_q0 = plan["target_q0"]
        replacement_fraction = plan["target_residual_raw"].new_zeros(())

    q0_logits = q0_model(
        context["history_q0"],
        context["history_valid"],
        second_input_q0,
        context["state_normalized"],
    )
    q0_loss = F.cross_entropy(
        q0_logits.reshape(-1, q0_logits.shape[-1]),
        plan["target_q0"].reshape(-1),
    )
    levels = independent_token_noise_levels(
        plan["target_q0"].shape[0],
        PLAN_TOKENS,
        diffusion.timesteps,
        plan["target_q0"].device,
    )
    target = residual_target_for_input(
        codec,
        plan["target_q0"],
        plan["target_residual_raw"],
        second_input_q0,
        residual_mean,
        residual_std,
        rebase=rebase_residual,
    )

    def second_model_fn(noisy, noise_levels):
        return residual_model(
            context["history_q0"],
            context["history_residual"],
            context["history_valid"],
            second_input_q0,
            noisy,
            noise_levels,
            context["state_normalized"],
        )

    _, residual_token_loss = diffusion.training_loss(
        second_model_fn,
        target,
        levels,
    )
    residual_loss = residual_token_loss.mean()
    total = q0_loss + residual_loss
    return total, {
        "loss/main": total.detach(),
        "loss/q0": q0_loss.detach(),
        "loss/residual": residual_loss.detach(),
        "two_forward/replacement_fraction": replacement_fraction.detach(),
        "two_forward/residual_rebasing": total.new_tensor(
            float(rebase_residual)
        ),
        "two_forward/recomputed_state": total.new_tensor(
            float(recompute_state)
        ),
        "q0/accuracy": q0_logits.argmax(dim=-1).eq(
            plan["target_q0"]
        ).float().mean().detach(),
    }


@torch.no_grad()
def build_commit_forcing_context(
    *,
    codec,
    statistics,
    plan,
    generated_plan_q0,
    generated_plan_residual,
    residual_mean,
    residual_std,
    motion_mean,
    motion_std,
    replacement_fraction,
    generator=None,
    bernoulli_replacement=False,
):
    batch_size = int(plan["batch_size"])
    transitions = ROLLOUT_COMMITS - 1
    expected = batch_size * transitions
    if generated_plan_q0.shape != (expected, PLAN_TOKENS):
        raise ValueError("generated Commit Forcing q0 shape is invalid")
    if generated_plan_residual.shape != (
        expected,
        PLAN_TOKENS,
        residual_mean.shape[-1],
    ):
        raise ValueError("generated Commit Forcing residual shape is invalid")

    def sequence(value):
        return value.view(batch_size, ROLLOUT_COMMITS, *value.shape[1:])

    current_history_q0 = sequence(plan["history_q0"])[:, :-1].reshape(
        expected,
        HISTORY_TOKENS,
    )
    current_history_residual = sequence(
        plan["history_residual"]
    )[:, :-1].reshape(
        expected,
        HISTORY_TOKENS,
        residual_mean.shape[-1],
    )
    current_history_valid = sequence(plan["history_valid"])[:, :-1].reshape(
        expected,
        HISTORY_TOKENS,
    )
    current_state = sequence(plan["state_physical"])[:, :-1].reshape(
        expected,
        plan["state_physical"].shape[-1],
    )
    current_state_normalized = statistics.normalize_state(current_state)

    decoded, _ = codec.decode_components(
        q0_latent(codec, generated_plan_q0),
        denormalize_residual(
            generated_plan_residual,
            residual_mean,
            residual_std,
        ),
        current_state_normalized,
    )
    decoded_raw = decoded * motion_std + motion_mean
    generated_state = codec.derive_commit_state(
        decoded_raw[:, :COMMIT_FRAMES],
        current_state,
        fps=statistics.fps,
    )
    generated_history_q0 = torch.cat(
        (
            current_history_q0,
            generated_plan_q0[:, :COMMIT_TOKENS],
        ),
        dim=1,
    )[:, -HISTORY_TOKENS:]
    generated_history_residual = torch.cat(
        (
            current_history_residual,
            generated_plan_residual[:, :COMMIT_TOKENS],
        ),
        dim=1,
    )[:, -HISTORY_TOKENS:]
    generated_history_valid = torch.cat(
        (
            current_history_valid,
            torch.ones(
                expected,
                COMMIT_TOKENS,
                device=current_history_valid.device,
                dtype=torch.bool,
            ),
        ),
        dim=1,
    )[:, -HISTORY_TOKENS:]

    oracle_history_q0 = sequence(plan["history_q0"])[:, 1:].reshape(
        expected,
        HISTORY_TOKENS,
    )
    oracle_history_residual = sequence(
        plan["history_residual"]
    )[:, 1:].reshape(
        expected,
        HISTORY_TOKENS,
        residual_mean.shape[-1],
    )
    oracle_history_valid = sequence(plan["history_valid"])[:, 1:].reshape(
        expected,
        HISTORY_TOKENS,
    )
    oracle_state = sequence(plan["state_physical"])[:, 1:].reshape(
        expected,
        plan["state_physical"].shape[-1],
    )

    replacement_mask = complete_commit_replacement_mask(
        batch_size,
        transitions,
        fraction=replacement_fraction,
        device=current_history_q0.device,
        generator=generator,
        bernoulli=bernoulli_replacement,
    ).reshape(expected)
    token_mask = replacement_mask[:, None]
    residual_mask = token_mask.unsqueeze(-1)
    context_state = torch.where(
        token_mask,
        generated_state,
        oracle_state,
    )
    context = {
        "batch_size": expected,
        "history_q0": torch.where(
            token_mask,
            generated_history_q0,
            oracle_history_q0,
        ),
        "history_residual": torch.where(
            residual_mask,
            generated_history_residual,
            oracle_history_residual,
        ),
        "history_valid": torch.where(
            token_mask,
            generated_history_valid,
            oracle_history_valid,
        ),
        "state_physical": context_state,
        "state_normalized": statistics.normalize_state(context_state),
        "target_q0": sequence(plan["target_q0"])[:, 1:].reshape(
            expected,
            PLAN_TOKENS,
        ),
        "target_residual_raw": sequence(
            plan["target_residual_raw"]
        )[:, 1:].reshape(
            expected,
            PLAN_TOKENS,
            residual_mean.shape[-1],
        ),
        "replacement_mask": replacement_mask,
    }
    return context


def commit_forcing_generated_context_fraction(
    global_update,
    curriculum_updates,
    generated_context_ceiling,
):
    curriculum_updates = int(curriculum_updates)
    if curriculum_updates <= 0:
        raise ValueError("Commit-Forcing curriculum updates must be positive")
    ceiling = float(generated_context_ceiling)
    if not 0.0 <= ceiling <= 1.0:
        raise ValueError("Commit-Forcing generated-context ceiling must be in [0,1]")
    update = min(max(int(global_update), 0), curriculum_updates)
    progress = update / curriculum_updates
    return ceiling * 0.5 * (1.0 - math.cos(math.pi * progress))


def _boundary_pose_motion(state):
    if state.ndim != 2 or state.shape[-1] != 66:
        raise ValueError("boundary pose conversion requires [B,66]")
    motion = state.new_zeros(state.shape[0], 34)
    motion[:, 2] = state[:, 0]
    motion[:, 4] = 1.0
    motion[:, 5:34] = state[:, 5:34]
    return motion


def _last_frame_binary_contact(raw_motion, kinematics, fps):
    if raw_motion.ndim != 3 or raw_motion.shape[1] < 2:
        raise ValueError("contact diagnostics require at least two frames")
    decoded = decode_g1_motion(
        raw_motion[:, -2:],
        motion_format="g1_yaw_delta",
    )
    feet = kinematics(
        decoded["root_pos"],
        decoded["root_rot"],
        decoded["dof_pos"],
    )["feet"]
    ground = feet[..., 2].amin(dim=(1, 2), keepdim=True)
    vertical_speed = (
        feet[:, -1, :, 2] - feet[:, -2, :, 2]
    ).abs() * float(fps)
    return (
        (feet[:, -1, :, 2] <= ground[:, 0] + 0.03)
        & (vertical_speed < 0.2)
    )


@torch.no_grad()
def smcf_target_and_trust(
    *,
    codec,
    statistics,
    kinematics,
    context,
    second_input_q0,
    residual_mean,
    residual_std,
    motion_mean,
    motion_std,
    thresholds,
    policy,
    training_seed,
    global_update,
):
    ordinary_target = residual_target_for_input(
        codec,
        context["target_q0"],
        context["target_residual_raw"],
        second_input_q0,
        residual_mean,
        residual_std,
        rebase=True,
    )
    full_latent, _ = state_matched_full_latent(
        codec,
        context["target_motion"],
        statistics.normalize_state(context["generated_state"]),
    )
    state_matched_target, state_matched_raw = normalized_rebased_residual(
        codec,
        full_latent,
        second_input_q0,
        residual_mean,
        residual_std,
    )
    reconstruction, _ = codec.decode_components(
        q0_latent(codec, second_input_q0),
        state_matched_raw,
        statistics.normalize_state(context["generated_state"]),
        target_frames=context["target_motion"].shape[1],
    )
    target_raw = context["target_motion"] * motion_std + motion_mean
    reconstruction_raw = reconstruction * motion_std + motion_mean
    target_decoded = decode_g1_motion(
        target_raw,
        motion_format="g1_yaw_delta",
    )
    reconstructed_decoded = decode_g1_motion(
        reconstruction_raw,
        motion_format="g1_yaw_delta",
    )
    target_fk = kinematics(
        target_decoded["root_pos"],
        target_decoded["root_rot"],
        target_decoded["dof_pos"],
    )
    reconstructed_fk = kinematics(
        reconstructed_decoded["root_pos"],
        reconstructed_decoded["root_rot"],
        reconstructed_decoded["dof_pos"],
    )
    reconstruction_fk = (
        reconstructed_fk["keypoints"] - target_fk["keypoints"]
    ).square().flatten(1).mean(dim=-1)

    generated_contact = _last_frame_binary_contact(
        context["generated_commit_raw"],
        kinematics,
        statistics.fps,
    )
    oracle_to_target = torch.stack(
        (
            _boundary_pose_motion(context["oracle_state"]),
            target_raw[:, 0],
        ),
        dim=1,
    )
    target_contact = _last_frame_binary_contact(
        oracle_to_target,
        kinematics,
        statistics.fps,
    )
    contact_equal = generated_contact.eq(target_contact).all(dim=-1)
    batch_size = context["generated_state"].shape[0]
    normalization_valid = torch.full(
        (batch_size,),
        bool(
            torch.isfinite(residual_mean).all()
            and torch.isfinite(residual_std).all()
            and torch.all(residual_std > 0)
        ),
        device=context["generated_state"].device,
        dtype=torch.bool,
    )
    diagnostics = smcf_numeric_diagnostics(
        generated_state_normalized=statistics.normalize_state(
            context["generated_state"]
        ),
        oracle_state_normalized=statistics.normalize_state(
            context["oracle_state"]
        ),
        state_matched_residual_raw=state_matched_raw,
        target_motion=context["target_motion"],
        reconstruction=reconstruction,
        reconstruction_fk=reconstruction_fk,
        contact_equal=contact_equal,
        causal=torch.ones(
            batch_size,
            device=context["generated_state"].device,
            dtype=torch.bool,
        ),
        normalization_valid=normalization_valid,
    )
    checks = continuous_trust_checks(diagnostics, thresholds)
    trust = evaluate_hard_trust_region(
        checks,
        eligible=context["replacement_mask"],
    )
    selection = select_smcf_target(
        ordinary_target,
        state_matched_target,
        trust,
        policy=policy,
        dataset_index=context["dataset_index"],
        transition_index=context["transition_index"],
        training_seed=training_seed,
        global_update=global_update,
    )
    return selection, diagnostics


def commit_forcing_loss(
    *,
    q0_model,
    residual_model,
    codec,
    diffusion,
    plan,
    residual_mean,
    residual_std,
    statistics,
    motion_mean,
    motion_std,
    global_update,
    total_updates,
    first_pass_nfe=10,
    full_generated_from_update=0,
    generated_context_ceiling=1.0,
    bernoulli_replacement=False,
):
    batch_size = int(plan["batch_size"])
    transitions = ROLLOUT_COMMITS - 1
    expected = batch_size * transitions

    def sequence(value):
        return value.view(batch_size, ROLLOUT_COMMITS, *value.shape[1:])

    current_history_q0 = sequence(plan["history_q0"])[:, :-1].reshape(
        expected,
        HISTORY_TOKENS,
    )
    current_history_residual = sequence(
        plan["history_residual"]
    )[:, :-1].reshape(
        expected,
        HISTORY_TOKENS,
        residual_mean.shape[-1],
    )
    current_history_valid = sequence(plan["history_valid"])[:, :-1].reshape(
        expected,
        HISTORY_TOKENS,
    )
    current_state_normalized = sequence(
        plan["state_normalized"]
    )[:, :-1].reshape(expected, plan["state_normalized"].shape[-1])

    with torch.no_grad():
        generated_plan_q0 = q0_model.generate(
            current_history_q0,
            current_history_valid,
            current_state_normalized,
            length=PLAN_TOKENS,
            greedy=False,
            temperature=1.0,
        )

        def first_model_fn(noisy, noise_levels):
            return residual_model(
                current_history_q0,
                current_history_residual,
                current_history_valid,
                generated_plan_q0,
                noisy,
                noise_levels,
                current_state_normalized,
            )

        generated_plan_residual = diffusion.sample(
            first_model_fn,
            (
                expected,
                PLAN_TOKENS,
                residual_mean.shape[-1],
            ),
            int(first_pass_nfe),
        )
        requested_fraction = (
            1.0
            if int(full_generated_from_update) > 0
            and int(global_update) >= int(full_generated_from_update)
            else commit_forcing_generated_context_fraction(
                global_update,
                total_updates,
                generated_context_ceiling,
            )
        )
        context = build_commit_forcing_context(
            codec=codec,
            statistics=statistics,
            plan=plan,
            generated_plan_q0=generated_plan_q0,
            generated_plan_residual=generated_plan_residual,
            residual_mean=residual_mean,
            residual_std=residual_std,
            motion_mean=motion_mean,
            motion_std=motion_std,
            replacement_fraction=requested_fraction,
            bernoulli_replacement=bernoulli_replacement,
        )
        second_input_q0 = q0_model.generate(
            context["history_q0"],
            context["history_valid"],
            context["state_normalized"],
            length=PLAN_TOKENS,
            greedy=False,
            temperature=1.0,
        )

    q0_logits = q0_model(
        context["history_q0"],
        context["history_valid"],
        second_input_q0,
        context["state_normalized"],
    )
    q0_loss = F.cross_entropy(
        q0_logits.reshape(-1, q0_logits.shape[-1]),
        context["target_q0"].reshape(-1),
    )
    levels = independent_token_noise_levels(
        expected,
        PLAN_TOKENS,
        diffusion.timesteps,
        context["target_q0"].device,
    )
    target = residual_target_for_input(
        codec,
        context["target_q0"],
        context["target_residual_raw"],
        second_input_q0,
        residual_mean,
        residual_std,
        rebase=True,
    )

    def second_model_fn(noisy, noise_levels):
        return residual_model(
            context["history_q0"],
            context["history_residual"],
            context["history_valid"],
            second_input_q0,
            noisy,
            noise_levels,
            context["state_normalized"],
        )

    _, residual_token_loss = diffusion.training_loss(
        second_model_fn,
        target,
        levels,
    )
    residual_loss = residual_token_loss.mean()
    total = q0_loss + residual_loss
    return total, {
        "loss/main": total.detach(),
        "loss/q0": q0_loss.detach(),
        "loss/residual": residual_loss.detach(),
        "commit_forcing/requested_fraction": total.new_tensor(
            requested_fraction
        ),
        "commit_forcing/actual_fraction": (
            context["replacement_mask"].float().mean().detach()
        ),
        "commit_forcing/generated_transition_count": (
            context["replacement_mask"]
            .view(batch_size, transitions)
            .sum(dim=1)
            .float()
            .mean()
            .detach()
        ),
        "commit_forcing/first_pass_nfe": total.new_tensor(
            float(first_pass_nfe)
        ),
        "commit_forcing/generated_context_ceiling": total.new_tensor(
            float(generated_context_ceiling)
        ),
        "commit_forcing/oracle_context_floor": total.new_tensor(
            1.0 - float(generated_context_ceiling)
        ),
        "q0/accuracy": q0_logits.argmax(dim=-1).eq(
            context["target_q0"]
        ).float().mean().detach(),
    }


def self_forcing_generated_rollout(
    *,
    student,
    q0_model,
    codec,
    diffusion,
    batch,
    residual_mean,
    residual_std,
    statistics,
    motion_mean,
    motion_std,
    student_steps,
):
    history_q0 = batch["history_q0"]
    history_residual = normalize_residual(
        batch["history_residual"],
        residual_mean,
        residual_std,
    )
    history_valid = batch["history_valid"]
    state_physical = batch["initial_state"]
    timeline = diffusion.ddim_timesteps(
        student_steps,
        history_q0.device,
    )[:-1]

    def q0_plan_fn(
        current_q0,
        current_residual,
        current_valid,
        current_state,
        length,
    ):
        return q0_model.generate(
            current_q0,
            current_valid,
            statistics.normalize_state(current_state),
            length=length,
            greedy=False,
            temperature=1.0,
        )

    def residual_plan_fn(
        current_q0,
        current_residual,
        current_valid,
        current_state,
        plan_q0,
        enable_gradient,
    ):
        exit_index = int(
            torch.randint(
                0,
                student_steps,
                (),
                device=plan_q0.device,
            )
        )

        def model_fn(noisy, levels):
            return student(
                current_q0,
                current_residual,
                current_valid,
                plan_q0,
                noisy,
                levels,
                statistics.normalize_state(current_state),
            )

        noisy = torch.randn(
            plan_q0.shape[0],
            PLAN_TOKENS,
            residual_mean.shape[-1],
            device=plan_q0.device,
        )
        return stochastic_denoising_exit(
            diffusion,
            model_fn,
            noisy,
            timeline,
            exit_index,
            enable_gradient=enable_gradient,
        )

    def state_update_fn(current_state, plan_q0, plan_residual):
        decoded, _ = codec.decode_components(
            q0_latent(codec, plan_q0),
            denormalize_residual(
                plan_residual,
                residual_mean,
                residual_std,
            ),
            statistics.normalize_state(current_state),
        )
        raw = decoded * motion_std + motion_mean
        return codec.derive_commit_state(
            raw[:, :COMMIT_FRAMES],
            current_state,
            fps=statistics.fps,
        )

    return self_forcing_rollout(
        history_q0=history_q0,
        history_residual=history_residual,
        history_valid=history_valid,
        state=state_physical,
        q0_plan_fn=q0_plan_fn,
        residual_plan_fn=residual_plan_fn,
        state_update_fn=state_update_fn,
    )


def self_forcing_generator_loss(
    *,
    stage,
    student,
    real_score,
    fake_score,
    gan_critic,
    q0_model,
    codec,
    diffusion,
    batch,
    residual_mean,
    residual_std,
    statistics,
    motion_mean,
    motion_std,
    kinematics,
    student_steps,
    physical_gradient_ratio,
):
    rollout = self_forcing_generated_rollout(
        student=student,
        q0_model=q0_model,
        codec=codec,
        diffusion=diffusion,
        batch=batch,
        residual_mean=residual_mean,
        residual_std=residual_std,
        statistics=statistics,
        motion_mean=motion_mean,
        motion_std=motion_std,
        student_steps=student_steps,
    )
    generated_raw_residual = denormalize_residual(
        rollout["residual"],
        residual_mean,
        residual_std,
    )
    generated_latent = (
        q0_latent(codec, rollout["q0"]) + generated_raw_residual
    )
    initial_state_normalized = statistics.normalize_state(batch["initial_state"])
    score_levels = homogeneous_sequence_noise_levels(
        generated_latent.shape[0],
        generated_latent.shape[1],
        diffusion.timesteps,
        generated_latent.device,
    )
    score_noise = torch.randn_like(rollout["residual"]).clamp(-20.0, 20.0)
    noisy_residual = diffusion.q_sample(
        rollout["residual"],
        score_levels,
        score_noise,
    )
    with torch.no_grad():
        real_residual_x0 = real_score(
            batch["history_q0"],
            normalize_residual(
                batch["history_residual"],
                residual_mean,
                residual_std,
            ),
            batch["history_valid"],
            rollout["q0"],
            noisy_residual,
            score_levels,
            initial_state_normalized,
        )
    if stage == "self_forcing_dmd":
        fake_residual_x0 = fake_score(
            batch["history_q0"],
            normalize_residual(
                batch["history_residual"],
                residual_mean,
                residual_std,
            ),
            batch["history_valid"],
            rollout["q0"],
            noisy_residual,
            score_levels,
            initial_state_normalized,
        )
        real_latent_x0 = q0_latent(codec, rollout["q0"]) + denormalize_residual(
            real_residual_x0,
            residual_mean,
            residual_std,
        )
        fake_latent_x0 = q0_latent(codec, rollout["q0"]) + denormalize_residual(
            fake_residual_x0,
            residual_mean,
            residual_std,
        )
        main_loss, stats = dmd_surrogate_loss(
            generated_latent,
            real_latent_x0,
            fake_latent_x0,
            rollout["gradient_mask"],
        )
    elif stage == "self_forcing_gan":
        fake_logits = gan_critic(
            generated_latent,
            score_levels,
        )
        main_loss = gan_generator_loss(fake_logits)
        stats = {"gan/fake_logit": fake_logits.mean().detach()}
    else:
        raise ValueError(f"invalid Self Forcing stage {stage}")

    plan_q0 = rollout["plan_q0"].reshape(-1, PLAN_TOKENS)
    plan_residual = rollout["plan_residual"].reshape(
        -1,
        PLAN_TOKENS,
        residual_mean.shape[-1],
    )
    plan_state = rollout["plan_states"].reshape(
        -1,
        rollout["plan_states"].shape[-1],
    )
    target_motion = torch.stack(
        [
            batch["target_motion"][
                :,
                commit * COMMIT_FRAMES : commit * COMMIT_FRAMES + 16,
            ]
            for commit in range(ROLLOUT_COMMITS)
        ],
        dim=1,
    ).reshape(-1, 16, 34)
    physical_loss, physical_stats = decoded_physical_auxiliary(
        codec,
        kinematics,
        plan_q0,
        plan_residual,
        target_motion,
        plan_state,
        statistics,
        motion_mean,
        motion_std,
        residual_mean,
        residual_std,
    )
    scale, gradient_stats = physical_auxiliary_scale(
        main_loss,
        physical_loss,
        maximum_ratio=physical_gradient_ratio,
        main_reference=generated_latent,
        auxiliary_reference=plan_residual,
    )
    total = main_loss + scale * physical_loss
    stats.update(
        {
            "loss/main": main_loss.detach(),
            "loss/physical_unscaled": physical_loss.detach(),
            "loss/total": total.detach(),
            "self_forcing/gradient_tail_tokens": torch.tensor(
                SELF_FORCING_GRADIENT_TAIL,
                device=total.device,
            ),
            **physical_stats,
            **gradient_stats,
        }
    )
    return total, stats, rollout


def update_learning_rate(
    optimizer,
    update,
    total_updates,
    warmup_updates,
    maximum,
    minimum,
    constant_from_update=0,
):
    if int(constant_from_update) > 0 and int(update) >= int(
        constant_from_update
    ):
        value = minimum
    elif update < warmup_updates:
        value = maximum * float(update + 1) / max(warmup_updates, 1)
    else:
        progress = min(
            max(
                (update - warmup_updates)
                / max(total_updates - warmup_updates, 1),
                0.0,
            ),
            1.0,
        )
        value = minimum + 0.5 * (maximum - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def scheduled_learning_rate(optimizer, args, update, maximum):
    minimum = min(args.minimum_learning_rate, maximum * 0.1)
    constant_from_update = getattr(
        args,
        "constant_learning_rate_from_update",
        0,
    )
    if not isinstance(constant_from_update, (int, float)):
        constant_from_update = 0
    if (
        int(constant_from_update) > 0
        and int(update) >= int(constant_from_update)
    ):
        for group in optimizer.param_groups:
            group["lr"] = minimum
        return minimum
    if args.training_profile == MINIMUM_LATENT_WIDTH_FINAL_PROFILE:
        hold_from = {
            "q0_base": MINIMUM_LATENT_WIDTH_UPDATES,
            "teacher": args.total_updates,
            "commit_forcing": MINIMUM_LATENT_WIDTH_FINAL_CURRICULUM_UPDATES,
        }[args.stage]
        if update >= hold_from:
            for group in optimizer.param_groups:
                group["lr"] = minimum
            return minimum
        return update_learning_rate(
            optimizer,
            update,
            hold_from,
            args.warmup_updates,
            maximum,
            minimum,
        )
    if (
        args.training_profile
        in (
            MINIMUM_LATENT_WIDTH_EXTENSION_PROFILE,
            MINIMUM_LATENT_WIDTH_COMMIT_FORCING_PROFILE,
        )
        and update >= MINIMUM_LATENT_WIDTH_UPDATES
    ):
        for group in optimizer.param_groups:
            group["lr"] = minimum
        return minimum
    return update_learning_rate(
        optimizer,
        update,
        args.total_updates,
        args.warmup_updates,
        maximum,
        minimum,
    )


def two_forward_curriculum_total_updates(args):
    if args.training_profile in (
        MINIMUM_LATENT_WIDTH_EXTENSION_PROFILE,
        MINIMUM_LATENT_WIDTH_COMMIT_FORCING_PROFILE,
    ):
        return MINIMUM_LATENT_WIDTH_UPDATES
    return args.total_updates


def authenticate_resume_checkpoint(path, expected_sha256):
    actual_sha256 = sha256(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            "resume checkpoint SHA256 mismatch: "
            f"expected {expected_sha256}, actual {actual_sha256}"
        )
    return actual_sha256


def aggregate_stats(total, stats):
    for key, value in stats.items():
        scalar = float(value.detach().float().mean().cpu())
        total[key] = total.get(key, 0.0) + scalar


def averaged_stats(total, count):
    return {key: value / max(int(count), 1) for key, value in total.items()}


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def init_wandb(args, run_dir, rank):
    if rank != 0 or args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_pj_name,
        name=args.exp_name,
        dir=str(run_dir),
        mode=args.wandb_mode,
        config=vars(args),
    )


def checkpoint_payload(
    *,
    args,
    update,
    config,
    codec_sha,
    provenance,
    q0_source_sha,
    residual_mean,
    residual_std,
    primary,
    optimizer,
    scaler,
    sampler_epoch,
    batch_in_epoch,
    resume_checkpoint_sha256="",
    ema=None,
    fake_score=None,
    fake_optimizer=None,
    gan_critic=None,
    critic_optimizer=None,
):
    payload = {
        "experiment_id": args.experiment_id,
        "stage": args.stage,
        "update": int(update),
        "config": config.manifest(),
        "training_config": vars(args),
        "model": unwrap(primary).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "codec_checkpoint": args.codec_checkpoint,
        "codec_sha256": codec_sha,
        **provenance,
        "q0_checkpoint": args.q0_checkpoint,
        "q0_source_sha256": q0_source_sha,
        "residual_mean": residual_mean.detach().cpu(),
        "residual_std": residual_std.detach().cpu(),
        "rng_state": rng_state(),
        "sampler_epoch": int(sampler_epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if fake_score is not None:
        payload["fake_score"] = unwrap(fake_score).state_dict()
        payload["fake_optimizer"] = fake_optimizer.state_dict()
    if gan_critic is not None:
        payload["gan_critic"] = unwrap(gan_critic).state_dict()
        payload["critic_optimizer"] = critic_optimizer.state_dict()
    return payload


def minimum_width_provenance(args, codec, dataset):
    metadata = dataset.metadata
    source_split = metadata["source_split"]
    return {
        "training_profile": args.training_profile,
        "cache_schema_version": metadata["cache_schema_version"],
        "cache_manifest_digest": dataset.manifest_digest,
        "source_split_digest": manifest_sha256(source_split),
        "code_dim": int(metadata.get("code_dim", codec.config.code_dim)),
        "codebook_size": int(
            metadata.get("codebook_size", codec.config.codebook_size)
        ),
        "codec_training_seed": metadata.get("codec_training_seed"),
        "generator_training_seed": int(args.seed),
        "normalizer_sha256": metadata.get(
            "normalizer_sha256",
            metadata.get("normalization_digest"),
        ),
        "streaming_statistics_sha256": metadata.get(
            "streaming_statistics_sha256",
            metadata.get("statistics_digest"),
        ),
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_source_sha256": optional_sha256(args.teacher_checkpoint),
        "residual_initialization_checkpoint": (
            args.residual_initialization_checkpoint
        ),
        "residual_initialization_sha256": optional_sha256(
            args.residual_initialization_checkpoint
        ),
    }


def validate_minimum_width_parent(
    checkpoint,
    *,
    expected_stage,
    args,
    codec_sha,
    provenance,
    q0_source_sha="",
):
    if args.training_profile not in MINIMUM_LATENT_WIDTH_PROFILES:
        return
    final_parent = args.training_profile == MINIMUM_LATENT_WIDTH_FINAL_PROFILE
    expected = {
        "experiment_id": (
            MINIMUM_LATENT_WIDTH_FINAL_EXPERIMENT_ID
            if final_parent
            else MINIMUM_LATENT_WIDTH_EXPERIMENT_ID
        ),
        "stage": expected_stage,
        "update": (
            MINIMUM_LATENT_WIDTH_EXTENSION_UPDATES
            if final_parent
            else MINIMUM_LATENT_WIDTH_UPDATES
        ),
        "training_profile": (
            MINIMUM_LATENT_WIDTH_FINAL_PROFILE
            if final_parent
            else MINIMUM_LATENT_WIDTH_PROFILE
        ),
        "codec_sha256": codec_sha,
        "cache_schema_version": args.cache_schema_version,
        "cache_manifest_digest": provenance["cache_manifest_digest"],
        "source_split_digest": provenance["source_split_digest"],
        "code_dim": provenance["code_dim"],
        "generator_training_seed": (
            1234 if final_parent else provenance["generator_training_seed"]
        ),
        "normalizer_sha256": provenance["normalizer_sha256"],
        "streaming_statistics_sha256": provenance[
            "streaming_statistics_sha256"
        ],
    }
    if final_parent and expected_stage == "teacher":
        expected["q0_source_sha256"] = q0_source_sha
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "minimum-latent-width parent provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_structural_quantizer_parent(
    checkpoint,
    *,
    expected_stage,
    args,
    codec_sha,
    provenance,
    q0_source_sha="",
):
    if args.training_profile != STRUCTURAL_QUANTIZER_PROFILE:
        return
    expected = {
        "experiment_id": STRUCTURAL_QUANTIZER_EXPERIMENT_ID,
        "stage": expected_stage,
        "update": STRUCTURAL_QUANTIZER_PARENT_UPDATES,
        "training_profile": STRUCTURAL_QUANTIZER_PROFILE,
        "codec_sha256": codec_sha,
        "cache_schema_version": STRUCTURAL_QUANTIZER_CACHE_VERSION,
        "cache_manifest_digest": provenance["cache_manifest_digest"],
        "source_split_digest": provenance["source_split_digest"],
        "code_dim": 16,
        "generator_training_seed": int(args.seed),
        "normalizer_sha256": provenance["normalizer_sha256"],
        "streaming_statistics_sha256": provenance[
            "streaming_statistics_sha256"
        ],
    }
    if expected_stage == "teacher":
        expected["q0_source_sha256"] = q0_source_sha
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "SQA parent provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_structural_quantizer_resume(
    checkpoint,
    *,
    args,
    codec_sha,
    provenance,
    q0_source_sha,
):
    if args.training_profile != STRUCTURAL_QUANTIZER_PROFILE:
        return
    expected = {
        "experiment_id": STRUCTURAL_QUANTIZER_EXPERIMENT_ID,
        "stage": args.stage,
        "training_profile": STRUCTURAL_QUANTIZER_PROFILE,
        "codec_sha256": codec_sha,
        "cache_schema_version": STRUCTURAL_QUANTIZER_CACHE_VERSION,
        "cache_manifest_digest": provenance["cache_manifest_digest"],
        "source_split_digest": provenance["source_split_digest"],
        "code_dim": 16,
        "generator_training_seed": int(args.seed),
        "normalizer_sha256": provenance["normalizer_sha256"],
        "streaming_statistics_sha256": provenance[
            "streaming_statistics_sha256"
        ],
        "q0_source_sha256": q0_source_sha,
    }
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    saved = checkpoint.get("training_config", {})
    for field in (
        "total_updates",
        "save_updates",
        "seed",
        "commit_forcing_generated_ceiling",
        "commit_forcing_curriculum_updates",
        "gradient_accumulation_steps",
    ):
        if saved.get(field) != getattr(args, field):
            mismatches[f"training_config.{field}"] = {
                "expected": getattr(args, field),
                "actual": saved.get(field),
            }
    if checkpoint.get("rng_state") is None:
        mismatches["rng_state"] = {"expected": "present", "actual": None}
    if mismatches:
        raise ValueError(
            "SQA exact-resume provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_minimum_width_extension_resume(
    checkpoint,
    *,
    args,
    codec_sha,
    provenance,
    q0_source_sha,
):
    if args.training_profile != MINIMUM_LATENT_WIDTH_EXTENSION_PROFILE:
        return
    expected = {
        "experiment_id": MINIMUM_LATENT_WIDTH_EXPERIMENT_ID,
        "stage": "two_forward",
        "update": MINIMUM_LATENT_WIDTH_UPDATES,
        "training_profile": MINIMUM_LATENT_WIDTH_PROFILE,
        "codec_sha256": codec_sha,
        "cache_schema_version": args.cache_schema_version,
        "cache_manifest_digest": provenance["cache_manifest_digest"],
        "source_split_digest": provenance["source_split_digest"],
        "code_dim": provenance["code_dim"],
        "generator_training_seed": provenance["generator_training_seed"],
        "normalizer_sha256": provenance["normalizer_sha256"],
        "streaming_statistics_sha256": provenance[
            "streaming_statistics_sha256"
        ],
        "teacher_source_sha256": provenance["teacher_source_sha256"],
        "residual_initialization_sha256": provenance[
            "residual_initialization_sha256"
        ],
        "q0_source_sha256": q0_source_sha,
    }
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    saved = checkpoint.get("training_config", {})
    saved_expected = {
        "experiment_id": MINIMUM_LATENT_WIDTH_EXPERIMENT_ID,
        "training_profile": MINIMUM_LATENT_WIDTH_PROFILE,
        "total_updates": MINIMUM_LATENT_WIDTH_UPDATES,
        "save_updates": MINIMUM_LATENT_WIDTH_SAVE_UPDATES,
        "learning_rate": 2e-4,
        "minimum_learning_rate": 2e-5,
        "warmup_updates": 10_000,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "q0_generation_batch_factor": 1,
        "seed": 1234,
    }
    saved_mismatches = {
        key: {"expected": value, "actual": saved.get(key)}
        for key, value in saved_expected.items()
        if saved.get(key) != value
    }
    if mismatches or saved_mismatches:
        raise ValueError(
            "minimum-latent-width extension resume contract mismatch: "
            + json.dumps(
                {"checkpoint": mismatches, "training_config": saved_mismatches},
                sort_keys=True,
            )
        )
    for key in (
        "model",
        "q0_model",
        "optimizer",
        "rng_state",
        "sampler_epoch",
        "batch_in_epoch",
    ):
        if key not in checkpoint:
            raise ValueError(f"extension resume checkpoint is missing {key}")


def validate_minimum_width_final_q0_resume(
    checkpoint,
    *,
    args,
    codec_sha,
    provenance,
):
    if not (
        args.training_profile == MINIMUM_LATENT_WIDTH_FINAL_PROFILE
        and args.stage == "q0_base"
    ):
        return
    expected = {
        "experiment_id": MINIMUM_LATENT_WIDTH_EXPERIMENT_ID,
        "stage": "q0_base",
        "update": MINIMUM_LATENT_WIDTH_UPDATES,
        "training_profile": MINIMUM_LATENT_WIDTH_PROFILE,
        "codec_sha256": codec_sha,
        "cache_schema_version": args.cache_schema_version,
        "cache_manifest_digest": provenance["cache_manifest_digest"],
        "source_split_digest": provenance["source_split_digest"],
        "code_dim": provenance["code_dim"],
        "generator_training_seed": 1234,
        "normalizer_sha256": provenance["normalizer_sha256"],
        "streaming_statistics_sha256": provenance[
            "streaming_statistics_sha256"
        ],
    }
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    saved = checkpoint.get("training_config", {})
    saved_expected = {
        "experiment_id": MINIMUM_LATENT_WIDTH_EXPERIMENT_ID,
        "training_profile": MINIMUM_LATENT_WIDTH_PROFILE,
        "stage": "q0_base",
        "total_updates": MINIMUM_LATENT_WIDTH_UPDATES,
        "save_updates": MINIMUM_LATENT_WIDTH_SAVE_UPDATES,
        "learning_rate": 2e-4,
        "minimum_learning_rate": 2e-5,
        "warmup_updates": 10_000,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "q0_generation_batch_factor": 1,
        "seed": 1234,
    }
    saved_mismatches = {
        key: {"expected": value, "actual": saved.get(key)}
        for key, value in saved_expected.items()
        if saved.get(key) != value
    }
    if mismatches or saved_mismatches:
        raise ValueError(
            "final q0 resume contract mismatch: "
            + json.dumps(
                {"checkpoint": mismatches, "training_config": saved_mismatches},
                sort_keys=True,
            )
        )
    for key in (
        "model",
        "optimizer",
        "rng_state",
        "sampler_epoch",
        "batch_in_epoch",
    ):
        if key not in checkpoint:
            raise ValueError(f"final q0 resume checkpoint is missing {key}")


def validate_minimum_width_final_route_resume(
    checkpoint,
    *,
    args,
    codec_sha,
    provenance,
    q0_source_sha,
):
    if not (
        args.training_profile == MINIMUM_LATENT_WIDTH_FINAL_PROFILE
        and args.stage in ("teacher", "commit_forcing")
    ):
        return
    milestones = (
        {50_000, 100_000}
        if args.stage == "teacher"
        else parse_update_set(MINIMUM_LATENT_WIDTH_FINAL_CF_SAVE_UPDATES)
    )
    if int(checkpoint.get("update", -1)) not in milestones:
        raise ValueError("final route resume update is not a frozen milestone")
    expected = {
        "experiment_id": MINIMUM_LATENT_WIDTH_FINAL_EXPERIMENT_ID,
        "stage": args.stage,
        "training_profile": MINIMUM_LATENT_WIDTH_FINAL_PROFILE,
        "codec_sha256": codec_sha,
        "cache_schema_version": args.cache_schema_version,
        "cache_manifest_digest": provenance["cache_manifest_digest"],
        "source_split_digest": provenance["source_split_digest"],
        "code_dim": provenance["code_dim"],
        "generator_training_seed": args.seed,
        "normalizer_sha256": provenance["normalizer_sha256"],
        "streaming_statistics_sha256": provenance[
            "streaming_statistics_sha256"
        ],
        "q0_source_sha256": q0_source_sha,
    }
    if args.stage == "commit_forcing":
        expected.update(
            {
                "teacher_source_sha256": provenance["teacher_source_sha256"],
                "residual_initialization_sha256": provenance[
                    "residual_initialization_sha256"
                ],
            }
        )
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    saved = checkpoint.get("training_config", {})
    saved_expected = {
        "experiment_id": MINIMUM_LATENT_WIDTH_FINAL_EXPERIMENT_ID,
        "training_profile": MINIMUM_LATENT_WIDTH_FINAL_PROFILE,
        "stage": args.stage,
        "total_updates": args.total_updates,
        "save_updates": args.save_updates,
        "learning_rate": 2e-4,
        "minimum_learning_rate": 2e-5,
        "warmup_updates": 10_000,
        "batch_size": 1 if args.stage == "commit_forcing" else 4,
        "gradient_accumulation_steps": 1,
        "q0_generation_batch_factor": 1,
        "seed": args.seed,
    }
    if args.stage == "commit_forcing":
        saved_expected.update(
            {
                "commit_forcing_curriculum_updates": (
                    MINIMUM_LATENT_WIDTH_FINAL_CURRICULUM_UPDATES
                ),
                "commit_forcing_generated_ceiling": (
                    args.commit_forcing_generated_ceiling
                ),
            }
        )
    saved_mismatches = {
        key: {"expected": value, "actual": saved.get(key)}
        for key, value in saved_expected.items()
        if checkpoint.get("training_config", {}).get(key) != value
    }
    if mismatches or saved_mismatches:
        raise ValueError(
            "final route resume contract mismatch: "
            + json.dumps(
                {"checkpoint": mismatches, "training_config": saved_mismatches},
                sort_keys=True,
            )
        )
    required = [
        "model",
        "optimizer",
        "rng_state",
        "sampler_epoch",
        "batch_in_epoch",
    ]
    if args.stage == "commit_forcing":
        required.extend(("q0_model", "scaler"))
    for key in required:
        if key not in checkpoint:
            raise ValueError(f"final route resume checkpoint is missing {key}")


def validate_minimum_width_commit_forcing_resume(
    checkpoint,
    *,
    args,
    codec_sha,
    provenance,
    q0_source_sha,
):
    if args.training_profile != MINIMUM_LATENT_WIDTH_COMMIT_FORCING_PROFILE:
        return
    update = int(checkpoint.get("update", -1))
    if update not in {10_000, 25_000, 50_000, 75_000, 100_000}:
        raise ValueError(
            "minimum-latent-width Commit Forcing resume update is not a "
            "frozen checkpoint milestone"
        )
    expected = {
        "experiment_id": MINIMUM_LATENT_WIDTH_COMMIT_FORCING_EXPERIMENT_ID,
        "stage": "commit_forcing",
        "training_profile": MINIMUM_LATENT_WIDTH_COMMIT_FORCING_PROFILE,
        "codec_sha256": codec_sha,
        "cache_schema_version": args.cache_schema_version,
        "cache_manifest_digest": provenance["cache_manifest_digest"],
        "source_split_digest": provenance["source_split_digest"],
        "code_dim": provenance["code_dim"],
        "generator_training_seed": provenance["generator_training_seed"],
        "normalizer_sha256": provenance["normalizer_sha256"],
        "streaming_statistics_sha256": provenance[
            "streaming_statistics_sha256"
        ],
        "teacher_source_sha256": provenance["teacher_source_sha256"],
        "residual_initialization_sha256": provenance[
            "residual_initialization_sha256"
        ],
        "q0_source_sha256": q0_source_sha,
    }
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    saved = checkpoint.get("training_config", {})
    saved_expected = {
        "experiment_id": MINIMUM_LATENT_WIDTH_COMMIT_FORCING_EXPERIMENT_ID,
        "training_profile": MINIMUM_LATENT_WIDTH_COMMIT_FORCING_PROFILE,
        "total_updates": MINIMUM_LATENT_WIDTH_COMMIT_FORCING_UPDATES,
        "save_updates": MINIMUM_LATENT_WIDTH_COMMIT_FORCING_SAVE_UPDATES,
        "learning_rate": 2e-4,
        "minimum_learning_rate": 2e-5,
        "warmup_updates": 10_000,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "q0_generation_batch_factor": 1,
        "seed": 1234,
    }
    saved_mismatches = {
        key: {"expected": value, "actual": saved.get(key)}
        for key, value in saved_expected.items()
        if saved.get(key) != value
    }
    if mismatches or saved_mismatches:
        raise ValueError(
            "minimum-latent-width Commit Forcing resume contract mismatch: "
            + json.dumps(
                {"checkpoint": mismatches, "training_config": saved_mismatches},
                sort_keys=True,
            )
        )
    for key in (
        "model",
        "q0_model",
        "optimizer",
        "scaler",
        "rng_state",
        "sampler_epoch",
        "batch_in_epoch",
    ):
        if key not in checkpoint:
            raise ValueError(
                f"Commit Forcing resume checkpoint is missing {key}"
            )


def main():
    args = prepare_args(parse_args())
    rank, world_size, _, device = initialize_distributed()
    if args.checkpoint and world_size > 1:
        raise ValueError(
            "multi-GPU resume is unsupported because checkpoints contain "
            "single-rank RNG state"
        )
    set_seed(args.seed, rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    codec_sha = sha256(args.codec_checkpoint)
    codec_checkpoint = torch.load(
        args.codec_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    codec = build_g1_hybrid_streaming_codec_from_checkpoint(
        codec_checkpoint
    ).to(device).eval()
    codec.requires_grad_(False)
    if args.training_profile in MINIMUM_LATENT_WIDTH_PROFILES:
        if int(codec.config.code_dim) not in (256, 64, 32, 16, 8):
            raise ValueError(
                "minimum-latent-width codec width must be 256/64/32/16/8"
            )
        if (
            int(codec.config.codebook_size) != 512
            or int(codec.config.num_codebooks) != 1
        ):
            raise ValueError(
                "minimum-latent-width profile requires a single 512-way q0"
            )
    if args.training_profile == STRUCTURAL_QUANTIZER_PROFILE:
        if (
            int(codec.config.code_dim) != 16
            or int(codec.config.codebook_size) != 512
            or int(codec.config.num_codebooks) != 1
        ):
            raise ValueError("SQA requires one packed 512-way d16 structural token")
        codec_training = codec_checkpoint.get("training_config", {})
        if (
            codec_training.get("experiment_id")
            != STRUCTURAL_QUANTIZER_EXPERIMENT_ID
            or int(codec_training.get("seed", -1)) != int(args.seed)
            or codec_training.get("sqa_stripped_bsq_diagnostic")
        ):
            raise ValueError("SQA codec provenance or eligibility mismatch")
    statistics = G1StreamingStateStatistics.from_state_dict(
        codec_checkpoint["streaming_statistics"]
    )
    config = model_config(args, codec)
    cache_device = device.type if args.cache_device == "auto" else args.cache_device
    dataset_kwargs = dict(
        codec_checkpoint=args.codec_checkpoint,
        data_path=args.data_path,
        generator_cache_dir=args.generator_cache_dir,
        experiment_id=(
            args.cache_provenance_experiment_id or args.experiment_id
        ),
        cache_schema_version=args.cache_schema_version,
        cache_batch_size=args.cache_batch_size,
        cache_device=cache_device,
        cache_data_len=args.cache_data_len,
    )
    train_dataset = G1PaperFaithfulDCDataset(
        split="train",
        rebuild_cache=args.rebuild_generator_cache,
        data_len=args.data_len,
        **dataset_kwargs,
    )
    if args.prepare_cache_only:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "codec_sha256": codec_sha,
                        "cache_manifest_digest": train_dataset.manifest_digest,
                        "train_examples": len(train_dataset),
                    },
                    indent=2,
                )
            )
        return
    test_dataset = G1PaperFaithfulDCDataset(
        split="test",
        rebuild_cache=False,
        data_len=args.eval_data_len,
        **dataset_kwargs,
    )
    if train_dataset.manifest_digest != test_dataset.manifest_digest:
        raise ValueError("train/test paper D+C cache manifests differ")
    provenance = minimum_width_provenance(args, codec, train_dataset)
    residual_mean_np, residual_std_np, _ = train_dataset.residual_statistics
    residual_mean = torch.as_tensor(
        residual_mean_np,
        device=device,
    ).view(1, 1, -1)
    residual_std = torch.as_tensor(
        residual_std_np,
        device=device,
    ).view(1, 1, -1)
    motion_mean = torch.as_tensor(
        codec_checkpoint["normalizer"]["mean"],
        device=device,
    ).view(1, 1, -1)
    motion_std = torch.as_tensor(
        codec_checkpoint["normalizer"]["std"],
        device=device,
    ).view(1, 1, -1)
    kinematics = G1TorchKinematics(
        args.g1_fk_model_path,
        root_quat_order=args.g1_root_quat_order,
    ).to(device)
    diffusion = PredX0CosineDiffusion(
        timesteps=DIFFUSION_TIMESTEPS,
        eta=0.0,
    ).to(device)

    q0_source_sha = ""
    frozen_q0 = None
    q0_source_checkpoint = None
    residual_source_checkpoint = None
    initialization_manifest = None
    parent_provenance = {
        "expected_codec_sha": codec_sha,
        "expected_cache_manifest": train_dataset.manifest_digest,
    }
    if args.stage == "q0_base":
        primary = G1PaperQ0Generator(config)
        if args.training_profile in (
            MINIMUM_LATENT_WIDTH_PROFILE,
            MINIMUM_LATENT_WIDTH_FINAL_PROFILE,
            STRUCTURAL_QUANTIZER_PROFILE,
        ):
            initialization_manifest = initialize_paper_name_stable(
                primary,
                args.seed,
            )
        primary = primary.to(device)
    elif args.q0_checkpoint:
        q0_source_sha = sha256(args.q0_checkpoint)
        frozen_q0, q0_source_checkpoint = load_q0(
            args.q0_checkpoint,
            config,
            device,
            **parent_provenance,
        )
        primary = None

    teacher = None
    fake_score = None
    gan_critic = None
    ema = None
    if args.stage == "teacher":
        if args.residual_initialization_checkpoint:
            primary, _ = load_residual(
                args.residual_initialization_checkpoint,
                config,
                device,
                expected_stages=("teacher",),
                expected_q0_source_sha=q0_source_sha,
                **parent_provenance,
            )
        else:
            primary = G1ResidualDiffusionGenerator(config)
            if args.training_profile in (
                MINIMUM_LATENT_WIDTH_PROFILE,
                MINIMUM_LATENT_WIDTH_FINAL_PROFILE,
                STRUCTURAL_QUANTIZER_PROFILE,
            ):
                initialization_manifest = initialize_paper_name_stable(
                    primary,
                    args.seed,
                )
            primary = primary.to(device)
    elif args.stage == "ode_distill":
        teacher, _ = load_residual(
            args.teacher_checkpoint,
            config,
            device,
            expected_stages=("teacher",),
            expected_q0_source_sha=q0_source_sha,
            **parent_provenance,
        )
        teacher.eval().requires_grad_(False)
        primary = G1ResidualDiffusionGenerator(config).to(device)
        primary.load_state_dict(teacher.state_dict(), strict=True)
    elif args.stage in ("diffusion_forcing", "df_homogeneous_control"):
        primary = G1ResidualDiffusionGenerator(config).to(device)
    elif args.stage in JOINT_Q0_RESIDUAL_STAGES:
        primary, residual_source_checkpoint = load_residual(
            args.residual_initialization_checkpoint,
            config,
            device,
            expected_stages=("teacher", "ode_distill"),
            expected_q0_source_sha=q0_source_sha,
            **parent_provenance,
        )
        frozen_q0.requires_grad_(True).train()
        if (
            args.training_profile in MINIMUM_LATENT_WIDTH_PROFILES
            or args.training_profile == STRUCTURAL_QUANTIZER_PROFILE
        ):
            initialization_manifest = {
                "version": "teacher_checkpoint_parent_v1",
                "parent_sha256": optional_sha256(
                    args.residual_initialization_checkpoint
                ),
            }
    elif args.stage.startswith("self_forcing_"):
        primary, _ = load_residual(
            args.student_checkpoint,
            config,
            device,
            expected_stages=("ode_distill",),
            expected_q0_source_sha=q0_source_sha,
            **parent_provenance,
        )
        teacher, _ = load_residual(
            args.teacher_checkpoint,
            config,
            device,
            expected_stages=("teacher",),
            expected_q0_source_sha=q0_source_sha,
            **parent_provenance,
        )
        teacher.eval().requires_grad_(False)
        if args.stage == "self_forcing_dmd":
            fake_score, _ = load_residual(
                args.teacher_checkpoint,
                config,
                device,
                expected_stages=("teacher",),
                expected_q0_source_sha=q0_source_sha,
                **parent_provenance,
            )
        else:
            gan_critic = G1LatentSequenceCritic(config).to(device)
        ema = ExponentialMovingAverage(primary, decay=args.ema_decay)

    if q0_source_checkpoint is not None:
        validate_minimum_width_parent(
            q0_source_checkpoint,
            expected_stage="q0_base",
            args=args,
            codec_sha=codec_sha,
            provenance=provenance,
            q0_source_sha=q0_source_sha,
        )
        validate_structural_quantizer_parent(
            q0_source_checkpoint,
            expected_stage="q0_base",
            args=args,
            codec_sha=codec_sha,
            provenance=provenance,
            q0_source_sha=q0_source_sha,
        )
    if residual_source_checkpoint is not None:
        validate_minimum_width_parent(
            residual_source_checkpoint,
            expected_stage="teacher",
            args=args,
            codec_sha=codec_sha,
            provenance=provenance,
            q0_source_sha=q0_source_sha,
        )
        validate_structural_quantizer_parent(
            residual_source_checkpoint,
            expected_stage="teacher",
            args=args,
            codec_sha=codec_sha,
            provenance=provenance,
            q0_source_sha=q0_source_sha,
        )
    if (
        args.training_profile in MINIMUM_LATENT_WIDTH_PROFILES
        or args.training_profile == STRUCTURAL_QUANTIZER_PROFILE
    ):
        if initialization_manifest is None:
            raise RuntimeError(
                "minimum-latent-width stage is missing initialization provenance"
            )
        provenance["initialization_manifest"] = initialization_manifest
        # Codec/model construction consumes width-dependent RNG amounts. Reset
        # after all initialization/loading so data order, diffusion noise,
        # replacement masks, and physical-loss sampling stay paired.
        set_seed(args.seed, rank)

    if args.torch_compile:
        primary = torch.compile(primary)
    if args.dry_run:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "config": config.manifest(),
                        "training_config": vars(args),
                        "codec_sha256": codec_sha,
                        "q0_source_sha256": q0_source_sha,
                        **provenance,
                        "train_examples": len(train_dataset),
                        "test_examples": len(test_dataset),
                        "primary_parameters": sum(
                            parameter.numel() for parameter in primary.parameters()
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return

    primary = maybe_ddp(primary, device, world_size)
    if args.stage in JOINT_Q0_RESIDUAL_STAGES:
        frozen_q0 = maybe_ddp(frozen_q0, device, world_size)
        trainable_parameters = list(primary.parameters()) + list(
            frozen_q0.parameters()
        )
    else:
        trainable_parameters = list(primary.parameters())
    if fake_score is not None:
        fake_score = maybe_ddp(fake_score, device, world_size)
    if gan_critic is not None:
        gan_critic = maybe_ddp(gan_critic, device, world_size)

    sf_stage = args.stage.startswith("self_forcing_")
    if sf_stage:
        generator_lr = 2e-6 * args.lr_scale
        optimizer_betas = (0.0, 0.999)
    else:
        generator_lr = args.learning_rate
        optimizer_betas = (0.9, 0.999)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=generator_lr,
        betas=optimizer_betas,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    fake_optimizer = None
    critic_optimizer = None
    if fake_score is not None:
        fake_optimizer = torch.optim.AdamW(
            fake_score.parameters(),
            lr=args.critic_learning_rate * args.lr_scale,
            betas=(0.0, 0.999),
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
    if gan_critic is not None:
        critic_optimizer = torch.optim.AdamW(
            gan_critic.parameters(),
            lr=generator_lr,
            betas=(0.0, 0.999),
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
    amp_enabled = device.type == "cuda" and args.mixed_precision != "no"
    amp_dtype = (
        torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    )
    scaler = GradScaler(
        "cuda",
        enabled=amp_enabled and args.mixed_precision == "fp16",
    )
    start_update = 0
    sampler_epoch = 0
    batch_in_epoch = 0
    resume_checkpoint_sha256 = ""
    if args.checkpoint:
        resume_checkpoint_sha256 = authenticate_resume_checkpoint(
            args.checkpoint,
            args.expected_resume_sha256,
        )
        resume = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if resume.get("stage") != args.stage:
            raise ValueError("resume stage mismatch")
        if resume.get("codec_sha256") != codec_sha:
            raise ValueError("resume codec provenance mismatch")
        if resume.get("experiment_id") != args.experiment_id:
            raise ValueError("resume experiment provenance mismatch")
        _verify_parent_checkpoint(
            resume,
            config,
            expected_codec_sha=codec_sha,
            expected_cache_manifest=train_dataset.manifest_digest,
            expected_q0_source_sha=(
                q0_source_sha if args.stage != "q0_base" else None
            ),
        )
        if args.commit_forcing_full_generated_from_update:
            required_resume_keys = {
                "model",
                "q0_model",
                "optimizer",
                "scaler",
                "rng_state",
                "sampler_epoch",
                "batch_in_epoch",
            }
            missing_resume_keys = sorted(required_resume_keys - set(resume))
            if missing_resume_keys:
                raise ValueError(
                    "Commit Forcing exact-resume parent is missing state: "
                    + ", ".join(missing_resume_keys)
                )
            parent_update = int(resume["update"])
            expected_parent_update = (
                COMMIT_FORCING_CONTINUATION_PARENT_UPDATES[
                    args.total_updates
                ]
            )
            if parent_update != expected_parent_update:
                raise ValueError(
                    "Commit Forcing continuation checkpoint update does not "
                    "match the frozen exact-resume parent"
                )
            parent_run = str(
                resume.get("training_config", {}).get("exp_name", "")
            )
            expected_parent_run = COMMIT_FORCING_CONTINUATION_PARENT_RUNS[
                args.total_updates
            ]
            if parent_run != expected_parent_run:
                raise ValueError(
                    "Commit Forcing continuation checkpoint run does not "
                    "match the frozen exact-resume parent"
                )
        if args.training_profile in MINIMUM_LATENT_WIDTH_PROFILES:
            validate_minimum_width_extension_resume(
                resume,
                args=args,
                codec_sha=codec_sha,
                provenance=provenance,
                q0_source_sha=q0_source_sha,
            )
            validate_minimum_width_commit_forcing_resume(
                resume,
                args=args,
                codec_sha=codec_sha,
                provenance=provenance,
                q0_source_sha=q0_source_sha,
            )
            validate_minimum_width_final_q0_resume(
                resume,
                args=args,
                codec_sha=codec_sha,
                provenance=provenance,
            )
            validate_minimum_width_final_route_resume(
                resume,
                args=args,
                codec_sha=codec_sha,
                provenance=provenance,
                q0_source_sha=q0_source_sha,
            )
        validate_structural_quantizer_resume(
            resume,
            args=args,
            codec_sha=codec_sha,
            provenance=provenance,
            q0_source_sha=q0_source_sha,
        )
        if args.training_profile == MINIMUM_LATENT_WIDTH_PROFILE:
            resume_mismatches = {
                key: {"expected": value, "actual": resume.get(key)}
                for key, value in provenance.items()
                if resume.get(key) != value
            }
            if resume_mismatches:
                raise ValueError(
                    "resume frozen provenance mismatch: "
                    + json.dumps(resume_mismatches, sort_keys=True)
                )
            if resume.get("q0_source_sha256", "") != q0_source_sha:
                raise ValueError("resume q0 parent provenance mismatch")
        saved_factor = int(
            resume.get("training_config", {}).get(
                "q0_generation_batch_factor",
                1,
            )
        )
        if (
            args.resume_optimizer
            and saved_factor != args.q0_generation_batch_factor
        ):
            raise ValueError(
                "optimizer resume requires the checkpoint q0 generation "
                "batch factor"
            )
        if args.resume_optimizer and "batch_in_epoch" not in resume:
            raise ValueError(
                "optimizer resume requires an exact batch_in_epoch cursor"
            )
        unwrap(primary).load_state_dict(resume["model"], strict=True)
        if args.stage in JOINT_Q0_RESIDUAL_STAGES:
            unwrap(frozen_q0).load_state_dict(resume["q0_model"], strict=True)
        if args.resume_optimizer:
            optimizer.load_state_dict(resume["optimizer"])
            if resume.get("scaler") is not None:
                scaler.load_state_dict(resume["scaler"])
            if fake_optimizer is not None:
                unwrap(fake_score).load_state_dict(resume["fake_score"], strict=True)
                fake_optimizer.load_state_dict(resume["fake_optimizer"])
            if critic_optimizer is not None:
                unwrap(gan_critic).load_state_dict(resume["gan_critic"], strict=True)
                critic_optimizer.load_state_dict(resume["critic_optimizer"])
        if ema is not None and "ema" in resume:
            ema.load_state_dict(resume["ema"])
        start_update = int(resume["update"])
        sampler_epoch = int(resume.get("sampler_epoch", 0))
        batch_in_epoch = int(resume.get("batch_in_epoch", 0))
        restore_rng_state(resume["rng_state"])

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    loader_generator = torch.Generator()
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=loader_generator,
    )
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = DataLoader(train_dataset, **loader_kwargs)
    generation_factor = (
        args.q0_generation_batch_factor
        if args.stage in STATIC_GENERATED_Q0_STAGES
        else 1
    )
    configured_target_updates = (
        min(args.total_updates, args.max_updates)
        if args.max_updates > 0
        else args.total_updates
    )
    checkpoint_updates = {
        *parse_update_set(args.save_updates),
        configured_target_updates,
    }
    validate_generation_batching(
        batches_per_epoch=len(loader),
        generation_factor=generation_factor,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_updates=checkpoint_updates,
        batch_in_epoch=batch_in_epoch,
    )
    run_dir = Path(args.project) / args.exp_name
    weights_dir = run_dir / "weights"
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": config.manifest(),
                    "training": vars(args),
                    "codec_sha256": codec_sha,
                    "q0_source_sha256": q0_source_sha,
                    **provenance,
                    "world_size": world_size,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    wandb_run = init_wandb(args, run_dir, rank)
    save_updates = parse_update_set(args.save_updates)
    target_updates = (
        min(args.total_updates, args.max_updates)
        if args.max_updates > 0
        else args.total_updates
    )
    update = start_update
    accumulated = 0
    log_total = {}
    log_count = 0
    started_at = time.time()
    last_log_at = started_at
    input_pipeline_wait_seconds = 0.0
    host_active_seconds = 0.0
    ready_for_batch_at = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    learning_rate = scheduled_learning_rate(
        optimizer,
        args,
        update,
        generator_lr,
    )
    if rank == 0:
        print(
            f"Paper D+C stage={args.stage} device={device} world={world_size} "
            f"train={len(train_dataset)} updates={start_update}->{target_updates} "
            f"params={sum(parameter.numel() for parameter in trainable_parameters)} "
            f"q0_generation_batch_factor={args.q0_generation_batch_factor}",
            flush=True,
    )

    while update < target_updates:
        sampler.set_epoch(sampler_epoch)
        loader_generator.manual_seed(args.seed + sampler_epoch)
        epoch_loader = islice(iter(loader), batch_in_epoch, None)
        for (
            batch,
            plan,
            rollout_plan,
            prefetched_generated_q0,
            is_generation_group_end,
        ) in prepared_training_batches(
            loader=epoch_loader,
            stage=args.stage,
            q0_generation_batch_factor=args.q0_generation_batch_factor,
            q0_model=frozen_q0,
            device=device,
            statistics=statistics,
            residual_mean=residual_mean,
            residual_std=residual_std,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        ):
            batch_received_at = time.perf_counter()
            input_pipeline_wait_seconds += (
                batch_received_at - ready_for_batch_at
            )
            host_active_started_at = time.perf_counter()
            batch_in_epoch += 1
            is_last_microbatch = (
                accumulated + 1 >= args.gradient_accumulation_steps
            )
            with ExitStack() as sync_context:
                if not is_last_microbatch:
                    if isinstance(primary, DistributedDataParallel):
                        sync_context.enter_context(primary.no_sync())
                    if (
                        args.stage in JOINT_Q0_RESIDUAL_STAGES
                        and isinstance(frozen_q0, DistributedDataParallel)
                    ):
                        sync_context.enter_context(frozen_q0.no_sync())
                with autocast(
                    device.type,
                    enabled=amp_enabled,
                    dtype=amp_dtype,
                ):
                    if args.stage == "q0_base":
                        loss, stats = q0_stage_loss(primary, rollout_plan)
                    elif args.stage == "teacher":
                        loss, stats = residual_training_loss(
                            model=primary,
                            q0_model=frozen_q0,
                            codec=codec,
                            diffusion=diffusion,
                            plan=rollout_plan,
                            residual_mean=residual_mean,
                            residual_std=residual_std,
                            statistics=statistics,
                            motion_mean=motion_mean,
                            motion_std=motion_std,
                            kinematics=kinematics,
                            noise_mode="homogeneous",
                            physical_gradient_ratio=args.physical_gradient_ratio,
                            prefetched_generated_q0=prefetched_generated_q0,
                        )
                    elif args.stage == "ode_distill":
                        loss, stats = ode_distillation_loss(
                            student=primary,
                            teacher=teacher,
                            q0_model=frozen_q0,
                            codec=codec,
                            diffusion=diffusion,
                            plan=rollout_plan,
                            residual_mean=residual_mean,
                            residual_std=residual_std,
                            statistics=statistics,
                            motion_mean=motion_mean,
                            motion_std=motion_std,
                            kinematics=kinematics,
                            student_steps=args.student_steps,
                            physical_gradient_ratio=args.physical_gradient_ratio,
                            prefetched_generated_q0=prefetched_generated_q0,
                        )
                    elif args.stage in (
                        "diffusion_forcing",
                        "df_homogeneous_control",
                    ):
                        loss, stats = residual_training_loss(
                            model=primary,
                            q0_model=frozen_q0,
                            codec=codec,
                            diffusion=diffusion,
                            plan=rollout_plan,
                            residual_mean=residual_mean,
                            residual_std=residual_std,
                            statistics=statistics,
                            motion_mean=motion_mean,
                            motion_std=motion_std,
                            kinematics=kinematics,
                            noise_mode=(
                                "independent"
                                if args.stage == "diffusion_forcing"
                                else "homogeneous"
                            ),
                            physical_gradient_ratio=args.physical_gradient_ratio,
                            prefetched_generated_q0=prefetched_generated_q0,
                        )
                    elif args.stage == "commit_forcing":
                        commit_curriculum_updates = (
                            args.commit_forcing_curriculum_updates
                            or two_forward_curriculum_total_updates(args)
                        )
                        loss, stats = commit_forcing_loss(
                            q0_model=frozen_q0,
                            residual_model=primary,
                            codec=codec,
                            diffusion=diffusion,
                            plan=plan,
                            residual_mean=residual_mean,
                            residual_std=residual_std,
                            statistics=statistics,
                            motion_mean=motion_mean,
                            motion_std=motion_std,
                            global_update=update,
                            total_updates=commit_curriculum_updates,
                            first_pass_nfe=10,
                            full_generated_from_update=(
                                args.commit_forcing_full_generated_from_update
                            ),
                            generated_context_ceiling=(
                                args.commit_forcing_generated_ceiling
                            ),
                            bernoulli_replacement=(
                                args.training_profile
                                == MINIMUM_LATENT_WIDTH_FINAL_PROFILE
                            ),
                        )
                    elif args.stage in (
                        "two_forward",
                        "one_forward_control",
                        "two_forward_no_rebasing",
                        "two_forward_oracle_state",
                    ):
                        loss, stats = two_forward_loss(
                            q0_model=frozen_q0,
                            residual_model=primary,
                            codec=codec,
                            diffusion=diffusion,
                            batch=batch,
                            plan=plan,
                            residual_mean=residual_mean,
                            residual_std=residual_std,
                            statistics=statistics,
                            motion_mean=motion_mean,
                            motion_std=motion_std,
                            global_update=update,
                            total_updates=two_forward_curriculum_total_updates(args),
                            two_forward=args.stage != "one_forward_control",
                            rebase_residual=(
                                args.stage != "two_forward_no_rebasing"
                            ),
                            recompute_state=(
                                args.stage != "two_forward_oracle_state"
                            ),
                        )
                    else:
                        loss, stats, rollout = self_forcing_generator_loss(
                            stage=args.stage,
                            student=primary,
                            real_score=teacher,
                            fake_score=fake_score,
                            gan_critic=gan_critic,
                            q0_model=frozen_q0,
                            codec=codec,
                            diffusion=diffusion,
                            batch=batch,
                            residual_mean=residual_mean,
                            residual_std=residual_std,
                            statistics=statistics,
                            motion_mean=motion_mean,
                            motion_std=motion_std,
                            kinematics=kinematics,
                            student_steps=args.student_steps,
                            physical_gradient_ratio=args.physical_gradient_ratio,
                        )
                    scaled_loss = loss / args.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
            aggregate_stats(log_total, stats)
            log_count += 1
            accumulated += 1
            if not is_last_microbatch:
                host_active_seconds += (
                    time.perf_counter() - host_active_started_at
                )
                ready_for_batch_at = time.perf_counter()
                continue

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            accumulated = 0
            next_learning_rate = scheduled_learning_rate(
                optimizer,
                args,
                update,
                generator_lr,
            )

            if args.stage == "self_forcing_dmd":
                generated = rollout["residual"].detach()
                for _ in range(args.dmd_fake_updates_per_generator):
                    fake_optimizer.zero_grad(set_to_none=True)
                    levels = homogeneous_sequence_noise_levels(
                        generated.shape[0],
                        generated.shape[1],
                        diffusion.timesteps,
                        generated.device,
                    )
                    noisy = diffusion.q_sample(generated, levels)
                    predicted = fake_score(
                        batch["history_q0"],
                        normalize_residual(
                            batch["history_residual"],
                            residual_mean,
                            residual_std,
                        ),
                        batch["history_valid"],
                        rollout["q0"].detach(),
                        noisy,
                        levels,
                        statistics.normalize_state(batch["initial_state"]),
                    )
                    critic_loss = fake_score_denoising_loss(
                        predicted,
                        generated,
                    )
                    critic_loss.backward()
                    fake_optimizer.step()
                stats["loss/fake_score"] = critic_loss.detach()
                aggregate_stats(
                    log_total,
                    {"loss/fake_score": critic_loss.detach()},
                )
            elif args.stage == "self_forcing_gan":
                critic_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_latent = (
                        q0_latent(codec, rollout["q0"])
                        + denormalize_residual(
                            rollout["residual"],
                            residual_mean,
                            residual_std,
                        )
                    )
                    real_latent = (
                        q0_latent(codec, batch["target_q0"])
                        + batch["target_residual"]
                    )
                    levels = homogeneous_sequence_noise_levels(
                        real_latent.shape[0],
                        real_latent.shape[1],
                        diffusion.timesteps,
                        real_latent.device,
                    )
                    fake_noisy = diffusion.q_sample(fake_latent, levels)
                    real_noisy = diffusion.q_sample(real_latent, levels)
                fake_logits = gan_critic(fake_noisy, levels)
                real_logits = gan_critic(real_noisy, levels)
                critic_loss = gan_critic_loss(real_logits, fake_logits)
                critic_loss.backward()
                critic_optimizer.step()
                stats["loss/gan_critic"] = critic_loss.detach()
                aggregate_stats(
                    log_total,
                    {"loss/gan_critic": critic_loss.detach()},
                )
            if ema is not None and update >= args.ema_start_update:
                ema.update(unwrap(primary))

            if (
                rank == 0
                and (
                    update % args.log_interval == 0
                    or update == target_updates
                )
            ):
                now = time.time()
                elapsed = now - started_at
                interval = now - last_log_at
                completed = update - start_update
                updates_per_second = completed / max(elapsed, 1e-8)
                eta = (target_updates - update) / max(updates_per_second, 1e-8)
                payload = {
                    "update": update,
                    "optimization/learning_rate": learning_rate,
                    "progress/percent": update / args.total_updates * 100.0,
                    "progress/eta_seconds": eta,
                    "throughput/updates_per_second": updates_per_second,
                    "throughput/samples_per_second": (
                        args.batch_size
                        * world_size
                        * args.gradient_accumulation_steps
                        * updates_per_second
                    ),
                    "throughput/q0_generation_batch_factor": (
                        args.q0_generation_batch_factor
                    ),
                    "time/log_interval_seconds": interval,
                    "time/input_pipeline_wait_seconds": (
                        input_pipeline_wait_seconds
                    ),
                    "time/host_active_seconds": host_active_seconds,
                    "efficiency/input_pipeline_wait_fraction": (
                        input_pipeline_wait_seconds / max(elapsed, 1e-8)
                    ),
                    **averaged_stats(log_total, log_count),
                }
                if device.type == "cuda":
                    payload["memory/max_allocated_gb"] = (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    )
                    payload["memory/max_reserved_gb"] = (
                        torch.cuda.max_memory_reserved(device) / 2**30
                    )
                print(
                    f"update {update}/{args.total_updates} "
                    f"loss={payload.get('loss/total', payload.get('loss/main', float('nan'))):.6f} "
                    f"lr={learning_rate:.3e} "
                    f"speed={updates_per_second:.3f} update/s "
                    f"eta={format_duration(eta)}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(payload, step=update)
                log_total = {}
                log_count = 0
                last_log_at = now
            learning_rate = next_learning_rate

            if update in save_updates or update == target_updates:
                (
                    checkpoint_sampler_epoch,
                    checkpoint_batch_in_epoch,
                ) = checkpoint_data_cursor(
                    sampler_epoch,
                    batch_in_epoch,
                    len(loader),
                    is_generation_group_end,
                )
                if world_size > 1:
                    dist.barrier()
                if rank == 0:
                    payload = checkpoint_payload(
                        args=args,
                        update=update,
                        config=config,
                        codec_sha=codec_sha,
                        provenance=provenance,
                        q0_source_sha=q0_source_sha,
                        residual_mean=residual_mean,
                        residual_std=residual_std,
                        primary=primary,
                        optimizer=optimizer,
                        scaler=scaler,
                        sampler_epoch=checkpoint_sampler_epoch,
                        batch_in_epoch=checkpoint_batch_in_epoch,
                        resume_checkpoint_sha256=resume_checkpoint_sha256,
                        ema=ema,
                        fake_score=fake_score,
                        fake_optimizer=fake_optimizer,
                        gan_critic=gan_critic,
                        critic_optimizer=critic_optimizer,
                    )
                    if args.stage in JOINT_Q0_RESIDUAL_STAGES:
                        payload["q0_model"] = unwrap(frozen_q0).state_dict()
                    checkpoint_path = weights_dir / f"update-{update}.pt"
                    save_checkpoint_atomic(payload, checkpoint_path)
                    print(f"saved {checkpoint_path}", flush=True)
                if world_size > 1:
                    dist.barrier()
            host_active_seconds += time.perf_counter() - host_active_started_at
            ready_for_batch_at = time.perf_counter()
            if update >= target_updates:
                break
        sampler_epoch += 1
        batch_in_epoch = 0

    if wandb_run is not None:
        wandb_run.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
