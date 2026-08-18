"""Checkpoint loading shared by V6f-X evaluation and online deployment."""

from __future__ import annotations

from dataclasses import fields

import torch

from model.g1_hybrid_streaming_codec import (
    build_g1_hybrid_streaming_codec_from_checkpoint,
)
from model.g1_paper_faithful_dc_streaming import (
    JOINT_Q0_RESIDUAL_STAGES,
    ExponentialMovingAverage,
    G1PaperFaithfulDCConfig,
    G1PaperQ0Generator,
    G1ResidualDiffusionGenerator,
)
from train_g1_paper_faithful_dc_streaming import sha256


DEFAULT_EXPERIMENT_ID = "EXP-20260723-v6f-w-paper-faithful-dc-streaming"


MINIMUM_LATENT_WIDTH_PROFILES = {
    "minimum_latent_width_screening",
    "minimum_latent_width_100k_extension",
    "minimum_latent_width_commit_forcing",
    "minimum_latent_width_final_commit_forcing",
}


def _config_from_checkpoint(checkpoint):
    allowed = {field.name for field in fields(G1PaperFaithfulDCConfig)}
    payload = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in allowed
    }
    return G1PaperFaithfulDCConfig(**payload)


def _load_q0_checkpoint(path, config, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "q0_base":
        raise ValueError("q0 checkpoint must come from q0_base")
    if _config_from_checkpoint(checkpoint) != config:
        raise ValueError("q0 and generator model configs differ")
    model = G1PaperQ0Generator(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval().requires_grad_(False), checkpoint


def is_minimum_latent_width_profile(value):
    return value in MINIMUM_LATENT_WIDTH_PROFILES


def load_generator_bundle(args, device):
    checkpoint = torch.load(
        args.generator_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if (
        checkpoint.get("experiment_id") != args.experiment_id
        and not args.allow_historical_provenance
    ):
        raise ValueError(
            "generator checkpoint experiment ID differs from the evaluation"
        )
    stage = checkpoint.get("stage")
    config = _config_from_checkpoint(checkpoint)
    if stage == "q0_base":
        q0_model = G1PaperQ0Generator(config)
        q0_model.load_state_dict(checkpoint["model"], strict=True)
        q0_checkpoint = checkpoint
        residual_model = None
    else:
        q0_path = args.q0_checkpoint or checkpoint.get("q0_checkpoint", "")
        if not q0_path:
            raise ValueError("residual evaluation requires a q0 checkpoint")
        q0_model, q0_checkpoint = _load_q0_checkpoint(
            q0_path,
            config,
            device,
        )
        residual_model = G1ResidualDiffusionGenerator(config)
        residual_model.load_state_dict(checkpoint["model"], strict=True)
        if stage in JOINT_Q0_RESIDUAL_STAGES:
            if "q0_model" not in checkpoint:
                raise ValueError("Two-Forward checkpoint is missing its q0 fork")
            q0_model.load_state_dict(checkpoint["q0_model"], strict=True)
        if "ema" in checkpoint:
            ema = ExponentialMovingAverage(
                residual_model,
                decay=float(checkpoint["ema"]["decay"]),
            )
            ema.load_state_dict(checkpoint["ema"])
            ema.copy_to(residual_model)
    q0_model = q0_model.to(device).eval().requires_grad_(False)
    if residual_model is not None:
        residual_model = residual_model.to(device).eval().requires_grad_(False)
    codec_path = args.codec_checkpoint or checkpoint["codec_checkpoint"]
    codec_digest = sha256(codec_path)
    if codec_digest != checkpoint.get("codec_sha256"):
        raise ValueError("generator checkpoint codec SHA256 mismatch")
    expected_q0_sha = checkpoint.get("q0_source_sha256")
    if (
        stage != "q0_base"
        and expected_q0_sha
        and sha256(args.q0_checkpoint or checkpoint["q0_checkpoint"])
        != expected_q0_sha
    ):
        raise ValueError("generator checkpoint q0 SHA256 mismatch")
    codec_checkpoint = torch.load(
        codec_path,
        map_location="cpu",
        weights_only=False,
    )
    codec = build_g1_hybrid_streaming_codec_from_checkpoint(
        codec_checkpoint
    ).to(device).eval()
    codec.requires_grad_(False)
    if is_minimum_latent_width_profile(checkpoint.get("training_profile")):
        required = (
            "cache_schema_version",
            "cache_manifest_digest",
            "source_split_digest",
            "code_dim",
            "normalizer_sha256",
            "streaming_statistics_sha256",
            "generator_training_seed",
        )
        missing = [key for key in required if checkpoint.get(key) is None]
        if missing:
            raise ValueError(
                "minimum-latent-width checkpoint provenance is incomplete: "
                + ",".join(missing)
            )
        if int(checkpoint["code_dim"]) != int(codec.config.code_dim):
            raise ValueError("generator and codec latent widths differ")
        if (
            q0_checkpoint.get("cache_manifest_digest")
            != checkpoint["cache_manifest_digest"]
            or q0_checkpoint.get("codec_sha256") != codec_digest
            or q0_checkpoint.get("code_dim") != checkpoint["code_dim"]
        ):
            raise ValueError(
                "q0 checkpoint does not share the frozen width/cache/codec"
            )
    return {
        "checkpoint": checkpoint,
        "q0_checkpoint": q0_checkpoint,
        "stage": stage,
        "config": config,
        "q0_model": q0_model,
        "residual_model": residual_model,
        "codec": codec,
        "codec_checkpoint": codec_checkpoint,
        "codec_path": codec_path,
        "codec_sha256": codec_digest,
    }
