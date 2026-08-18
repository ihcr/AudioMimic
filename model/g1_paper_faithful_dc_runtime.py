"""Strict-causal online runtime for the V6f-X paper-faithful D+C generator.

The generator's latent history represents commands that were generated and
committed to the downstream controller.  The physical S66 condition is a
separate execution-feedback channel: after every C4 commit it must be replaced
with the latest measured state, never reconstructed from the decoded plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from dataset.g1_streaming_state import (
    G1_STREAMING_STATE_DIM,
    boundary_state_from_motion,
)
from model.g1_paper_faithful_dc_streaming import (
    COMMIT_TOKENS,
    HISTORY_TOKENS,
    PLAN_TOKENS,
    diffusion_forcing_sample,
)


MOTION_FRAMES_PER_TOKEN = 2
COMMIT_FRAMES = COMMIT_TOKENS * MOTION_FRAMES_PER_TOKEN


def _append_history(history: torch.Tensor, committed: torch.Tensor) -> torch.Tensor:
    return torch.cat((history, committed), dim=1)[:, -history.shape[1] :]


def _normalized_zero_residual(
    shape: tuple[int, int, int],
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
) -> torch.Tensor:
    return -residual_mean.expand(shape) / residual_std.expand(shape)


def _validate_execution_state(execution_state: torch.Tensor) -> None:
    if execution_state.ndim != 2 or execution_state.shape[-1] != G1_STREAMING_STATE_DIM:
        raise ValueError(
            "execution_state must have shape [B, "
            f"{G1_STREAMING_STATE_DIM}]"
        )
    if not torch.isfinite(execution_state).all():
        raise FloatingPointError("execution_state contains non-finite values")


@torch.inference_mode()
def encode_k64_history_from_motion(
    *,
    codec,
    statistics,
    raw_motion: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the exact K64 causal context used by the paper-faithful evaluator.

    Fifteen overlapping H8 codec windows contribute their committed C4
    prefixes. The leading C4 remains invalid padding, exactly as in training.
    The returned residual history is in codec space, before generator residual
    normalization. The returned S66 is the source boundary diagnostic only;
    online deployment must condition on measured execution feedback instead.
    """
    if raw_motion.ndim != 3 or raw_motion.shape[-1] != 34:
        raise ValueError("raw_motion must have shape [B,T,34]")
    if motion_mean.shape != (1, 1, 34) or motion_std.shape != (1, 1, 34):
        raise ValueError("motion_mean and motion_std must have shape [1,1,34]")
    if raw_motion.device != motion_mean.device or raw_motion.device != motion_std.device:
        raise ValueError("raw_motion and motion statistics must share a device")
    if not torch.isfinite(raw_motion).all():
        raise FloatingPointError("raw_motion contains non-finite values")
    required_frames = HISTORY_TOKENS * MOTION_FRAMES_PER_TOKEN
    if raw_motion.shape[1] < required_frames:
        raise ValueError(
            "raw_motion needs at least "
            f"{required_frames} frames to construct a K{HISTORY_TOKENS} history"
        )

    batch = raw_motion.shape[0]
    plan_count = HISTORY_TOKENS // COMMIT_TOKENS - 1
    starts = (
        torch.arange(plan_count, device=raw_motion.device, dtype=torch.long)
        * COMMIT_FRAMES
    )
    expanded_motion = raw_motion[:, None].expand(-1, plan_count, -1, -1).reshape(
        batch * plan_count,
        raw_motion.shape[1],
        raw_motion.shape[2],
    )
    expanded_starts = starts[None].expand(batch, -1).reshape(-1)
    frame_offsets = torch.arange(
        PLAN_TOKENS * MOTION_FRAMES_PER_TOKEN,
        device=raw_motion.device,
        dtype=torch.long,
    )
    gather = expanded_starts[:, None] + frame_offsets[None]
    plans_raw = expanded_motion.gather(
        1,
        gather[..., None].expand(-1, -1, raw_motion.shape[-1]),
    )
    source_states = boundary_state_from_motion(
        expanded_motion,
        expanded_starts,
        fps=statistics.fps,
    )
    components = codec.encode_components(
        (plans_raw - motion_mean) / motion_std,
        statistics.normalize_state(source_states),
    )
    q0 = components["q0_ids"]
    residual = components["residual"]
    expected_tokens = batch * plan_count * PLAN_TOKENS
    if q0.dtype != torch.long or q0.numel() != expected_tokens:
        raise ValueError("codec q0 IDs do not match the expected K64 seed shape")
    if residual.ndim < 2 or residual.numel() % expected_tokens:
        raise ValueError("codec residuals do not match the expected K64 seed shape")
    q0 = q0.view(batch, plan_count, PLAN_TOKENS)
    residual = residual.view(batch, plan_count, PLAN_TOKENS, -1)
    committed_q0 = q0[:, :, :COMMIT_TOKENS].reshape(batch, -1)
    committed_residual = residual[:, :, :COMMIT_TOKENS].reshape(
        batch,
        -1,
        residual.shape[-1],
    )
    history_q0 = F.pad(committed_q0, (COMMIT_TOKENS, 0))
    history_residual = F.pad(committed_residual, (0, 0, COMMIT_TOKENS, 0))
    history_valid = torch.zeros(
        batch,
        HISTORY_TOKENS,
        device=raw_motion.device,
        dtype=torch.bool,
    )
    history_valid[:, COMMIT_TOKENS:] = True
    source_boundary_state = boundary_state_from_motion(
        raw_motion,
        torch.full(
            (batch,),
            required_frames,
            device=raw_motion.device,
            dtype=torch.long,
        ),
        fps=statistics.fps,
    )
    return history_q0, history_residual, history_valid, source_boundary_state


@dataclass(frozen=True)
class PaperFaithfulDCRuntimeState:
    """Persistent online state for one or more strictly causal streams."""

    history_q0: torch.Tensor
    history_residual: torch.Tensor
    history_valid: torch.Tensor
    execution_state: torch.Tensor
    step_index: int = 0
    awaiting_execution_feedback: bool = False

    def __post_init__(self) -> None:
        batch = int(self.execution_state.shape[0])
        _validate_execution_state(self.execution_state)
        if self.history_q0.ndim != 2 or self.history_q0.shape != (
            batch,
            HISTORY_TOKENS,
        ):
            raise ValueError(
                f"history_q0 must have shape [B, {HISTORY_TOKENS}]"
            )
        if self.history_residual.ndim != 3 or self.history_residual.shape[:2] != (
            batch,
            HISTORY_TOKENS,
        ):
            raise ValueError(
                f"history_residual must have shape [B, {HISTORY_TOKENS}, R]"
            )
        if self.history_valid.shape != (batch, HISTORY_TOKENS):
            raise ValueError(
                f"history_valid must have shape [B, {HISTORY_TOKENS}]"
            )
        if self.history_q0.dtype != torch.long:
            raise TypeError("history_q0 must use torch.long")
        if self.history_valid.dtype != torch.bool:
            raise TypeError("history_valid must use torch.bool")
        if self.history_q0.device != self.execution_state.device:
            raise ValueError("history_q0 and execution_state must share a device")
        if self.history_residual.device != self.execution_state.device:
            raise ValueError("history_residual and execution_state must share a device")
        if self.history_valid.device != self.execution_state.device:
            raise ValueError("history_valid and execution_state must share a device")
        if not torch.isfinite(self.history_residual).all():
            raise FloatingPointError("history_residual contains non-finite values")


@dataclass(frozen=True)
class PaperFaithfulDCCommit:
    """A planned H8 horizon and the C4 prefix that must be executed."""

    raw_commit: torch.Tensor
    raw_plan: torch.Tensor
    q0_plan: torch.Tensor
    residual_plan: torch.Tensor
    execution_state_before: torch.Tensor
    latency_ms: dict[str, float]


class PaperFaithfulDCRuntime:
    """Generate one C4 commit, then require an external S66 before replanning."""

    def __init__(
        self,
        *,
        q0_model,
        residual_model,
        stage: str,
        codec,
        diffusion,
        statistics,
        residual_mean: torch.Tensor,
        residual_std: torch.Tensor,
        motion_mean: torch.Tensor,
        motion_std: torch.Tensor,
        nfe: int,
        q0_policy: str = "sample",
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> None:
        self.q0_model = q0_model
        self.residual_model = residual_model
        self.stage = str(stage)
        self.codec = codec
        self.diffusion = diffusion
        self.statistics = statistics
        self.residual_mean = residual_mean
        self.residual_std = residual_std
        self.motion_mean = motion_mean
        self.motion_std = motion_std
        self.nfe = int(nfe)
        self.q0_policy = str(q0_policy)
        self.temperature = float(temperature)
        self.generator = generator
        if self.q0_policy not in ("greedy", "sample"):
            raise ValueError("q0_policy must be greedy or sample")
        if self.nfe <= 0:
            raise ValueError("nfe must be positive")
        if self.residual_mean.ndim != 3 or self.residual_mean.shape[:2] != (1, 1):
            raise ValueError("residual_mean must have shape [1, 1, R]")
        if self.residual_std.shape != self.residual_mean.shape:
            raise ValueError("residual_std must match residual_mean")
        if self.motion_mean.shape != (1, 1, 34) or self.motion_std.shape != (1, 1, 34):
            raise ValueError("motion_mean and motion_std must have shape [1, 1, 34]")
        if torch.any(self.residual_std <= 0) or torch.any(self.motion_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        if self.residual_model is not None and self.diffusion is None:
            raise ValueError("residual_model requires a diffusion sampler")

    @property
    def residual_dim(self) -> int:
        return int(self.residual_mean.shape[-1])

    def cold_start(self, execution_state: torch.Tensor) -> PaperFaithfulDCRuntimeState:
        """Create the explicitly supported no-oracle startup state."""
        _validate_execution_state(execution_state)
        batch = int(execution_state.shape[0])
        return PaperFaithfulDCRuntimeState(
            history_q0=torch.zeros(
                batch,
                HISTORY_TOKENS,
                device=execution_state.device,
                dtype=torch.long,
            ),
            history_residual=torch.zeros(
                batch,
                HISTORY_TOKENS,
                self.residual_dim,
                device=execution_state.device,
                dtype=execution_state.dtype,
            ),
            history_valid=torch.zeros(
                batch,
                HISTORY_TOKENS,
                device=execution_state.device,
                dtype=torch.bool,
            ),
            execution_state=execution_state.detach().clone(),
        )

    def warm_start(
        self,
        execution_state: torch.Tensor,
        *,
        history_q0: torch.Tensor,
        history_residual: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> PaperFaithfulDCRuntimeState:
        """Start from encoded K64 motion while conditioning on measured S66.

        `history_residual` must already be normalized with this generator's
        residual statistics. `execution_state` is always current external
        feedback, never the state reconstructed from seed motion.
        """
        _validate_execution_state(execution_state)
        if execution_state.shape[0] != history_q0.shape[0]:
            raise ValueError("execution_state and K64 history batch sizes differ")
        if history_residual.shape[-1] != self.residual_dim:
            raise ValueError("K64 residual history does not match the generator width")
        return PaperFaithfulDCRuntimeState(
            history_q0=history_q0.detach().clone(),
            history_residual=history_residual.detach().clone(),
            history_valid=history_valid.detach().clone(),
            execution_state=execution_state.detach().clone(),
        )

    @torch.inference_mode()
    def plan_commit(
        self,
        state: PaperFaithfulDCRuntimeState,
    ) -> tuple[PaperFaithfulDCRuntimeState, PaperFaithfulDCCommit]:
        """Generate H8, append only C4 latents, and wait for measured S66."""
        if state.awaiting_execution_feedback:
            raise RuntimeError(
                "cannot plan a second C4 commit before execution feedback arrives"
            )
        batch = int(state.execution_state.shape[0])
        device = state.execution_state.device
        condition = self.statistics.normalize_state(state.execution_state)
        started = time.perf_counter()
        q0_started = started
        q0_plan = self.q0_model.generate(
            state.history_q0,
            state.history_valid,
            condition,
            length=PLAN_TOKENS,
            greedy=self.q0_policy == "greedy",
            temperature=self.temperature,
            generator=self.generator,
        )
        if q0_plan.shape != (batch, PLAN_TOKENS) or q0_plan.dtype != torch.long:
            raise ValueError(
                f"q0 model must return long [B, {PLAN_TOKENS}] plans"
            )
        if q0_plan.device != device:
            raise ValueError("q0 model returned a plan on the wrong device")
        self._synchronize(device)
        q0_finished = time.perf_counter()

        if self.residual_model is None:
            residual_plan = _normalized_zero_residual(
                (batch, PLAN_TOKENS, self.residual_dim),
                self.residual_mean,
                self.residual_std,
            )
        else:
            noise = torch.randn(
                batch,
                PLAN_TOKENS,
                self.residual_dim,
                device=device,
                dtype=state.history_residual.dtype,
                generator=self.generator,
            )

            def model_fn(noisy: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
                return self.residual_model(
                    state.history_q0,
                    state.history_residual,
                    state.history_valid,
                    q0_plan,
                    noisy,
                    levels,
                    condition,
                )

            if self.stage == "diffusion_forcing":
                residual_plan = diffusion_forcing_sample(
                    self.diffusion,
                    model_fn,
                    noise.shape,
                    self.nfe,
                    uncertainty_scale=1.0,
                    noise=noise,
                )
            else:
                residual_plan = self.diffusion.sample(
                    model_fn,
                    noise.shape,
                    self.nfe,
                    noise=noise,
                )
        if residual_plan.shape != (batch, PLAN_TOKENS, self.residual_dim):
            raise ValueError(
                f"residual plan must have shape [B, {PLAN_TOKENS}, R]"
            )
        if not torch.isfinite(residual_plan).all():
            raise FloatingPointError("residual plan contains non-finite values")
        self._synchronize(device)
        residual_finished = time.perf_counter()

        q0_latent = F.embedding(q0_plan, self.codec.quantizer.codebooks[0])
        residual_raw = residual_plan * self.residual_std + self.residual_mean
        decoded, _ = self.codec.decode_components(q0_latent, residual_raw, condition)
        raw_plan = decoded * self.motion_std + self.motion_mean
        if raw_plan.shape[:2] != (batch, PLAN_TOKENS * MOTION_FRAMES_PER_TOKEN) or raw_plan.shape[-1] != 34:
            raise ValueError("codec must decode an H8 [B, 16, 34] motion plan")
        if not torch.isfinite(raw_plan).all():
            raise FloatingPointError("decoded motion plan contains non-finite values")
        self._synchronize(device)
        finished = time.perf_counter()

        next_state = replace(
            state,
            history_q0=_append_history(
                state.history_q0,
                q0_plan[:, :COMMIT_TOKENS],
            ),
            history_residual=_append_history(
                state.history_residual,
                residual_plan[:, :COMMIT_TOKENS],
            ),
            history_valid=_append_history(
                state.history_valid,
                torch.ones(
                    batch,
                    COMMIT_TOKENS,
                    device=device,
                    dtype=torch.bool,
                ),
            ),
            step_index=int(state.step_index) + 1,
            awaiting_execution_feedback=True,
        )
        commit = PaperFaithfulDCCommit(
            raw_commit=raw_plan[:, :COMMIT_FRAMES].detach().clone(),
            raw_plan=raw_plan.detach().clone(),
            q0_plan=q0_plan.detach().clone(),
            residual_plan=residual_plan.detach().clone(),
            execution_state_before=state.execution_state.detach().clone(),
            latency_ms={
                "q0": (q0_finished - q0_started) * 1000.0,
                "residual": (residual_finished - q0_finished) * 1000.0,
                "decode": (finished - residual_finished) * 1000.0,
                "total": (finished - started) * 1000.0,
            },
        )
        return next_state, commit

    def accept_execution_feedback(
        self,
        state: PaperFaithfulDCRuntimeState,
        execution_state: torch.Tensor,
    ) -> PaperFaithfulDCRuntimeState:
        """Install measured S66 without re-encoding any physical motion."""
        if not state.awaiting_execution_feedback:
            raise RuntimeError("execution feedback is only valid after a planned commit")
        _validate_execution_state(execution_state)
        if execution_state.shape != state.execution_state.shape:
            raise ValueError("execution feedback batch shape does not match runtime state")
        if execution_state.device != state.execution_state.device:
            raise ValueError("execution feedback must share the runtime device")
        return replace(
            state,
            execution_state=execution_state.detach().clone(),
            awaiting_execution_feedback=False,
        )

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
