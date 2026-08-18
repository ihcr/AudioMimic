import unittest

import torch

from run_g1_paper_faithful_dc_sonic import (
    _apply_delayed_residual,
    _committed_reference_frame_count,
    _jit_inference_start_deadline,
)

from model.g1_paper_faithful_dc_runtime import (
    COMMIT_FRAMES,
    PaperFaithfulDCRuntime,
    encode_k64_history_from_motion,
)


class _Statistics:
    def __init__(self):
        self.observed_states = []
        self.fps = 30.0

    def normalize_state(self, state):
        self.observed_states.append(state.detach().clone())
        return state + 10.0


class _Q0Model:
    def __init__(self):
        self.history_q0 = []
        self.history_valid = []
        self.conditions = []

    def generate(
        self,
        history_q0,
        history_valid,
        condition,
        *,
        length,
        greedy,
        temperature,
        generator,
    ):
        del greedy, temperature, generator
        self.history_q0.append(history_q0.detach().clone())
        self.history_valid.append(history_valid.detach().clone())
        self.conditions.append(condition.detach().clone())
        values = torch.arange(length, device=history_q0.device, dtype=torch.long)
        return values.unsqueeze(0).expand(history_q0.shape[0], -1).clone()


class _Quantizer:
    def __init__(self):
        self.codebooks = [torch.arange(32, dtype=torch.float32).view(32, 1)]


class _Codec:
    def __init__(self):
        self.quantizer = _Quantizer()
        self.decode_states = []

    def decode_components(self, q0_latent, residual, state):
        self.decode_states.append(state.detach().clone())
        batch, tokens, _ = q0_latent.shape
        decoded = torch.zeros(batch, tokens * 2, 34, device=q0_latent.device)
        decoded[..., 0] = q0_latent[..., 0].repeat_interleave(2, dim=1)
        decoded[..., 2] = state[:, 0:1]
        decoded[..., 4] = 1.0
        decoded[..., 5] = residual[..., 0].repeat_interleave(2, dim=1)
        return decoded, {}

    def encode_components(self, motion, state):
        del state
        batch, frames, _ = motion.shape
        tokens = frames // 2
        q0_ids = torch.arange(
            batch * tokens,
            device=motion.device,
            dtype=torch.long,
        ).view(batch, tokens, 1) % 32
        residual = motion[:, ::2, :1].reshape(batch * tokens, 1)
        return {"q0_ids": q0_ids, "residual": residual}


class PaperFaithfulDCRuntimeTests(unittest.TestCase):
    def test_delayed_residual_uses_previous_error_and_current_prediction(self):
        current = torch.full((1, 66), 10.0)
        measured_previous = torch.full((1, 66), 5.0)
        synthetic_previous = torch.full((1, 66), 3.0)
        corrected, residual = _apply_delayed_residual(
            current,
            measured_previous=measured_previous,
            synthetic_previous=synthetic_previous,
            alpha=0.25,
        )
        self.assertTrue(torch.equal(residual, torch.full((1, 66), 2.0)))
        self.assertTrue(torch.equal(corrected, torch.full((1, 66), 10.5)))

    def test_delayed_residual_first_step_is_open_loop(self):
        current = torch.randn(1, 66)
        corrected, residual = _apply_delayed_residual(
            current,
            measured_previous=torch.randn(1, 66),
            synthetic_previous=None,
            alpha=1.0,
        )
        self.assertTrue(torch.equal(corrected, current))
        self.assertIsNone(residual)

    def test_chunked_reference_clock_preserves_30_to_50hz_phase(self):
        frame_index = 0
        counts = []
        for _ in range(6):
            count = _committed_reference_frame_count(
                frame_index=frame_index,
                source_frames=8,
                source_fps=30.0,
                target_fps=50.0,
            )
            counts.append(count)
            frame_index += count
        self.assertEqual(counts, [13, 13, 14, 13, 13, 14])
        self.assertEqual(frame_index, 80)

    def test_jit_inference_starts_one_budget_before_commit_deadline(self):
        start = _jit_inference_start_deadline(
            stream_started=100.0,
            commit_index=1,
            commit_seconds=8.0 / 30.0,
            budget_ms=80.0,
        )
        self.assertAlmostEqual(start, 100.0 + 8.0 / 30.0 - 0.080)

    def _runtime(self):
        self.statistics = _Statistics()
        self.q0_model = _Q0Model()
        self.codec = _Codec()
        return PaperFaithfulDCRuntime(
            q0_model=self.q0_model,
            residual_model=None,
            stage="q0_base",
            codec=self.codec,
            diffusion=None,
            statistics=self.statistics,
            residual_mean=torch.zeros(1, 1, 1),
            residual_std=torch.ones(1, 1, 1),
            motion_mean=torch.zeros(1, 1, 34),
            motion_std=torch.ones(1, 1, 34),
            nfe=10,
            q0_policy="greedy",
        )

    def test_commit_advances_generated_history_but_requires_external_s66(self):
        runtime = self._runtime()
        measured_start = torch.zeros(1, 66)
        measured_start[:, 0] = 0.78
        state = runtime.cold_start(measured_start)

        pending, commit = runtime.plan_commit(state)

        self.assertEqual(tuple(commit.raw_commit.shape), (1, COMMIT_FRAMES, 34))
        self.assertTrue(pending.awaiting_execution_feedback)
        self.assertTrue(torch.equal(pending.history_q0[0, -4:], torch.arange(4)))
        self.assertTrue(bool(pending.history_valid[0, -4:].all()))
        self.assertFalse(bool(pending.history_valid[0, :-4].any()))
        self.assertTrue(torch.equal(self.q0_model.conditions[0], measured_start + 10.0))
        with self.assertRaisesRegex(RuntimeError, "before execution feedback"):
            runtime.plan_commit(pending)

        measured_after = torch.full((1, 66), 3.0)
        resumed = runtime.accept_execution_feedback(pending, measured_after)
        self.assertFalse(resumed.awaiting_execution_feedback)
        self.assertTrue(torch.equal(resumed.execution_state, measured_after))
        _, second_commit = runtime.plan_commit(resumed)
        self.assertTrue(torch.equal(self.q0_model.conditions[1], measured_after + 10.0))
        self.assertTrue(torch.equal(second_commit.execution_state_before, measured_after))

    def test_execution_feedback_requires_a_pending_commit_and_matching_shape(self):
        runtime = self._runtime()
        state = runtime.cold_start(torch.zeros(1, 66))
        with self.assertRaisesRegex(RuntimeError, "only valid after"):
            runtime.accept_execution_feedback(state, torch.zeros(1, 66))
        pending, _ = runtime.plan_commit(state)
        with self.assertRaisesRegex(ValueError, "batch shape"):
            runtime.accept_execution_feedback(pending, torch.zeros(2, 66))

    def test_k64_seed_uses_training_causal_padding_and_measured_s66(self):
        runtime = self._runtime()
        raw_motion = torch.zeros(1, 128, 34)
        raw_motion[..., 2] = 0.78
        raw_motion[..., 4] = 1.0
        raw_motion[..., 5] = torch.arange(128, dtype=torch.float32)
        history_q0, history_residual_raw, history_valid, source_state = (
            encode_k64_history_from_motion(
                codec=runtime.codec,
                statistics=runtime.statistics,
                raw_motion=raw_motion,
                motion_mean=torch.zeros(1, 1, 34),
                motion_std=torch.ones(1, 1, 34),
            )
        )
        self.assertEqual(tuple(history_q0.shape), (1, 64))
        self.assertEqual(tuple(history_residual_raw.shape), (1, 64, 1))
        self.assertFalse(bool(history_valid[0, :4].any()))
        self.assertTrue(bool(history_valid[0, 4:].all()))
        self.assertEqual(int(history_valid.sum()), 60)
        self.assertEqual(tuple(source_state.shape), (1, 66))

        measured_start = torch.full((1, 66), 2.0)
        measured_start[:, 0] = 0.78
        state = runtime.warm_start(
            measured_start,
            history_q0=history_q0,
            history_residual=history_residual_raw,
            history_valid=history_valid,
        )
        _, commit = runtime.plan_commit(state)
        self.assertTrue(torch.equal(self.q0_model.history_q0[0], history_q0))
        self.assertTrue(torch.equal(self.q0_model.history_valid[0], history_valid))
        self.assertTrue(torch.equal(commit.execution_state_before, measured_start))


if __name__ == "__main__":
    unittest.main()
