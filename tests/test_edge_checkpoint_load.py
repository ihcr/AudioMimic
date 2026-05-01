import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import EDGE as edge_module


class EdgeCheckpointLoadTests(unittest.TestCase):
    def test_checkpoint_load_disables_weights_only(self):
        fake_model = MagicMock()
        fake_model.parameters.return_value = []
        fake_diffusion = MagicMock()
        fake_diffusion.to.return_value = fake_diffusion
        fake_accelerator = MagicMock()
        fake_accelerator.device = "cpu"
        fake_accelerator.prepare.side_effect = lambda x: x

        checkpoint = {
            "normalizer": object(),
            "ema_state_dict": {},
            "model_state_dict": {},
        }

        with patch.object(edge_module, "Accelerator", return_value=fake_accelerator), patch.object(
            edge_module, "AcceleratorState", return_value=SimpleNamespace(num_processes=1)
        ), patch.object(edge_module, "DanceDecoder", return_value=fake_model), patch.object(
            edge_module, "SMPLSkeleton", return_value=MagicMock()
        ), patch.object(
            edge_module, "GaussianDiffusion", return_value=fake_diffusion
        ), patch.object(
            edge_module, "Adan", return_value=MagicMock()
        ), patch.object(
            edge_module.torch, "load", return_value=checkpoint
        ) as mock_load:
            edge_module.EDGE("jukebox", "checkpoint.pt")

        _, kwargs = mock_load.call_args
        self.assertIn("weights_only", kwargs)
        self.assertFalse(kwargs["weights_only"])


if __name__ == "__main__":
    unittest.main()
