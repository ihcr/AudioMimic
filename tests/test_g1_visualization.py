import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


def reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def minimal_motion(frames=3):
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    dof_pos = np.zeros((frames, 29), dtype=np.float32)
    return {
        "motion_format": "g1",
        "fps": 30.0,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }


def fake_fk_result(frames=3):
    bodies = np.zeros((frames, 4, 3), dtype=np.float32)
    bodies[:, 1, 2] = 0.8
    bodies[:, 2, 2] = 0.4
    bodies[:, 3, 2] = 0.1
    bodies[:, 3, 0] = np.linspace(0.0, 0.2, frames, dtype=np.float32)
    return {
        "bodies": bodies,
        "body_names": ["world", "pelvis", "left_knee_link", "left_ankle_roll_link"],
        "body_parent_ids": np.array([0, 0, 1, 2], dtype=np.int64),
    }


class G1VisualizationTests(unittest.TestCase):
    def test_render_g1_motion_defaults_to_native_mujoco_mp4_with_audio(self):
        sys.modules.pop("vis", None)
        visualization = reload_module("eval.g1_visualization")

        def fake_render_mujoco_video(motion, output_path, **kwargs):
            Path(output_path).write_bytes(b"video")

        def fake_mux_video_with_audio(video_path, audio_path, output_path):
            self.assertEqual(Path(video_path).read_bytes(), b"video")
            self.assertEqual(Path(audio_path).name, "song.wav")
            Path(output_path).write_bytes(b"muxed")

        with TemporaryDirectory() as tmpdir, patch.object(
            visualization,
            "_render_mujoco_video",
            side_effect=fake_render_mujoco_video,
        ) as render_mock, patch.object(
            visualization,
            "_mux_video_with_audio",
            side_effect=fake_mux_video_with_audio,
        ):
            audio_path = Path(tmpdir) / "song.wav"
            audio_path.write_bytes(b"wav")
            output = visualization.render_g1_motion(
                minimal_motion(),
                out=tmpdir,
                epoch="sample",
                num=2,
                name=str(audio_path),
                model_path="robot.xml",
            )
            output_bytes = Path(output).read_bytes()

        self.assertEqual(output, str(Path(tmpdir) / "sample_2_song_g1.mp4"))
        self.assertEqual(output_bytes, b"muxed")
        self.assertEqual(render_mock.call_args.kwargs["model_path"], "robot.xml")
        self.assertEqual(render_mock.call_args.kwargs["root_quat_order"], "xyzw")
        self.assertEqual(render_mock.call_args.kwargs["width"], 960)
        self.assertEqual(render_mock.call_args.kwargs["height"], 720)
        self.assertEqual(render_mock.call_args.kwargs["mujoco_gl"], "egl")
        self.assertNotIn("vis", sys.modules)

    def test_render_g1_motion_native_accepts_custom_backend_options(self):
        visualization = reload_module("eval.g1_visualization")

        with TemporaryDirectory() as tmpdir, patch.object(
            visualization,
            "_render_mujoco_video",
            side_effect=lambda motion, output_path, **kwargs: Path(output_path).write_bytes(b"video"),
        ) as render_mock:
            output = visualization.render_g1_motion(
                minimal_motion(),
                out=tmpdir,
                epoch="sample",
                num=0,
                name="song.wav",
                sound=False,
                render_backend="mujoco",
                width=320,
                height=240,
                mujoco_gl="glfw",
            )

        self.assertEqual(output, str(Path(tmpdir) / "sample_0_song_g1.mp4"))
        self.assertEqual(render_mock.call_args.kwargs["width"], 320)
        self.assertEqual(render_mock.call_args.kwargs["height"], 240)
        self.assertEqual(render_mock.call_args.kwargs["mujoco_gl"], "glfw")

    def test_render_g1_motion_can_use_separate_audio_and_output_names(self):
        visualization = reload_module("eval.g1_visualization")

        with TemporaryDirectory() as tmpdir, patch.object(
            visualization,
            "_render_mujoco_video",
            side_effect=lambda motion, output_path, **kwargs: Path(output_path).write_bytes(b"video"),
        ), patch.object(
            visualization,
            "_mux_video_with_audio",
            side_effect=lambda video_path, audio_path, output_path: Path(output_path).write_bytes(b"muxed"),
        ):
            audio_path = Path(tmpdir) / "mHO5.mp3"
            audio_path.write_bytes(b"audio")
            output = visualization.render_g1_motion(
                minimal_motion(),
                out=tmpdir,
                epoch="fullsong",
                num=0,
                name=str(audio_path),
                output_name=["gHO_sBM_cAll_d21_mHO5_ch02_slice0.wav"],
                sound=True,
            )

        self.assertEqual(
            output,
            str(Path(tmpdir) / "fullsong_0_gHO_sBM_cAll_d21_mHO5_ch02_g1.mp4"),
        )

    def test_default_mujoco_camera_views_front_side(self):
        visualization = reload_module("eval.g1_visualization")
        fake_mujoco = SimpleNamespace(
            MjvCamera=lambda: SimpleNamespace(
                type=None,
                distance=0,
                azimuth=0,
                elevation=0,
                lookat=np.zeros(3, dtype=np.float64),
            ),
            mjtCamera=SimpleNamespace(mjCAMERA_FREE=0),
        )

        camera = visualization._mujoco_camera(fake_mujoco)

        self.assertEqual(camera.azimuth, 115)

    def test_native_renderer_expands_mujoco_offscreen_buffer(self):
        visualization = reload_module("eval.g1_visualization")
        model = SimpleNamespace(
            nq=36,
            vis=SimpleNamespace(global_=SimpleNamespace(offwidth=640, offheight=480)),
        )
        rendered_shapes = []

        class FakeData:
            def __init__(self, fake_model):
                self.qpos = np.zeros(fake_model.nq, dtype=np.float64)

        class FakeRenderer:
            def __init__(self, fake_model, height, width):
                rendered_shapes.append((fake_model.vis.global_.offwidth, fake_model.vis.global_.offheight, width, height))

            def update_scene(self, data, camera=None):
                pass

            def render(self):
                return np.zeros((720, 960, 3), dtype=np.uint8)

            def close(self):
                pass

        fake_mujoco = SimpleNamespace(
            MjData=FakeData,
            Renderer=FakeRenderer,
            MjvCamera=lambda: SimpleNamespace(
                type=None,
                distance=0,
                azimuth=0,
                elevation=0,
                lookat=np.zeros(3, dtype=np.float64),
            ),
            mjtCamera=SimpleNamespace(mjCAMERA_FREE=0),
            mj_forward=lambda model, data: None,
        )

        def fake_write_video(video_path, frames, fps, width, height):
            list(frames)
            Path(video_path).write_bytes(b"video")

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "robot.xml"
            model_path.write_text("<mujoco/>")
            with patch.dict(sys.modules, {"mujoco": fake_mujoco}), patch.object(
                visualization,
                "load_g1_mujoco_model",
                return_value=model,
            ), patch.object(
                visualization,
                "_write_video_frames",
                side_effect=fake_write_video,
            ):
                visualization._render_mujoco_video(
                    minimal_motion(frames=1),
                    Path(tmpdir) / "out.mp4",
                    fps=30,
                    model_path=model_path,
                    root_quat_order="xyzw",
                    width=960,
                    height=720,
                    mujoco_gl="egl",
                )

        self.assertEqual(rendered_shapes, [(960, 720, 960, 720)])

    def test_render_body_selection_drops_world_and_preserves_tree(self):
        visualization = reload_module("eval.g1_visualization")

        poses, parents, names = visualization.select_g1_render_bodies(fake_fk_result())

        self.assertEqual(names, ["pelvis", "left_knee_link", "left_ankle_roll_link"])
        self.assertEqual(parents.tolist(), [-1, 0, 1])
        self.assertEqual(poses.shape, (3, 3, 3))

    def test_compute_axes_limits_keeps_static_g1_readable(self):
        visualization = reload_module("eval.g1_visualization")
        poses = np.array(
            [
                [
                    [0.0, 0.0, 0.85],
                    [0.1, 0.1, 1.2],
                    [-0.1, 0.1, 0.1],
                    [0.1, -0.1, 0.1],
                ]
            ],
            dtype=np.float32,
        )

        axes = visualization.compute_axes_limits(poses)
        spans = [high - low for low, high in axes]

        self.assertTrue(all(span <= 2.0 for span in spans))

    def test_render_g1_motion_writes_silent_gif_without_importing_smpl_vis(self):
        sys.modules.pop("vis", None)
        visualization = reload_module("eval.g1_visualization")

        def fake_save(animation, filename, *args, **kwargs):
            animation._draw_was_started = True
            Path(filename).write_bytes(b"gif")

        with TemporaryDirectory() as tmpdir, patch.object(
            visualization,
            "forward_g1_kinematics",
            return_value=fake_fk_result(),
        ) as fk_mock, patch(
            "matplotlib.animation.Animation.save",
            new=fake_save,
        ):
            output = visualization.render_g1_motion(
                minimal_motion(),
                out=tmpdir,
                epoch="sample",
                num=2,
                name="song_slice0.wav",
                sound=False,
                render_backend="stick",
                model_path="robot.xml",
                root_quat_order="xyzw",
            )
            output_bytes = Path(output).read_bytes()

        self.assertEqual(output, str(Path(tmpdir) / "sample_2_song_slice0_g1.gif"))
        self.assertEqual(output_bytes, b"gif")
        fk_mock.assert_called_once()
        self.assertEqual(fk_mock.call_args.kwargs["model_path"], "robot.xml")
        self.assertEqual(fk_mock.call_args.kwargs["root_quat_order"], "xyzw")
        self.assertNotIn("vis", sys.modules)

    def test_render_g1_motion_rejects_malformed_payload_before_fk(self):
        visualization = reload_module("eval.g1_visualization")
        bad_motion = minimal_motion()
        bad_motion.pop("root_rot")

        with patch.object(visualization, "forward_g1_kinematics") as fk_mock:
            with self.assertRaisesRegex(ValueError, "root_rot"):
                visualization.render_g1_motion(
                    bad_motion,
                    out="renders",
                    epoch="sample",
                    num=0,
                    name="song.wav",
                    sound=False,
                    render_backend="stick",
                    model_path="robot.xml",
                )

        fk_mock.assert_not_called()

    def test_render_g1_motion_defaults_to_g1_dataset_xyzw_quaternions(self):
        visualization = reload_module("eval.g1_visualization")

        def fake_save(animation, filename, *args, **kwargs):
            animation._draw_was_started = True
            Path(filename).write_bytes(b"gif")

        with TemporaryDirectory() as tmpdir, patch.object(
            visualization,
            "forward_g1_kinematics",
            return_value=fake_fk_result(),
        ) as fk_mock, patch(
            "matplotlib.animation.Animation.save",
            new=fake_save,
        ):
            visualization.render_g1_motion(
                minimal_motion(),
                out=tmpdir,
                epoch="sample",
                num=0,
                name="song.wav",
                sound=False,
                render_backend="stick",
                model_path="robot.xml",
            )

        self.assertEqual(fk_mock.call_args.kwargs["root_quat_order"], "xyzw")

    def test_render_g1_motion_rejects_missing_audio_before_native_render(self):
        visualization = reload_module("eval.g1_visualization")

        with TemporaryDirectory() as tmpdir, patch.object(
            visualization,
            "_render_mujoco_video",
        ) as render_mock:
            with self.assertRaisesRegex(FileNotFoundError, "audio file not found"):
                visualization.render_g1_motion(
                    minimal_motion(),
                    out=tmpdir,
                    epoch="sample",
                    num=0,
                    name=Path(tmpdir) / "missing.wav",
                    sound=True,
                )

        render_mock.assert_not_called()

    def test_render_g1_motion_rejects_missing_mujoco_model_path(self):
        visualization = reload_module("eval.g1_visualization")

        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "G1 MuJoCo model not found"):
                visualization.render_g1_motion(
                    minimal_motion(),
                    out=tmpdir,
                    epoch="sample",
                    num=0,
                    name="song.wav",
                    sound=False,
                    model_path=Path(tmpdir) / "missing.xml",
                )

    def test_render_g1_motion_rejects_native_init_failure_clearly(self):
        visualization = reload_module("eval.g1_visualization")

        with TemporaryDirectory() as tmpdir, patch.object(
            visualization,
            "_render_mujoco_video",
            side_effect=RuntimeError("Native G1 MuJoCo rendering failed to initialize"),
        ):
            with self.assertRaisesRegex(RuntimeError, "MuJoCo rendering failed"):
                visualization.render_g1_motion(
                    minimal_motion(),
                    out=tmpdir,
                    epoch="sample",
                    num=0,
                    name="song.wav",
                    sound=False,
                )


if __name__ == "__main__":
    unittest.main()
