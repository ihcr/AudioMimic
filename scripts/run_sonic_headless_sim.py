"""Run the GR00T MuJoCo simulator headlessly and release its elastic band on stdin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groot_root",
        default="~/GR00T-WholeBodyControl",
        help="GR00T-WholeBodyControl checkout",
    )
    parser.add_argument("--sonic_state_port", type=int, default=5559)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    groot_root = Path(args.groot_root).expanduser().resolve()
    if str(groot_root) not in sys.path:
        sys.path.insert(0, str(groot_root))

    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.scripts.run_sim_loop import SimWrapper
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

    config = SimLoopConfig(
        enable_onscreen=False,
        enable_offscreen=False,
        enable_image_publish=False,
        enable_sonic_state_publish=True,
        sonic_state_port=args.sonic_state_port,
    )
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name
    wbc_config["SONIC_STATE_PUBLISH"] = True
    wbc_config["SONIC_STATE_PORT"] = args.sonic_state_port
    wbc_config["SONIC_STATE_TOPIC"] = config.sonic_state_topic

    simulator = SimWrapper(
        robot_model=instantiate_g1_robot_model(),
        env_name=config.env_name,
        config=wbc_config,
        onscreen=False,
        offscreen=False,
        enable_image_publish=False,
    ).sim
    SimulatorFactory.start_simulator(simulator, as_thread=True)

    try:
        print("HEADLESS_SIM_READY elastic_band=enabled", flush=True)
        print("Commands: release, hold, reset, status, quit", flush=True)
        while simulator.sim_thread is not None and simulator.sim_thread.is_alive():
            command = input().strip().lower()
            band = simulator.sim_env.elastic_band
            if command in ("release", "9"):
                band.enable = False
                print("HEADLESS_SIM_RELEASED elastic_band=disabled", flush=True)
            elif command == "hold":
                band.enable = True
                print("HEADLESS_SIM_HELD elastic_band=enabled", flush=True)
            elif command == "reset":
                simulator.reset()
                print(f"HEADLESS_SIM_RESET elastic_band={band.enable}", flush=True)
            elif command == "status":
                height = float(simulator.sim_env.mj_data.qpos[2])
                print(
                    f"HEADLESS_SIM_STATUS height={height:.4f} elastic_band={band.enable}",
                    flush=True,
                )
            elif command in ("quit", "exit"):
                break
            elif command:
                print(f"Unknown command: {command}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        simulator.close()


if __name__ == "__main__":
    main(parse_args())
