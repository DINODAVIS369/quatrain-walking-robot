"""Evaluate / Play the quadruped walking robot in simulation."""

import argparse
import sys
from pathlib import Path

# Add mjlab src to path if needed
mjlab_src = Path(__file__).resolve().parent.parent / "mjlab" / "src"
if mjlab_src.exists():
    sys.path.insert(0, str(mjlab_src))

import mjlab.scripts.play as mjlab_play
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.viewer.native.keys import KEY_A, KEY_D, KEY_E, KEY_M, KEY_Q, KEY_S, KEY_W, KEY_X

# Register custom Robodog task
import env_cfg


def main():
    parser = argparse.ArgumentParser(description="Visualize and test the quadruped robot.")
    parser.add_argument(
        "--task",
        type=str,
        default="Mjlab-Velocity-Flat-Robodog",
        help="Task ID (e.g. Mjlab-Velocity-Flat-Robodog)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["trained", "zero", "random"],
        default=None,
        help="Policy to run. Defaults to 'trained' when --checkpoint is supplied, otherwise 'random'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments to render",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        choices=["auto", "native", "viser"],
        default="auto",
        help="Simulation viewer backend (keyboard control requires native)",
    )
    parser.add_argument(
        "--command-mode",
        choices=["keyboard", "automatic"],
        default="keyboard",
        help="Start with keyboard velocity control or automatic velocity commands; press M to toggle",
    )
    args = parser.parse_args()

    if args.viewer == "viser" and args.command_mode == "keyboard":
        parser.error("keyboard control requires --viewer native or --viewer auto")

    # A supplied checkpoint should never be silently ignored by the random
    # dummy policy. Keep random as the no-argument behavior for diagnostics.
    agent = args.agent or ("trained" if args.checkpoint else "random")
    if agent == "trained" and not args.checkpoint:
        parser.error("--agent trained requires --checkpoint PATH")

    cfg = PlayConfig(
        agent=agent,
        checkpoint_file=args.checkpoint,
        num_envs=args.num_envs,
        viewer="native" if args.command_mode == "keyboard" else args.viewer,
    )

    command_state = {
        "mode": args.command_mode,
        "vx": 0.0,
        "vy": 0.0,
        "wz": 0.0,
    }

    class KeyboardCommandViewer(mjlab_play.NativeMujocoViewer):
        """Native viewer that steers the learned policy with velocity commands."""

        def __init__(self, env, policy):
            super().__init__(env, policy, key_callback=self._on_key)

        def _on_key(self, key: int) -> None:
            if key == KEY_M:
                command_state["mode"] = (
                    "automatic" if command_state["mode"] == "keyboard" else "keyboard"
                )
                command_state["vx"] = command_state["vy"] = command_state["wz"] = 0.0
                print(f"\nCommand mode: {command_state['mode'].upper()}")
            elif key == KEY_X:
                command_state["vx"] = command_state["vy"] = command_state["wz"] = 0.0
                print("\nVelocity command stopped")
            elif command_state["mode"] == "keyboard":
                if key == KEY_W:
                    command_state["vx"] = min(1.0, command_state["vx"] + 0.1)
                elif key == KEY_S:
                    command_state["vx"] = max(-1.0, command_state["vx"] - 0.1)
                elif key == KEY_A:
                    command_state["vy"] = min(0.5, command_state["vy"] + 0.1)
                elif key == KEY_D:
                    command_state["vy"] = max(-0.5, command_state["vy"] - 0.1)
                elif key == KEY_Q:
                    command_state["wz"] = min(0.8, command_state["wz"] + 0.1)
                elif key == KEY_E:
                    command_state["wz"] = max(-0.8, command_state["wz"] - 0.1)
                else:
                    return
                print(
                    f"\rCommand vx={command_state['vx']:+.1f} m/s, "
                    f"vy={command_state['vy']:+.1f} m/s, "
                    f"yaw={command_state['wz']:+.1f} rad/s   ",
                    end="",
                    flush=True,
                )

        def _execute_step(self) -> bool:
            if command_state["mode"] == "keyboard":
                term = self.env.unwrapped.command_manager.get_term("twist")
                term.vel_command_b[:, 0].fill_(command_state["vx"])
                term.vel_command_b[:, 1].fill_(command_state["vy"])
                term.vel_command_b[:, 2].fill_(command_state["wz"])
                term.is_heading_env[:] = False
                term.is_standing_env[:] = False
                term.is_world_env[:] = False
            return super()._execute_step()

    # run_play constructs this class when native viewing is selected.
    mjlab_play.NativeMujocoViewer = KeyboardCommandViewer

    print(f"Running simulation for {args.task} with agent='{agent}'")
    print("\nKeyboard controls: W/S forward/back, A/D strafe, Q/E turn, X stop, M toggle keyboard/automatic commands.")
    if agent != "trained":
        print("Note: keyboard commands move the policy target; use a trained checkpoint for learned walking.")
    run_play(args.task, cfg)


if __name__ == "__main__":
    main()
