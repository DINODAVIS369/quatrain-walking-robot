"""Train the quadruped walking robot using RL (PPO)."""

import argparse
import sys
from pathlib import Path

# Add mjlab src to path if needed
mjlab_src = Path(__file__).resolve().parent.parent / "mjlab" / "src"
if mjlab_src.exists():
    sys.path.insert(0, str(mjlab_src))

from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import os

os.environ.setdefault("WANDB_MODE", "disabled")

# Register custom Robodog task
import env_cfg


def main():
    parser = argparse.ArgumentParser(description="Train quadruped walking robot.")
    parser.add_argument(
        "--task",
        type=str,
        default="Mjlab-Velocity-Flat-Robodog",
        help="Task ID to train (e.g. Mjlab-Velocity-Flat-Robodog)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1024,
        help="Number of parallel environments to simulate on GPU",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5000,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/quatrain_walking_robot",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint in --log-dir",
    )
    parser.add_argument(
        "--load-run",
        type=str,
        default=".*",
        help=(
            "Regex to match the run folder to resume from "
            "(default '.*' picks the most recent run)"
        ),
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="model_.*.pt",
        help=(
            "Regex to match the checkpoint file to load "
            "(default 'model_.*.pt' picks the latest)"
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force training on CPU even if a GPU is available",
    )
    args = parser.parse_args()

    # Auto-select GPU if available, unless --cpu is set
    if not args.cpu:
        try:
            import torch
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                print("GPU detected: training on cuda:0")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                print("No GPU found: training on CPU")
        except ImportError:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("--cpu flag set: training on CPU")

    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs

    rl_cfg = load_rl_cfg(args.task)
    rl_cfg.max_iterations = args.max_iterations
    rl_cfg.resume = args.resume
    rl_cfg.load_run = args.load_run
    rl_cfg.load_checkpoint = args.load_checkpoint

    cfg = TrainConfig(
        env=env_cfg,
        agent=rl_cfg,
        log_root=args.log_dir,
    )

    log_path = Path(args.log_dir) / rl_cfg.experiment_name
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting training task: {args.task}")
    print(f"Num environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    if args.resume:
        print(f"Resuming from run='{args.load_run}', checkpoint='{args.load_checkpoint}'")
    print(f"Logging to: {log_path}")

    run_train(args.task, cfg, log_path)


if __name__ == "__main__":
    main()
