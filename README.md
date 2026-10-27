# Quadruped Walking Robot

A 12 degree of freedom quadruped simulation and reinforcement learning project. It defines a custom MJLab velocity tracking task, trains a PPO walking policy, and provides scripts to inspect the robot, run a hand-coded gait, and play a trained policy.

This repository contains the quadruped project only. **MJLab is a separate dependency** and must be cloned next to this repository because the scripts look for a sibling `mjlab/src` checkout.

## Requirements

- Git
- Linux x86_64 recommended for training
- Python 3.10–3.13
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python environment and dependency management
- NVIDIA GPU and a compatible driver for GPU-accelerated training (CUDA 12.4+ recommended by MJLab). CPU mode can be used for small checks, but training will be much slower.

MJLab, MuJoCo, MuJoCo Warp, PyTorch, and the remaining Python packages are installed by `uv` in the MJLab checkout below; there is no need to install each one separately.

## Download and install

Install Git, Python, and `uv` for your operating system first. Then clone both repositories into the same parent directory and install MJLab's dependencies:

```bash
mkdir -p ~/robotics
cd ~/robotics
git clone https://github.com/mujocolab/mjlab.git
git clone https://github.com/DINODAVIS369/quatrain-walking-robot.git
cd mjlab
uv sync
```

The expected layout is:

```text
~/robotics/
├── mjlab/                   # MJLab framework and its uv environment
└── quatrain-walking-robot/  # This quadruped project
```

`uv sync` creates the environment in `mjlab/.venv` and installs MJLab plus its dependencies. The project scripts detect the sibling `mjlab/src` source checkout. Run the project commands from the quadruped folder and tell `uv` to use the MJLab environment:

```bash
cd ~/robotics/quatrain-walking-robot
uv run --project ../mjlab python inspect_robot.py
```

## Use the project

### Inspect the robot

```bash
uv run --project ../mjlab python inspect_robot.py
uv run --project ../mjlab python check_model.py --view
```

### Try the hand-coded gait

```bash
# Headless gait test
uv run --project ../mjlab python walk_test.py --duration 5.0

# Open the viewer
uv run --project ../mjlab python walk_test.py --view
```

### Play the included trained policy

The `logs/quatrain_walking_robot/robodog_velocity/model_4999.pt` checkpoint is included in this repository.

```bash
uv run --project ../mjlab python play.py \
  --checkpoint logs/quatrain_walking_robot/robodog_velocity/model_4999.pt
```

The native viewer starts in keyboard command mode:

| Key | Velocity command |
| --- | --- |
| `W` / `S` | Forward / backward |
| `A` / `D` | Strafe left / right |
| `Q` / `E` | Turn left / right |
| `X` | Stop |
| `M` | Toggle keyboard / automatic commands |

For automatic command mode at startup:

```bash
uv run --project ../mjlab python play.py \
  --checkpoint logs/quatrain_walking_robot/robodog_velocity/model_4999.pt \
  --command-mode automatic
```

### Train a policy

```bash
uv run --project ../mjlab python train.py --num-envs 256 --max-iterations 5000
```

Training needs an NVIDIA GPU for practical performance. Reduce `--num-envs` if GPU memory is limited. Add `--cpu` to force CPU mode for a small run or configuration check.

## Main files

- `urdf/robot.urdf`: active quadruped model
- `robot_cfg.py`: robot, sensor, and actuator configuration
- `env_cfg.py` and `configs/`: MJLab task and environment configuration
- `train.py`: PPO training entry point
- `play.py`: trained policy playback and velocity controls
- `walk_test.py`: hand-coded gait and standing tests
- `inspect_robot.py` and `check_model.py`: model inspection tools
- `GUIDE.md`: longer explanation of the model, task, training loop, and limitations

## Hardware note

This project trains and evaluates a simulated robot policy. The included `.pt` checkpoint is not firmware and cannot be flashed directly to an ESP32 or used to drive motors by itself. Real hardware needs a compatible inference runtime, sensor and state-estimation code, motor drivers, and a safe control interface.
