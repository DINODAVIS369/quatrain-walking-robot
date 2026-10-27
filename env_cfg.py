"""Custom Environment and RL Runner configurations for Robodog quadruped."""

import math
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from configs.velocity_env_cfg import make_velocity_env_cfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from robot_cfg import ROBODOG_ACTION_SCALE, get_robodog_robot_cfg


def robodog_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Robodog flat terrain velocity tracking configuration."""
    cfg = make_velocity_env_cfg()

    # Simulation solver settings
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Attach our custom 12-DOF robot entity
    cfg.scene.entities = {"robot": get_robodog_robot_cfg()}

    # Flat ground terrain
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Configure foot contact sensors
    foot_geom_names = (
        "FL_foot_collision",
        "FR_foot_collision",
        "RL_foot_collision",
        "RR_foot_collision",
    )
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=foot_geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Clean up sensors for flat ground
    remove_sensors = {
        "terrain_scan",
        "foot_height_scan",
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
    }
    existing_sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
    )
    cfg.scene.sensors = existing_sensors + (feet_ground_cfg,)

    # Observations: remove height scan from flat terrain
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("foot_height", None)

    # Events: configure body and feet geom names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names

    # Rewards: pose tracking for standing/walking
    cfg.rewards["pose"].params["std_standing"] = {
        r".*(hip|thigh)_joint.*": 0.05,
        r".*calf_joint.*": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*(hip|thigh)_joint.*": 0.3,
        r".*calf_joint.*": 0.6,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*(hip|thigh)_joint.*": 0.3,
        r".*calf_joint.*": 0.6,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

    # Remove height-dependent rewards on flat ground
    cfg.rewards.pop("foot_clearance", None)
    cfg.rewards.pop("foot_swing_height", None)
    cfg.rewards.pop("foot_slip", None)

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0

    # Curriculum & terminations
    cfg.curriculum.pop("terrain_levels", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations.pop("illegal_contact", None)

    # Set action scaling for Robodog actuators
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ROBODOG_ACTION_SCALE

    # Termination: fell over if base tilts past 70 degrees
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )

    # Tracking camera
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.8
    cfg.viewer.elevation = -12.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        # Disable random pushes in play mode so user can freely control the robot
        cfg.events.pop("push_robot", None)
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
        twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)

    return cfg


def robodog_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RSL-RL PPO configuration for Robodog."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="robodog_velocity",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=5000,
    )


# Register custom Robodog task
TASK_ID = "Mjlab-Velocity-Flat-Robodog"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=robodog_flat_env_cfg(),
    play_env_cfg=robodog_flat_env_cfg(play=True),
    rl_cfg=robodog_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
