"""Robot configuration for the custom 12-DOF Robodog quadruped."""

from pathlib import Path
import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

URDF_PATH: Path = Path(__file__).resolve().parent / "urdf" / "robot.urdf"
assert URDF_PATH.exists(), f"URDF file not found at {URDF_PATH}"


def get_robodog_spec() -> mujoco.MjSpec:
    """Load URDF and equip with IMU sensor sites and foot contact sites."""
    spec = mujoco.MjSpec.from_file(str(URDF_PATH))
    base = spec.body("base_link")

    # Add a freejoint so the robot is floating-base (falls under gravity,
    # can make ground contact, and can walk). Without this, base_link is
    # welded to the world origin and the robot just hovers in the air.
    fj = base.add_freejoint()
    fj.name = "base_freejoint"

    # Add IMU site at center of chassis
    imu = base.add_site()
    imu.name = "imu"
    imu.pos = (0, 0, 0)

    # Add Gyro sensor (angular velocity)
    gyro = spec.add_sensor()
    gyro.name = "imu_ang_vel"
    gyro.type = mujoco.mjtSensor.mjSENS_GYRO
    gyro.objtype = mujoco.mjtObj.mjOBJ_SITE
    gyro.objname = "imu"

    # Add Velocimeter sensor (linear velocity)
    vel = spec.add_sensor()
    vel.name = "imu_lin_vel"
    vel.type = mujoco.mjtSensor.mjSENS_VELOCIMETER
    vel.objtype = mujoco.mjtObj.mjOBJ_SITE
    vel.objname = "imu"

    # Add foot contact and tracking sites
    for leg in ("FL", "FR", "RL", "RR"):
        calf = spec.body(f"{leg}_calf")
        site = calf.add_site()
        site.name = leg
        site.pos = (0, 0, -0.213)
        site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        site.size = (0.025, 0, 0)

    return spec


# Go2-convention initial standing pose.
# Base height puts the 0.025 m foot spheres on the ground at thigh=0.8, calf=-1.6.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.322),
    joint_pos={
        ".*hip_joint": 0.0,
        ".*thigh_joint": 0.80,
        ".*calf_joint": -1.60,
    },
    joint_vel={".*": 0.0},
)

# Actuators (PD gains tuned for 12.8kg quadruped dynamics)
ROBODOG_HIP_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
    stiffness=80.0,
    damping=4.0,
    effort_limit=23.7,
)

ROBODOG_KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_calf_joint",),
    stiffness=100.0,
    damping=5.0,
    effort_limit=45.43,
)

ROBODOG_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(ROBODOG_HIP_ACTUATOR, ROBODOG_KNEE_ACTUATOR),
    soft_joint_pos_limit_factor=0.9,
)

# Action scaling for RL policy outputs: 0.25 * effort / stiffness
ROBODOG_ACTION_SCALE: dict[str, float] = {
    ".*_hip_joint": 0.25 * 23.7 / 80.0,    # ~0.074 rad
    ".*_thigh_joint": 0.25 * 23.7 / 80.0,  # ~0.074 rad
    ".*_calf_joint": 0.25 * 45.43 / 100.0, # ~0.114 rad
}


def get_robodog_robot_cfg() -> EntityCfg:
    """Get fresh Robodog Entity configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_robodog_spec,
        articulation=ROBODOG_ARTICULATION,
    )
