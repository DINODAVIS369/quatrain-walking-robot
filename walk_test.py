"""Test quadruped walking without RL using periodic joint position trajectory (Trot Gait).

This script demonstrates classical robotic gait generation:
1. Implements a PD joint controller: tau = Kp * (q_target - q) - Kd * q_vel
2. Generates an alternating diagonal Trot Gait (Pair 1: FL+RR, Pair 2: FR+RL)
3. Computes forward displacement, velocity, and dynamic stability
4. Includes an interactive 3D browser viewer (--view) to watch the robot walk!
"""

import argparse
import time
from pathlib import Path
import mujoco
import numpy as np


def build_walking_scene(urdf_path: str = "urdf/robot.urdf") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the URDF with floating base and ground plane for walking."""
    spec = mujoco.MjSpec.from_file(urdf_path)

    # Floating base on chassis
    base = spec.body("base_link")
    base.add_freejoint()

    # Ground floor with realistic contact friction
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = (20, 20, 0.05)
    floor.condim = 3
    floor.friction = (1.2, 0.005, 0.0001)

    # MuJoCo suppresses automatic contacts between parent-child bodies.
    # Add explicit knee pairs so the calf cannot pass through its thigh.
    for leg in ("FL", "FR", "RL", "RR"):
        spec.add_pair(
            name=f"{leg}_knee_contact",
            geomname1=f"{leg}_thigh_collision",
            geomname2=f"{leg}_calf_collision",
            condim=3,
        )

    # Add torque actuators for all 12 revolute joints
    for jnt in spec.joints:
        if jnt.type == mujoco.mjtJoint.mjJNT_HINGE:
            act = spec.add_actuator()
            act.name = f"{jnt.name}_actuator"
            act.target = jnt.name
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            effort = 35.55 if ("calf" in jnt.name or "knee" in jnt.name) else 23.70
            act.ctrlrange = (-effort, effort)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


class TrotGaitController:
    """Procedural diagonal Trot Gait generator for quadruped robots."""

    def __init__(
        self,
        freq: float = 1.0,
        step_length: float = 0.12,
        step_height: float = 0.05,
        stand_thigh: float = 0.80,
        stand_calf: float = -1.60,
    ):
        self.freq = freq
        self.step_length = step_length
        self.step_height = step_height
        self.stand_thigh = stand_thigh
        self.stand_calf = stand_calf

        # PD gains
        # Hip roll, Thigh pitch, Calf pitch (repeated for 4 legs)
        self.Kp = np.array([80.0, 160.0, 160.0] * 4)
        self.Kd = np.array([4.0, 6.0, 6.0] * 4)

    def compute_target_angles(self, t: float) -> np.ndarray:
        """Drive stride mainly with the thigh and use modest knee flex for swing."""
        length = 0.213
        nominal_drop = length * (
            np.cos(self.stand_thigh)
            + np.cos(self.stand_thigh + self.stand_calf)
        )
        # Convert the requested foot travel into a thigh angle change around
        # the nominal pose. Clamp it to keep the leg away from its limits.
        x_per_thigh_rad = max(nominal_drop, 0.1)
        thigh_amplitude = min(0.35, 0.5 * self.step_length / x_per_thigh_rad)
        knee_lift = min(0.42, 1.6 * self.step_height / length)
        hip_swing = 0.12

        phi_1 = (self.freq * t) % 1.0
        phi_2 = (phi_1 + 0.5) % 1.0

        def leg_targets(phase: float, side: float) -> tuple[float, float, float]:
            wave = np.sin(2.0 * np.pi * phase)
            swing = max(0.0, wave)
            thigh = self.stand_thigh - thigh_amplitude * wave
            # Bend the knee and move the hip outward during swing for clearance.
            calf = self.stand_calf - knee_lift * swing
            hip = side * hip_swing * swing
            return hip, thigh, calf

        fl = leg_targets(phi_1, 1.0)
        fr = leg_targets(phi_2, -1.0)
        rl = leg_targets(phi_2, 1.0)
        rr = leg_targets(phi_1, -1.0)

        # Target order is hip, thigh, calf for FL, FR, RL, then RR.
        return np.array([*fl, *fr, *rl, *rr])

    def compute_standing_torques(self, data: mujoco.MjData) -> np.ndarray:
        """Hold the nominal four-leg standing pose with the PD controller."""
        q_stand = np.array([0.0, self.stand_thigh, self.stand_calf] * 4)
        q_current = data.qpos[7:19]
        v_current = data.qvel[6:18]
        tau = self.Kp * (q_stand - q_current) - self.Kd * v_current
        return np.clip(tau, -35.55, 35.55)

    def compute_knee_sweep_targets(self, model: mujoco.MjModel, t: float) -> np.ndarray:
        """Return a kinematic pose with only the knees sweeping through their limits."""
        q_des = np.array([0.0, self.stand_thigh, self.stand_calf] * 4)
        phase = (self.freq * t) % 1.0
        ramp = 1.0 - abs(2.0 * phase - 1.0)  # lower -> upper -> lower
        for jnt_id in range(1, model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if "calf" in name:
                q_index = model.jnt_qposadr[jnt_id] - 7
                lower, upper = model.jnt_range[jnt_id]
                q_des[q_index] = lower + ramp * (upper - lower)
        return q_des

    def compute_torques(self, model: mujoco.MjModel, data: mujoco.MjData, t: float) -> np.ndarray:
        """Compute PD joint motor torques."""
        q_des = self.compute_target_angles(t)
        q_current = data.qpos[7:19]  # 12 joint positions (after 7 freejoint qpos)
        v_current = data.qvel[6:18]  # 12 joint velocities (after 6 freejoint qvel)

        tau = self.Kp * (q_des - q_current) - self.Kd * v_current
        return np.clip(tau, -35.55, 35.55)


def reset_to_standing(model: mujoco.MjModel, data: mujoco.MjData, controller: TrotGaitController):
    """Place robot in nominal standing posture above floor."""
    leg_length = 0.213
    foot_drop = leg_length * (
        np.cos(controller.stand_thigh)
        + np.cos(controller.stand_thigh + controller.stand_calf)
    )
    # foot_drop locates the foot center; only add the foot radius to touch ground.
    data.qpos[0:3] = [0.0, 0.0, foot_drop + 0.025]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Upright quaternion
    data.qvel[:] = 0.0

    # Set standing angles
    for j in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        q_idx = model.jnt_qposadr[j]
        if "thigh" in name:
            data.qpos[q_idx] = controller.stand_thigh
        elif "calf" in name:
            data.qpos[q_idx] = controller.stand_calf
        elif "hip" in name:
            data.qpos[q_idx] = 0.0

    # Settle for 0.4 seconds
    for _ in range(200):
        q = data.qpos[7:19]
        v = data.qvel[6:18]
        q_stand = np.array([0.0, controller.stand_thigh, controller.stand_calf] * 4)
        data.ctrl[:] = np.clip(controller.Kp * (q_stand - q) - controller.Kd * v, -35.55, 35.55)
        mujoco.mj_step(model, data)


def main():
    parser = argparse.ArgumentParser(description="Test quadruped walking without RL using Trot Gait.")
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration in seconds")
    parser.add_argument("--freq", type=float, default=1.0, help="Step frequency in Hz (cycles/sec)")
    parser.add_argument("--step-length", type=float, default=0.12, help="Foot stride length in meters")
    parser.add_argument("--step-height", type=float, default=0.05, help="Foot lift height in meters")
    parser.add_argument("--view", action="store_true", help="Launch interactive 3D browser viewer")
    parser.add_argument("--stand", action="store_true", help="Hold the standing pose instead of walking")
    parser.add_argument("--knees-only", action="store_true", help="Hold hips/thighs still and sweep knees across their joint limits")
    args = parser.parse_args()

    print("=" * 65)
    print("      QUADRUPED OPEN-LOOP WALKING TEST (NO RL REQUIRED)      ")
    print("=" * 65)
    print(f"  Mode:              {'Knees only' if args.knees_only else ('Standing hold' if args.stand else 'Diagonal Trot (FL+RR / FR+RL)')}")
    print(f"  Step Frequency:    {args.freq} Hz ({args.freq * 2:.1f} steps/second)")
    print(f"  Stride Length:     {args.step_length} m")
    print(f"  Foot Lift Height:  {args.step_height} m")
    print("=" * 65)

    model, data = build_walking_scene("urdf/robot.urdf")
    controller = TrotGaitController(
        freq=args.freq,
        step_length=args.step_length,
        step_height=args.step_height,
        stand_thigh=0.80,
        stand_calf=-1.60,
    )

    reset_to_standing(model, data, controller)

    if args.view:
        from mjviser import Viewer

        print("\n[INFO] Launching 3D Viser Viewer...")
        print(">> Open your browser at: http://localhost:8080 <<\n")

        # Custom step function that computes PD walking gait on every frame
        sim_time = [0.0]

        def step_with_gait(m: mujoco.MjModel, d: mujoco.MjData):
            if args.knees_only:
                d.qpos[7:19] = controller.compute_knee_sweep_targets(m, sim_time[0])
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
            else:
                d.ctrl[:] = (controller.compute_standing_torques(d) if args.stand
                             else controller.compute_torques(m, d, sim_time[0]))
                mujoco.mj_step(m, d)
            sim_time[0] += m.opt.timestep

        viewer = Viewer(model=model, data=data, step_fn=step_with_gait)
        viewer.run()
        return

    # Headless simulation mode
    dt = model.opt.timestep
    total_steps = int(args.duration / dt)

    start_x = data.qpos[0]
    root_pose = data.qpos[:7].copy()
    activity = "knee limit sweep" if args.knees_only else ("standing hold" if args.stand else "forward walking")
    print(f"\nSimulating {args.duration:.1f} seconds of {activity} ({total_steps} steps)...")

    for step in range(total_steps):
        t = step * dt
        if args.knees_only:
            q_des = controller.compute_knee_sweep_targets(model, t)
            q = data.qpos[7:19]
            v = data.qvel[6:18]
            data.ctrl[:] = np.clip(controller.Kp * (q_des - q) - controller.Kd * v, -35.55, 35.55)
            mujoco.mj_step(model, data)
            data.qpos[:7] = root_pose
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)
        else:
            data.ctrl[:] = (controller.compute_standing_torques(data) if args.stand
                            else controller.compute_torques(model, data, t))
            mujoco.mj_step(model, data)

        # Print progress every 1 second
        if step % int(1.0 / dt) == 0 and step > 0:
            if not args.knees_only:
                current_x = data.qpos[0]
                current_z = data.qpos[2]
                print(f"  t = {step*dt:4.1f}s | Forward Distance: {current_x - start_x:+.3f}m | Body Height: {current_z:.3f}m")

    end_x = data.qpos[0]
    final_z = data.qpos[2]
    distance_walked = end_x - start_x
    speed = distance_walked / args.duration

    print("\n" + "=" * 65)
    if args.knees_only:
        print("                    KNEE SWEEP RESULTS                       ")
        for leg in ("FL", "FR", "RL", "RR"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_calf_joint")
            q = data.qpos[model.jnt_qposadr[jid]]
            lo, hi = model.jnt_range[jid]
            print(f"  {leg}: knee angle {q:+.3f} rad (limits {lo:+.3f} to {hi:+.3f})")
        print("  Hips/thighs held at standing angles; torso held fixed; knee contacts enabled.")
    else:
        print("                     WALKING TEST RESULTS                     ")
        print(f"  Total Distance Walked:  {distance_walked:+.3f} meters")
        print(f"  Average Walking Speed:  {speed:+.3f} m/s ({speed * 3.6:.2f} km/h)")
        print(f"  Final Body Height:      {final_z:.3f} m (Upright: {'YES' if final_z > 0.18 else 'NO'})")
        print(f"  Completed Strides:      ~{int(args.duration * args.freq * 2)} steps")
    print("=" * 65)

    if not args.knees_only:
        if distance_walked > 0.5:
            print("\n>> SUCCESS: The robot walks forward stably without RL!")
        else:
            print("\n>> NOTE: Adjust --step-length or --freq to tune forward speed.")


if __name__ == "__main__":
    main()
