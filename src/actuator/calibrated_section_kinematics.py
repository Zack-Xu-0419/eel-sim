"""Geometry and prescribed-pose playback, independent of pressure relations.

Mirrors the established section mapping; the existing pressure runner is unchanged.
"""
import math
from pathlib import Path
import re
import time
import mujoco
import numpy as np
from src.actuator.calibrated_actuator_model import _finite as finite

from src.kinematics.chain_kinematics import AngleLimitError


class SectionKinematics:
    """Map one total bending angle to a compatible fixed-base slice chain.

    Kinematics means calculating positions/orientations from supplied angles.
    This class does not calculate bending from forces or evolve fluid volume.
    """

    def __init__(self, model_path):
        """Load the generated MJCF XML and check its named slice/hinge layout."""
        self.path = Path(model_path).expanduser().resolve()
        # MjModel stores compiled geometry/joints/assets. MjData stores the
        # changing pose, velocities and derived quantities for that model.
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        joints = []
        # Discover bend_1, bend_2, ... from their names rather than assuming a
        # particular XML ordering or number of slices. Numeric sort is needed
        # because alphabetical ordering would put bend_10 before bend_2.
        for joint_id in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            match = re.fullmatch(r"bend_([1-9][0-9]*)", name)
            if match:
                joints.append((int(match[1]), joint_id, name))
        joints.sort()
        if not joints:
            raise ValueError("Model needs at least two slices and one named bend_1 hinge")
        if [i for i, _, _ in joints] != list(range(1, len(joints)+1)):
            raise ValueError("Bending joint names must be consecutive: bend_1, bend_2, ...")
        if m.nq != len(joints) or m.nv != len(joints):
            raise ValueError("Model must be a fixed-base section with no additional degrees of freedom")
        self.joint_ids = np.array([j for _, j, _ in joints], dtype=int)
        self.joint_names = [name for _, _, name in joints]
        # MuJoCo joint IDs are not generally the same as qpos/qvel addresses.
        # Resolve both explicitly so assignment targets the intended hinges.
        self.qpos_addresses = m.jnt_qposadr[self.joint_ids].copy()
        self.dof_addresses = m.jnt_dofadr[self.joint_ids].copy()
        try:
            # The total-angle mapping assumes one fixed first slice and one
            # hinge per subsequent slice, connected in a single serial chain.
            if m.body("slice_0").jntnum[0] != 0:
                raise ValueError("slice_0 must be fixed")
            for i, j, _ in joints:
                body = m.body(f"slice_{i}")
                if (body.id != m.jnt_bodyid[j]
                        or body.parentid[0] != m.body(f"slice_{i-1}").id):
                    raise ValueError("Bending hinges must connect consecutive slices in one chain")
        except KeyError as exc:
            raise ValueError("Model is missing the expected slice_0, slice_1, ... bodies") from exc
        if np.any(m.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Every bending joint must be a hinge")
        if not np.allclose(m.jnt_axis[self.joint_ids], [0, 0, 1]):
            raise ValueError("Bending hinges must rotate about positive local z")
        if not np.allclose(m.qpos0[self.qpos_addresses], 0):
            raise ValueError("This balanced-reference mapping requires zero reference joint angles")
        self.apply(0., 0.)

    def angles(self, total_angle_rad):
        """Return per-hinge radians without changing the displayed pose.

        n slices have n-1 hinges. Equal angular increments approximate a
        constant-curvature bend; they are not a solution of elastic mechanics.
        """
        theta = finite(total_angle_rad, "total angle")
        angles = np.full(len(self.joint_ids), theta/len(self.joint_ids))
        m = self.model
        for joint, name, angle in zip(self.joint_ids, self.joint_names, angles):
            low, high = m.jnt_range[joint]
            if m.jnt_limited[joint] and not low-1e-12 <= angle <= high+1e-12:
                raise AngleLimitError(
                    f"Requested total bend {math.degrees(theta):.6g} deg puts {name} at "
                    f"{math.degrees(angle):.6g} deg, outside [{math.degrees(low):.6g}, "
                    f"{math.degrees(high):.6g}] deg. Simulation stopped; angle and volume were not clamped.")
        return angles

    def apply(self, total_angle_rad, time_s, angle_velocity_rad_s=0.):
        """Assign a pose/rate, then refresh derived MuJoCo geometry once."""
        angles = self.angles(total_angle_rad)  # Check before changing the displayed pose.
        velocity = finite(angle_velocity_rad_s, "angle velocity") / len(angles)
        clock = finite(time_s, "time_s")
        self.data.qpos[self.qpos_addresses] = angles
        self.data.qvel[self.dof_addresses] = velocity
        # Clear control/external-force slots because motion is prescribed here.
        # There is no force-based actuation or water-force feedback in this path.
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.data.time = clock
        # mj_forward updates geometry/derived values at the assigned coordinates.
        # mj_step would integrate dynamics and is intentionally not used.
        mujoco.mj_forward(self.model, self.data)



def playback(section, samples):
    """Display existing samples once at real time; never rerun the equations.

    macOS requires the mjpython launcher for the native viewer. Video export
    uses its own uniform frame times; this live viewer selects solver samples
    at roughly 60 Hz to avoid updating the UI on every 0.001-second step.
    """
    import mujoco.viewer
    frames = [samples[0]]
    for sample in samples[1:]:
        if sample.time_s-frames[-1].time_s >= 1/60-1e-12:
            frames.append(sample)
    if frames[-1] is not samples[-1]:
        frames.append(samples[-1])
    section.apply(frames[0].angle_rad, frames[0].time_s)
    with mujoco.viewer.launch_passive(section.model, section.data) as viewer:
        viewer.cam.lookat[:] = [.06, 0, 0]
        viewer.cam.distance = .36
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -18
        start = time.monotonic()
        previous = frames[0]
        for frame in frames:
            # Match recorded simulation time to elapsed wall-clock time. The
            # short sleeps also allow the user to close the viewer mid-run.
            while viewer.is_running():
                remaining = frame.time_s-frames[0].time_s-(time.monotonic()-start)
                if remaining <= 0:
                    break
                time.sleep(min(remaining, .01))
            if not viewer.is_running():
                break
            elapsed = frame.time_s-previous.time_s
            velocity = (frame.angle_rad-previous.angle_rad)/elapsed if elapsed > 0 else 0.
            # The viewer may read the same data from another thread; lock while
            # changing its pose, then synchronize the visible scene.
            with viewer.lock():
                section.apply(frame.angle_rad, frame.time_s, velocity)
            viewer.sync()
            previous = frame
