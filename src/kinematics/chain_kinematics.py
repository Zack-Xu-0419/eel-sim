"""Shared CAD-chain positioning; no actuator or pump imports.

Legacy callers retain full forward refresh. Angle playback explicitly selects
kinematics_only=True, which never evaluates forces or integrates dynamics.
"""
import math
from pathlib import Path
import re
import mujoco
import numpy as np

class AngleLimitError(ValueError):
    """Requested hinge angle exceeds its configured limit."""

def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value

def _vector(values, count, name):
    result = np.asarray(values, dtype=float)
    if result.shape != (count,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} requires {count} finite values')
    return result


class ChainKinematics:
    """Assign relative bends to a serial MJCF chain, with optional prescribed root pose."""

    def __init__(self, model_path, *, allow_root=False, kinematics_only=False):
        self.kinematics_only = kinematics_only
        self.path = Path(model_path).expanduser().resolve()
        self.model = m = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(m)
        groups = {}
        for joint in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint) or ''
            match = re.fullmatch(r'module_(\d+)_bend_([1-9]\d*)', name)
            if match:
                module, hinge = map(int, match.groups())
                groups.setdefault(module, []).append((hinge, joint, name))
        if not groups or sorted(groups) != list(range(len(groups))):
            raise ValueError('Chain needs consecutive module_0, module_1, ... bending groups')
        self.module_count = len(groups)
        self.joint_groups, group_ids = [], []
        for module in range(self.module_count):
            joints = sorted(groups[module])
            if [j[0] for j in joints] != list(range(1, len(joints)+1)):
                raise ValueError(f'Module {module} hinges must be consecutive from bend_1')
            self.joint_groups.append([j[2] for j in joints])
            group_ids.append(np.array([j[1] for j in joints], dtype=int))
        self.group_joint_ids = tuple(group_ids)
        self.joint_ids = np.concatenate(group_ids)
        self.joint_names = [name for group in self.joint_groups for name in group]
        roots = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.root_joint_id = roots[0] if allow_root and len(roots) == 1 else None
        extra = self.root_joint_id is not None
        if (m.nq != len(self.joint_ids)+7*extra or m.nv != len(self.joint_ids)+6*extra
                or m.njnt != len(self.joint_ids)+extra):
            raise ValueError('Chain must contain only bending hinges and an optional permitted root free joint')
        if extra and (m.joint(self.root_joint_id).name != 'eel_root'
                      or m.jnt_bodyid[self.root_joint_id] != m.body('fixture').id):
            raise ValueError('Root free joint must be eel_root on fixture')
        if (np.any(m.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE)
                or not np.allclose(m.jnt_axis[self.joint_ids], [0, 0, 1])):
            raise ValueError('All module bends must be positive local-z hinges')
        self.qpos_addresses = m.jnt_qposadr[self.joint_ids].copy()
        self.dof_addresses = m.jnt_dofadr[self.joint_ids].copy()
        if not np.allclose(m.qpos0[self.qpos_addresses], 0):
            raise ValueError('Balanced chain requires zero reference hinge angles')
        self.base_site_ids, self.tip_site_ids, self.module_colors = [], [], []
        previous_end = None
        try:
            for module, ids in enumerate(group_ids):
                first = m.body(f'module_{module}_slice_0')
                if first.jntnum[0] != 0:
                    raise ValueError('First slice of each module must be welded to its parent')
                if previous_end is not None and first.parentid[0] != previous_end:
                    raise ValueError('Modules must be connected end-to-end in a serial chain')
                for i, joint in enumerate(ids, start=1):
                    body = m.body(f'module_{module}_slice_{i}')
                    if (body.id != m.jnt_bodyid[joint]
                            or body.parentid[0] != m.body(f'module_{module}_slice_{i-1}').id):
                        raise ValueError('Internal hinges must join consecutive module slices')
                previous_end = m.body(f'module_{module}_slice_{len(ids)}').id
                self.base_site_ids.append(m.site(f'module_{module}_section_base').id)
                self.tip_site_ids.append(m.site(f'module_{module}_section_tip').id)
                geom = m.geom(f'module_{module}_slice_visual_0').id
                self.module_colors.append(tuple(float(c) for c in m.geom_rgba[geom]))
        except KeyError as exc:
            raise ValueError('Chain is missing its named module bodies, sites, or mesh geoms') from exc
        self.apply(np.zeros(self.module_count), 0.)
        self.module_lengths_m = tuple(float(np.linalg.norm(self.data.site_xpos[b]-self.data.site_xpos[a]))
                                      for a, b in zip(self.base_site_ids, self.tip_site_ids))
        self.total_length_m = sum(self.module_lengths_m)
        if self.total_length_m <= 0:
            raise ValueError('Chain must have positive module lengths')

    def angles(self, total_angles_rad):
        """Flatten per-module relative bends into equal local hinge increments."""
        totals = _vector(total_angles_rad, self.module_count, 'Module angles')
        result = np.concatenate([np.full(len(ids), theta/len(ids))
                                 for theta, ids in zip(totals, self.group_joint_ids)])
        for joint, name, angle in zip(self.joint_ids, self.joint_names, result):
            low, high = self.model.jnt_range[joint]
            if self.model.jnt_limited[joint] and not low-1e-12 <= angle <= high+1e-12:
                raise AngleLimitError(f'{name}: requested {math.degrees(angle):.6g} deg, '
                                      f'outside [{math.degrees(low):.6g}, {math.degrees(high):.6g}] deg; '
                                      'angle and transferred volume are not clamped')
        return result

    def apply(self, total_angles_rad, time_s, angle_velocities_rad_s=None,
              root_pose=None, root_velocity=None):
        """Refresh the connected pose; kinematics_only skips force evaluation entirely."""
        angles = self.angles(total_angles_rad)
        velocities = (np.zeros(self.module_count) if angle_velocities_rad_s is None
                      else _vector(angle_velocities_rad_s, self.module_count, 'Module angular velocities'))
        rates = np.concatenate([np.full(len(ids), rate/len(ids))
                                for rate, ids in zip(velocities, self.group_joint_ids)])
        clock = finite(time_s, 'time_s')
        if root_pose is not None:
            pose = _vector(root_pose, 7, 'Root pose')
            velocity = _vector(np.zeros(6) if root_velocity is None else root_velocity, 6, 'Root velocity')
            if self.root_joint_id is None:
                raise ValueError('Overall pose requires a chain generated with root_freejoint=True')
            qa, va = (self.model.jnt_qposadr[self.root_joint_id], self.model.jnt_dofadr[self.root_joint_id])
            self.data.qpos[qa:qa+7] = pose
            self.data.qvel[va:va+6] = velocity
        self.data.qpos[self.qpos_addresses] = angles
        self.data.qvel[self.dof_addresses] = rates
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.data.time = clock
        self.refresh()

    def refresh(self):
        if self.kinematics_only:
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_comVel(self.model, self.data)
            mujoco.mj_camlight(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)

    def centerline(self):
        """Current world positions of module bases, slice centers and final tip."""
        points = [self.data.site_xpos[self.base_site_ids[0]].copy()]
        for module, ids in enumerate(self.group_joint_ids):
            for i in range(len(ids)+1):
                points.append(self.data.site(f'module_{module}_marker_{i}').xpos.copy())
            points.append(self.data.site_xpos[self.tip_site_ids[module]].copy())
        return np.array(points)
