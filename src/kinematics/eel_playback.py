"""Time-indexed measured-angle playback, independent of actuator dynamics.

Playback uses SI units. The fit helper accepts measured angles in degrees.
Quaternions are w,x,y,z.
Snapshots own read-only copies, never views into mutable MuJoCo data.
"""
from dataclasses import dataclass
import csv
from pathlib import Path
import mujoco
import numpy as np
from src.kinematics.chain_kinematics import ChainKinematics, finite


def frozen(values):
    out = np.array(values, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def _times(times):
    out = frozen(times)
    if out.ndim != 1 or len(out) < 2 or not np.all(np.isfinite(out)) or np.any(np.diff(out) <= 0):
        raise ValueError('Require at least two finite, strictly increasing timestamps')
    return out


def _array(values, shape, label):
    out = frozen(values)
    if out.shape != shape or not np.all(np.isfinite(out)):
        raise ValueError(f'{label} must have shape {shape} and finite values')
    return out


def _interval(times, clock):
    clock = finite(clock, 'time_s')
    if not times[0] <= clock <= times[-1]:
        raise ValueError('Query time outside trajectory range')
    i = min(int(np.searchsorted(times, clock, side='right'))-1, len(times)-2)
    return i, (clock-times[i])/(times[i+1]-times[i])


@dataclass(frozen=True)
class AngleSample:
    time_s: float
    bends_rad: np.ndarray
    bend_rates_rad_s: np.ndarray
    interpolated_gap: bool


class AngleTrajectory:
    """Linear section bends; right-hand rates at knots, left rate at final time."""
    def __init__(self, times_s, bends_rad, *, gap_intervals=None):
        self.times_s = _times(times_s)
        bends = np.asarray(bends_rad, dtype=float)
        if bends.ndim != 2 or bends.shape[1] < 1:
            raise ValueError('bends_rad must be a time-by-section matrix')
        self.bends_rad = _array(bends, (len(self.times_s), bends.shape[1]), 'bends_rad')
        self.section_count = bends.shape[1]
        self.rates = frozen(np.diff(bends, axis=0)/np.diff(self.times_s)[:, None])
        self.gap_intervals = np.zeros(len(self.times_s)-1, dtype=bool)
        if gap_intervals is not None:
            g = np.asarray(gap_intervals, dtype=bool)
            if g.shape != self.gap_intervals.shape:
                raise ValueError('One gap flag required per interval')
            self.gap_intervals[:] = g
        self.gap_intervals.setflags(write=False)

    def at(self, time_s):
        i, f = _interval(self.times_s, time_s)
        return AngleSample(float(time_s), frozen((1-f)*self.bends_rad[i]+f*self.bends_rad[i+1]),
                           frozen(self.rates[i]), bool(self.gap_intervals[i] and 0 < f < 1))

    @classmethod
    def from_recording_csv(cls, path, *, max_gap_s=.15, allow_camera_angle=False):
        """Load t_s, act1..N_bend_deg; usable=0 and missing bends are excluded.

        Source timestamp order is validated before filtering. No smoothing,
        angle re-zeroing, head-offset estimation, or tail-angle summation.
        Opt in to Time(s), Camera_Angle for a single-actuator recording.
        """
        import re
        max_gap_s = finite(max_gap_s, 'max_gap_s')
        if max_gap_s <= 0:
            raise ValueError('max_gap_s must be positive')
        with Path(path).open(newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            columns = sorted((int(m[1]), key) for key in headers
                             if (m := re.fullmatch(r'act(\d+)_bend_deg', key)))
            if 't_s' in headers and columns and [n for n, _ in columns] == list(range(1, len(columns)+1)):
                time_column, bend_columns = 't_s', [key for _, key in columns]
            elif allow_camera_angle and {'Time(s)', 'Camera_Angle'}.issubset(headers):
                time_column, bend_columns = 'Time(s)', ['Camera_Angle']
            else:
                expected = 't_s and consecutive act1_bend_deg ... actN_bend_deg'
                if allow_camera_angle:
                    expected += ', or Time(s) and Camera_Angle'
                raise ValueError(f'Recording requires {expected}')
            rows = list(reader)
        clocks = _times([float(row[time_column]) for row in rows])
        kept, bends = [], []
        for i, row in enumerate(rows):
            values = [float(row[key]) if (row[key] or '').strip() else np.nan for key in bend_columns]
            if float(row.get('usable', '1')) == 1 and np.all(np.isfinite(values)):
                kept.append(i); bends.append(values)
        if len(kept) < 2:
            raise ValueError('Recording requires at least two usable bend samples')
        intervals = np.diff(clocks[kept])
        if np.any(intervals > max_gap_s+1e-12):
            raise ValueError(f'Recording gap {intervals.max():.6g} s exceeds max_gap_s={max_gap_s:g}')
        result = cls(clocks[kept], np.deg2rad(bends), gap_intervals=np.diff(kept) > 1)
        result.source_path = str(Path(path).resolve())
        result.excluded_rows = len(rows)-len(kept)
        return result


def fit_bends(times_s, measured_bends_deg, *, frequency_hz=.3):
    """Fit a single trace or a time-by-section matrix of measured degrees."""
    bends = np.deg2rad(measured_bends_deg)
    if bends.ndim == 1:
        bends = bends[:, None]
    return HarmonicTrajectory(AngleTrajectory(times_s, bends), frequency_hz=frequency_hz)


class HarmonicTrajectory:
    """Centered one-period mean plus independently phased first and third harmonics."""
    def __init__(self, measured, *, frequency_hz=.3):
        self.frequency_hz = finite(frequency_hz, 'frequency_hz')
        if self.frequency_hz <= 0:
            raise ValueError('frequency_hz must be positive')
        self.section_count = measured.section_count
        self._measured = measured
        clocks = measured.times_s
        self.phase_origin_s = float(clocks[0])
        self._half_period = .5 / self.frequency_hz
        start, end = clocks[0]+self._half_period, clocks[-1]-self._half_period
        full_window = (clocks >= start) & (clocks <= end)
        fit_times = clocks[full_window]
        if len(fit_times) < 4:
            raise ValueError('Harmonic fit needs at least four samples with a full centered period')
        self.times_s = frozen(np.concatenate(([start], fit_times[(fit_times > start) & (fit_times < end)], [end])))

        # Integrate the piecewise-linear measurements, including partial sample intervals.
        areas = .5 * (measured.bends_rad[:-1] + measured.bends_rad[1:]) * np.diff(clocks)[:, None]
        self._cumulative_area = np.vstack((np.zeros(self.section_count), np.cumsum(areas, axis=0)))
        self._gap_starts = clocks[:-1][measured.gap_intervals]
        self._gap_ends = clocks[1:][measured.gap_intervals]
        residual = measured.bends_rad[full_window] - self._baseline(fit_times)
        coefficients, _, rank, _ = np.linalg.lstsq(self._terms(fit_times), residual, rcond=None)
        if rank < 4:
            raise ValueError('Harmonic fit cannot distinguish both phases of the first and third harmonics')
        # Rows are cos(wt), sin(wt), cos(3wt), sin(3wt), in radians per section.
        self.coefficients_rad = frozen(coefficients)

    def _terms(self, time_s):
        phase = 2*np.pi*self.frequency_hz*(time_s-self.phase_origin_s)
        return np.stack((np.cos(phase), np.sin(phase), np.cos(3*phase), np.sin(3*phase)), axis=-1)

    def _linear_sample(self, time_s):
        clocks = self._measured.times_s
        # Roundoff at full-window boundaries can put a window edge just outside the raw data.
        time_s = np.clip(time_s, clocks[0], clocks[-1])
        index = np.searchsorted(clocks, time_s, side='right').clip(1, len(clocks)-1)-1
        elapsed = np.asarray(time_s-clocks[index])[..., None]
        bend = self._measured.bends_rad[index] + elapsed*self._measured.rates[index]
        return index, elapsed, bend

    def _integral_at(self, time_s):
        index, elapsed, bend = self._linear_sample(time_s)
        return self._cumulative_area[index] + .5*elapsed*(self._measured.bends_rad[index] + bend)

    def _baseline(self, time_s):
        return self.frequency_hz*(self._integral_at(time_s+self._half_period)
                                  - self._integral_at(time_s-self._half_period))

    def at(self, time_s):
        time_s = finite(time_s, 'time_s')
        if not self.times_s[0] <= time_s <= self.times_s[-1]:
            raise ValueError('Query time outside fitted trajectory range')
        left, right = time_s-self._half_period, time_s+self._half_period
        terms = self._terms(time_s)
        cos1, sin1, cos3, sin3 = terms
        derivative = 2*np.pi*self.frequency_hz*np.array([-sin1, cos1, -3*sin3, 3*cos3])
        bends = self._baseline(time_s) + np.einsum('i,ij->j', terms, self.coefficients_rad)
        rate = self.frequency_hz*(self._linear_sample(right)[2] - self._linear_sample(left)[2])
        rate += np.einsum('i,ij->j', derivative, self.coefficients_rad)
        gap = self._gap_starts.searchsorted(right, side='left')-1
        return AngleSample(time_s, frozen(bends), frozen(rate), bool(gap >= 0 and self._gap_ends[gap] > left))


def _mul(a, b):
    out = np.empty(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


class PoseTrajectory:
    """First-section base pose: linear world translation and shortest-path SLERP.

    +x points downstream. No pose means a stationary identity base. Root linear
    velocities are world-frame, angular velocities are local-frame (MuJoCo free
    joint convention). Reported object velocities are always world-frame.
    """
    def __init__(self, times_s, positions_m, quaternions_wxyz):
        self.times_s = _times(times_s)
        self.positions_m = _array(positions_m, (len(self.times_s), 3), 'positions_m')
        q = np.array(_array(quaternions_wxyz, (len(self.times_s), 4), 'quaternions_wxyz'))
        norms = np.linalg.norm(q, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError('Orientation quaternion must be nonzero')
        q /= norms[:, None]
        for i in range(1, len(q)):
            if np.dot(q[i-1], q[i]) < 0:
                q[i] *= -1
        self.quaternions_wxyz = frozen(q)
        self.local_rotvec = []
        for a, b in zip(q, q[1:]):
            conj = a*np.array([1, -1, -1, -1])
            rel = _mul(conj, b)
            norm = np.linalg.norm(rel[1:])
            self.local_rotvec.append(np.zeros(3) if norm < 1e-15 else
                                     rel[1:]/norm*(2*np.arctan2(norm, max(0., rel[0]))))
        self.local_rotvec = frozen(self.local_rotvec)

    def at(self, time_s):
        i, f = _interval(self.times_s, time_s)
        dt = self.times_s[i+1]-self.times_s[i]
        v = self.local_rotvec[i]*f
        theta = np.linalg.norm(v)
        dq = np.r_[np.cos(theta/2), v*(np.sin(theta/2)/theta if theta else .5)]
        quat = _mul(self.quaternions_wxyz[i], dq)
        pos = (1-f)*self.positions_m[i]+f*self.positions_m[i+1]
        velocity = np.r_[(self.positions_m[i+1]-self.positions_m[i])/dt, self.local_rotvec[i]/dt]
        return frozen(np.r_[pos, quat]), frozen(velocity)


@dataclass(frozen=True)
class ObjectState:
    name: str
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray


@dataclass(frozen=True)
class SectionState:
    index: int
    bend_rad: float
    bend_rate_rad_s: float
    base: ObjectState
    tip: ObjectState
    bodies: tuple
    geoms: tuple


@dataclass(frozen=True)
class EelState:
    time_s: float
    bends_rad: np.ndarray
    bend_rates_rad_s: np.ndarray
    hinge_angles_rad: np.ndarray
    hinge_rates_rad_s: np.ndarray
    sections: tuple
    interpolated_gap: bool


class EelPlayback:
    """Random-access geometry from angles; independent of all pump models."""
    def __init__(self, model_path, trajectory, pose_trajectory=None):
        self.trajectory, self.pose_trajectory = trajectory, pose_trajectory
        self.chain = ChainKinematics(model_path, allow_root=True, kinematics_only=True)
        self.model, self.data = self.chain.model, self.chain.data
        if trajectory.section_count != self.chain.module_count:
            raise ValueError('Trajectory section count does not match chain')
        if pose_trajectory is not None:
            if self.chain.root_joint_id is None:
                raise ValueError('Pose input requires root_freejoint=True in chain generator')
            if (pose_trajectory.times_s[0] > trajectory.times_s[0]
                    or pose_trajectory.times_s[-1] < trajectory.times_s[-1]):
                raise ValueError('Pose trajectory must cover the entire bend trajectory')
        # Resolve object IDs once. The CAD geom pose includes MuJoCo mesh-asset
        # transforms, so it need not equal the owning body's origin/orientation.
        self._objects = []
        for i in range(self.chain.module_count):
            prefix = f'module_{i}_'
            bodies = [j for j in range(self.model.nbody) if self.model.body(j).name.startswith(prefix+'slice_')]
            geoms = [j for j in range(self.model.ngeom) if self.model.geom(j).name.startswith(prefix)]
            self._objects.append((bodies, geoms))

    def _object(self, kind, ident):
        m, d = self.model, self.data
        if kind == 'body':
            pos, mat, jac = d.xpos[ident], d.xmat[ident], mujoco.mj_jacBody
            name = m.body(ident).name
        elif kind == 'geom':
            pos, mat, jac = d.geom_xpos[ident], d.geom_xmat[ident], mujoco.mj_jacGeom
            name = m.geom(ident).name
        else:
            pos, mat, jac = d.site_xpos[ident], d.site_xmat[ident], mujoco.mj_jacSite
            name = m.site(ident).name
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        jac(m, d, jp, jr, ident)
        quat = np.empty(4); mujoco.mju_mat2Quat(quat, mat)
        return ObjectState(name, frozen(pos), frozen(quat), frozen(jp@d.qvel), frozen(jr@d.qvel))

    def at(self, time_s):
        sample = self.trajectory.at(time_s)
        pose, velocity = (np.r_[np.zeros(3), 1., 0., 0., 0.], np.zeros(6))
        if self.pose_trajectory is not None:
            pose, velocity = self.pose_trajectory.at(time_s)
        self.chain.apply(sample.bends_rad, time_s, sample.bend_rates_rad_s,
                         root_pose=pose if self.chain.root_joint_id is not None else None,
                         root_velocity=velocity)
        mujoco.mj_subtreeVel(self.model, self.data)
        sections = []
        for i, (bodies, geoms) in enumerate(self._objects):
            sections.append(SectionState(i, float(sample.bends_rad[i]), float(sample.bend_rates_rad_s[i]),
                self._object('site', self.chain.base_site_ids[i]), self._object('site', self.chain.tip_site_ids[i]),
                tuple(self._object('body', j) for j in bodies), tuple(self._object('geom', j) for j in geoms)))
        return EelState(float(time_s), sample.bends_rad, sample.bend_rates_rad_s,
                        frozen(self.data.qpos[self.chain.qpos_addresses]),
                        frozen(self.data.qvel[self.chain.dof_addresses]), tuple(sections), sample.interpolated_gap)
