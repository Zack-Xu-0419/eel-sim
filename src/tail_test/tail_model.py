"""Tail Testing Simulation: Model, Kinematics, and Hydrodynamics."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mujoco

from src.paths import ROOT

WATER_DENSITY = 998.0  # kg/m^3
WATER_VISCOSITY = 0.001  # Pa*s

# Default geometry matching fish simulation caudal tail
DEFAULT_TAIL_LENGTH_M = 0.230   # 230 mm length
DEFAULT_TAIL_HEIGHT_M = 0.160   # 160 mm height
DEFAULT_TAIL_WIDTH_M = 0.032    # 32 mm max thickness (2 * 16 mm semi-axis)
DEFAULT_PIVOT_RADIUS_M = 0.025  # 25 mm pivot radius


@dataclass
class TailSample:
    time_s: float
    angle_rad: float
    angle_deg: float
    angular_velocity_rad_s: float
    angular_velocity_deg_s: float
    angular_acceleration_rad_s2: float
    angular_acceleration_deg_s2: float

    # Total Net Hydrodynamic Force (Cartesian)
    fluid_force_x_n: float           # Net X force (forward thrust if positive)
    fluid_force_y_n: float           # Net Y force (lateral resistance)
    fluid_force_z_n: float           # Net Z force
    fluid_force_mag_n: float         # Net resultant fluid force magnitude |F|

    # Body & Aerodynamic Frame Projections
    tail_normal_force_n: float       # Force component normal to fin surface
    tail_axial_force_n: float        # Force component along fin chord
    hydrodynamic_lift_n: float       # Lift force perpendicular to instantaneous velocity
    hydrodynamic_drag_n: float       # Drag force opposing instantaneous velocity

    # Individual Hydrodynamic Force Components (Decomposed from MuJoCo fluid model)
    blunt_drag_force_x_n: float      # Blunt pressure drag X
    blunt_drag_force_y_n: float      # Blunt pressure drag Y
    slender_drag_force_x_n: float    # Slender skin-friction drag X
    slender_drag_force_y_n: float    # Slender skin-friction drag Y
    kutta_lift_force_x_n: float      # Kutta circulation lift X
    kutta_lift_force_y_n: float      # Kutta circulation lift Y
    magnus_lift_force_x_n: float     # Magnus rotational lift X
    magnus_lift_force_y_n: float     # Magnus rotational lift Y
    viscous_base_force_x_n: float    # Base viscous Stokes force X
    viscous_base_force_y_n: float    # Base viscous Stokes force Y

    # Individual Torques on Pivot Joint (N*m)
    total_fluid_torque_nm: float     # Net hydrodynamic resistive torque on pivot
    angular_drag_torque_nm: float    # Hydrodynamic rotational damping torque
    blunt_drag_torque_nm: float      # Torque produced by blunt pressure drag
    joint_damping_torque_nm: float   # Mechanical joint damping torque (-d * omega)
    inertial_torque_nm: float        # Inertial reaction torque (I * alpha)
    actuator_required_torque_nm: float  # Drive torque needed to sustain the flapping motion

    # Positions and Arrow Vectors
    fluid_geom_pos: np.ndarray       # 3D position of fluid geometry center of pressure
    arrow_origin: np.ndarray         # Origin point for force arrow drawing
    arrow_vector: np.ndarray         # Scaled force vector for rendering


class TailSimulator:
    """Simulates a single pivoted fish tail fin flapping in water with MuJoCo fluid dynamics."""

    def __init__(self, xml_path=None, *, frequency_hz=1.0, amplitude_deg=25.0,
                 force_arrow_scale=0.08):
        if xml_path is None:
            xml_path = ROOT / "Geometry/tail_test/tail.xml"
        self.xml_path = Path(xml_path).resolve()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.frequency_hz = float(frequency_hz)
        self.amplitude_rad = np.deg2rad(float(amplitude_deg))
        self.force_arrow_scale = float(force_arrow_scale)

        # Joint and body address lookups
        self.root_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.pivot_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pivot_joint")
        self.fluid_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fluid_geometry")

        if self.root_joint < 0 or self.pivot_joint < 0:
            raise ValueError("Model must contain 'root' freejoint and 'pivot_joint' hinge")

        self.root_qpos = self.model.jnt_qposadr[self.root_joint]
        self.root_dof = self.model.jnt_dofadr[self.root_joint]
        self.pivot_qpos = self.model.jnt_qposadr[self.pivot_joint]
        self.pivot_dof = self.model.jnt_dofadr[self.pivot_joint]

        # Cache original fluid coefficients
        self.orig_fluid_coefs = self.model.geom_fluid[self.fluid_geom_id, 1:6].copy()
        self.joint_damping = float(self.model.dof_damping[self.pivot_dof])

        # Preallocate mass matrix buffer
        self._mass_matrix = np.empty((self.model.nv, self.model.nv))

        # Reset state to origin
        self.reset()

    def reset(self):
        """Reset base fixture to world origin and tail angle to neutral."""
        mujoco.mj_resetData(self.model, self.data)
        # Root fixture pinned at origin
        self.data.qpos[self.root_qpos:self.root_qpos+3] = [0.0, 0.0, 0.0]
        self.data.qpos[self.root_qpos+3:self.root_qpos+7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.root_dof:self.root_dof+6] = 0.0

        self.data.qpos[self.pivot_qpos] = 0.0
        self.data.qvel[self.pivot_dof] = 0.0
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_fwdVelocity(self.model, self.data)

    def evaluate_at(self, t_s: float) -> TailSample:
        """Evaluate tail kinematics and exact MuJoCo hydrodynamic forces at time t."""
        omega_flapping = 2.0 * np.pi * self.frequency_hz
        theta = self.amplitude_rad * np.sin(omega_flapping * t_s)
        theta_dot = omega_flapping * self.amplitude_rad * np.cos(omega_flapping * t_s)
        theta_ddot = -(omega_flapping ** 2) * self.amplitude_rad * np.sin(omega_flapping * t_s)

        # Pin fixture at origin
        self.data.qpos[self.root_qpos:self.root_qpos+3] = [0.0, 0.0, 0.0]
        self.data.qpos[self.root_qpos+3:self.root_qpos+7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.root_dof:self.root_dof+6] = 0.0

        # Apply hinge state
        self.data.qpos[self.pivot_qpos] = theta
        self.data.qvel[self.pivot_dof] = theta_dot

        # Restore original fluid coefficients for total force evaluation
        self.model.geom_fluid[self.fluid_geom_id, 1:6] = self.orig_fluid_coefs

        # Forward kinematics and fluid force evaluation
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_fwdVelocity(self.model, self.data)

        # 1. Total generalized fluid forces:
        # Translation DOFs [0:3] give net Cartesian fluid force (Fx, Fy, Fz)
        F_xyz = self.data.qfrc_fluid[self.root_dof:self.root_dof+3].copy()
        fx, fy, fz = float(F_xyz[0]), float(F_xyz[1]), float(F_xyz[2])
        f_mag = float(np.linalg.norm(F_xyz))

        # Pivot fluid resistance torque
        total_fluid_torque = float(self.data.qfrc_fluid[self.pivot_dof])

        # 2. Decompose individual hydrodynamic force contributions
        # Base viscous Stokes force (all drag/lift shape coefficients zero)
        self.model.geom_fluid[self.fluid_geom_id, 1:6] = 0.0
        mujoco.mj_fwdVelocity(self.model, self.data)
        base_viscous_F = self.data.qfrc_fluid[self.root_dof:self.root_dof+3].copy()
        base_viscous_tau = float(self.data.qfrc_fluid[self.pivot_dof])

        # Blunt drag alone (index 0)
        coef_vec = np.zeros(5)
        coef_vec[0] = self.orig_fluid_coefs[0]
        self.model.geom_fluid[self.fluid_geom_id, 1:6] = coef_vec
        mujoco.mj_fwdVelocity(self.model, self.data)
        blunt_F = self.data.qfrc_fluid[self.root_dof:self.root_dof+3] - base_viscous_F
        blunt_tau = float(self.data.qfrc_fluid[self.pivot_dof]) - base_viscous_tau

        # Slender drag alone (index 1)
        coef_vec = np.zeros(5)
        coef_vec[1] = self.orig_fluid_coefs[1]
        self.model.geom_fluid[self.fluid_geom_id, 1:6] = coef_vec
        mujoco.mj_fwdVelocity(self.model, self.data)
        slender_F = self.data.qfrc_fluid[self.root_dof:self.root_dof+3] - base_viscous_F

        # Angular drag alone (index 2)
        coef_vec = np.zeros(5)
        coef_vec[2] = self.orig_fluid_coefs[2]
        self.model.geom_fluid[self.fluid_geom_id, 1:6] = coef_vec
        mujoco.mj_fwdVelocity(self.model, self.data)
        angular_tau = float(self.data.qfrc_fluid[self.pivot_dof]) - base_viscous_tau

        # Kutta lift alone (index 3)
        coef_vec = np.zeros(5)
        coef_vec[3] = self.orig_fluid_coefs[3]
        self.model.geom_fluid[self.fluid_geom_id, 1:6] = coef_vec
        mujoco.mj_fwdVelocity(self.model, self.data)
        kutta_F = self.data.qfrc_fluid[self.root_dof:self.root_dof+3] - base_viscous_F

        # Magnus lift alone (index 4)
        coef_vec = np.zeros(5)
        coef_vec[4] = self.orig_fluid_coefs[4]
        self.model.geom_fluid[self.fluid_geom_id, 1:6] = coef_vec
        mujoco.mj_fwdVelocity(self.model, self.data)
        magnus_F = self.data.qfrc_fluid[self.root_dof:self.root_dof+3] - base_viscous_F

        # Restore original coefficients
        self.model.geom_fluid[self.fluid_geom_id, 1:6] = self.orig_fluid_coefs
        mujoco.mj_fwdVelocity(self.model, self.data)

        # 3. Aerodynamic and Blade Frame Projections
        tangent = np.array([np.cos(theta), np.sin(theta), 0.0])
        normal = np.array([-np.sin(theta), np.cos(theta), 0.0])

        f_normal = float(np.dot(F_xyz, normal))
        f_axial = float(np.dot(F_xyz, tangent))

        # Velocity vector of the fluid geometry center of pressure
        geom_pos = self.data.geom_xpos[self.fluid_geom_id].copy()
        r_cp = float(np.linalg.norm(geom_pos[:2]))
        v_cp = np.array([-r_cp * theta_dot * np.sin(theta), r_cp * theta_dot * np.cos(theta), 0.0])
        speed_cp = float(np.linalg.norm(v_cp))

        if speed_cp > 1e-6:
            v_hat = v_cp / speed_cp
            lift_dir = np.cross(np.array([0.0, 0.0, 1.0]), v_hat)
            hydro_drag = -float(np.dot(F_xyz, v_hat))
            hydro_lift = float(np.dot(F_xyz, lift_dir))
        else:
            hydro_drag = 0.0
            hydro_lift = 0.0

        # 4. Joint Torques (Damping, Inertia, Actuator required drive)
        joint_damping_torque = -self.joint_damping * theta_dot

        # Inertia about pivot
        mujoco.mj_fullM(self.model, self._mass_matrix, self.data.qM)
        I_pivot = float(self._mass_matrix[self.pivot_dof, self.pivot_dof])
        inertial_torque = I_pivot * theta_ddot

        # Required actuator drive torque to achieve this motion against inertia, fluid resistance, and damping
        # I * theta_ddot = tau_actuator + tau_fluid + tau_damping
        # tau_actuator = I * theta_ddot - tau_fluid - tau_damping
        actuator_required_torque = inertial_torque - total_fluid_torque - joint_damping_torque

        # Arrow rendering origin (elevated slightly above the fin surface in Z)
        arrow_origin = geom_pos.copy()
        arrow_origin[2] += 0.085

        # Rendered arrow end vector
        arrow_vec = F_xyz * self.force_arrow_scale

        return TailSample(
            time_s=t_s,
            angle_rad=theta,
            angle_deg=float(np.rad2deg(theta)),
            angular_velocity_rad_s=theta_dot,
            angular_velocity_deg_s=float(np.rad2deg(theta_dot)),
            angular_acceleration_rad_s2=theta_ddot,
            angular_acceleration_deg_s2=float(np.rad2deg(theta_ddot)),
            fluid_force_x_n=fx,
            fluid_force_y_n=fy,
            fluid_force_z_n=fz,
            fluid_force_mag_n=f_mag,
            tail_normal_force_n=f_normal,
            tail_axial_force_n=f_axial,
            hydrodynamic_lift_n=hydro_lift,
            hydrodynamic_drag_n=hydro_drag,
            blunt_drag_force_x_n=float(blunt_F[0]),
            blunt_drag_force_y_n=float(blunt_F[1]),
            slender_drag_force_x_n=float(slender_F[0]),
            slender_drag_force_y_n=float(slender_F[1]),
            kutta_lift_force_x_n=float(kutta_F[0]),
            kutta_lift_force_y_n=float(kutta_F[1]),
            magnus_lift_force_x_n=float(magnus_F[0]),
            magnus_lift_force_y_n=float(magnus_F[1]),
            viscous_base_force_x_n=float(base_viscous_F[0]),
            viscous_base_force_y_n=float(base_viscous_F[1]),
            total_fluid_torque_nm=total_fluid_torque,
            angular_drag_torque_nm=angular_tau,
            blunt_drag_torque_nm=blunt_tau,
            joint_damping_torque_nm=joint_damping_torque,
            inertial_torque_nm=inertial_torque,
            actuator_required_torque_nm=actuator_required_torque,
            fluid_geom_pos=geom_pos,
            arrow_origin=arrow_origin,
            arrow_vector=arrow_vec,
        )
