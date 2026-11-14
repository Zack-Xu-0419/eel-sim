"""Swimming with prescribed bends and MuJoCo's fluid loads."""
import csv
from pathlib import Path

import mujoco
import numpy as np

from src.kinematics.eel_playback import PoseTrajectory

WATER_DENSITY = 998.  # kg/m³
WATER_VISCOSITY = .001  # Pa·s


def integrate_inertia_swim(playback, output_dir, *, dt_s=.005, mass_kg=None):
    """Integrate planar motion using MuJoCo's inertia-box loads."""
    return integrate_planar_swim(playback, output_dir, dt_s=dt_s, mass_kg=mass_kg,
                                 ellipsoids=False)


def integrate_planar_swim(playback, output_dir, *, dt_s=.005, mass_kg=None, ellipsoids=True):
    """Advance the planar COM, yaw and momentum with prescribed hinge motion."""
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('Fluid timestep must be positive and finite')
    model, data, chain = playback.model, playback.data, playback.chain
    root = chain.root_joint_id
    if root is None:
        raise ValueError('Planar swim requires an eel_root free joint')
    fluid_geoms = model.geom_fluid[:, 0] > 0
    if ellipsoids and not fluid_geoms.any():
        raise ValueError('Planar ellipsoid swim requires fluid-enabled ellipsoids')
    if mass_kg is not None:
        if not np.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError('Eel mass must be positive and finite')
        mujoco.mj_setTotalmass(model, mass_kg)
        mujoco.mj_setConst(model, data)
    trajectory = playback.trajectory
    root_dof = int(model.jnt_dofadr[root])
    root_qpos = int(model.jnt_qposadr[root])
    free = root_dof + np.array([0, 1, 5])
    hinges = chain.dof_addresses
    mass_matrix = np.empty((model.nv, model.nv))
    interaction = model.geom_fluid[:, 0].copy()
    original_fluid = model.opt.density, model.opt.viscosity
    represented = np.zeros(model.nbody, dtype=bool)
    represented[model.geom_bodyid[fluid_geoms]] = True
    unrepresented = ~represented & (model.body_mass > 0)
    omitted_masses = model.body_mass[unrepresented].copy()
    root_body = model.jnt_bodyid[root]
    mass = float(model.body_subtreemass[root_body])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('Inertia swim requires a positive, finite subtree mass')

    def evaluate(clock, state):
        com_x, com_y, yaw, px, py, angular_momentum = state
        sample = trajectory.at(clock)
        pose = [0., 0., 0., np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
        chain.apply(sample.bends_rad, clock, sample.bend_rates_rad_s,
                    root_pose=pose)
        rx, ry = data.subtree_com[root_body, :2]
        root_xy = np.array([com_x-rx, com_y-ry])
        data.qpos[root_qpos:root_qpos+2] = root_xy
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fullM(model, mass_matrix, data.qM)
        free_mass = mass_matrix[np.ix_(free, free)]
        coupling = mass_matrix[np.ix_(free, hinges)]
        # The generalized yaw momentum is about the free joint origin. Shift
        # from the COM, then subtract the contribution of the prescribed hinges.
        root_momentum = np.array([px, py, angular_momentum+rx*py-ry*px])
        shape_momentum = np.einsum('ij,j->i', coupling, data.qvel[hinges])
        root_velocity = np.linalg.solve(free_mass, root_momentum-shape_momentum)
        data.qvel[free] = root_velocity
        if ellipsoids:
            # Compute velocities with the whole body mass distribution, then
            # suppress fallback box drag on links lacking an ellipsoid.
            mujoco.mj_comVel(model, data)
            model.body_mass[unrepresented] = 0.
            try:
                mujoco.mj_passive(model, data)
            finally:
                model.body_mass[unrepresented] = omitted_masses
        else:
            mujoco.mj_fwdVelocity(model, data)
        fx, fy, moment_root = data.qfrc_fluid[free]
        moment_com = moment_root-rx*fy+ry*fx
        derivative = np.array([px/mass, py/mass, root_velocity[2], fx, fy, moment_com])
        return derivative, root_xy, root_velocity, moment_root

    # The piecewise-linear trajectory changes bend rate at knots. Land on
    # these times so the midpoint samples only one side of each change.
    knots = trajectory.times_s
    intervals = [np.linspace(a, b, int(np.ceil((b-a)/dt_s))+1)[1:]
                 for a, b in zip(knots, knots[1:])]
    times = np.concatenate(([knots[0]], *intervals))
    states = np.empty((len(times), 6))
    root_positions = np.empty((len(times), 2))
    root_velocities = np.empty((len(times), 3))
    loads = np.empty((len(times), 4))
    first = trajectory.at(times[0])
    chain.apply(first.bends_rad, times[0], first.bend_rates_rad_s,
                root_pose=[0., 0., 0., 1., 0., 0., 0.])
    states[0] = 0.
    states[0, :2] = data.subtree_com[root_body, :2]

    # A geom with enabled ellipsoid interaction otherwise opts its entire
    # body out of the inertia-box fallback.
    if not ellipsoids:
        model.geom_fluid[:, 0] = 0.
    model.opt.density, model.opt.viscosity = WATER_DENSITY, WATER_VISCOSITY
    try:
        for i, clock in enumerate(times):
            derivative, root_positions[i], root_velocities[i], moment_root = evaluate(clock, states[i])
            loads[i] = [*derivative[3:], moment_root]
            if i+1 < len(times):
                dt = times[i+1]-clock
                midpoint, *_ = evaluate(clock+.5*dt, states[i]+.5*dt*derivative)
                states[i+1] = states[i]+dt*midpoint
                if not np.isfinite(states[i+1]).all():
                    raise ValueError('Planar swim diverged. Reduce the fluid timestep.')
    finally:
        model.geom_fluid[:, 0] = interaction
        model.opt.density, model.opt.viscosity = original_fluid

    zeros = np.zeros(len(times))
    yaw = states[:, 2]
    pose = PoseTrajectory(times, np.column_stack((root_positions, zeros)),
                          np.column_stack((np.cos(yaw/2), zeros, zeros, np.sin(yaw/2))))
    with (Path(output_dir)/'swim.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'root_x_m', 'root_y_m', 'yaw_rad', 'root_vx_m_s',
                         'root_vy_m_s', 'yaw_rate_rad_s', 'fluid_force_x_n',
                         'fluid_force_y_n', 'fluid_moment_about_com_z_nm',
                         'fluid_moment_about_root_z_nm', 'com_x_m', 'com_y_m',
                         'com_vx_m_s', 'com_vy_m_s', 'angular_momentum_about_com_z_kg_m2_s'])
        writer.writerows(zip(times, *root_positions.T, yaw, *root_velocities.T, *loads.T,
                             *states[:, :2].T, *(states[:, 3:5]/mass).T, states[:, 5]))
    record = dict(dt_max_s=dt_s, water_density_kg_m3=WATER_DENSITY,
                      water_viscosity_pa_s=WATER_VISCOSITY,
                      mass_kg=mass,
                      mass_rescaled=mass_kg is not None,
                      fluid_model='ellipsoid' if ellipsoids else 'inertia_box',
                      initial_planar_momentum='zero')
    if ellipsoids:
        record['fluid_geom_count'] = int(fluid_geoms.sum())
        record['coefficient_sets'] = np.unique(model.geom_fluid[fluid_geoms, 1:6], axis=0).tolist()
    return pose, record


def inertia_box_forces(playback):
    """Return body IDs and world-frame forces for the current playback pose."""
    model, data = playback.model, playback.data
    root_dof = model.jnt_dofadr[playback.chain.root_joint_id]
    translation = slice(root_dof, root_dof+3)
    interaction = model.geom_fluid[:, 0].copy()
    mass = model.body_mass.copy()
    bodies = np.flatnonzero(mass)
    original_fluid = model.opt.density, model.opt.viscosity
    forces = np.empty((len(bodies), 3))
    try:
        # Preserve the current poses and velocities. Evaluate the same
        # inertia-box force as the integrator, isolating one body at a time.
        model.opt.density, model.opt.viscosity = WATER_DENSITY, WATER_VISCOSITY
        model.geom_fluid[:, 0] = 0.
        model.body_mass[:] = 0.
        for i, body in enumerate(bodies):
            model.body_mass[body] = mass[body]
            mujoco.mj_passive(model, data)
            forces[i] = data.qfrc_fluid[translation]
            model.body_mass[body] = 0.
    finally:
        model.geom_fluid[:, 0] = interaction
        model.body_mass[:] = mass
        model.opt.density, model.opt.viscosity = original_fluid
        mujoco.mj_passive(model, data)
    return bodies, forces


def ellipsoid_forces(playback):
    """Return geom IDs and world-frame forces for the current playback pose."""
    model, data = playback.model, playback.data
    root_dof = model.jnt_dofadr[playback.chain.root_joint_id]
    translation = slice(root_dof, root_dof+3)
    interaction = model.geom_fluid[:, 0].copy()
    geom_ids = np.flatnonzero(interaction)
    mass = model.body_mass.copy()
    original_fluid = model.opt.density, model.opt.viscosity
    forces = np.empty((len(geom_ids), 3))
    mujoco.mj_fwdPosition(model, data)
    model.opt.density, model.opt.viscosity = WATER_DENSITY, WATER_VISCOSITY
    try:
        # Retain one geom and its parent per evaluation to isolate its load
        # without triggering inertia-box drag on any other body.
        model.geom_fluid[:, 0] = 0.
        model.body_mass[:] = 0.
        for i, geom_id in enumerate(geom_ids):
            body = model.geom_bodyid[geom_id]
            model.geom_fluid[geom_id, 0] = interaction[geom_id]
            model.body_mass[body] = mass[body]
            mujoco.mj_fwdVelocity(model, data)
            forces[i] = data.qfrc_fluid[translation]
            model.geom_fluid[geom_id, 0] = 0.
            model.body_mass[body] = 0.
    finally:
        model.geom_fluid[:, 0] = interaction
        model.body_mass[:] = mass
        model.opt.density, model.opt.viscosity = original_fluid
        mujoco.mj_fwdVelocity(model, data)
    return geom_ids, forces


def integrate_straight_swim(playback, output_dir, *, dt_s=.005):
    """Integrate world-X COM motion, holding COM Y/Z and root heading fixed."""
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('Fluid timestep must be positive and finite')
    model, data, chain = playback.model, playback.data, playback.chain
    root = chain.root_joint_id
    if root is None:
        raise ValueError('Swimming requires an eel_root free joint')
    fluid_geoms = model.geom_fluid[:, 0] > 0
    if not fluid_geoms.any():
        raise ValueError('Swimming requires fluid-enabled geometry')
    ellipsoid_bodies = np.zeros(model.nbody, dtype=bool)
    ellipsoid_bodies[model.geom_bodyid[fluid_geoms]] = True
    unrepresented = ~ellipsoid_bodies & (model.body_mass > 0)
    omitted_masses = model.body_mass[unrepresented].copy()
    body = model.jnt_bodyid[root]
    root_dof = model.jnt_dofadr[root]
    translation = slice(root_dof, root_dof+3)
    mass = float(model.body_subtreemass[body])
    root_identity = [0., 0., 0., 1., 0., 0., 0.]

    def evaluate(clock, speed):
        sample = playback.trajectory.at(clock)
        chain.apply(sample.bends_rad, clock, sample.bend_rates_rad_s, root_pose=root_identity)
        mujoco.mj_subtreeVel(model, data)
        shape_com = data.subtree_com[body].copy()
        # The bending links already move the COM relative to the root.
        # Compensate for that motion before evaluating the water-relative loads.
        data.qvel[translation] = [speed, 0., 0.] - data.subtree_linvel[body]
        mujoco.mj_fwdPosition(model, data)
        # MuJoCo otherwise assigns inertia-box drag to bodies with no fluid geom.
        # Skip that fallback only while evaluating the fluid loads. The COM
        # and position-dependent computations above use all physical masses.
        model.body_mass[unrepresented] = 0.
        try:
            mujoco.mj_fwdVelocity(model, data)
        finally:
            model.body_mass[unrepresented] = omitted_masses
        # The translation DOFs move every link equally, so their generalized
        # fluid forces equal the net Cartesian force on the entire eel.
        return data.qfrc_fluid[translation].copy(), shape_com

    # Include trajectory knots so integration never crosses a bend-rate jump
    # in the middle of a step.
    knots = playback.trajectory.times_s
    times = [knots[0]]
    for left, right in zip(knots, knots[1:]):
        count = int(np.ceil((right-left)/dt_s))
        times.extend(np.linspace(left, right, count+1)[1:])
    root_positions, rows = [], []
    distance = speed = 0.
    density, viscosity = WATER_DENSITY, WATER_VISCOSITY
    original_fluid = model.opt.density, model.opt.viscosity
    model.opt.density, model.opt.viscosity = density, viscosity
    try:
        for i, clock in enumerate(times):
            force, shape_com = evaluate(clock, speed)
            if i == 0:
                initial_com = shape_com.copy()
            root_positions.append(initial_com + [distance, 0., 0.] - shape_com)
            # Local +X points from head toward tail, so headward speed is -X.
            rows.append((clock, -distance, -speed, -force[0], *force))
            if i+1 == len(times):
                break
            dt = times[i+1]-clock
            midpoint_speed = speed + .5*dt*force[0]/mass
            midpoint_force, _ = evaluate(clock+.5*dt, midpoint_speed)
            distance += dt*midpoint_speed
            speed += dt*midpoint_force[0]/mass
            if not np.isfinite(speed) or not np.isfinite(distance):
                raise ValueError('Swim integration diverged. Reduce the fluid timestep.')
    finally:
        model.opt.density, model.opt.viscosity = original_fluid

    pose = PoseTrajectory(times, root_positions, np.tile(root_identity[3:], (len(times), 1)))
    output = Path(output_dir)/'swim.csv'
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'headward_displacement_m', 'headward_speed_m_s',
                         'headward_fluid_force_n', 'fluid_force_x_n', 'fluid_force_y_n', 'fluid_force_z_n'])
        writer.writerows(rows)
    return pose, dict(dt_max_s=dt_s, water_density_kg_m3=density, water_viscosity_pa_s=viscosity,
                      mass_kg=mass, fluid_geom_count=int(fluid_geoms.sum()),
                      coefficient_sets=np.unique(model.geom_fluid[fluid_geoms, 1:6], axis=0).tolist())
