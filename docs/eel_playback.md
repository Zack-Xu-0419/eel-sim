# Angle-driven eel: measured motion → CAD geometry

This system prescribes the bends from a recording and can optionally integrate
forward swimming from fluid loads. It is independent of the hydraulic runners.

## Launch the recording

Run from the repository root with the [five-actuator lab recording](../src/5_030Hz_LinearA_16L_test3_pose_bends.csv) or your own CSV.

```sh
./.venv/bin/python examples/recorded_eel_run.py --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv
./.venv/bin/python examples/recorded_eel_run.py --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv --video
./.venv/bin/python .venv/bin/mjpython examples/recorded_eel_run.py --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv --view
```

The first command exports states without opening a graphics context. The second
also produces a 1920×1080 MP4 at 30 fps. The third opens the native rotatable
viewer; the Python-wrapper invocation avoids macOS shebang issues in paths with
spaces. Use `--duration 6` for a shorter export, `--fps 60` for more video samples,
`--csv /absolute/path.csv` for another recording, and `--output-dir /path` to save
elsewhere. Duration changes the export only; the viewer replays the full input.

Default outputs under `Outputs/recorded_eel/`:

- `eel.xml`, mesh assets and `eel.json`: generated CAD chain and geometry record.
- `state.csv`: long-form state rows for every section base, tip, soft-division body,
  and attached geom. Each row includes frame/time, section bend/rate, gap flag,
  world position, quaternion, and world linear/angular velocity.
- `hinges.csv`: internal relative hinge angles and rates for each exported frame.
- `motion.mp4` and `snapshot.jpg`: generated only with `--video`.
- `run.json`: source hash, units, sampling settings, and assumptions.

Video frame n and CSV frame n use exactly the same evaluated snapshot at
`t_start + n/fps`. The final frame is the last such time not exceeding the
requested duration. Video holds each frame for 1/fps, so file duration can exceed
the last sampled timestamp by up to one frame. Inputs are never extrapolated.

## Replay the harmonic fit

```sh
./.venv/bin/python examples/recorded_eel_run.py --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv --fit-harmonics --duration 5 --video
```

Add `--csv` to select a different recording. In this mode, `Time(s)` and
`Camera_Angle` also work for a single actuator. Camera angles keep their recorded
sign and are interpreted in degrees. Fitted exports default to
`Outputs/recorded_eel_fit/`, with the fit settings and coefficients in `run.json`.

For arrays, use `fit_bends(times_s, measured_bends_deg)` from `src.kinematics.eel_playback`.
Pass the returned trajectory to `EelPlayback`, or call `trajectory.at(t_s)`
directly to get the section bends and bend rates in radians and radians per second.
The fit uses a time-weighted mean over a centered period and least-squares
harmonics at 0.3 and 0.9 Hz with independent phases. Each baseline window needs
about 1.67 seconds of recording on each side at 0.3 Hz. Queries are limited to
that interior range. The fitted gap flag means the baseline window overlaps an
interval with missing tracking.

## Predict straight swimming

```sh
./.venv/bin/python examples/recorded_eel_run.py --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv --fit-harmonics --swim --video
```

This starts the center of mass from rest and integrates its longitudinal motion.
The bends are prescribed. Heading, lateral COM position, and depth are fixed.
MuJoCo evaluates the ellipsoid loads at each integration step using still fresh
water (998 kg/m³, 0.001 Pa·s) and the model's fluid coefficients. The generated
model uses MuJoCo's default coefficients and placeholder actuator masses.

`swim.csv` reports net fluid force and integrated COM motion at the physics
timestamps. Headward is negative world X. The regular state CSV and optional
video replay the computed root path at the export frame rate. With the fit,
outputs default to `Outputs/recorded_eel_fit_swim/`. Use `--fluid-dt` to change
the maximum integration step from its 0.005 s default.

The generated chain uses one fluid ellipsoid on the middle slice of each soft
span, plus a separate ellipsoid on each rigid end. The bend still uses ten
slices per actuator. Use `--slice-fluid-geometry` to compare with the earlier
per-slice fluid geometry. Swimming sums ellipsoid loads only, so the intermediate
slices without fluid geoms do not receive MuJoCo's fallback inertia-box drag.

Add `--force-arrows` with `--swim` and either `--view` or `--video` to draw the
five largest fluid force vectors just above the ellipsoids so the meshes do not
obscure them. Arrow length defaults to 0.8 m per newton and can be adjusted with
`--force-arrow-scale`. The loads are evaluated at the playback state.
Use `--all-force-arrows` to draw every ellipsoid's force vector.
Add `--ellipsoid-view` with `--video` for a top-down view of Group 3.

By default, video and viewer exports also render a **green velocity vector arrow**
at the eel's Center of Mass (length scale defaults to 1.0 m per (m/s), adjustable
with `--velocity-arrow-scale` or toggleable with `--no-velocity-arrow`). The video
overlay also reports instantaneous COM and base $(\Delta x, \Delta y)$ displacement
compared to the start, alongside the 3D velocity vector and speed. `state.csv`
includes `displacement_x_m`, `displacement_y_m`, and `displacement_z_m` columns for
every tracked object relative to frame 0.

## Predict planar swimming with ellipsoid loads

```sh
./.venv/bin/python examples/recorded_eel_run.py --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv --fit-harmonics --planar-swim --mass-kg 4.116 --video --ellipsoid-view --all-force-arrows
```

The recorded bends prescribe the shape. MuJoCo evaluates passive ellipsoid
forces and torques, and the solver integrates world X/Y and yaw from rest
in total planar momentum. Depth, roll and pitch are fixed. Only the enabled
ellipsoids contribute water loads. The solver uses the robot's modeled inertia
and the loads reported by MuJoCo's passive stage. It does not solve the
acceleration-dependent added-mass terms from the fluid equations.

The harmonic run writes to `Outputs/recorded_eel_fit_planar_swim/`. Its
`swim.csv` records signed world-frame COM position and velocity, yaw, and net
fluid load. At the initial heading, negative world X points toward the head.

## Predict planar swimming with inertia-box drag

```sh
./.venv/bin/python examples/recorded_eel_run.py --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv --fit-harmonics --inertia-swim --mass-kg 4.116
```

This baseline uses the recording's bend angles and initially zero net planar
momentum. Each section keeps its prescribed bend while the eel translates in
world X and Y and rotates in yaw. Depth, roll and pitch stay fixed. MuJoCo
calculates drag from a box inferred from each moving body's mass and inertia.
The integrator advances linear and angular momentum from the resulting fluid
forces. Use `--fluid-dt` to change the maximum step.

The generated model distributes the supplied whole-robot mass equally across
actuators. The head and tail meshes are visible and contribute no independent
drag or inertia. Each inertia box receives local drag independently, so axial
drag can accumulate across the discretization of the continuous body.

With `--fit-harmonics`, outputs go to `Outputs/recorded_eel_fit_inertia_swim/`.
`swim.csv` reports force, yaw moment about the COM and root, root and COM motion,
and planar angular momentum. Drop `--fit-harmonics` to use linearly interpolated
measured bends and write to `Outputs/recorded_eel_inertia_swim/`. Use `--video`
for MP4 or `--view` under `mjpython` for interactive playback.

Under **Geom groups**, **Group 3** shows the original ellipsoid overlay and
**Group 4** shows the inertia boxes used for drag. In this mode the ellipsoids
are for visual comparison. **Group 0** toggles the CAD meshes.

Add `--force-arrows` to show the five largest inertia-box forces at each frame.
Orange arrows sit above the corresponding body centers of mass. The visual
scale defaults to 40 m/N for this mode and can be changed with
`--force-arrow-scale`.

## Query the API

```python
import numpy as np
from src.kinematics.eel_playback import AngleTrajectory, EelPlayback

angles = AngleTrajectory([0., 1., 2.], np.deg2rad([[0]*5, [10, 5, 0, -5, -10], [0]*5]))
eel = EelPlayback('Outputs/recorded_eel/eel.xml', angles)
state = eel.at(.5)
section = state.sections[0]
print(section.bend_rad, section.tip.position_m, section.tip.linear_velocity_m_s)
```

Use `AngleTrajectory.from_recording_csv(path, max_gap_s=.15)` for the supplied
CSV. The adapter discovers consecutive `act1_bend_deg` through `actN_bend_deg`,
reads `t_s`, converts degrees to radians, and filters `usable=0` or missing bends.
It does not smooth, subtract offsets, include tail angles, or infer head geometry.
The example defaults to a fixed base. Measured bends are interpreted as total
relative section bends, not absolute world headings.

`eel.at(t)` can be called at arbitrary times and in any order. No step or previous
state is needed. Snapshots contain read-only copies; subsequent queries do not
alter previous snapshots. Section indices are zero-based. Base/tip poses refer to
section boundary frames, not segment centers. Body poses refer to soft-division
body origins; geom poses include compiled mesh offsets/rotations. To reconstruct
surface vertices, transform compiled local mesh vertices using the geom pose.

Each section exposes `base`, `tip`, `bodies`, and `geoms`. Each object has `name`,
`position_m`, `quaternion_wxyz`, `linear_velocity_m_s`, `angular_velocity_rad_s`.
`state.hinge_angles_rad` and `state.hinge_rates_rad_s` follow
`eel.chain.joint_names`. All object velocities are world-frame and measured at
that object's own origin. There are no acceleration outputs.

## Optional overall pose

Pass `PoseTrajectory(times_s, positions_m, quaternions_wxyz)` as the third
argument to `EelPlayback`. It may use a different sample grid, but must cover the
entire bend trajectory. Quaternions are normalized and interpreted as **w,x,y,z**.
Translation is linear; orientation follows the shortest quaternion arc.

Overall pose means the **first section's base frame**, with local +x pointing
from first section toward last section in the straight reference configuration.
This is not automatically the tracked head center or nose. Transform measured
head pose into this frame using a separately established rigid offset before
passing it in. No offset is guessed here. Without a pose input the root stays at
origin with identity orientation, even if the generated model has a free joint.

Custom chain XML may be loaded with `--model`. Optional overall pose requires a
model generated by `generate_chain(..., root_freejoint=True)`. The old generator
API still defaults to a fixed base. Both variants are accepted for angle-only
playback. The default recording runner generates the movable-root variant but
prescribes a stationary root unless the API supplies a pose trajectory.

## Geometry and interpolation assumptions

Default: five 157 mm sections, each 26 rigid + 105 soft + 26 rigid, 80 mm wide,
160 mm tall. Each soft span has ten divisions and nine equal-angle internal
hinges. Rigid ends stay welded to nearest soft divisions; module boundaries add
no hinge. More/fewer sections are supported through the generator and trajectory
column count. These remain image-derived meshes, not a new imported CAD file.

Angles vary linearly between samples; rates are constant within each interval.
At interior samples the rate comes from the following interval; the final sample
uses the preceding interval. Rates can jump at sample times, so this is not a
smooth acceleration model. Quaternion rotations use the analogous constant local
angular velocity for each interpolation interval.

The supplied recording has two invalid frames (738–739). The interval bridging
them is about 0.1001 s and is accepted by the default 0.15 s gap threshold.
`interpolated_gap=True` strictly inside that interval; measured endpoints are not
flagged. A gap longer than the threshold raises an error; change `max_gap_s`
explicitly if justified. Invalid leading/trailing rows trim the supported range;
there is no extrapolation. Generic `AngleTrajectory` inputs must be finite and
complete; only the recording adapter filters invalid rows.

Limits, duplicate timestamps, nonfinite input, missing groups, unsupported extra
joints, and section-count mismatches fail explicitly. Angles are never clamped.

## Files to edit

- `src/kinematics/eel_playback.py`: trajectory interpolation, recording adapter, and state API.
- `src/kinematics/chain_kinematics.py`: shared hinge mapping and MuJoCo geometry refresh. Old
  hydraulic imports are re-exported through `src/kinematics/traveling_wave.py`; they retain their
  original forward-refresh behavior. New playback explicitly uses only
  `mj_kinematics`, `mj_comPos`, `mj_comVel`, and `mj_camlight`.
- `src/export/eel_playback_export.py`: camera, native viewer, video overlay, CSV format.
- `examples/recorded_eel_run.py`: recording selection, dimensions, export options.
- `src/geometry/auto_hydraulic_section.py` and `src/geometry/auto_hydraulic_chain.py`:
  profile/mesh generation, discretization, rigid-end dimensions and assembly.
