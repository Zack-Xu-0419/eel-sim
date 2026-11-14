# Eel simulation

Prescribe measured section bends, replay the CAD geometry in MuJoCo, or predict planar swimming from ellipsoid fluid loads. The project also includes hydraulic actuator examples and a standalone fixed-pivot tail test.

## Setup

Use Python 3.11. Install FFmpeg and make it available on PATH for MP4 export.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Recorded ellipsoid swimming

The recorded eel uses the measured bend of each section to prescribe its shape.
MuJoCo evaluates the load on each fluid ellipsoid. The solver sums the forces
and the moments about the center of mass, including each force’s moment arm
and each ellipsoid’s direct torque. It integrates the resulting motion in
world X, world Y, and yaw. Depth, roll, and pitch stay fixed.

### Open the interactive viewer on macOS

Run from the repository root. The examples use the [five-actuator lab recording](src/5_030Hz_LinearA_16L_test3_pose_bends.csv).
Pass a different path to `--csv` to use your own recording.

```sh
./.venv/bin/python .venv/bin/mjpython examples/recorded_eel_run.py \
  --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv \
  --planar-swim --mass-kg 4.116 --fit-harmonics --view --force-arrows
```

- `--planar-swim` selects ellipsoid loads with free translation and yaw.
- `--mass-kg 4.116` sets the **whole robot’s** mass for the lab eel. The generated
  chain distributes that mass equally across its actuators.
- `--fit-harmonics` fits the bends at 0.3 and 0.9 Hz with a moving baseline.
  Leave it out to replay the measured bends with linear interpolation. The fit
  uses a centered window and trims about 1.67 seconds from each end.
- `--force-arrows` displays the five largest local forces at each frame.
  Use `--all-force-arrows` to show every nonzero ellipsoid force.
  Set the visual scale with `--force-arrow-scale` in m/N.

Enable **Group 3** under **Geom groups** to show the fluid ellipsoids. Toggle
**Group 0** to hide the CAD meshes. The green arrow shows COM velocity. The
second `mjpython` path in the command launches MuJoCo’s macOS viewer through the
same virtual environment and works with spaces in the project path.

### Export video or data

Use ordinary Python for a synchronized MP4 and CSV export. `--ellipsoid-view`
renders a top-down view of the ellipsoids with the force arrows.

```sh
./.venv/bin/python examples/recorded_eel_run.py \
  --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv \
  --planar-swim --mass-kg 4.116 --fit-harmonics \
  --video --ellipsoid-view --all-force-arrows
```

Omit the three video options on the last line for a data-only run. Use
`--output-dir Outputs/my_run` to keep separate runs. With the harmonic fit,
the default folder is `Outputs/recorded_eel_fit_planar_swim/`.

| Output | Contents |
| --- | --- |
| `swim.csv` | Net fluid force, yaw moment, and integrated COM and root motion |
| `state.csv`, `hinges.csv` | Body and hinge states at export timestamps |
| `motion.mp4`, `snapshot.jpg` | Video and still image with `--video` |
| `run.json`, `eel.xml` | Run settings and generated model |

In the CSVs, world +X points toward the tail at the initial heading. The video
overlay reports COM X displacement and velocity with positive values toward
the initial head direction.

### Recording format and other modes

Supply `t_s` in seconds and one signed relative bend in degrees for each
actuator, named consecutively from `act1_bend_deg` through `actN_bend_deg`.
An optional `usable` column flags excluded rows with zero. The lab recording
also includes measured head heading and tail tracking for comparison.

Omit the swimming mode to replay the body shape with a fixed base. Use
`--swim` for longitudinal motion with fixed heading or `--inertia-swim` for
the separate planar baseline that uses MuJoCo’s inertia-box drag.
See the [playback guide](docs/eel_playback.md) for interpolation and solver details.

### Choose an ellipsoid configuration

The five ready-to-run XML files in `Geometry/ellipsoids/` share the CAD assets
under `Geometry/`. Each model has five actuators with ten soft divisions
apiece and a nominal total mass of 4.116 kg.

| Model | Fluid geometry | Ellipsoids |
| --- | --- | ---: |
| [soft_with_tail.xml](Geometry/ellipsoids/soft_with_tail.xml) | One per soft span, two rigid ends per actuator, head and tail | 17 |
| [soft_without_tail.xml](Geometry/ellipsoids/soft_without_tail.xml) | One per soft span, two rigid ends per actuator, head | 16 |
| [full_with_tail.xml](Geometry/ellipsoids/full_with_tail.xml) | One per complete actuator, head and tail | 7 |
| [full_without_tail.xml](Geometry/ellipsoids/full_without_tail.xml) | One per complete actuator, head | 6 |
| [per_slice_with_tail.xml](Geometry/ellipsoids/per_slice_with_tail.xml) | One per soft division, two rigid ends per actuator, head and tail | 62 |

The last row is the original per-slice layout with 50 soft-slice ellipsoids,
10 rigid-end ellipsoids, and one each for the head and tail. The two tailless
models omit the tail mesh as well as its fluid ellipsoid. All five use the
same ten-division bending resolution.

```sh
./.venv/bin/python .venv/bin/mjpython examples/recorded_eel_run.py \
  --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv --model Geometry/ellipsoids/per_slice_with_tail.xml \
  --planar-swim --mass-kg 4.116 --fit-harmonics --view --all-force-arrows \
  --output-dir Outputs/per_slice_with_tail
```

Swap the XML and output folder to compare variants. For an MP4, use ordinary
Python and replace `--view` with `--video --ellipsoid-view`. With no
`--model`, the runner generates the soft-span layout for the recording’s
section count.

### Reproduce the tail-force audit

Run the full-actuator model with the tail attached. The 13.5–13.9 s interval
refers to the recording timestamps shown in the video overlay.

```sh
./.venv/bin/python examples/recorded_eel_run.py \
  --csv src/5_030Hz_LinearA_16L_test3_pose_bends.csv \
  --model Geometry/ellipsoids/full_with_tail.xml \
  --planar-swim --mass-kg 4.116 --fit-harmonics \
  --video --ellipsoid-view --all-force-arrows \
  --output-dir Outputs/audit_full_with_tail
```

For the tailless comparison, use `Geometry/ellipsoids/full_without_tail.xml`
and save to `Outputs/audit_full_without_tail` with the same remaining options.

## Run without a recording

The pressure-based hydraulic demo uses a built-in flow sequence. The
calibrated demo uses the volume-to-angle actuator model.

```sh
./.venv/bin/python -m src.hydraulic.run_hydraulic_section
./.venv/bin/python -m src.actuator.run_calibrated_section
./.venv/bin/python examples/traveling_wave_run.py --data-only
```

For an isolated fixed-pivot sweep, see the [tail test guide](docs/tail_test.md).

## Project layout

| Folder | Contents |
| --- | --- |
| `src/actuator/` | Calibrated volume-to-angle model, ESC-to-flow mapping, and runner |
| `src/hydraulic/` | Pressure-based hydraulic model, flow signals, and runner |
| `src/kinematics/` | Recorded angles, hinge poses, and traveling-wave motion |
| `src/hydrodynamics/` | Fluid forces and planar swimming integration |
| `src/export/` | Video, viewer, and telemetry export |
| `src/geometry/` | Section and chain generators |
| `src/tail_test/` | Isolated tail sweep and its diagnostics |
| `Geometry/` | Model XMLs, CAD meshes, and profile data |
| `configs/` | Default parameters and fitted coefficients |
| `examples/`, `docs/` | Run scripts and guides |

Generated exports go to `Outputs/`.

## Attribution

This project includes work derived from [FishSim](https://github.com/srl-ethz/fishsim) by the Soft Robotics Lab, ETH Zürich. See [LICENSE](LICENSE).
