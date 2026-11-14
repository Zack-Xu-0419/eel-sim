"""Run the reduced hydraulic model and optionally play prescribed poses.

Headless: .venv/bin/python -m src.hydraulic.run_hydraulic_section
macOS viewer: .venv/bin/python .venv/bin/mjpython -m src.hydraulic.run_hydraulic_section --view
Export: .venv/bin/python -m src.hydraulic.run_hydraulic_section --save-video --save-jpg

This uses mj_forward, never mj_step: passive mechanics do not determine motion.
"""

import argparse
from bisect import bisect_right
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import re
import time

import mujoco
import numpy as np

from src.hydraulic.hydraulic_model import HydraulicModel, HydraulicParameters, HydraulicRelations, finite


from src.paths import ROOT
SAMPLE_FIELDS = ["time_s", "volume_m3", "angle_rad", "pressure_difference_pa",
                 "pump_flow_m3_s", "leak_flow_m3_s", "net_flow_m3_s"]


class AngleLimitError(ValueError):
    pass


class SectionKinematics:
    """Map a total bend to the generated section using named joint addresses."""

    def __init__(self, model_path):
        self.path = Path(model_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        joints = []
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
        self.qpos_addresses = m.jnt_qposadr[self.joint_ids].copy()
        self.dof_addresses = m.jnt_dofadr[self.joint_ids].copy()
        try:
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
        angles = self.angles(total_angle_rad)  # Check before changing the displayed pose.
        velocity = finite(angle_velocity_rad_s, "angle velocity") / len(angles)
        clock = finite(time_s, "time_s")
        self.data.qpos[self.qpos_addresses] = angles
        self.data.qvel[self.dof_addresses] = velocity
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.data.time = clock
        mujoco.mj_forward(self.model, self.data)


def demo_flow(time_s):
    """Actual signed pump flow, not a motor on/off command."""
    if time_s < 3.:
        return 2e-6
    if time_s < 6.:
        return 0.
    if time_s < 9.:
        return -2e-6
    return 0.


def simulate(section, hydraulic, flow, duration_s=12., dt_s=.001, flow_breakpoints_s=()):
    """Yield samples; sample flow is the held input of the completed step.

    Evaluate the input at step midpoints (zero-order hold). Split the default
    demo and declared waveform discontinuities at their switching times so
    even a non-divisor timestep stays accurate. Callable waveform objects can
    supply switch_times(start_s, end_s); plain functions can pass explicit
    flow_breakpoints_s.
    """
    duration = finite(duration_s, "duration_s")
    dt = finite(dt_s, "dt_s")
    if duration <= 0 or dt <= 0:
        raise ValueError("duration_s and dt_s must be positive")
    if duration <= hydraulic.state.time_s:
        raise ValueError("duration_s must exceed the current hydraulic time")
    boundaries = [finite(value, 'flow breakpoint') for value in flow_breakpoints_s]
    if flow is demo_flow:
        boundaries.extend((3., 6., 9.))
    if callable(getattr(flow, 'switch_times', None)):
        boundaries.extend(finite(value, 'flow breakpoint')
                          for value in flow.switch_times(hydraulic.state.time_s, duration))
    boundaries = sorted(set(b for b in boundaries if hydraulic.state.time_s < b < duration))
    first_target = min(hydraulic.state.time_s+dt, duration)
    if boundaries:
        first_target = min(first_target, boundaries[0])
    initial = hydraulic.sample(flow((hydraulic.state.time_s+first_target)/2))
    section.apply(initial.angle_rad, initial.time_s)
    yield initial
    origin, tick = hydraulic.state.time_s, 1
    while hydraulic.state.time_s < duration:
        t = hydraulic.state.time_s
        regular_target = origin + tick*dt
        target = min(regular_target, duration)
        next_boundary = bisect_right(boundaries, t)
        if next_boundary < len(boundaries) and boundaries[next_boundary] < target:
            target = boundaries[next_boundary]
        h = target-t
        pump = finite(flow(t+h/2), "pump flow signal")
        previous_angle = hydraulic.sample(pump).angle_rad
        sample = hydraulic.step(pump, h)
        section.apply(sample.angle_rad, sample.time_s, (sample.angle_rad-previous_angle)/h)
        yield sample
        if target == regular_target:
            tick += 1


def load_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    if not isinstance(config.get("calibration_status"), str) or not config["calibration_status"].strip():
        raise ValueError("Configuration needs an explicit calibration_status")
    try:
        parameters = HydraulicParameters(**config["parameters"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid hydraulic parameters: {exc}") from exc
    return config, parameters


@dataclass
class ExperimentResult:
    """Calculated samples, run metadata and model for further export/playback."""
    section: SectionKinematics
    samples: list
    record: dict
    exit_code: int


def run_experiment(section, hydraulic, flow, output_dir, *, duration_s=12., dt_s=.001,
                   config=None, config_path=None, flow_metadata=None,
                   export_options=None, flow_breakpoints_s=()):
    """Run a Python-defined experiment and save its synchronized data/media.

    section is a SectionKinematics object; hydraulic is a fresh HydraulicModel;
    flow is a callable time_s -> actual pump_flow_m3_s. SineFlow/SquareFlow
    provide input metadata and square-wave switch times automatically. A plain
    callable can declare discontinuities using flow_breakpoints_s.

    Returned exit_code is 0 on success, 2 on a simulation/export failure; valid
    samples and run.json are preserved in the failure case. The duration is an
    absolute simulation end time, matching simulate(). New models start at 0.
    config records metadata; construct hydraulic with the desired parameters
    first. The provided example constructs both from the same parameter object.
    """
    duration, dt = finite(duration_s, 'duration_s'), finite(dt_s, 'dt_s')
    if duration <= hydraulic.state.time_s or dt <= 0:
        raise ValueError('duration_s must exceed current time and dt_s must be positive')
    output = Path(output_dir).expanduser().resolve()
    config = config if config is not None else {'calibration_status': 'uncalibrated_custom_relations'}
    if (not isinstance(config, dict) or not isinstance(config.get('calibration_status'), str)
            or not config['calibration_status'].strip()):
        raise ValueError('Provide an explicit calibration_status in config')
    flow_breakpoints_s = tuple(finite(value, 'flow breakpoint') for value in flow_breakpoints_s)
    if flow_metadata is None and callable(getattr(flow, 'metadata', None)):
        flow_metadata = flow.metadata()
    if export_options is not None and export_options.snapshot_time_s is not None:
        if not hydraulic.state.time_s <= export_options.snapshot_time_s <= duration:
            raise ValueError('Snapshot time lies outside the experiment interval')
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "mode": "prescribed_angle_kinematics", "status": "running",
        "model_path": str(section.path), "config_path": str(Path(config_path).expanduser().resolve()) if config_path is not None else None,
        "config": config, "dt_s": dt, "duration_s": duration,
        "initial_hydraulic_state": asdict(hydraulic.state),
        "explicit_flow_breakpoints_s": list(flow_breakpoints_s),
        "joint_names": section.joint_names, "flow_input_units": "m^3/s internally; CLI uses mL/s",
        "flow_input": flow_metadata or {"type": "python_callable", "name": getattr(flow, "__name__", type(flow).__name__)},
        "equations": ["theta = angle_from_volume(v)", "delta_p = pressure_from_angle(theta)", "Q_leak = leak_flow_from_pressure(delta_p)", "v_dot = Q_p - Q_leak"],
        "assumptions": ["balanced zero-volume and zero-angle reference", "relations configured by caller; no experimental validation",
                        "uniform joint-angle increments approximate constant curvature", "first rigid slice is fixed",
                        "no pressure storage, hysteresis or load-dependent bending", "no surrounding-water forces",
                        "zero Q_p means zero actual pump flow, not necessarily a switched-off physical pump"],
        "csv_flow_convention": "At t>0, pump flow is the held input of the completed step; initial row uses the first interval; other flows are endpoint values",
        "csv_angle_velocity_convention": "Finite difference of total angle over the completed step; first interval at t=0; blank if no interval completed",
        "media_export": {"status": "pending" if export_options is not None else "not_requested"},
    }
    samples, failure = [], None
    joint_fields = [f"{name}_rad" for name in section.joint_names]
    fields = SAMPLE_FIELDS+['angle_velocity_rad_s']+joint_fields
    try:
        with (output/"trajectory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            pending_row = None
            try:
                for sample in simulate(section, hydraulic, flow, duration, dt, flow_breakpoints_s):
                    row = asdict(sample)
                    row['angle_velocity_rad_s'] = (float(section.data.qvel[section.dof_addresses].sum())
                                                  if samples else None)
                    row.update(zip(joint_fields, section.data.qpos[section.qpos_addresses]))
                    if pending_row is not None:
                        if pending_row['angle_velocity_rad_s'] is None:
                            pending_row['angle_velocity_rad_s'] = row['angle_velocity_rad_s']
                        writer.writerow(pending_row)
                    pending_row = row
                    samples.append(sample)
            finally:
                if pending_row is not None:
                    writer.writerow(pending_row)
    except (ValueError, OverflowError) as exc:
        failure = str(exc)
        record["failure"] = {"message": failure, "attempted_state": asdict(hydraulic.state)}
    record["status"] = "stopped_with_error" if failure else "completed"
    if failure and export_options is not None:
        record['media_export'] = {"status": "skipped", "message": "Hydraulic simulation did not complete"}
    record["samples_written"] = len(samples)
    record["last_valid_time_s"] = samples[-1].time_s if samples else None
    (output/"run.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(f"Calibration: {config['calibration_status']}; motion is prescribed, not predicted by mechanical forces.")
    print(f"CSV: {output/'trajectory.csv'}")
    print(f"Run record: {output/'run.json'}")
    if failure:
        print(f"ERROR: {failure}")
        return ExperimentResult(section, samples, record, 2)
    print(f"Completed {samples[-1].time_s:g} s; peak total bend {max(abs(s.angle_rad) for s in samples)*180/math.pi:.3f} deg")
    if export_options is not None:
        from src.export.hydraulic_export import export_media
        try:
            record['media_export'] = export_media(section, samples, export_options)
        except (ValueError, RuntimeError, OSError, ImportError, mujoco.FatalError) as exc:
            record['media_export'] = {"status": "failed", "message": str(exc)}
            (output/'run.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
            print(f'ERROR: Media export failed: {exc}. The calculated CSV is preserved.')
            return ExperimentResult(section, samples, record, 2)
        (output/'run.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        for name in ('video_path', 'jpg_path', 'frames_dir', 'video_telemetry_csv'):
            if record['media_export'].get(name):
                print(f"{name}: {record['media_export'][name]}")
    return ExperimentResult(section, samples, record, 0)


def playback(section, samples):
    """Play one pass at real time; macOS requires the mjpython launcher."""
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
            while viewer.is_running():
                remaining = frame.time_s-frames[0].time_s-(time.monotonic()-start)
                if remaining <= 0:
                    break
                time.sleep(min(remaining, .01))
            if not viewer.is_running():
                break
            elapsed = frame.time_s-previous.time_s
            velocity = (frame.angle_rad-previous.angle_rad)/elapsed if elapsed > 0 else 0.
            with viewer.lock():
                section.apply(frame.angle_rad, frame.time_s, velocity)
            viewer.sync()
            previous = frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT/"Geometry/generated/section_n10.xml")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/hydraulic_config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"Outputs/hydraulic_section")
    parser.add_argument("--flow-ml-s", type=float, help="Constant actual pump flow in mL/s; otherwise use the demo")
    parser.add_argument("--duration", type=float, default=12., help="Run duration in seconds")
    parser.add_argument("--dt", type=float, default=.001, help="Integration timestep in seconds")
    parser.add_argument("--view", action="store_true", help="Play poses once at real time; use mjpython on macOS")
    parser.add_argument("--save-video", nargs="?", const=True, type=Path, metavar="MP4",
                        help="Save an MP4; default path is OUTPUT_DIR/motion.mp4")
    parser.add_argument("--save-jpg", nargs="?", const=True, type=Path, metavar="JPG",
                        help="Save a JPG at maximum absolute bend; default OUTPUT_DIR/snapshot.jpg")
    parser.add_argument("--save-frames", nargs="?", const=True, type=Path, metavar="DIR",
                        help="Save a JPG sequence and timestamps in a new/empty directory; default OUTPUT_DIR/frames")
    parser.add_argument("--snapshot-time", type=float, metavar="SECONDS",
                        help="Choose the --save-jpg time instead of the maximum bend")
    parser.add_argument("--width", type=int, default=1920, help="Export width in pixels (default 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Export height in pixels (default 1080)")
    parser.add_argument("--fps", type=float, default=60., help="Video/JPG sequence frame rate (default 60)")
    parser.add_argument("--video-crf", type=int, default=18, help="H.264 quality, 0–51; lower is better (default 18)")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPG quality, 1–100 (default 95)")
    parser.add_argument("--overlay", action="store_true",
                        help="Show time, flow, angle, angular velocity, volume and pressure on exported media")
    args = parser.parse_args(argv)
    try:
        output = args.output_dir.expanduser().resolve()
        export_options = None
        if any(value is not None for value in (args.save_video, args.save_jpg, args.save_frames)):
            from src.export.hydraulic_export import ExportOptions
            export_options = ExportOptions(
                video_path=output/'motion.mp4' if args.save_video is True else args.save_video,
                jpg_path=output/'snapshot.jpg' if args.save_jpg is True else args.save_jpg,
                frames_dir=output/'frames' if args.save_frames is True else args.save_frames,
                width=args.width, height=args.height, fps=args.fps,
                video_crf=args.video_crf, jpg_quality=args.jpg_quality,
                snapshot_time_s=args.snapshot_time, overlay=args.overlay)
        elif args.snapshot_time is not None:
            raise ValueError('--snapshot-time requires --save-jpg')
        elif args.overlay:
            raise ValueError('--overlay requires --save-video, --save-jpg or --save-frames')
        config, parameters = load_config(args.config)
        duration, dt = finite(args.duration, "duration"), finite(args.dt, "dt")
        if duration <= 0 or dt <= 0:
            raise ValueError("duration and dt must be positive")
        if args.snapshot_time is not None and not 0 <= args.snapshot_time <= duration:
            raise ValueError('Snapshot time must be between zero and the run duration')
        section = SectionKinematics(args.model)
        hydraulic = HydraulicModel(HydraulicRelations.linear(parameters), parameters.initial_volume_m3)
        flow = demo_flow if args.flow_ml_s is None else (
            lambda t, q=finite(args.flow_ml_s, "flow-ml-s")*1e-6: q)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    result = run_experiment(
        section, hydraulic, flow, output, duration_s=duration, dt_s=dt,
        config=config, config_path=args.config,
        flow_metadata={"type": "demo" if args.flow_ml_s is None else "constant",
                       "constant_flow_ml_s": args.flow_ml_s},
        export_options=export_options)
    if result.exit_code:
        return result.exit_code
    if args.view:
        playback(result.section, result.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
