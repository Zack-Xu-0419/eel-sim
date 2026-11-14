"""Separate volume-angle actuator API, CLI and prescribed MuJoCo playback.

The calibrated model has no pressure state/output. Existing pressure-based
modules, defaults and commands are not used or modified by this runner.
All solver values use SI; CLI flow/volume/angle inputs use mL/s, mL, degrees.

Public entry points:
    load_config()    read coefficients and their provenance
    simulate()       produce successive model samples and prescribed poses
    run_experiment() save that trajectory and optionally render it
    main()           translate terminal options into those same API calls
"""
import argparse
from bisect import bisect_right
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path

import mujoco

from src.actuator.calibrated_actuator_model import CalibratedActuatorModel, CalibratedParameters, _finite as finite
from src.actuator.calibrated_section_kinematics import AngleLimitError, SectionKinematics, playback
from src.actuator.calibrated_signals import SineFlow, SquareFlow

from src.paths import ROOT
# Resolve defaults from this file's location, not the terminal's current folder.
DEFAULT_CONFIG = ROOT/'configs/calibrated_actuator_config.json'
DEFAULT_MODEL = ROOT/'Geometry/generated/section_n10.xml'
SAMPLE_FIELDS = ['time_s', 'volume_m3', 'angle_rad', 'pump_flow_m3_s',
                 'leak_flow_m3_s', 'net_flow_m3_s']


def load_config(path=DEFAULT_CONFIG, medium='air'):
    """Return (provenance dictionary, validated parameter object).

    The JSON status describes calibration limits; parameters_si is the numeric
    source of truth. Choosing water selects fitted coefficients only, not fluid
    forces or an underwater environment. This does not perform a new data fit.
    """
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or not isinstance(config.get('status'), str) or not config['status'].strip():
        raise ValueError('Volume-angle config requires a nonempty status string')
    try:
        # Unpack the selected JSON dictionary into the named dataclass fields.
        parameters = CalibratedParameters(**config['parameters_si'][medium.lower()])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Use a volume-angle config with parameters_si[medium]; the pressure config is incompatible') from exc
    return config, parameters


def demo_flow(time_s):
    """Actual pump flow: +2 mL/s, off, -2 mL/s, off, in three-second blocks."""
    if time_s < 3:
        return 2e-6
    if time_s < 6:
        return 0.0
    if time_s < 9:
        return -2e-6
    return 0.0


def simulate(section, actuator, flow, duration_s=12., dt_s=.001, flow_breakpoints_s=(), *, stop_when=None):
    """Yield one trajectory; input flow(t_s) returns actual signed m3/s.

    Exact constant-input integration is performed by the calibrated actuator.
    Smooth flow is sampled at interval midpoints. Known discontinuities split
    intervals exactly. Duration is the absolute simulation end time. Initial
    pump flow matches the first held interval. No MuJoCo dynamics are stepped.
    Optional stop_when(sample) is checked once after each completed step,
    excluding the initial sample. The triggering sample is retained.

    This is a generator: each yield returns one immutable sample, and execution
    resumes at the following statement when the caller asks for the next one.
    """
    end, dt = finite(duration_s, 'duration_s'), finite(dt_s, 'dt_s')
    if end <= actuator.time_s or dt <= 0:
        raise ValueError('duration_s must exceed current time and dt_s must be positive')
    if stop_when is not None and not callable(stop_when):
        raise ValueError('stop_when must be callable or None')
    # Smooth inputs can be sampled at midpoints. For a discontinuity, split at
    # the switch first: otherwise one held input could span opposite flow signs.
    boundaries = [finite(b, 'flow breakpoint') for b in flow_breakpoints_s]
    if flow is demo_flow:
        boundaries.extend((3., 6., 9.))
    if callable(getattr(flow, 'switch_times', None)):
        boundaries.extend(finite(b, 'waveform switch') for b in flow.switch_times(actuator.time_s, end))
    boundaries = sorted(set(b for b in boundaries if actuator.time_s < b < end))
    # Frame/CSV zero needs the input of the first integration interval. It uses
    # the same midpoint as the upcoming step, rather than an unrelated Q(0).
    first_end = min(actuator.time_s+dt, end)
    if boundaries:
        first_end = min(first_end, boundaries[0])
    sample = actuator.sample(flow((actuator.time_s+first_end)/2))
    section.apply(sample.angle_rad, sample.time_s)
    yield sample
    # Compute regular targets from an integer tick and fixed origin instead of
    # repeatedly adding dt. A switch can split a tick without shifting the grid.
    origin, tick = actuator.time_s, 1
    while actuator.time_s < end:
        t = actuator.time_s
        regular = origin+tick*dt
        target = min(regular, end)
        # bisect_right skips a boundary already reached, avoiding zero-length steps.
        index = bisect_right(boundaries, t)
        if index < len(boundaries) and boundaries[index] < target:
            target = boundaries[index]
        h = target-t
        # flow() returns actual signed m3/s. It is held constant during this step;
        # the exact linear solution applies to that held value, not any arbitrary
        # continuous waveform between samples.
        pump = finite(flow(t+h/2), 'pump flow')
        previous_angle = actuator.sample(pump).angle_rad
        # This is the only call that advances hydraulic time/volume. Applying the
        # returned angle changes MuJoCo's pose but does not integrate it again.
        sample = actuator.step(pump, h)
        section.apply(sample.angle_rad, sample.time_s, (sample.angle_rad-previous_angle)/h)
        yield sample
        # Evaluate after yielding so every valid endpoint remains in the
        # history, including one at which a user predicate raises an error.
        if stop_when is not None and bool(stop_when(sample)):
            return
        if target == regular:
            tick += 1


@dataclass
class ExperimentResult:
    """Return calculated history and status to Python callers.

    exit_code=0 means success; 2 means a handled simulation/export failure.
    Samples and metadata remain available for inspection in either case.
    """
    section: SectionKinematics
    samples: list
    record: dict
    exit_code: int


def run_experiment(section, actuator, flow, output_dir, *, duration_s=12., dt_s=.001,
                   config=None, medium=None, config_path=None, flow_metadata=None,
                   export_options=None, flow_breakpoints_s=(), stop_when=None):
    """Pressure-free CSV + optional synchronized video/JPG experiment API.

    Construct the actuator with desired CalibratedParameters first. config is
    provenance only; actual parameters are always recorded from the object.
    If a config/medium pair is supplied its SI coefficients must match, avoiding
    a fitted label on unrelated custom coefficients. Reusing an output folder
    replaces its trajectory.csv/run.json and requested successful media.
    stop_when(sample) can end a run normally after a completed step.
    duration_s remains an absolute end-time cap; CSV, media, and playback
    all use the actual retained history.
    """
    if not isinstance(actuator, CalibratedActuatorModel):
        raise ValueError('This runner requires CalibratedActuatorModel, not the pressure-based model')
    duration, dt = finite(duration_s, 'duration_s'), finite(dt_s, 'dt_s')
    if duration <= actuator.time_s or dt <= 0:
        raise ValueError('duration_s must exceed current time and dt_s must be positive')
    if stop_when is not None and not callable(stop_when):
        raise ValueError('stop_when must be callable or None')
    parameters = asdict(actuator.parameters)
    if config is not None:
        # Provenance is not an alternate parameter source inside this function.
        # Reject mismatches so run.json cannot claim one fit while using another.
        if not isinstance(config, dict) or not isinstance(config.get('status'), str) or not config['status'].strip():
            raise ValueError('Config provenance requires a nonempty status string')
        if medium is None or config.get('parameters_si', {}).get(medium.lower()) != parameters:
            raise ValueError('Config coefficients must match actuator.parameters for the selected medium')
    if export_options is not None and export_options.snapshot_time_s is not None:
        if not actuator.time_s <= export_options.snapshot_time_s <= duration:
            raise ValueError('Snapshot time lies outside the experiment interval')
    if flow_metadata is None and callable(getattr(flow, 'metadata', None)):
        flow_metadata = flow.metadata()
    boundaries = tuple(finite(b, 'flow breakpoint') for b in flow_breakpoints_s)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Record actual coefficients, units, assumptions and input definition so the
    # result can be understood later without reconstructing terminal arguments.
    record = {'mode': 'prescribed_angle_kinematics', 'model_type': 'volume_angle_leak',
              'status': 'running', 'calibration_status': config['status'] if config is not None else 'custom_coefficients',
              'medium': medium, 'parameters_si': parameters, 'config': config,
              'config_path': str(Path(config_path).expanduser().resolve()) if config_path else None,
              'model_path': str(section.path), 'dt_s': dt, 'duration_s': duration,
              'initial_state': {'time_s': actuator.time_s, 'volume_m3': actuator.volume_m3},
              'flow_input': flow_metadata or {'type': 'python_callable', 'name': getattr(flow, '__name__', type(flow).__name__)},
              'explicit_flow_breakpoints_s': list(boundaries), 'joint_names': section.joint_names,
              'stop_condition': getattr(stop_when, '__name__', type(stop_when).__name__) if stop_when is not None else None,
              'equations': ['theta=kV*v', 'Qleak=G*(theta-theta_eq)', 'v_dot=Qpump-Qleak'],
              'units': {'time_s': 's', 'volume_m3': 'm^3', 'angle_rad': 'rad',
                        'angle_velocity_rad_s': 'rad/s', 'all_flows': 'm^3/s'},
              'integration': 'Exact linear solution with held midpoint pump flow per interval',
              'csv_flow_convention': 'Completed-step held input; initial sample uses the first held input; leak/net are endpoint values',
              'angle_velocity_convention': 'Completed-interval finite difference; initial row uses the first interval',
              'assumptions': ['Actual pump flow is supplied; no ESC/PWM conversion',
                              'v is net transferred volume; chamber difference changes by 2*v',
                              'One common total angle; equal increments over n-1 hinges; first slice fixed',
                              'Fixed linear gain and effective leakage; no hysteresis or load-dependent bending',
                              'Stopped-pump leakage coefficient is also used while pumping',
                              'No surrounding-water forces; medium only selects fitted coefficients',
                              'Fitted rest angle can include camera/reference bias',
                              'MuJoCo updates prescribed geometry, not force-driven bending'],
              'media_export': {'status': 'pending' if export_options is not None else 'not_requested'}}
    samples, failure = [], None
    stop_triggered = False

    def check_stop(sample):
        nonlocal stop_triggered
        stop_triggered = bool(stop_when(sample))
        return stop_triggered

    try:
        # Calculate once. Both CSV export and rendering below use this list.
        for sample in simulate(section, actuator, flow, duration, dt, boundaries,
                               stop_when=check_stop if stop_when is not None else None):
            samples.append(sample)
    except (ValueError, OverflowError) as exc:
        # A joint-limit failure can occur after the hydraulic state advanced.
        # Keep its attempted volume in metadata, while CSV keeps valid poses only.
        failure = str(exc)
        record['failure'] = {'message': failure,
                             'attempted_state': {'time_s': actuator.time_s, 'volume_m3': actuator.volume_m3}}
    # Derive angular velocity from consecutive total angles. This is the same
    # piecewise-linear rate used by media telemetry, not a new motion simulation.
    rates = [(b.angle_rad-a.angle_rad)/(b.time_s-a.time_s) for a, b in zip(samples, samples[1:])]
    joint_fields = [name+'_rad' for name in section.joint_names]
    with (output/'trajectory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS+['angle_velocity_rad_s']+joint_fields)
        writer.writeheader()
        for i, sample in enumerate(samples):
            row = asdict(sample)
            # A normal row uses the interval ending there; row zero uses the
            # first interval because no completed interval exists before t=0.
            row['angle_velocity_rad_s'] = rates[max(0, i-1)] if rates else None
            row.update(zip(joint_fields, section.angles(sample.angle_rad)))
            writer.writerow(row)
    record.update(status='stopped_with_error' if failure else 'completed', samples_written=len(samples),
                  last_valid_time_s=samples[-1].time_s if samples else None,
                  actual_duration_s=samples[-1].time_s-record['initial_state']['time_s'] if samples else 0.,
                  termination_reason='error' if failure else ('stop_condition' if stop_triggered else 'duration_limit'))

    def save_record():
        """Persist current status; disallow non-finite JSON numeric values."""
        (output/'run.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')

    if failure and export_options is not None:
        record['media_export'] = {'status': 'skipped', 'message': 'Simulation did not complete'}
    save_record()
    print(f"Model: {record['calibration_status']}; prescribed volume-angle motion.")
    print(f"CSV: {output/'trajectory.csv'}")
    if failure:
        print(f'ERROR: {failure}. Valid samples and attempted volume are preserved; no clamping.')
        return ExperimentResult(section, samples, record, 2)
    if export_options is not None:
        # Import renderer/encoder code only when media were requested. Export
        # samples are interpolated for frame timing, never hydraulically rerun.
        from src.export.calibrated_section_export import export_media
        try:
            record['media_export'] = export_media(section, samples, export_options)
        except (ValueError, RuntimeError, OSError, ImportError, mujoco.FatalError) as exc:
            # Encoding and simulation have independent status: a failed video
            # must not destroy a successful trajectory.csv.
            record['media_export'] = {'status': 'failed', 'message': str(exc)}
            save_record()
            print(f'ERROR: Media export failed: {exc}. Calculated CSV is preserved.')
            return ExperimentResult(section, samples, record, 2)
        save_record()
    print(f"Completed {samples[-1].time_s:g} s ({record['termination_reason']}); peak total bend {max(abs(s.angle_rad) for s in samples)*180/math.pi:.3f} deg")
    return ExperimentResult(section, samples, record, 0)


def main(argv=None):
    """Command-line wrapper around the importable API; no separate physics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--medium', choices=('air', 'water'), default='air')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'Outputs/calibrated_section')
    parser.add_argument('--waveform', choices=('demo', 'constant', 'sine', 'square'), default='demo')
    parser.add_argument('--flow-ml-s', type=float, help='Actual constant signed flow; selects constant waveform')
    parser.add_argument('--amplitude-ml-s', type=float, default=2.)
    parser.add_argument('--frequency-hz', type=float, default=.3)
    parser.add_argument('--offset-ml-s', type=float, default=0.)
    parser.add_argument('--phase-deg', type=float, default=0.)
    parser.add_argument('--duty-cycle', type=float, default=.5)
    initial = parser.add_mutually_exclusive_group()
    # Either choose an initial transferred volume or an equivalent initial bend.
    initial.add_argument('--initial-volume-ml', type=float)
    initial.add_argument('--initial-angle-deg', type=float)
    parser.add_argument('--duration', type=float, default=12.)
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--view', action='store_true', help='macOS: launch using mjpython')
    for name, metavar in [('video', 'MP4'), ('jpg', 'JPG'), ('frames', 'DIR')]:
        # Flag absent -> None; flag alone -> True/default path; flag + path -> Path.
        parser.add_argument('--save-'+name, nargs='?', const=True, type=Path, metavar=metavar)
    parser.add_argument('--snapshot-time', type=float)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--video-crf', type=int, default=18)
    parser.add_argument('--jpg-quality', type=int, default=95)
    parser.add_argument('--overlay', action='store_true')
    args = parser.parse_args(argv)
    try:
        config, parameters = load_config(args.config, args.medium)
        duration, dt = finite(args.duration, 'duration'), finite(args.dt, 'dt')
        if duration <= 0 or dt <= 0:
            raise ValueError('duration and dt must be positive')
        initial_volume = finite(args.initial_volume_ml or 0., 'initial-volume-ml')*1e-6
        if args.initial_angle_deg is not None:
            # theta=kV*v, so this initializes the same state through v=theta/kV.
            initial_volume = math.radians(finite(args.initial_angle_deg, 'initial-angle-deg'))/parameters.volume_to_angle_rad_per_m3
        if args.flow_ml_s is not None:
            if args.waveform not in ('demo', 'constant'):
                raise ValueError('--flow-ml-s cannot be combined with sine/square waveform')
            # Capture the converted constant once; t is accepted to match the
            # same callable interface used for sine/square/custom waveforms.
            flow = lambda t, q=finite(args.flow_ml_s, 'flow-ml-s')*1e-6: q
            flow_metadata = {'type': 'constant', 'flow_ml_s': args.flow_ml_s}
        elif args.waveform == 'constant':
            raise ValueError('--waveform constant requires --flow-ml-s')
        elif args.waveform == 'demo':
            flow, flow_metadata = demo_flow, {'type': 'demo', 'segments_s': [0, 3, 6, 9, 12], 'flow_ml_s': [2, 0, -2, 0]}
        else:
            # Waveform constructors accept display units; their __call__ methods
            # convert to m3/s when simulate() samples them.
            options = dict(amplitude_ml_s=args.amplitude_ml_s, frequency_hz=args.frequency_hz,
                           offset_ml_s=args.offset_ml_s, phase_deg=args.phase_deg)
            flow = SineFlow(**options) if args.waveform == 'sine' else SquareFlow(**options, duty_cycle=args.duty_cycle)
            flow_metadata = flow.metadata()
        output = args.output_dir.expanduser().resolve()
        export = None
        if any(v is not None for v in (args.save_video, args.save_jpg, args.save_frames)):
            # ExportOptions describes the desired output; creating it does not
            # launch MuJoCo rendering or run the actuator model.
            from src.export.calibrated_section_export import ExportOptions
            export = ExportOptions(video_path=output/'motion.mp4' if args.save_video is True else args.save_video,
                                   jpg_path=output/'snapshot.jpg' if args.save_jpg is True else args.save_jpg,
                                   frames_dir=output/'frames' if args.save_frames is True else args.save_frames,
                                   width=args.width, height=args.height, fps=args.fps,
                                   video_crf=args.video_crf, jpg_quality=args.jpg_quality,
                                   snapshot_time_s=args.snapshot_time, overlay=args.overlay,
                                   overlay_label=f'MODEL VALUES • PROVISIONAL • {args.medium.upper()}')
            if args.snapshot_time is not None and not 0 <= args.snapshot_time <= duration:
                raise ValueError('Snapshot time lies outside the experiment interval')
        elif args.snapshot_time is not None or args.overlay:
            raise ValueError('--snapshot-time/--overlay requires a requested media output')
        # The XML already exists (made by the geometry generator). Loading it
        # creates the mechanical scene; this does not generate a new CAD shape.
        section = SectionKinematics(args.model)
        actuator = CalibratedActuatorModel(parameters, initial_volume)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    result = run_experiment(section, actuator, flow, output, duration_s=duration, dt_s=dt,
                            config=config, medium=args.medium, config_path=args.config,
                            flow_metadata=flow_metadata, export_options=export)
    if result.exit_code:
        return result.exit_code
    if args.view:
        playback(result.section, result.samples)
    return 0


if __name__ == '__main__':
    # Importing run_experiment does not execute main(); launching this file does.
    raise SystemExit(main())
