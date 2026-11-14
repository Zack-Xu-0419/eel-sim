"""Run a connected chain with independently phased hydraulic actuators.

Default geometry: five 157 mm actuators, each with 26 mm rigid ends and ten
divisions of the 105 mm soft region. Gait settings are selected below.
Outputs are separate from custom_calibrated. --data-only skips MuJoCo rendering;
--view opens freely rotatable playback (use .venv/bin/mjpython on macOS).
"""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.actuator.calibrated_actuator_model import CalibratedActuatorModel
from src.export.calibrated_section_export import ExportOptions
from src.actuator.esc_pump_model import StaticEscParameters
from src.geometry.auto_hydraulic_chain import generate_chain
from src.actuator.run_calibrated_section import load_config
from src.kinematics.traveling_wave import (ChainKinematics, TravelingWaveCommand,
                            periodic_initial_volumes, simulate_chain)


def write_trajectory(path, chain, samples):
    """One CSV row per solver timestamp, with module states and local hinges."""
    state_names = ('volume_m3', 'angle_rad', 'pump_flow_m3_s', 'leak_flow_m3_s', 'net_flow_m3_s')
    fields = ['time_s']
    for module in range(chain.module_count):
        fields += [f'module_{module}_{name}' for name in
                   ('pwm_command', *state_names, 'angle_velocity_rad_s', 'tip_heading_rad')]
    fields += [name+'_rad' for name in chain.joint_names]
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(samples):
            if len(samples) > 1:
                a, b = (samples[max(0, index-1)], sample) if index else samples[:2]
                rates = (np.array(b.angles_rad)-a.angles_rad)/(b.time_s-a.time_s)
            else:
                rates = np.zeros(chain.module_count)
            headings = np.cumsum(sample.angles_rad)
            row = {'time_s': sample.time_s}
            for module, state in enumerate(sample.actuator_samples):
                prefix = f'module_{module}_'
                row[prefix+'pwm_command'] = sample.commands[module]
                row.update({prefix+name: getattr(state, name) for name in state_names})
                row[prefix+'angle_velocity_rad_s'] = rates[module]
                row[prefix+'tip_heading_rad'] = headings[module]
            row.update(zip((name+'_rad' for name in chain.joint_names), chain.angles(sample.angles_rad)))
            writer.writerow(row)


def plot_summary(path, chain, samples, period_s):
    """Plot calculated module angles and connected centerlines, not raw data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    clocks = np.array([s.time_s for s in samples])
    angles = np.array([s.angles_rad for s in samples])
    figure = plt.figure(figsize=(13, 7), layout='constrained')
    layout = figure.add_gridspec(2, 4, height_ratios=[1.3, 1])
    axis = figure.add_subplot(layout[0, :])
    for module in range(chain.module_count):
        axis.plot(clocks, np.rad2deg(angles[:, module]), color=chain.module_colors[module],
                  label=f'Actuator {module+1}')
    axis.set(xlabel='Time (s)', ylabel='Module relative bend (deg)',
             title='Independent hydraulic actuators: phase-delayed bending')
    axis.grid(alpha=.2)
    axis.legend(ncol=min(chain.module_count, 6), loc='upper right')
    extent = chain.total_length_m*1000
    snapshots = []
    for clock in np.linspace(clocks[0], min(clocks[-1], clocks[0]+.75*period_s), 4):
        totals = [np.interp(clock, clocks, angles[:, i]) for i in range(chain.module_count)]
        chain.apply(totals, float(clock))
        parts = []
        for module, group in enumerate(chain.joint_groups):
            points = [chain.data.site_xpos[chain.base_site_ids[module]]]
            points += [chain.data.site(f'module_{module}_marker_{i}').xpos
                       for i in range(len(group)+1)]
            points += [chain.data.site_xpos[chain.tip_site_ids[module]]]
            points = np.array(points)*1000
            parts.append(points)
        snapshots.append((float(clock), parts))
    y_limit = max(.1*extent, max(np.max(np.abs(points[:, 1]))
                  for _, parts in snapshots for points in parts))+.05*extent
    for panel, (clock, parts) in enumerate(snapshots):
        ax = figure.add_subplot(layout[1, panel])
        for module, points in enumerate(parts):
            ax.plot(points[:, 0], points[:, 1], color=chain.module_colors[module], lw=3)
        ax.plot(0, 0, 'ks', ms=4)
        ax.set(title=f't = {clock:.2f} s', xlabel='x (mm)', xlim=(-.03*extent, 1.03*extent),
               ylim=(-y_limit, y_limit), aspect='equal')
        if panel == 0:
            ax.set_ylabel('y (mm)')
        ax.grid(alpha=.2)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modules', type=int, default=5)
    parser.add_argument('--slices-per-module', '--soft-divisions', type=int, default=10,
                        help='Soft-region divisions per actuator; rigid ends add no joints')
    parser.add_argument('--soft-length-mm', type=float, default=105.,
                        help='Active soft length; total module length adds both rigid ends')
    parser.add_argument('--rigid-start-mm', type=float, default=26.)
    parser.add_argument('--rigid-end-mm', type=float, default=26.)
    parser.add_argument('--gap-mm', type=float, default=0.,
                        help='Optional visual spacing within the soft region; default is continuous')
    parser.add_argument('--duration', type=float, default=16.)
    parser.add_argument('--period', type=float, default=3.3)
    parser.add_argument('--phase-lag-deg', type=float, default=60.)
    parser.add_argument('--amplitude-command', type=float, default=80)
    parser.add_argument('--phase0-deg', type=float, default=-90.)
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--medium', choices=('air', 'water'), default='air')
    parser.add_argument('--config', type=Path,
                        default=ROOT/'configs/fitted_model.json')
    parser.add_argument('--pump-config', type=Path,
                        default=ROOT/'configs/fitted_pump_model.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'Outputs/traveling_wave')
    parser.add_argument('--start-straight', action='store_true',
                        help='Start every volume at zero instead of the periodic response')
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--fps', type=float, default=60.)
    args = parser.parse_args(argv)
    try:
        if not all(math.isfinite(v) and v > 0 for v in (args.duration, args.dt, args.fps)):
            raise ValueError('Duration, timestep and frame rate must be positive and finite')
        config, parameters = load_config(args.config, args.medium)
        pump_record = json.loads(args.pump_config.read_text())
        pump = StaticEscParameters.from_json(args.pump_config, args.medium)
        if pump_record.get('frozen_actuator_parameters_si', {}).get(args.medium) != asdict(parameters):
            raise ValueError('Pump fit assumed different actuator coefficients; use its matching actuator config')
        command = TravelingWaveCommand(pump.neutral_command, args.amplitude_command, args.period,
                                       args.phase_lag_deg, args.phase0_deg)
        output = args.output_dir.expanduser().resolve()
        geometry = generate_chain(output/'model'/'hydraulic_chain.xml', n_modules=args.modules,
                                  slices_per_module=args.slices_per_module,
                                  length_mm=args.soft_length_mm+args.rigid_start_mm+args.rigid_end_mm,
                                  rigid_start_mm=args.rigid_start_mm, rigid_end_mm=args.rigid_end_mm,
                                  gap_mm=args.gap_mm)
        chain = ChainKinematics(geometry['xml_path'])
        volumes = ((0.,)*args.modules if args.start_straight else
                   periodic_initial_volumes(parameters, pump, command, args.modules))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    actuators = [CalibratedActuatorModel(parameters, volume) for volume in volumes]
    condition = pump_record.get('operating_conditions', {}).get(args.medium, {})
    command_range = [pump.neutral_command-args.amplitude_command,
                     pump.neutral_command+args.amplitude_command]
    record = {
        'status': 'running', 'motion_mode': 'prescribed multi-actuator kinematics',
        'module_count': args.modules, 'slices_per_module': args.slices_per_module,
        'soft_divisions_per_module': args.slices_per_module,
        'soft_length_m': geometry['soft_length_m'],
        'rigid_start_mm': args.rigid_start_mm, 'rigid_end_mm': args.rigid_end_mm,
        'gap_mm': args.gap_mm,
        'module_length_m': geometry['module_length_m'], 'total_length_m': chain.total_length_m,
        'duration_s': args.duration, 'dt_s': args.dt, 'medium': args.medium,
        'command': command.metadata(), 'command_range': command_range,
        'actuator_parameters_si': asdict(parameters), 'pump_parameters_si': asdict(pump),
        'actuator_calibration_status': config['status'], 'pump_calibration_status': pump_record['status'],
        'actuator_config': str(args.config.resolve()), 'pump_config': str(args.pump_config.resolve()),
        'config_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (args.config, args.pump_config)},
        'command_extrapolates_recorded_range': (
            command_range[0] < condition.get('command_min', -math.inf)
            or command_range[1] > condition.get('command_max', math.inf)),
        'initialization': 'all volumes zero' if args.start_straight else 'analytical periodic hydraulic response',
        'initial_volumes_m3': list(volumes), 'joint_groups': chain.joint_groups,
        'geometry': geometry, 'media_export': {'status': 'not_requested'},
        'units': {'time_s': 's', 'pwm_command': 'raw logged ESC command units',
                  'angle_rad': 'rad', 'tip_heading_rad': 'rad in world x-y plane',
                  'volume_m3': 'm^3', 'all_flows': 'm^3/s', 'angle_velocity_rad_s': 'rad/s'},
        'flow_convention': 'Command and pump flow held at interval midpoint; endpoint leak/net; initial uses first interval',
        'assumptions': [
            'One independent pump/closed hydraulic circuit per actuator; no shared hydraulics',
            'All modules reuse the same chosen single-actuator coefficients; not calibrated as an assembled body',
            'Positive phase lag delays each downstream command and its relative bending angle',
            'Equal local hinge increments inside each soft region; rigid ends welded to nearest division',
            'Physical module boundaries remain rigidly joined without extra bending joints',
            'Relative module angles add along the serial body to produce world headings',
            'Pump map is instantaneous with no lag, deadband, saturation or measured pressure',
            'No external water forces, mechanical load feedback, self-contact, propulsion or free-body motion',
            'Choosing water changes coefficients only; it does not enable surrounding water forces',
        ],
    }
    record_path = output/'run.json'

    def save_record():
        record_path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')

    save_record()
    print(f'{args.modules} independent actuators; {args.slices_per_module} soft divisions each; '
          f'{chain.total_length_m:.3f} m total', flush=True)
    print(f'Each actuator: {args.rigid_start_mm:g} mm rigid + '
          f'{geometry["soft_length_m"]*1000:g} mm soft + {args.rigid_end_mm:g} mm rigid', flush=True)
    print(f'Period {args.period:g} s; downstream lag {args.phase_lag_deg:g} deg '
          f'({command.metadata()["adjacent_delay_s"]:.3f} s per actuator)', flush=True)
    samples = []
    try:
        for sample in simulate_chain(chain, actuators, [pump]*args.modules, command,
                                     duration_s=args.duration, dt_s=args.dt):
            samples.append(sample)
    except (ValueError, OverflowError) as exc:
        record.update(status='stopped_with_error', error=str(exc),
                      attempted_volumes_m3=[a.volume_m3 for a in actuators])
    write_trajectory(output/'trajectory.csv', chain, samples)
    record.update(samples_written=len(samples), last_valid_time_s=samples[-1].time_s if samples else None)
    if record['status'] == 'stopped_with_error':
        save_record()
        print(f'ERROR: {record["error"]}. Valid samples saved; no clamping.')
        return 2
    record.update(status='completed', peak_module_bend_deg=float(np.max(np.abs(
        np.rad2deg([s.angles_rad for s in samples])))))
    plot_summary(output/'wave_summary.png', chain, samples, args.period)
    save_record()
    if not args.data_only:
        from src.export.traveling_wave_export import export_chain_media
        try:
            options = ExportOptions(video_path=output/'motion.mp4', jpg_path=output/'snapshot.jpg',
                                    fps=args.fps, snapshot_time_s=min(args.period/4, args.duration),
                                    overlay=True, overlay_label=f'MODEL VALUES | {args.period:g} s period | '
                                    f'{args.phase_lag_deg:g} deg delay per actuator | {args.medium.upper()} coefficients')
            record['media_export'] = export_chain_media(chain, samples, options)
        except Exception as exc:
            record['media_export'] = {'status': 'failed', 'error': str(exc)}
            save_record()
            raise
        save_record()
    print(f'CSV: {output/"trajectory.csv"}', flush=True)
    if args.view:
        from src.export.traveling_wave_export import playback_chain
        playback_chain(chain, samples)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
