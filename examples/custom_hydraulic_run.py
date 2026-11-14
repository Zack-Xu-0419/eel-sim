"""Editable experiment: coefficients + waveform -> synchronized CSV and MP4.

From the project root:
    .venv/bin/python examples/custom_hydraulic_run.py --waveform sine
    .venv/bin/python examples/custom_hydraulic_run.py --waveform square

Edit PARAMETERS below, or pass --config path/to/your_config.json.
Use --data-only to skip rendering. Outputs are replaced when reusing a folder.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export.hydraulic_export import ExportOptions
from src.hydraulic.hydraulic_model import HydraulicModel, HydraulicParameters, HydraulicRelations
from src.hydraulic.hydraulic_signals import SineFlow, SquareFlow
from src.hydraulic.run_hydraulic_section import SectionKinematics, load_config, playback, run_experiment


# YOUR PHYSICAL COEFFICIENTS: all in SI; these are uncalibrated demo values.
PARAMETERS = HydraulicParameters(
    volume_to_angle_rad_per_m3=200000.,      # 0.2 rad/mL
    pressure_per_angle_pa_per_rad=20000.,    # 20 kPa/rad
    leak_resistance_pa_s_per_m3=4e9,         # Pa s/m^3
    initial_volume_m3=0.,                   # transferred volume relative to balance
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--waveform', choices=('sine', 'square'), default='sine')
    parser.add_argument('--amplitude-ml-s', type=float, default=2.)
    parser.add_argument('--frequency-hz', type=float, default=.5)
    parser.add_argument('--offset-ml-s', type=float, default=0.)
    parser.add_argument('--phase-deg', type=float, default=0.)
    parser.add_argument('--duty-cycle', type=float, default=.5,
                        help='Positive fraction of the square-wave period (default 0.5)')
    parser.add_argument('--duration', type=float, default=12.)
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--model', type=Path, default=ROOT/'Geometry/generated/section_n10.xml')
    parser.add_argument('--config', type=Path, help='Override the PARAMETERS block using an existing hydraulic JSON')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--no-overlay', action='store_true')
    parser.add_argument('--view', action='store_true', help='Use mjpython on macOS for live playback')
    args = parser.parse_args(argv)
    output = (args.output_dir or ROOT/'Outputs/custom_hydraulic'/args.waveform).expanduser().resolve()

    try:
        if args.config is not None:
            config, parameters = load_config(args.config)
        else:
            parameters = PARAMETERS
            config = {
                'calibration_status': 'uncalibrated_demo',
                'description': 'Coefficients from PARAMETERS in examples/custom_hydraulic_run.py',
                'parameters': asdict(parameters),
                'units': {'volume_to_angle_rad_per_m3': 'rad/m^3',
                          'pressure_per_angle_pa_per_rad': 'Pa/rad',
                          'leak_resistance_pa_s_per_m3': 'Pa*s/m^3',
                          'initial_volume_m3': 'm^3'},
            }

        # INPUT: an actual signed pump-flow waveform. Constructors accept mL/s;
        # calling flow(time_s) converts to m^3/s for the hydraulic integrator.
        signal_parameters = dict(amplitude_ml_s=args.amplitude_ml_s,
                                 frequency_hz=args.frequency_hz,
                                 offset_ml_s=args.offset_ml_s, phase_deg=args.phase_deg)
        flow = (SineFlow(**signal_parameters) if args.waveform == 'sine' else
                SquareFlow(**signal_parameters, duty_cycle=args.duty_cycle))

        # PHYSICS: replace the three relations here when you have nonlinear data.
        relations = HydraulicRelations.linear(parameters)
        hydraulic = HydraulicModel(relations, initial_volume_m3=parameters.initial_volume_m3)

        # GEOMETRY: loads the generated MJCF XML, mesh and named bending hinges.
        section = SectionKinematics(args.model)

        # OUTPUT: render the already-calculated samples, never a separate run.
        export = None if args.data_only else ExportOptions(
            video_path=output/'motion.mp4', jpg_path=output/'snapshot.jpg',
            width=args.width, height=args.height, fps=args.fps, overlay=not args.no_overlay)

        result = run_experiment(
            section, hydraulic, flow, output,
            duration_s=args.duration, dt_s=args.dt,
            config=config, config_path=args.config, export_options=export)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if result.exit_code:
        return result.exit_code
    if args.view:
        playback(result.section, result.samples)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
