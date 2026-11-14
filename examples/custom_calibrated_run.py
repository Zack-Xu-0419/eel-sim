"""Editable, pressure-free experiment: custom coefficients and actual flow.

Default: pump toward 30 deg, then relax below 5 deg, with MP4/JPG/telemetry.
--data-only skips rendering. DURATION_S is the maximum end time.
Default outputs are separate from the old pressure-based example.

This is the recommended starting file for a new experiment. Edit the four
blocks below; the imported modules handle integration, geometry and export.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Allow direct execution from examples/ to import the project-root modules.
# No separately installed "actuator library" is required.
sys.path.insert(0, str(ROOT))
from src.actuator.calibrated_actuator_model import CalibratedActuatorModel, CalibratedParameters
from src.export.calibrated_section_export import ExportOptions
from src.actuator.calibrated_signals import SineFlow, SquareFlow
from src.actuator.run_calibrated_section import SectionKinematics, load_config, playback, run_experiment

# EDIT THESE: SI coefficients (initial values copied from the provisional air fit).
# To use an air/water JSON configuration instead, pass --config and --medium.

PARAMETERS = CalibratedParameters(
    volume_to_angle_rad_per_m3=27465.796146022065,
    leak_conductance_m3_s_per_rad=9.674674949569114e-7,
    equilibrium_angle_rad=0,
)
# Constructors accept mL/s. Calling FLOW(time_s) returns m3/s.
# Replace SineFlow with SquareFlow, or with a function of time that returns
# actual flow. An ESC/PWM command is not a flow measurement.


import numpy as np

# def FLOW(t):
#     direction = 1 if (t + 2) // 4 % 2 == 1 else -1
#     return direction * 10e-6  # ±10 mL/s

# FLOW = SineFlow(10, 0.25, 0, 90)


# FLOW = SineFlow(amplitude_ml_s=2., frequency_hz=.3)

# This is volume transferred relative to balance, not total chamber volume.
INITIAL_VOLUME_M3 = 0.0
DURATION_S = 10  # Maximum end time; STOP_WHEN can finish earlier.
# The solver timestep is independent of the video frame period (default 1/60 s).
DT_S = .001
calibrated_model = CalibratedActuatorModel(PARAMETERS, INITIAL_VOLUME_M3)

# reached = False
# def FLOW(t):
#     global reached
#     print(calibrated_model.sample().angle_rad * 180/3.1415)
#     if calibrated_model.sample().angle_rad * 180/3.1415 <= 30 and reached == False:
#         return 10e-6
#     else:
#         reached = True
#         return 0

from src.actuator.esc_pump_model import StaticEscParameters

PUMP = StaticEscParameters.from_json(
    ROOT/"configs/fitted_pump_model.json", "air")

def FLOW(t):
    direction = 1 if (t + 2) // 4 % 2 == 1 else -1
    pwm = 94 + 59*np.sin(2*np.pi*t/8-np.pi/2)
    return PUMP.flow_from_command(pwm)

def STOP_WHEN(sample):
    """After reaching the pump target, finish when the bend relaxes below 5 deg."""
    return reached and abs(np.degrees(sample.angle_rad)) < 5.0


def main(argv=None):
    """Read optional launch settings, construct objects, then run once.

    Calling main() from the terminal uses command-line arguments. Tests can
    pass an explicit list through argv without modifying the global settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'Geometry/generated/section_n10.xml')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'Outputs/custom_calibrated')
    parser.add_argument('--config', type=Path, help='Optional JSON overrides the PARAMETERS block')
    parser.add_argument('--medium', choices=('air', 'water'), default='air',
                        help='Select coefficients when --config is supplied; otherwise edit PARAMETERS')
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--view', action='store_true')
    args = parser.parse_args(argv)
    config = None
    parameters = PARAMETERS
    if args.config is None and args.medium != 'air':
        parser.error('--medium water requires --config; otherwise edit the PARAMETERS block')
    if args.config is not None:
        # JSON parameters override the editable PARAMETERS block; configuration
        # provenance is passed alongside the model so the result is traceable.
        config, parameters = load_config(args.config, args.medium)
    # Construct the feedback state after choosing coefficients and reset it
    # for every run, including repeated main() calls and --config overrides.
    global calibrated_model, reached
    calibrated_model = CalibratedActuatorModel(parameters, INITIAL_VOLUME_M3)
    reached = False
    output = args.output_dir.expanduser().resolve()
    export = None if args.data_only else ExportOptions(
        video_path=output/'motion.mp4', jpg_path=output/'snapshot.jpg',
        overlay=True, overlay_label='MODEL VALUES • CUSTOM COEFFICIENTS')
    # Each object has one job: section maps angles to hinges; actuator evolves
    # volume; FLOW supplies the input; export describes the requested media.
    # run_experiment records one trajectory and reuses it for CSV and rendering.
    result = run_experiment(SectionKinematics(args.model),
                            calibrated_model, FLOW, output,
                            duration_s=DURATION_S, dt_s=DT_S, config=config,
                            medium=args.medium if config else None, config_path=args.config,
                            export_options=export, stop_when=STOP_WHEN)
    # Successful calculations may also be played in a live MuJoCo window.
    # On macOS use mjpython for this option (see the usage guide).
    if args.view and not result.exit_code:
        playback(result.section, result.samples)
    return result.exit_code


if __name__ == '__main__':
    # This block runs when launched as a script, but not when imported by Python.
    raise SystemExit(main())
