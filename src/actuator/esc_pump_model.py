"""Instantaneous, effective ESC-command to signed pump-flow mapping.

This module adds no pump lag. Downstream actuator parameters stay separate.
Commands use the raw CSV PWM units; these are not assumed to be percent duty
cycle, volts, or microseconds. Flow is inferred, not independently measured.
"""
from bisect import bisect_right
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


@dataclass(frozen=True)
class StaticEscParameters:
    """Qpump = gain*(command-neutral), in m3/s; no clipping or response delay.

    neutral_command is an effective fitted zero, not a measured firmware stop
    command. Applying this map outside its fitted conditions is extrapolation.
    """
    gain_m3_s_per_command: float
    neutral_command: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.gain_m3_s_per_command <= 0:
            raise ValueError('Pump command-to-flow gain must be positive')

    def flow_from_command(self, command):
        return _finite(self.gain_m3_s_per_command*(_finite(command, 'command')-self.neutral_command), 'pump flow')

    @classmethod
    def from_json(cls, path, medium='water'):
        record = json.loads(Path(path).read_text())
        if record.get('pump_dynamics') != 'instantaneous_static':
            raise ValueError('Expected an instantaneous static ESC pump configuration')
        try:
            return cls(**record['pump_parameters_si'][medium.lower()])
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('Pump configuration requires pump_parameters_si[medium]') from exc


class RecordedEscFlow:
    """Convert a recorded, left-held PWM history to flow(t_s) for the runner.

    Use elapsed times beginning at zero. Each command is held until the next
    recorded timestamp; switch_times lets the runner split steps at switches.
    No command is inferred before/after the supplied recording, or auto-treated
    as "pump off". In particular PWM91 is not silently overridden to zero flow.
    """
    def __init__(self, times_s, commands, parameters):
        self.times_s = tuple(_finite(t, 'recorded time') for t in times_s)
        self.commands = tuple(_finite(u, 'recorded command') for u in commands)
        if (len(self.times_s) < 2 or len(self.commands) != len(self.times_s)
                or self.times_s[0] != 0
                or any(b <= a for a, b in zip(self.times_s, self.times_s[1:]))):
            raise ValueError('Need equal command/time histories with increasing times starting at zero')
        if not isinstance(parameters, StaticEscParameters):
            raise ValueError('parameters must be StaticEscParameters')
        self.parameters = parameters

    def __call__(self, time_s):
        time = _finite(time_s, 'time_s')
        if not self.times_s[0] <= time <= self.times_s[-1]:
            raise ValueError('Requested time is outside recorded ESC history')
        index = bisect_right(self.times_s, time)-1
        return self.parameters.flow_from_command(self.commands[index])

    def switch_times(self, start_s, end_s):
        start, end = _finite(start_s, 'start_s'), _finite(end_s, 'end_s')
        if end <= start:
            raise ValueError('end_s must exceed start_s')
        return tuple(t for t in self.times_s if start < t < end)

    def metadata(self):
        return {'type': 'recorded_pwm_through_fitted_static_pump',
                'pump_dynamics': 'instantaneous_static', 'tau_p_s': 0.,
                'parameters_si': asdict(self.parameters),
                'command_units': 'raw logged PWM command units', 'flow_units': 'm^3/s',
                'command_interpolation': 'left-held between recorded timestamps',
                'recorded_samples': len(self.times_s), 'end_time_s': self.times_s[-1],
                'scope': 'Effective driven map conditional on frozen actuator model; flow unmeasured'}
