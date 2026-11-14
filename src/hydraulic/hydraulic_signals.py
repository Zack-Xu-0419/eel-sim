"""Actual signed pump-flow waveforms; constructor flow values use mL/s.

Calling a signal with time in seconds returns flow in m^3/s for the solver.
Positive flow transfers liquid B -> A. These are flow commands, not PWM models.
"""
from dataclasses import asdict, dataclass
import math

from src.hydraulic.hydraulic_model import finite


def _validate(signal):
    for name in ('amplitude_ml_s', 'frequency_hz', 'offset_ml_s', 'phase_deg'):
        object.__setattr__(signal, name, finite(getattr(signal, name), name))
    if signal.amplitude_ml_s < 0:
        raise ValueError('amplitude_ml_s must be nonnegative')
    if signal.frequency_hz <= 0:
        raise ValueError('frequency_hz must be positive')


@dataclass(frozen=True)
class SineFlow:
    amplitude_ml_s: float = 2.
    frequency_hz: float = .5
    offset_ml_s: float = 0.
    phase_deg: float = 0.

    def __post_init__(self):
        _validate(self)

    def __call__(self, time_s):
        phase = 2*math.pi*self.frequency_hz*finite(time_s, 'time_s') + math.radians(self.phase_deg)
        return (self.offset_ml_s+self.amplitude_ml_s*math.sin(phase))*1e-6

    def switch_times(self, start_s, end_s):
        return ()

    def metadata(self):
        return dict(type='sine', **asdict(self))


@dataclass(frozen=True)
class SquareFlow:
    amplitude_ml_s: float = 2.
    frequency_hz: float = .5
    offset_ml_s: float = 0.
    phase_deg: float = 0.
    duty_cycle: float = .5

    def __post_init__(self):
        _validate(self)
        duty = finite(self.duty_cycle, 'duty_cycle')
        if not 0 < duty < 1:
            raise ValueError('duty_cycle must lie strictly between 0 and 1')
        object.__setattr__(self, 'duty_cycle', duty)

    def __call__(self, time_s):
        phase = (self.frequency_hz*finite(time_s, 'time_s') + self.phase_deg/360) % 1
        direction = 1. if phase < self.duty_cycle else -1.
        return (self.offset_ml_s+direction*self.amplitude_ml_s)*1e-6

    def switch_times(self, start_s, end_s):
        """All interior discontinuities, including phase shifts and duty cycle."""
        start, end = finite(start_s, 'start_s'), finite(end_s, 'end_s')
        if end <= start:
            raise ValueError('end_s must exceed start_s')
        phase = self.phase_deg/360
        times = []
        for edge in (0., self.duty_cycle):
            first = math.floor(self.frequency_hz*start+phase-edge)+1
            last = math.ceil(self.frequency_hz*end+phase-edge)-1
            for cycle in range(first, last+1):
                time = (cycle+edge-phase)/self.frequency_hz
                if start < time < end:
                    times.append(time)
        return tuple(sorted(set(times)))

    def metadata(self):
        return dict(type='square', **asdict(self))
