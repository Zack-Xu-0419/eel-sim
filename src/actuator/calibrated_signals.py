"""Actual signed pump-flow waveforms; constructor flow values use mL/s.

Calling a signal with time in seconds returns flow in m^3/s for the solver.
Positive flow transfers liquid B -> A. These are flow commands, not PWM models.
"""
from dataclasses import asdict, dataclass
import math

from src.actuator.calibrated_actuator_model import _finite as finite


def _validate(signal):
    """Reject invalid waveform settings before an experiment starts.

    The objects are frozen dataclasses: object.__setattr__ is used only during
    initialization to store the validated numeric values.
    """
    for name in ('amplitude_ml_s', 'frequency_hz', 'offset_ml_s', 'phase_deg'):
        object.__setattr__(signal, name, finite(getattr(signal, name), name))
    if signal.amplitude_ml_s < 0:
        raise ValueError('amplitude_ml_s must be nonnegative')
    if signal.frequency_hz <= 0:
        raise ValueError('frequency_hz must be positive')


@dataclass(frozen=True)
class SineFlow:
    """A smooth pump-flow input, not a prediction of actual pump hardware.

    amplitude_ml_s is the distance from offset to a peak (half the peak-to-peak
    range). frequency_hz is cycles per second. phase_deg chooses where in the
    cycle the signal starts: 0 degrees starts at the offset and rises; 90 starts
    at the positive peak. A zero offset permits flow in both directions.
    """
    amplitude_ml_s: float = 2.
    frequency_hz: float = .5
    offset_ml_s: float = 0.
    phase_deg: float = 0.

    def __post_init__(self):
        _validate(self)

    def __call__(self, time_s):
        """Evaluate flow at a time in seconds and return SI flow in m^3/s."""
        # Angular frequency 2*pi*f converts elapsed seconds to phase in radians.
        phase = 2*math.pi*self.frequency_hz*finite(time_s, 'time_s') + math.radians(self.phase_deg)
        # 1 mL = 1e-6 m^3; the hydraulic integrator always receives SI values.
        return (self.offset_ml_s+self.amplitude_ml_s*math.sin(phase))*1e-6

    def switch_times(self, start_s, end_s):
        """A sine is continuous, so no input jumps need special step boundaries."""
        return ()

    def metadata(self):
        """Record human-readable input settings alongside the output data."""
        return dict(type='sine', **asdict(self))


@dataclass(frozen=True)
class SquareFlow:
    """Alternate between offset + amplitude and offset - amplitude in mL/s.

    duty_cycle is the fraction of each period spent at the upper value. The
    default 0.5 spends equal time in both directions. For a 0 to 2 mL/s pulse,
    use offset_ml_s=1 and amplitude_ml_s=1. Phase shifts the entire waveform,
    including its switching times.
    """
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
        """Return the selected plateau value in m^3/s at the requested time."""
        # Express phase as a fraction of a cycle, wrapped into [0, 1).
        phase = (self.frequency_hz*finite(time_s, 'time_s') + self.phase_deg/360) % 1
        # At the duty boundary the new lower plateau starts immediately.
        direction = 1. if phase < self.duty_cycle else -1.
        return (self.offset_ml_s+direction*self.amplitude_ml_s)*1e-6

    def switch_times(self, start_s, end_s):
        """Return every input jump strictly inside the requested time span.

        The runner splits integration steps at these events. Otherwise a step
        that straddles a jump could apply one plateau value across both sides
        and transfer the wrong amount of liquid.
        """
        start, end = finite(start_s, 'start_s'), finite(end_s, 'end_s')
        if end <= start:
            raise ValueError('end_s must exceed start_s')
        phase = self.phase_deg/360
        times = []
        # Each cycle has two edges: wraparound to the upper plateau at 0, and
        # the change to the lower plateau at duty_cycle. Solve f*t+phase=n+edge.
        for edge in (0., self.duty_cycle):
            # Integer cycle bounds exclude events already at start or end.
            first = math.floor(self.frequency_hz*start+phase-edge)+1
            last = math.ceil(self.frequency_hz*end+phase-edge)-1
            for cycle in range(first, last+1):
                time = (cycle+edge-phase)/self.frequency_hz
                if start < time < end:
                    times.append(time)
        return tuple(sorted(set(times)))

    def metadata(self):
        """Save the waveform settings so the experiment can be reproduced."""
        return dict(type='square', **asdict(self))
