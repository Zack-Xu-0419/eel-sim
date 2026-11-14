"""Independent hydraulic actuators connected into one prescribed body wave.

An actuator module is a complete 120 mm section, not one rigid slice. Each
module has its own transferred-volume state and its own pump command. Within
that module, its total relative angle is shared by its internal hinges.
MuJoCo updates the connected geometry; no mechanical/fluid dynamics are stepped.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import re

import mujoco
import numpy as np

from src.actuator.calibrated_actuator_model import CalibratedActuatorModel, _finite as finite
from src.kinematics.chain_kinematics import ChainKinematics, AngleLimitError


@dataclass(frozen=True)
class TravelingWaveCommand:
    """Raw PWM sine; positive phase_lag_deg delays each downstream actuator.

    For 8 s period and 60 deg lag, adjacent actuators are delayed by 1.333 s.
    neutral_command should normally be the selected fitted pump's zero-flow
    command. This number is not a voltage, measured pulse width or flow rate.
    """
    neutral_command: float
    amplitude_command: float = 59.
    period_s: float = 8.
    phase_lag_deg: float = 60.
    phase0_deg: float = -90.

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.amplitude_command < 0 or self.period_s <= 0:
            raise ValueError('Command amplitude must be nonnegative and period positive')

    def phase(self, module_index):
        if isinstance(module_index, bool) or not isinstance(module_index, (int, np.integer)) or module_index < 0:
            raise ValueError('module_index must be a nonnegative integer')
        return math.radians(self.phase0_deg-module_index*self.phase_lag_deg)

    def __call__(self, time_s, module_index):
        return self.neutral_command+self.amplitude_command*math.sin(
            2*math.pi*finite(time_s, 'time_s')/self.period_s+self.phase(module_index))

    def metadata(self):
        return dict(type='phase_delayed_sine_pwm', **self.__dict__,
                    frequency_hz=1/self.period_s,
                    adjacent_delay_s=self.period_s*self.phase_lag_deg/360,
                    direction='first to last module for positive lag; negative reverses direction')


def periodic_initial_volumes(parameters, pump, command, module_count):
    """Initialize the continuous sinusoid's steady periodic hydraulic response.

    This avoids presenting a pump-start transient as a coherent traveling wave.
    These are deliberately nonzero initial volumes, not extra fitted parameters.
    The normal simulator still advances the hydraulic equations at every step.
    """
    if isinstance(module_count, bool) or not isinstance(module_count, int) or module_count < 1:
        raise ValueError('module_count must be a positive integer')
    kv = parameters.volume_to_angle_rad_per_m3
    conductance = parameters.leak_conductance_m3_s_per_rad
    rate, omega = kv*conductance, 2*math.pi/command.period_s
    mean_flow = pump.flow_from_command(command.neutral_command)
    if rate == 0 and abs(mean_flow) > 1e-18:
        raise ValueError('A zero-leak actuator with nonzero mean flow has no periodic steady state')
    mean_volume = (parameters.equilibrium_angle_rad/kv+mean_flow/rate
                   if rate else 0.)
    amplitude = pump.gain_m3_s_per_command*command.amplitude_command
    return tuple(mean_volume+amplitude*(rate*math.sin(command.phase(i))-omega*math.cos(command.phase(i)))
                 /(rate*rate+omega*omega) for i in range(module_count))


@dataclass(frozen=True)
class ChainSample:
    """One shared timestamp; commands/flows describe the interval ending here."""
    time_s: float
    commands: tuple
    actuator_samples: tuple

    @property
    def angles_rad(self):
        return tuple(sample.angle_rad for sample in self.actuator_samples)


def simulate_chain(chain, actuators, pumps, command, duration_s=16., dt_s=.001):
    """Yield one synchronized trajectory from independent actuator states.

    command(t_s, module_index) returns a raw ESC command; each pump converts it
    to m3/s. Smooth commands are held at integration-interval midpoints, exactly
    as in the single-module runner. Equal phases do not create a traveling wave.
    """
    actuators, pumps = tuple(actuators), tuple(pumps)
    if len(actuators) != chain.module_count or len(pumps) != chain.module_count:
        raise ValueError('Need one actuator state and one pump map per module')
    if len({id(a) for a in actuators}) != len(actuators):
        raise ValueError('Each module requires an independent actuator state')
    if any(not isinstance(a, CalibratedActuatorModel) for a in actuators):
        raise ValueError('Actuators must be CalibratedActuatorModel objects')
    if not callable(command):
        raise ValueError('command must be callable as command(time_s, module_index)')
    end, dt = finite(duration_s, 'duration_s'), finite(dt_s, 'dt_s')
    start = actuators[0].time_s
    if end <= start or dt <= 0 or start+dt <= start:
        raise ValueError('duration_s must exceed current time and dt_s must be positive and advance time')
    if any(a.time_s != start for a in actuators):
        raise ValueError('All actuator clocks must start synchronized')

    def commands_and_flows(clock):
        values = tuple(finite(command(clock, i), 'PWM command') for i in range(chain.module_count))
        return values, tuple(p.flow_from_command(u) for p, u in zip(pumps, values))

    commands, flows = commands_and_flows(start+min(dt, end-start)/2)
    initial = ChainSample(start, commands, tuple(a.sample(q) for a, q in zip(actuators, flows)))
    chain.apply(initial.angles_rad, initial.time_s)
    yield initial
    tick, clock = 1, start
    while clock < end:
        target = min(start+tick*dt, end)
        h = target-clock
        if h <= 0:
            raise ValueError('Timestep no longer advances simulation time')
        commands, flows = commands_and_flows(clock+h/2)
        before = np.array([a.sample().angle_rad for a in actuators])
        observations = tuple(a.step(q, h) for a, q in zip(actuators, flows))
        sample = ChainSample(target, commands, observations)
        rates = (np.array(sample.angles_rad)-before)/h
        chain.apply(sample.angles_rad, target, rates)
        yield sample
        clock, tick = target, tick+1
