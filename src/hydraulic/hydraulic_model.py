"""A replaceable, lumped volume-to-angle model; no MuJoCo dependency.

Positive pump flow transfers liquid B -> A; positive leakage returns A -> B.
Volume is net liquid transferred relative to balance, not cumulative pump output.
The three callable relations can be replaced independently of RK4 or playback.
"""

from dataclasses import dataclass
from functools import partial
import math
from typing import Callable


def finite(value, name):
    """Reject invalid physical inputs/results before changing model state."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class HydraulicParameters:
    """SI parameters for the initial linear relations, not calibrated defaults."""

    volume_to_angle_rad_per_m3: float
    pressure_per_angle_pa_per_rad: float
    leak_resistance_pa_s_per_m3: float
    initial_volume_m3: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.volume_to_angle_rad_per_m3 <= 0:
            raise ValueError("volume_to_angle_rad_per_m3 must be positive")
        if self.pressure_per_angle_pa_per_rad < 0:
            raise ValueError("pressure_per_angle_pa_per_rad cannot be negative")
        if self.leak_resistance_pa_s_per_m3 <= 0:
            raise ValueError("leak_resistance_pa_s_per_m3 must be positive")


def linear_angle_from_volume(volume_m3, *, gain_rad_per_m3):
    return gain_rad_per_m3 * volume_m3


def linear_pressure_from_angle(angle_rad, *, gain_pa_per_rad):
    return gain_pa_per_rad * angle_rad


def linear_leak_flow_from_pressure(pressure_difference_pa, *, resistance_pa_s_per_m3):
    return pressure_difference_pa / resistance_pa_s_per_m3


@dataclass(frozen=True)
class HydraulicRelations:
    """Pure scalar functions, with SI inputs and outputs.

    Replace functions here to try nonlinear or tabulated relations. RK4 calls
    them at intermediate states, so functions must not mutate hidden state.
    Stateful hysteresis would require extending the explicit model state.
    """

    angle_from_volume: Callable[[float], float]
    pressure_from_angle: Callable[[float], float]
    leak_flow_from_pressure: Callable[[float], float]

    @classmethod
    def linear(cls, parameters):
        return cls(
            angle_from_volume=partial(linear_angle_from_volume,
                gain_rad_per_m3=parameters.volume_to_angle_rad_per_m3),
            pressure_from_angle=partial(linear_pressure_from_angle,
                gain_pa_per_rad=parameters.pressure_per_angle_pa_per_rad),
            leak_flow_from_pressure=partial(linear_leak_flow_from_pressure,
                resistance_pa_s_per_m3=parameters.leak_resistance_pa_s_per_m3),
        )


@dataclass(frozen=True)
class HydraulicState:
    time_s: float
    volume_m3: float


@dataclass(frozen=True)
class HydraulicSample:
    time_s: float
    volume_m3: float
    angle_rad: float
    pressure_difference_pa: float
    pump_flow_m3_s: float
    leak_flow_m3_s: float
    net_flow_m3_s: float


class HydraulicModel:
    def __init__(self, relations, initial_volume_m3=0.0):
        self.relations = relations
        self.state = HydraulicState(0.0, finite(initial_volume_m3, "initial_volume_m3"))
        self.sample(0.0)  # Validate relations at the starting state too.

    def _evaluate(self, state, pump_flow):
        theta = finite(self.relations.angle_from_volume(state.volume_m3), "angle relation output")
        pressure = finite(self.relations.pressure_from_angle(theta), "pressure relation output")
        leak = finite(self.relations.leak_flow_from_pressure(pressure), "leak relation output")
        net = finite(pump_flow - leak, "net flow")
        return HydraulicSample(state.time_s, state.volume_m3, theta, pressure, pump_flow, leak, net)

    def sample(self, pump_flow_m3_s):
        return self._evaluate(self.state, finite(pump_flow_m3_s, "pump_flow_m3_s"))

    def step(self, pump_flow_m3_s, dt_s):
        """Advance v_dot=Q_p-Q_leak with classical RK4 and held pump flow.

        The returned flows are evaluated at the final state, for the flow held
        during this step. A rejected/non-finite step leaves state unchanged.
        """
        pump = finite(pump_flow_m3_s, "pump_flow_m3_s")
        dt = finite(dt_s, "dt_s")
        if dt <= 0:
            raise ValueError("dt_s must be positive")
        t, v = self.state.time_s, self.state.volume_m3
        next_time = finite(t+dt, "next time")
        if next_time <= t:
            raise ValueError("dt_s is too small to advance the simulation clock")

        def rate(volume):
            state = HydraulicState(t, finite(volume, "RK4 volume"))
            return self._evaluate(state, pump).net_flow_m3_s

        k1 = rate(v)
        k2 = rate(v + dt*k1/2)
        k3 = rate(v + dt*k2/2)
        k4 = rate(v + dt*k3)
        volume = finite(v + dt*(k1+2*k2+2*k3+k4)/6, "integrated volume")
        next_state = HydraulicState(next_time, volume)
        result = self._evaluate(next_state, pump)
        self.state = next_state
        return result
