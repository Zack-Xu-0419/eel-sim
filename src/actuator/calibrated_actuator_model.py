"""Pressure-free, provisional actuator model. Public inputs and outputs are SI.

v is signed liquid transferred B -> A relative to the balanced reference;
the change in chamber-volume difference is 2*v. Positive leak returns A -> B.
The fitted JSON declares the source material, data range and assumptions.
This module is independent of MuJoCo and does not change simulator defaults.

Read in this order: CalibratedParameters (coefficients), CalibratedSample
(one returned observation), then CalibratedActuatorModel.step (time evolution).
Only time_s and volume_m3 evolve; angle and flows are calculated from them.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path


def _finite(value, name):
    """Convert a scalar to float and reject NaN/infinity before using it."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class CalibratedParameters:
    """Fixed linear coefficients, stored in SI rather than display units.

    frozen=True prevents accidental assignment to a coefficient after creation.
    The positive/negative sign convention belongs to the volume state and flow,
    so kV is positive and the effective leak conductance G is nonnegative.
    """
    # kV: radians of total bend per cubic metre transferred between chambers.
    volume_to_angle_rad_per_m3: float
    # G: signed return flow per radian away from the fitted equilibrium angle.
    leak_conductance_m3_s_per_rad: float
    # theta_eq: fitted camera/rest reference; not necessarily physical straight.
    equilibrium_angle_rad: float

    def __post_init__(self):
        """Dataclasses call this after construction to normalize/check values."""
        for key in self.__dataclass_fields__:
            object.__setattr__(self, key, _finite(getattr(self, key), key))
        if self.volume_to_angle_rad_per_m3 <= 0:
            raise ValueError("volume-to-angle gain must be positive")
        if self.leak_conductance_m3_s_per_rad < 0:
            raise ValueError("leak conductance cannot be negative")

    @classmethod
    def from_json(cls, path, medium="air"):
        """Construct parameters from the selected air/water SI dictionary.

        The ** operator expands JSON keys into constructor keyword arguments.
        This loads existing coefficients; it does not refit the experimental data.
        """
        record = json.loads(Path(path).read_text())
        return cls(**record["parameters_si"][medium.lower()])


@dataclass(frozen=True)
class CalibratedSample:
    """Immutable snapshot returned to callers, CSV writers and video exporters.

    Keeping past samples immutable lets a run advance without altering history.
    All quantities use SI; the renderer converts them only for display.
    """
    time_s: float
    volume_m3: float
    angle_rad: float
    pump_flow_m3_s: float
    leak_flow_m3_s: float
    net_flow_m3_s: float


class CalibratedActuatorModel:
    """theta=kV*v; v_dot=Qp-G*(theta-theta_eq).

    step() uses the exact linear solution with constant actual pump flow during
    the supplied interval. Pump-command calibration belongs upstream. Leakage
    is recalculated at the returned state. No pressure is inferred or reported.
    """

    def __init__(self, parameters, initial_volume_m3=0.0):
        """Start a new trajectory at t=0 with a chosen transferred volume.

        Zero means the balanced reference, not an empty chamber. Absolute
        chamber capacity is not an evolving state in this reduced model.
        """
        self.parameters = parameters
        self.time_s = 0.0
        self.volume_m3 = _finite(initial_volume_m3, "initial_volume_m3")

    def angle_from_volume(self, volume_m3):
        """Apply the fitted linear relation theta=kV*v; output is radians."""
        return self.parameters.volume_to_angle_rad_per_m3 * volume_m3

    def leak_flow_from_angle(self, angle_rad):
        """Return G*(theta-theta_eq), allowing return flow to reverse sign.

        G is constant here, but Qleak changes whenever the angle changes.
        """
        p = self.parameters
        return p.leak_conductance_m3_s_per_rad * (angle_rad-p.equilibrium_angle_rad)

    def sample(self, pump_flow_m3_s=0.0):
        """Observe the current state without advancing or changing it.

        Supplied pump flow is combined with current leakage to report net flow.
        No pump command/PWM conversion or pressure estimate happens here.
        """
        pump = _finite(pump_flow_m3_s, "pump_flow_m3_s")
        angle = self.angle_from_volume(self.volume_m3)
        leak = self.leak_flow_from_angle(angle)
        return CalibratedSample(self.time_s, self.volume_m3, angle, pump, leak, pump-leak)

    def step(self, pump_flow_m3_s, dt_s):
        """Advance once and return endpoint values for the held pump input.

        Substituting theta=kV*v gives v_dot + (kV*G)*v = Qp + G*theta_eq.
        For constant Qp over dt this has a closed-form solution, used below.
        If the relations become nonlinear, this exact update must be replaced
        with a suitable integrator; changing angle_from_volume alone is not enough.
        """
        pump = _finite(pump_flow_m3_s, "pump_flow_m3_s")
        dt = _finite(dt_s, "dt_s")
        if dt <= 0 or self.time_s+dt <= self.time_s:
            raise ValueError("dt_s must be positive and advance time")
        p = self.parameters
        # rate = 1/tau [1/s]; forcing has flow units [m3/s]. This algebra keeps
        # the equilibrium offset even when the pump flow is exactly zero.
        rate = p.volume_to_angle_rad_per_m3*p.leak_conductance_m3_s_per_rad
        forcing = pump+p.leak_conductance_m3_s_per_rad*p.equilibrium_angle_rad
        # This is (1-exp(-rate*dt))/rate, the integrated exponential response.
        # expm1(x) computes exp(x)-1 accurately for tiny x. With G=0 the limit
        # is dt, giving simple volume accumulation v_new=v_old+Qp*dt.
        increment = dt if rate == 0 else -math.expm1(-rate*dt)/rate
        volume = self.volume_m3*math.exp(-rate*dt)+forcing*increment
        # Check the proposed volume before committing it to the model state.
        _finite(volume, "integrated volume")
        self.volume_m3 = volume
        self.time_s += dt
        # Returned leakage/net flow describe the final angle, not its step average.
        return self.sample(pump)
