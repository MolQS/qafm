"""Default parameter values used across ``qafm``.

This module collects the default numerical parameters for the sensor
model and the various tip-sample interactions (van der Waals, Morse,
Lennard-Jones, and electrostatic). Individual simulations typically
import this module and override only the parameters they need.
"""

# ---------------------------------------------------------------------------
# Sensor parameters
# ---------------------------------------------------------------------------

f0 = 300000.0
"""Cantilever resonance frequency, in Hz."""

k0 = 35.0
"""Cantilever spring constant, in N/m."""

Q0 = 20000
"""Cantilever quality factor, dimensionless."""

A0 = 5e-9
"""Cantilever free oscillation amplitude, in m."""


# ---------------------------------------------------------------------------
# van der Waals parameters
# ---------------------------------------------------------------------------

H = 357.619e-21
"""Hamaker constant, in J."""

Theta = 29.7
"""Half-opening angle of the conical tip, in degree."""

R = 15e-9
"""Tip radius, in m."""

zoffset = 583.04e-12
"""Tip-sample distance offset, in m."""

default_h1 = None
"""Optional cone-to-apex transition height for the F6-type geometry, in m."""

default_L = None
"""Optional tip cone length for the F6-type geometry, in m."""


# ---------------------------------------------------------------------------
# Morse parameters
# ---------------------------------------------------------------------------

kappa = 4.25e9
"""Morse decay constant, in 1/m."""

sigma0 = 0.235e-9
"""Morse equilibrium distance, in m."""

ebond = 0.371e-18
"""Morse bond energy, in J."""


# ---------------------------------------------------------------------------
# Lennard-Jones parameters
# ---------------------------------------------------------------------------

sigma_tip: float = 0.3e-9
"""Lennard-Jones diameter of the tip atoms, in m."""

sigma_sample: float = 0.3e-9
"""Lennard-Jones diameter of the sample atoms, in m."""

epsilon_tip: float = 1.0e-21
"""Lennard-Jones well depth of the tip atoms, in J."""

epsilon_sample: float = 1.0e-21
"""Lennard-Jones well depth of the sample atoms, in J."""

r_offset: float | None = None
"""Optional distance offset applied to the Lennard-Jones potential, in m."""

r_min: float = 1.0e-12
"""Minimum tip-sample distance used to regularize the Lennard-Jones potential, in m."""


# ---------------------------------------------------------------------------
# Electrostatic parallel-plate parameters
# ---------------------------------------------------------------------------

tip_area: float = 5e-9 * 5e-9
"""Effective tip-sample plate area, in m^2."""

vbias: float = 1.0
"""Applied bias voltage, in V."""

eps_r: float = 1.0
"""Relative permittivity between the plates, unitless."""


# ---------------------------------------------------------------------------
# Electrostatic sphere-plane parameters
# ---------------------------------------------------------------------------

radius: float = 5e-9
"""Tip radius for the sphere-plane model, in m."""

vbias: float = 1.0
"""Applied bias voltage, in V."""

eps_tip: float = 1.0
"""Relative permittivity of the tip, unitless."""

eps_sample: float = 1.0
"""Relative permittivity of the sample, unitless."""

z_min: float = 1e-12
"""Minimum tip-sample distance used to regularize the sphere-plane potential, in m."""
