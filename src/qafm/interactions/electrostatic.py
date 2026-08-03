# src/qafm/forces/electrostatic.py

import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from scipy import constants as C

from qafm.numerics.utils import _as_array
from qafm.parameters import resolve_params



@dataclass(frozen=True)
class ElectrostaticParameters:
    """ Parameters of the parallel-plate electrostatic interaction model.

    Attributes
    ----------
    tip_area:
        Effective capacitor plate area in m².
    vbias:
        Applied bias voltage in V.
    eps_r:
        Relative permittivity of the medium between tip and sample.
    """
    tip_area: float = 5e-9 * 5e-9  # m^2
    vbias: float = 1.0             # V
    eps_r: float = 1.0             # unitless


@dataclass(frozen=True)
class SpherePlaneElectrostaticParameters:
    """Parameters of the electrostatic sphere-plane interaction model.

    Attributes
    ----------
    radius:
        Radius of the spherical tip in m.
    vbias:
        Applied bias voltage in V.
    eps_tip:
        Relative permittivity of the tip material.
    eps_sample:
        Relative permittivity of the sample material.
    z_min:
        Minimum allowed tip-sample distance in m. Distances below this value are
        clipped to avoid division by zero and numerical divergence.
    """
    radius: float = 5e-9        # m
    vbias: float = 1.0          # V
    eps_tip: float = 1.0        # unitless
    eps_sample: float = 1.0     # unitless
    z_min: float = 1e-12        # m

    @property
    def eps_r_eff(self) -> float:
        """Calculate the effective relative permittivity.

        The effective relative permittivity is calculated from the tip and
        sample permittivities using their harmonic mean and clipped to the
        interval ``[1, 100]``.

        Returns
        -------
        eps_r_eff:
            Effective relative permittivity of the tip-sample system.
        """
        eps_r = 2 * self.eps_tip * self.eps_sample / (
            self.eps_tip + self.eps_sample
        )
        return float(np.clip(eps_r, 1.0, 1e2))



def parallel_plates_force(
    z_axis: ArrayLike,
    model_par: ElectrostaticParameters | object = None,
) -> NDArray[np.float64]:
    """ Calculate the electrostatic force for a a parallel plate capacitor
        (two infinite metallic plates).

    The tip and sample are modeled as a parallel-plate capacitor with area
    ``tip_area``, separation ``z_axis``, and permittivity
    ``epsilon_0 * eps_r`` between the metallic plates. 
    The force is calculated at constant applied bias voltage:

        F =  1/2 dC / dz V^2
          = -1/2 eps * A * V^2 / z^2

    with the capacitance
        C = eps * A / z  (with eps = eps_0 * eps_r)

    Parameters
    ----------
    z_axis:
        Tip-sample distance axis in m.
    model_par:
        Parallel-plate electrostatic parameters. The relevant parameters are:
         - 'tip_area' (in m^2)
         - 'vbias' (in V)
         - 'eps_r' (unitless)

    Returns
    -------
    force:
        Electrostatic force in N.
    """

    model_par = resolve_params(
        model_par,
        expected_type=ElectrostaticParameters,
        default=ElectrostaticParameters(),
    )

    z_axis = _as_array(z_axis)

    return (
        -1.0
        / 2.0
        * C.epsilon_0
        * model_par.eps_r
        * model_par.tip_area
        * np.pow(model_par.vbias, 2)
        / np.pow(z_axis, 2)
    )



def parallel_plates_forcegradient(
    z_axis: ArrayLike,
    model_par: ElectrostaticParameters | object = None,
) -> NDArray[np.float64]:
    """ Calculate the force gradient of the parallel-plate capacitor.

    The force gradient is the derivative of the parallel-plate electrostatic
    force with respect to the tip-sample distance.

        k = dF / dz
          = 1/2 d^2 C / dz^2 V^2
          = eps * A * V^2 / z^3

    Parameters
    ----------
    z_axis:
        Tip-sample distance axis in m.
    model_par:
        Parallel-plate electrostatic parameters. The relevant parameters are:
        - 'tip_area' (in m^2)
        - 'vbias' (in V)
        - 'eps_r' (unitless)

    Returns
    -------
    force_gradient:
        Electrostatic force gradient in N/m.
    """

    model_par = resolve_params(
        model_par,
        expected_type=ElectrostaticParameters,
        default=ElectrostaticParameters(),
    )

    z_axis = _as_array(z_axis)

    return (
        C.epsilon_0
        * model_par.eps_r
        * model_par.tip_area
        * np.pow(model_par.vbias, 2)
        / np.pow(z_axis, 3)
    )



# TODO: Why eps_sample and eps_tip, I thought these are metallic?
def sphere_plane_force(
    z_axis: ArrayLike,
    model_par: SpherePlaneElectrostaticParameters | object = None,
) -> NDArray[np.float64]:
    """ Calculate the electrostatic force between a metallic spherical tip and a metallic plane.

    The model uses a spherical tip of radius ``radius`` above a flat sample.
    The dielectric response is described by the effective relative permittivity
    calculated from ``eps_tip`` and ``eps_sample``. Distances below ``z_min``
    are clipped to avoid numerical divergence.

        F(z) = - pi * eps * R^2 * V^2 / (z * (z + R))

        with:
        - eps = eps_0 * eps_r_eff
        - eps_r_eff = 2 * eps_tip * eps_sample / (eps_tip + eps_sample)

    Parameters
    ----------
    z_axis:
        Tip-sample distance axis in m.
    model_par:
        Sphere-plane electrostatic parameters.

    Returns
    -------
    force:
        Electrostatic sphere-plane force in N.
    """

    model_par = resolve_params(
        model_par,
        expected_type=SpherePlaneElectrostaticParameters,
        default=SpherePlaneElectrostaticParameters(),
    )

    z_axis = _as_array(z_axis)
    z_axis = np.maximum(z_axis, model_par.z_min)

    eps = C.epsilon_0 * model_par.eps_r_eff

    return (
        - np.pi
        * eps
        * model_par.radius**2
        * model_par.vbias**2
        / (z_axis * (z_axis + model_par.radius))
    )


# TODO: See TODO for sphere_plane_force
def sphere_plane_forcegradient(
    z_axis: ArrayLike,
    model_par: SpherePlaneElectrostaticParameters | object = None,
) -> NDArray[np.float64]:
    """ Calculate the force gradient of the sphere-plane interaction.

    The force gradient is the derivative of the electrostatic sphere-plane force
    with respect to the tip-sample distance. Distances below ``z_min`` are
    clipped to avoid division by zero and numerical divergence.

        F(z) = - pi * eps * R^2 * V^2 / (z * (z + R))

        dF/dz = pi * eps * R^2 * V^2 * (2z + R)
                / (z^2 * (z + R)^2)

    Parameters
    ----------
    z_axis:
        Tip-sample distance axis in m.
    model_par:
        Sphere-plane electrostatic parameters.

    Returns
    -------
    force_gradient:
        Electrostatic sphere-plane force gradient in N/m.
    """

    model_par = resolve_params(
        model_par,
        expected_type=SpherePlaneElectrostaticParameters,
        default=SpherePlaneElectrostaticParameters(),
    )

    z_axis = _as_array(z_axis)
    z_axis = np.maximum(z_axis, model_par.z_min)

    eps = C.epsilon_0 * model_par.eps_r_eff

    return (
        np.pi
        * eps
        * model_par.radius**2
        * model_par.vbias**2
        * (2 * z_axis + model_par.radius)
        / (z_axis**2 * (z_axis + model_par.radius) ** 2)
    )