# src/qafm/forces/chemical.py

import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray

from qafm.numerics.utils import _as_array
from qafm.parameters import resolve_params


@dataclass(frozen=True)
class MorseParameters:
    """Parameters of the Morse interaction model.

    Attributes
    ----------
    Kappa:
        Inverse decay length of the Morse interaction in 1/m.
    Sigma0:
        Equilibrium distance of the Morse potential in m.
    Ebond:
        Bond energy of the Morse potential in J.
    """
    kappa: float = 4.25e9
    sigma0: float = 0.235e-9
    ebond: float = 0.371e-18


@dataclass(frozen=True)
class LennardJonesParameters:
    """Parameters of the Lennard-Jones interaction model.

    The effective Lennard-Jones parameters are calculated from tip and sample
    parameters using the Lorentz-Berthelot mixing rules.

    Attributes
    ----------
    Sigma_tip:
        Lennard-Jones size parameter of the tip in m.
    Sigma_sample:
        Lennard-Jones size parameter of the sample in m.
    Epsilon_tip:
        Lennard-Jones energy parameter of the tip in J.
    Epsilon_sample:
        Lennard-Jones energy parameter of the sample in J.
    R_offset:
        Offset added to the surface-to-surface distance to obtain the effective
        interaction distance. If ``None``, the mixed ``sigma`` value is used.
    R_min:
        Minimum allowed interaction distance in m. Used to avoid division by
        zero or numerical divergence at very small distances.
    """
    sigma_tip: float = 0.3e-9          # m
    sigma_sample: float = 0.3e-9       # m
    epsilon_tip: float = 1.0e-21       # J
    epsilon_sample: float = 1.0e-21    # J
    r_offset: float | None = None      # m
    r_min: float = 1.0e-12             # m

    @property
    def sigma(self) -> float:
        """Lorentz mixing rule: sigma_ab = (sigma_tip + sigma_sample) / 2."""
        return 0.5 * (self.sigma_tip + self.sigma_sample)

    @property
    def epsilon(self) -> float:
        """Berthelot mixing rule: epsilon_ab = sqrt(epsilon_tip * epsilon_sample)."""
        return float(np.sqrt(self.epsilon_tip * self.epsilon_sample))

    @property
    def effective_r_offset(self) -> float:
        """Use sigma as default offset if no explicit r_offset is given."""
        if self.r_offset is None:
            return self.sigma

        return self.r_offset


def morse_force(
    z_axis: ArrayLike,
    model_par: MorseParameters | object = None,
) -> NDArray[np.float64]:
    """ Evaluate the interaction force of the Morse potential.

    The Morse force is evaluated as a function of distance ``z_axis`` with the
    following formula:
        Fts_Morse = 2*ebond*kappa*(-exp(-kappa*(z_axis - sigma0))
                    + exp(-2*kappa*(z_axis - sigma0)))

    Parameters
    ----------
    z_axis:
        Tip-sample distance axis in m.
    model_par:
        Morse interaction parameters.

    Returns
    -------
    force:
        Morse interaction force in N.
    """

    model_par = resolve_params(
        model_par,
        expected_type=MorseParameters,
        default=MorseParameters(),
    )

    z_axis = _as_array(z_axis)

    return (
        model_par.ebond
        * 2
        * model_par.kappa
        * (
            -np.exp(
                -model_par.kappa
                * (z_axis - model_par.sigma0)
            )
            + np.exp(
                -2
                * model_par.kappa
                * (z_axis - model_par.sigma0)
            )
        )
    )


def morse_forcegradient(
    z_axis: ArrayLike,
    model_par: MorseParameters | object = None,
) -> NDArray[np.float64]:
    """ Evaluate the interaction force gradient of the Morse potential.

    The force gradient is the derivative of the Morse force with respect to the
    tip-sample distance with the following formula:
        kts_Morse = ebond*2*kappa^2*exp(-kappa*(z_axis - sigma0))
                    -2*exp(-2*kappa*(z_axis - sigma0))

    Parameters
    ----------
    z_axis:
        Tip-sample distance axis in m.
    model_par:
        Morse interaction parameters.

    Returns
    -------
    force_gradient:
        Morse force gradient in N/m.
    """

    model_par = resolve_params(
        model_par,
        expected_type=MorseParameters,
        default=MorseParameters(),
    )

    z_axis = _as_array(z_axis)

    return (
        model_par.ebond
        * 2
        * (model_par.kappa**2)
        * (
            np.exp(
                -model_par.kappa
                * (z_axis - model_par.sigma0)
            )
            - 2
            * np.exp(
                -2
                * model_par.kappa
                * (z_axis - model_par.sigma0)
            )
        )
    )


# TODO: change distance from tip-sample to simply distance r
def lennard_jones_force(
    z_axis: ArrayLike,
    model_par: LennardJonesParameters | object = None,
) -> NDArray[np.float64]:
    """ Evaluate the interaction force of the Lennard-Jones potential.

    The interaction distance is calculated from the surface-to-surface distance
    as ``r = z_axis + R_offset``. If no explicit ``R_offset`` is given, the
    mixed Lennard-Jones size parameter ``Sigma`` is used. Very small distances
    are clipped to ``R_min`` to avoid numerical divergence.

    Parameters
    ----------
    z_axis:
        Surface-to-surface distance axis in m.
    model_par:
        Lennard-Jones interaction parameters.

    Returns
    -------
    force:
        Lennard-Jones interaction force in N.
    """

    model_par = resolve_params(
        model_par,
        expected_type=LennardJonesParameters,
        default=LennardJonesParameters(),
    )

    z_axis = _as_array(z_axis)

    r = z_axis + model_par.effective_r_offset

    # avoid division by zero or numerical divergence at very small distances
    r = np.maximum(r, model_par.r_min)

    return 24 * model_par.epsilon * (
        2 * (model_par.sigma**12) / (r**13)
        - (model_par.sigma**6) / (r**7)
    )


def lennard_jones_forcegradient(
    z_axis: ArrayLike,
    model_par: LennardJonesParameters | object = None,
) -> NDArray[np.float64]:
    """ Interaction force gradient of the Lennard-Jones potential.

    The force gradient is calculated with respect to the surface-to-surface
    distance ``z_axis``. Since ``r = z_axis + R_offset``, the derivative satisfies
    ``dr / dz = 1``. Very small interaction distances are clipped to ``R_min`` to
    avoid numerical divergence. For

        F(r) = 24 * epsilon * (2*sigma^12/r^13 - sigma^6/r^7)

    and r = z + r_offset, dr/dz = 1:

        dF/dz = 24 * epsilon * (
            -26*sigma^12/r^14 + 7*sigma^6/r^8
        )
   
    Parameters
    ----------
    z_axis:
        Surface-to-surface distance axis in m.
    model_par:
        Lennard-Jones interaction parameters.

    Returns
    -------
    force_gradient:
        Lennard-Jones force gradient in N/m.
    """

    model_par = resolve_params(
        model_par,
        expected_type=LennardJonesParameters,
        default=LennardJonesParameters(),
    )

    z_axis = _as_array(z_axis)

    r = z_axis + model_par.effective_r_offset
    r = np.maximum(r, model_par.r_min)

    return 24 * model_par.epsilon * (
        -26 * (model_par.sigma**12) / (r**14)
        + 7 * (model_par.sigma**6) / (r**8)
    )