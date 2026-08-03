# src/qafm/forces/vdw.py
#
# This module implements various van der Waals force models for AFM tips,
# including conical tips, spherical apex tips, and integrated geometries.
# The models are based on the work of
# Argento & French (J. Appl. Phys. 80, 6081 (1996))
# and the F4a, F5, F6a, and F6b geometries from PRB 89, 235417.
#
# force_f3 = force_cone_sphere
# force_f4a = force_cone_integrated
# force_f4b = force_cone
# force_f5 = force_truncated_cone
# force_f6a = force_spherical_cap_cone
# force_f6b = force_spherical_cap_cone_geometric

import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray

from qafm.numerics.utils import _as_array
from qafm.parameters import resolve_params


@dataclass(frozen=True)
class VdwParameters:
    """ Parameter set for van der Waals force models. Standard Parameters for vdW
    taken from http://dx.doi.org/10.1103/PhysRevB.103.075409

    Attributes
    ----------
    H:
        Hamaker constant in J.
    Theta:
        Tip opening angle in degree.
    R:
        Tip radius or characteristic tip length in m, depending on the model.
    zoffset:
        Offset added to the surface-to-surface distance z in m.
    h1:
        Optional height parameter for F6-type models.
        If not given, R / 2 is used.
    L:
        Optional lateral length parameter for F6a. If not given,
        sqrt(2*h1*R - h1**2) is used.
    """
    H: float = 357.619e-21
    Theta: float = 29.7       # degree
    R: float = 5e-9
    zoffset: float = 583.04e-12

    # Optional parameters for F6-type models.
    # If None, simple defaults are used.
    default_h1: float | None = None
    default_L: float | None = None

    @property
    def theta_rad(self) -> float:
        return float(np.deg2rad(self.Theta))

    @property
    def h1(self) -> float:
        if self.default_h1 is None:
            return self.R / 2.0
        return self.default_h1

    @property
    def L(self) -> float:
        if self.default_L is None:
            h1 = self.h1
            return float(np.sqrt(2 * h1 * self.R - h1**2))
        return self.default_L


def cone_force(
    z_axis: ArrayLike,
    model_par: VdwParameters | object = None,
) -> NDArray[np.float64]:
    """van der Waals force for a conical tip. This implements the solution from 
    Argento&French, J. Appl. Phys. 80, 6081 (1996) [10.1063/1.363680]
    (formula F4b in Kuhn & Rahe). The tip is described by the opening angle
    Theta and the Hamaker constant H. An offset zoffset defines the z=0 position.:

        F(z) = -H * tan(theta)^2 / (6 * z_hat)

        with z_hat = z + z_offset.

    Parameters
    ----------
    z_axis:
        Surface-to-surface distance in m.
    model_par:
        van der Waals model parameters. The opening angle Theta is given
        in degree.

    Returns
    -------
    numpy.ndarray
        van der Waals force in N.
    """

    model_par = resolve_params(
        model_par,
        expected_type=VdwParameters,
        default=VdwParameters(),
    )

    z_axis = _as_array(z_axis)

    z_hat = z_axis + model_par.zoffset

    F_vdW = (
        -1.0
        * model_par.H
        * (np.tan(model_par.Theta * np.pi / 180) ** 2)
        / (6 * z_hat)
    )

    return F_vdW


def cone_forcegradient(
    z_axis: ArrayLike,
    model_par: VdwParameters | object = None,
) -> NDArray[np.float64]:
    """ Force gradient of the sharp-cone van der Waals force.

    This is the derivative of force_cone with respect to z:

        dF/dz = H * tan(theta)^2 / (6 * z_hat**2)

        with z_hat = z + z_offset.

    Parameters
    ----------
    z_axis:
        Surface-to-surface distance in m.
    model_par:
        van der Waals model parameters. The opening angle Theta is given
        in degree.

    Returns
    -------
    numpy.ndarray
        Force gradient in N/m.
    """
    
    model_par = resolve_params(
        model_par,
        expected_type=VdwParameters,
        default=VdwParameters(),
    )

    z_axis = _as_array(z_axis)

    z_hat = z_axis + model_par.zoffset

    return (
        model_par.H
        * (np.tan(model_par.Theta * np.pi / 180) ** 2)
        / (6 * z_hat**2)
    )


def cone_sphere_force(
    z_axis: ArrayLike,
    model_par: VdwParameters | object = None,
) -> NDArray[np.float64]:
    """ van der Waals force for a conical tip terminated by a spherical apex.

    This implements the Argento & French solution for a cone with spherical
    apex, corresponding to formula F3 in Kuhn & Rahe.
    J. Appl. Phys. 80, 6081 (1996) [10.1063/1.363680].

    Parameters
    ----------
    z_axis:
        Surface-to-surface distance in m.
    model_par:
        van der Waals model parameters:
        - H: Hamaker constant in J
        - R: tip radius in m
        - Theta: opening angle in degree
        - zoffset: z offset in m

    Returns
    -------
    numpy.ndarray
        van der Waals force in N.
    """

    model_par = resolve_params(
        model_par,
        expected_type=VdwParameters,
        default=VdwParameters(),
    )

    z_axis = _as_array(z_axis)

    theta = model_par.Theta * np.pi / 180
    z_hat = z_axis + model_par.zoffset

    F_vdW = (
        model_par.H
        * (model_par.R**2)
        * (1 - np.sin(theta))
        * (
            model_par.R * np.sin(theta)
            - z_hat * np.sin(theta)
            - model_par.R
            - z_hat
        )
        / (
            (6 * (z_hat**2))
            * (
                model_par.R
                + z_hat
                - model_par.R * np.sin(theta)
            )**2
        )
        - (
            model_par.H
            * np.tan(theta)
            * (
                z_hat * np.sin(theta)
                + model_par.R * np.sin(theta)
                + model_par.R * np.cos(2 * theta)
            )
        )
        / (
            (6 * np.cos(theta))
            * (
                z_hat
                + model_par.R
                - model_par.R * np.sin(theta)
            )**2
        )
    )
    return F_vdW


def cone_sphere_forcegradient(
    z_axis: ArrayLike,
    model_par: VdwParameters | object = None,
) -> NDArray[np.float64]:
    """ Force gradient for the cone-sphere van der Waals force.

    This is the analytical force gradient corresponding to force_cone,
    i.e. the F3 Argento & French cone-sphere geometry.

    Parameters
    ----------
    z_axis:
        Surface-to-surface distance in m.
    model_par:
        van der Waals model parameters:
        - H: Hamaker constant in J
        - R: tip radius in m
        - Theta: opening angle in degree
        - zoffset: z offset in m

    Returns
    -------
    numpy.ndarray
        Force gradient in N/m.
    """

    model_par = resolve_params(
        model_par,
        expected_type=VdwParameters,
        default=VdwParameters(),
    )

    z_axis = _as_array(z_axis)

    theta = model_par.Theta * np.pi / 180
    z_hat = z_axis + model_par.zoffset

    k_vdW = (
        (
            model_par.H
            * model_par.R
            * (
                model_par.R
                * (2.0 * model_par.R + 3.0 * z_hat)
                + (
                    2 * model_par.R - z_hat
                )
                * np.sin(theta)
                * (
                    -2 * (model_par.R + z_hat)
                    + model_par.R * np.sin(theta)
                )
            )
            + model_par.H
            * z_hat**3
            * np.tan(theta)**2
        )
        / (
            6
            * (z_hat**3)
            * (
                model_par.R
                + z_hat
                - model_par.R * np.sin(theta)
            )**2
        )
    )

    return k_vdW


# TODO: change the name of this function to something more descriptive
def cone_integrated_force(
    z_axis: ArrayLike,
    model_par: VdwParameters | object = None,
) -> NDArray[np.float64]:
    """ van der Waals force for an integrated cone geometry.

    This implements the F4a geometry from PRB 89, 235417:

        F(z) = -2 * H * tan(theta)^2 / (3 * pi * z_hat)

    with z_hat = z + z_offset.

    Parameters
    ----------
    z_axis:
        Surface-to-surface distance in m.
    model_par:
        van der Waals model parameters. The opening angle Theta is given
        in degree.

    Returns
    -------
    numpy.ndarray
        van der Waals force in N.
    """
    
    model_par = resolve_params(
        model_par,
        expected_type=VdwParameters,
        default=VdwParameters(),
    )

    z_axis = _as_array(z_axis)

    z_hat = z_axis + model_par.zoffset
    theta = model_par.theta_rad
    H = model_par.H

    return -2.0 * H * np.tan(theta) ** 2 / (3.0 * np.pi * z_hat)


# TODO: not implemented yet
def cone_integrated_forcegradient():
    pass
        

def truncated_cone_force(
    z_axis: ArrayLike,
    model_par: VdwParameters | object = None,
) -> NDArray[np.float64]:
    """ van der Waals force for a truncated cone geometry.

    This implements the F5 geometry from PRB 89, 235417.

    The parameter L is currently taken as model_par.R. The opening angle
    Theta is given in degree and converted internally to radian.

    Parameters
    ----------
    z_axis:
        Surface-to-surface distance in m.
    model_par:
        van der Waals model parameters:
        - H: Hamaker constant in J
        - R: characteristic length L in m
        - Theta: opening angle in degree
        - zoffset: z offset in m

    Returns
    -------
    numpy.ndarray
        van der Waals force in N.
    """

    model_par = resolve_params(
        model_par,
        expected_type=VdwParameters,
        default=VdwParameters(),
    )

    z_axis = _as_array(z_axis)

    z_hat = z_axis + model_par.zoffset
    theta = model_par.theta_rad
    H = model_par.H
    L = model_par.R

    term = (
        1.0
        + (np.tan(theta) / L) * z_hat
        + (np.tan(theta) ** 2 / L**2) * z_hat**2
    )

    return -(2.0 * H * L**2) / (3.0 * np.pi * z_hat**3) * term


# TODO: not implemented yet
def truncated_cone_forcegradient():
    pass


def spherical_cap_cone_force(
    z_axis: ArrayLike,
    model_par: VdwParameters | object = None,
) -> NDArray[np.float64]:
    """ van der Waals force for a cone with spherical cap.

    This implements the F6a geometry from PRB 89, 235417.

    If no explicit h1 or L values are provided, the defaults are:
        h1 = R / 2
        L = sqrt(2*h1*R - h1**2)

    The opening angle Theta is given in degree and converted internally
    to radian.

    Parameters
    ----------
    z_axis:
        Surface-to-surface distance in m.
    model_par:
        van der Waals model parameters:
        - H: Hamaker constant in J
        - R: tip radius R in m
        - Theta: opening angle in degree
        - zoffset: z offset in m
        - h1: optional h1 parameter in m
        - L: optional L parameter in m

    Returns
    -------
    numpy.ndarray
        van der Waals force in N.
    """

    model_par = resolve_params(
        model_par,
        expected_type=VdwParameters,
        default=VdwParameters(),
    )

    z_axis = _as_array(z_axis)

    z_hat = z_axis + model_par.zoffset
    theta = model_par.theta_rad
    H = model_par.H
    R = model_par.R
    h1 = model_par.h1
    L = model_par.L

    term1 = h1**2 * (
        3.0 * R * z_hat + (R - z_hat) * h1
    ) / (
        z_hat**2 * (z_hat + h1) ** 3
    )

    term2 = L**2 / ((z_hat + h1) ** 3)

    term3 = (
        4.0
        * np.tan(theta)
        * (L + np.tan(theta) * (z_hat + h1))
        / (np.pi * (z_hat + h1) ** 2)
    )

    return -H / 6.0 * (term1 + term2 + term3)


# TODO: not implemented yet
def spherical_cap_cone_forcegradient():
    pass


def spherical_cap_cone_geometric_force(
    z_axis: ArrayLike,
    model_par: VdwParameters | object = None,
) -> NDArray[np.float64]:
    """ van der Waals force for a cone with spherical cap using geometric L.

    This implements the F6b geometry from PRB 89, 235417.

    If no explicit h1 value is provided, the default is:
        h1 = R / 2

    The opening angle Theta is given in degree and converted internally
    to radian.

    Parameters
    ----------
    z_axis:
        Surface-to-surface distance in m.
    model_par:
        van der Waals model parameters:
        - H: Hamaker constant in J
        - R: tip radius R in m
        - Theta: opening angle in degree
        - zoffset: z offset in m
        - h1: optional h1 parameter in m

    Returns
    -------
    numpy.ndarray
        van der Waals force in N.
    """

    model_par = resolve_params(
        model_par,
        expected_type=VdwParameters,
        default=VdwParameters(),
    )

    z_axis = _as_array(z_axis)

    z_hat = z_axis + model_par.zoffset
    theta = model_par.theta_rad
    H = model_par.H
    R = model_par.R
    h1 = model_par.h1

    sqrt_term = np.sqrt(R**2 - (R - h1) ** 2)

    term1 = h1**2 * (
        3.0 * R * z_hat + (R - z_hat) * h1
    ) / (
        z_hat**2 * (z_hat + h1) ** 3
    )

    term2 = (R**2 - (R - h1) ** 2) / ((z_hat + h1) ** 3)

    term3 = (
        4.0
        * np.tan(theta)
        * (sqrt_term + np.tan(theta) * (z_hat + h1))
        / (np.pi * (z_hat + h1) ** 2)
    )

    return -H / 6.0 * (term1 + term2 + term3)


# TODO: not implemented yet
def spherical_cap_cone_geometric_forcegradient():
    pass