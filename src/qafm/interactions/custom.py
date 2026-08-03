# custom.py

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qafm.interactions.chemical import morse_force, morse_forcegradient
from qafm.interactions.vdw import cone_force, cone_forcegradient
from qafm.interactions.vdw import cone_sphere_force, cone_sphere_forcegradient


def morse_vdw_cone_force(
    z_axis: ArrayLike,
    morse_par,
    vdw_par,
) -> NDArray[np.float64]:
    """ Definition of a Morse and vdW force law. 
        vdW force law is for conical tip following 
        Argento&French J. Appl. Phys. 80, 6081 (1996) [10.1063/1.363680] 
        (formula F_4b in Kuhn&Rahe)
    """
    return morse_force(z_axis, morse_par) + cone_force(z_axis, vdw_par)


def morse_vdw_cone_forcegradient(
    z_axis: ArrayLike,
    morse_par,
    vdw_par,
) -> NDArray[np.float64]:
    """ Morse and vdW force gradient for conical tip following 
        Argento&French J. Appl. Phys. 80, 6081 (1996) [10.1063/1.363680] 
        (formula F_4b in Kuhn&Rahe)
    """
    return morse_forcegradient(z_axis, morse_par) + cone_forcegradient(z_axis, vdw_par)


def morse_vdw_cone_sphere_force(
    z_axis: ArrayLike,
    morse_par,
    vdw_par,
    *,
    zero_for_negative_z: bool = True,
) -> NDArray[np.float64]:
    """ Definition of a Morse and vdW force law. 
        vdW force law is for conical tip with half-sphere as tip apex, following 
        Argento&French J. Appl. Phys. 80, 6081 (1996) [10.1063/1.363680] 
        (formula F_3 in Kuhn&Rahe)
    """
    force = morse_force(z_axis, morse_par) + cone_sphere_force(z_axis, vdw_par)

    if zero_for_negative_z:
        force = np.where(np.asarray(z_axis) < 0, 0.0, force)

    return force


def morse_vdw_cone_sphere_forcegradient(
    z_axis: ArrayLike,
    morse_par,
    vdw_par,
    *,
    zero_for_negative_z: bool = True,
) -> NDArray[np.float64]:
    """ Defintion of Morse and vdW force gradient for conical tip terminated by
        a half sphere following
        Argento&French J. Appl. Phys. 80, 6081 (1996) [10.1063/1.363680] 
        (formula F_3 in Kuhn&Rahe)
        Parameters are: H, R, Theta, zoffset
    """
    gradient = morse_forcegradient(z_axis, morse_par) + cone_sphere_forcegradient(z_axis, vdw_par)

    if zero_for_negative_z:
        gradient = np.where(np.asarray(z_axis) < 0, 0.0, gradient)

    return gradient


# TODO: implement viscous forces and gradients
def morse_viscous_force(z_axis, zp_axis, model_par):
    """ Morse and viscous force F_Viscous = 6*pi*r*nu*v """
    pass

def morse_viscous_forcegradient(z_axis, zp_axis, model_par):
    """ Morse and viscous force gradient F_Viscous = 6*pi*r*nu*v """
    pass