# src/qafm/fm/conversions.py

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qafm.numerics.utils import _as_array
from qafm.oscillator import OscillatorParameters
from qafm.parameters import resolve_params



def ktscap_to_fexc(
    ktscap: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    r"""Calculate the excitation frequency from the cap-averaged force gradient.

    Calculates the excitation frequency ``fexc`` from the cap-averaged force
    gradient ``<k_ts>_cap``. The implementation assumes the FM AFM mode with
    ``phi = -pi/2``, where ``F0 / A * cos(phi) = 0``.

    ``fexc`` follows from

        fexc = f0 * sqrt(1-1/k0*(ktscap))

    Parameters
    ----------
    ktscap:
        Cap-averaged force gradient in N/m.
    model_par:
        Sensor parameters or a qafm ``ParameterSet``. If ``None``, default
        ``OscillatorParameters`` are used.

    Returns
    -------
    fexc:
        Excitation frequency in Hz.

    Raises
    ------
    ValueError
        If the square-root argument becomes negative.
    """

    ktscap = _as_array(ktscap)

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    f0 = model_par.f0
    k0 = model_par.k0

    F0Acosphi = 0.0  # generally: F0/A*cos(phi) (FM mode: phi=-pi/2)

    sqrt_inner = 1 - 1 / k0 * (ktscap + F0Acosphi)

    if np.any(sqrt_inner < 0):
        raise ValueError(
            "Invalid ktscap: square-root argument became negative. "
            "Check signs or physical regime."
        )

    fexc = f0 * np.sqrt(sqrt_inner)

    return fexc



def ktscap_to_df_approx(
    ktscap: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    r"""Approximation for calculating the frequency shift from the cap-averaged force gradient.

    Calculates the frequency shift ``df`` from the cap-averaged force gradient
    ``<k_ts>_cap`` using the small frequency-shift approximation
    ``|f_exc - f0| << f0``, this is a linear re-scaling.

    The approximation is calculated as:

        <k_{ts}^\circ>_\cap \approx -2*k_0/f_0 * \Delta f

    Note: There is generally no need to use this approximation. 
    Use ktscap_to_fexc instead. 

    Parameters
    ----------
    ktscap:
        Cap-averaged force gradient in N/m.
    model_par:
        Sensor parameters or a qafm ``ParameterSet``. If ``None``, default
        ``OscillatorParameters`` are used.

    Returns
    -------
    df:
        Approximate frequency shift in Hz.
    """

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    ktscap = _as_array(ktscap)

    f0 = model_par.f0
    k0 = model_par.k0

    dfapprox = -f0 / (2 * k0) * ktscap

    return dfapprox


def fexc_to_ktscap(
    fexc: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    r"""Calculate the cap-averaged force gradient from the excitation frequency.

    Calculates ``<k_ts>_cap`` from the excitation frequency assuming FM mode
    with ``phi = -pi/2``, where ``F0 / A * cos(phi) = 0``.

    ``ktscap`` follows from

        ktscap = k0 * (1.0 - (fexc / f0)**2)

    Parameters
    ----------
    fexc:
        Excitation frequency in Hz.
    model_par:
        Sensor parameters used for the conversion.

    Returns
    -------
    ktscap:
        Cap-averaged force gradient in N/m.
    """

    fexc = _as_array(fexc)

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    f0 = model_par.f0
    k0 = model_par.k0

    F0Acosphi = 0.0  # If not FM mode with phi=-pi/2: F0/A*cos(phi)

    ktscap = k0 * (1.0 - np.pow(fexc / f0, 2)) - F0Acosphi

    return ktscap


def fexc_to_df(
    fexc: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    r"""Calculate the frequency shift from the excitation frequency.

    The frequency shift is calculated from

        \Delta f = f_\text{exc} - f_0 .

    Parameters
    ----------
    fexc:
        Excitation frequency in Hz.
    model_par:
        Sensor parameters or a qafm ``ParameterSet``. If ``None``, default
        ``OscillatorParameters`` are used.

    Returns
    -------
    df:
        Frequency shift in Hz.
    """

    fexc = _as_array(fexc)

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    f0 = model_par.f0

    return fexc - f0


def df_to_ktscap(
    df: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    r"""Calculate the cap-averaged force gradient from the frequency shift.

    Calculates the cap-averaged force gradient ``<k_ts>_cap`` from the frequency
    shift ``df``. The implementation assumes the FM mode with ``phi = -pi/2``, where
    ``F0 / A * cos(phi) = 0``.

    Uses fexc_to_ktscap with calculating ``fexc = f_0 + df``.

    Parameters
    ----------
    df:
        Frequency shift in Hz.
    model_par:
        Sensor parameters or a qafm ``ParameterSet``. If ``None``, default
        ``OscillatorParameters`` are used.

    Returns
    -------
    ktscap:
        Cap-averaged force gradient in N/m.
    """

    df = _as_array(df)

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    f0 = model_par.f0

    return fexc_to_ktscap(f0 + df, model_par)


def gammacap_to_F0(
    gammatscap: ArrayLike,
    fexc: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    """Calculate the excitation force amplitude F0.

    Calculates the excitation force amplitude from the cap-averaged damping
    coefficient and the excitation frequency for the FM AFM mode with
    phi = -pi/2.

    F0 follows from

        F0 = -2*pi * fexc * A0 * (gammatscap + k0 / (2*pi * f0 * Q0))

    Parameters
    ----------
    gammatscap:
        Cap-averaged damping coefficient ``<gamma_ts>_cap``.
    fexc:
        Excitation frequency in Hz.
    model_par:
        Sensor parameters or a qafm ``ParameterSet``.
        If ``None``, default ``OscillatorParameters`` are used.

    Returns
    -------
    F0:
        Excitation force amplitude in N.
    """

    gammatscap = _as_array(gammatscap)
    fexc = _as_array(fexc)

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    f0 = model_par.f0
    k0 = model_par.k0
    Q0 = model_par.Q0
    A0 = model_par.A0

    F0 = 2 * np.pi * fexc * A0 * (
        gammatscap + k0 / (2 * np.pi * f0 * Q0)
    )

    return F0


def F0_to_gammacap(
    F0: ArrayLike,
    fexc: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    """Calculate the cap-averaged damping coefficient.

    Calculates ``<gamma_ts>_cap`` from the excitation force amplitude and
    excitation frequency for the FM AFM mode with ``phi = -pi/2``.

    The damping coefficient follows from

        gammatscap = + F0 / (2*pi*fexc*A0)
                     - k0 / (2*pi*f0*Q0)

    Parameters
    ----------
    F0:
        Excitation force amplitude in N.
    fexc:
        Excitation frequency in Hz.
    model_par:
        Sensor parameters or a qafm ``ParameterSet``.
        If ``None``, default ``OscillatorParameters`` are used.

    Returns
    -------
    gammatscap:
        Cap-averaged damping coefficient ``<gamma_ts>_cap``.
    """

    F0 = _as_array(F0)
    fexc = _as_array(fexc)

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    f0 = model_par.f0
    k0 = model_par.k0
    Q0 = model_par.Q0
    A0 = model_par.A0

    gammatscap = (
        + F0 / (2 * np.pi * fexc * A0)
        - k0 / (2 * np.pi * f0 * Q0)
    )

    return gammatscap


def Fevencup_to_qs(
    Fevencup: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    """Calculate the static sensor displacement qs.

    The static displacement follows from

        qs = <F_even>_U / k0

    Parameters
    ----------
    Fevencup:
        Cup-averaged even tip-sample force ``<F_even>_U`` in N.
    model_par:
        Sensor parameters or a qafm ``ParameterSet``.
        If ``None``, default ``OscillatorParameters`` are used.

    Returns
    -------
    qs:
        Static sensor displacement in m.
    """

    Fevencup = _as_array(Fevencup)

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    k0 = model_par.k0

    if k0 <= 0:
        raise ValueError("k0 must be greater than zero.")

    qs = Fevencup / k0

    return qs


def qs_to_Fevencup(
    qs: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    """Calculate the cup-averaged even tip-sample force.

    The averaged force follows from

        <F_even>_U = k0 * qs

    Parameters
    ----------
    qs:
        Static sensor displacement in m.
    model_par:
        Sensor parameters or a qafm ``ParameterSet``.
        If ``None``, default ``OscillatorParameters`` are used.

    Returns
    -------
    Fevencup:
        Cup-averaged even tip-sample force ``<F_even>_U`` in N.
    """

    qs = _as_array(qs)

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    k0 = model_par.k0

    if k0 <= 0:
        raise ValueError("k0 must be greater than zero.")

    Fevencup = k0 * qs

    return Fevencup