# src/qafm/quant/oscillator.py

import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit
from .numerics.utils import _as_array
#from qafm.parameters import resolve_params


def resolve_params( model_par, expected_type, default):
    """This looses the dependency on qafm.parameters.
    
    TODO: Update in next release!
    """
    return model_par


@dataclass(frozen=True)
class OscillatorParameters:
    f0: float = 277203.0        # Hz
    k0: float = 18.58           # N/m
    Q0: float = 2.0e4           # unitless
    A0: float = 1.0e-9          # m


@dataclass(frozen=True)
class FitResult:
    f0: float
    k0: float
    Q0: float
    covariance: NDArray[np.float64]
    standard_deviation: NDArray[np.float64]

    @property
    def parameters(self) -> OscillatorParameters:
        return OscillatorParameters(
            f0=self.f0,
            k0=self.k0,
            Q0=self.Q0,
        )


def q_t(
    t: ArrayLike,
    qs: float,
    A: float,
    fexc: float,
    phi: float,
) -> NDArray[np.float64]:
    """Calculate the sensor deflection in the harmonic approximation.

    The deflection is modeled as a cosine oscillation around a static offset.

    Parameters
    ----------
    t:
        Time values at which the deflection is evaluated.
    qs:
        Static sensor deflection offset.
    A:
        Oscillation amplitude.
    fexc:
        Excitation frequency in Hz.
    phi:
        Phase offset in radians.

    Returns
    -------
    q_t:
        Sensor deflection values evaluated at `t`.
    """

    t = _as_array(t)

    return qs + A * np.cos(2 * np.pi * fexc * t + phi)


def Gho(
    fexc: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.complex128]:
    """Calculate the complex transfer function of a damped harmonic oscillator.

    Parameters
    ----------
    fexc:
        Excitation frequency or frequencies in Hz.
    model_par:
        Oscillator parameters containing eigenfrequency, spring constant, and
        quality factor.

    Returns
    -------
    Gho:
        Complex transfer function values evaluated at `fexc`.
    """

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    fexc = _as_array(fexc)

    f0 = model_par.f0
    k0 = model_par.k0
    Q0 = model_par.Q0

    return (
        1
        / k0
        * 1
        / (
            1
            - np.pow(fexc / f0, 2)
            + 1j / Q0 * fexc / f0
        )
    )


def Gho_A(
    fexc: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    """Calculate the amplitude of the oscillator transfer function.

    This is equivalent to the absolute value of `Gho`.

    Parameters
    ----------
    fexc:
        Excitation frequency or frequencies in Hz.
    model_par:
        Oscillator parameters containing eigenfrequency, spring constant, and
        quality factor.

    Returns
    -------
    A:
        Amplitude of the transfer function evaluated at `fexc`.
    """

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    fexc = _as_array(fexc)

    f0 = model_par.f0
    k0 = model_par.k0
    Q0 = model_par.Q0

    return (
        1
        / k0
        * 1
        / np.sqrt(
            np.pow(1 - np.pow(fexc / f0, 2), 2)
            + np.pow(fexc / (Q0 * f0), 2)
        )
    )


def Gho_phi(
    fexc: ArrayLike,
    model_par: OscillatorParameters | object = None,
) -> NDArray[np.float64]:
    """Calculate the phase of the oscillator transfer function.

    Parameters
    ----------
    fexc:
        Excitation frequency or frequencies in Hz.
    model_par:
        Oscillator parameters containing eigenfrequency and quality
            factor.

    Returns
    -------
    phi:
        Phase of the transfer function evaluated at `fexc`, in radians.
    """

    fexc = _as_array(fexc)

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    f0 = model_par.f0
    Q0 = model_par.Q0

    return np.atan2(
        -fexc / (Q0 * f0),
        1 - np.pow(fexc / f0, 2),
    )


def f_resonance(
    model_par: OscillatorParameters | object = None,
) -> float:
    """Calculate the resonance frequency of the damped harmonic oscillator.

    Parameters
    ----------
    model_par:
        Oscillator parameters containing eigenfrequency and quality factor.

    Returns
    -------
    f_r:
        Resonance frequency in Hz.
    """

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    f0 = model_par.f0
    Q0 = model_par.Q0

    return float(f0 * np.sqrt(1 - 1 / (2 * np.pow(Q0, 2))))


def Gho_fit(
    f: ArrayLike,
    Af: ArrayLike,
    verbose: bool = True,
) -> FitResult:
    """Fit the amplitude response of a damped harmonic oscillator.

    The function fits the amplitude of the harmonic oscillator transfer
    function, ``Gho_A``, to measured amplitude-response data. Initial values
    for the fit parameters are estimated from the maximum amplitude, the
    frequency position of the maximum, and the full width at half maximum
    (FWHM).

    Parameters
    ----------
    f:
        Excitation frequencies in Hz. Must be a one-dimensional array with the
        same shape as ``Af``.
    Af:
        Measured amplitude response values. All values must be positive and the
        array must have the same shape as ``f``.
    verbose:
        If ``True``, print initial parameter estimates, fitted parameters,
        covariances, and standard deviations.

    Returns
    -------
    result:
        Fit result containing the fitted oscillator parameters ``f0``, ``k0``,
        and ``Q0``, as well as the covariance matrix and parameter standard
        deviations.

    Raises
    ------
    ValueError
        If ``f`` and ``Af`` have different shapes, are not one-dimensional,
        contain fewer than five data points, or if ``Af`` contains non-positive
        values.
    """

    f = _as_array(f)
    Af = _as_array(Af)

    if f.shape != Af.shape:
        raise ValueError(
            f"f and Af must have the same shape. Got {f.shape=} and {Af.shape=}."
        )

    if f.ndim != 1:
        raise ValueError("f and Af must be one-dimensional arrays.")

    if len(f) < 5:
        raise ValueError("Need at least 5 data points for Gho_fit.")

    if np.any(Af <= 0):
        raise ValueError("Af must contain only positive values for Gho_fit.")

    ### estimate the initial fit parameters

    # f0 from maximum value
    f0guess = f[np.argmax(Af)]

    # Q0 from FWHM
    half_max = np.max(Af) / 2
    crossings = np.where(np.diff(np.sign(Af - half_max)))[0]

    if len(crossings) >= 2:
        fwhm = np.abs(f[crossings[-1]] - f[crossings[0]])
        Q0guess = np.sqrt(3) * f0guess / fwhm
    else:
        # fallback if FWHM cannot be estimated robustly
        Q0guess = 1.0e4

    # k0 from maximum value maximum of |G_ho| is Q0/k0
    k0guess = Q0guess / np.max(Af)

    if verbose:
        print(
            "starting curve_fit. Initial values: "
            f"f0={f0guess:.2f}Hz; "
            f"k0={k0guess:.0f}N/m; "
            f"Q0={Q0guess:.0f}"
        )

    def Ghofunc(f_values, f0, k0, Q0):
        params = OscillatorParameters(
            f0=f0,
            k0=k0,
            Q0=Q0,
        )

        return Gho_A(f_values, params)

    popt, pcov = curve_fit(
        Ghofunc,
        f,
        Af,
        p0=[f0guess, k0guess, Q0guess],
    )

    # popt: best-fit parameters [f0, k0, Q0]
    # pcov: covariance matrix diagonal = variance of each param
    perr = np.sqrt(np.diag(pcov))

    result = FitResult(
        f0=float(popt[0]),
        k0=float(popt[1]),
        Q0=float(popt[2]),
        covariance=pcov,
        standard_deviation=perr,
    )

    if verbose:
        print(
            "Fit parameters:   "
            f"f0={result.f0:.2f}Hz; "
            f"k0={result.k0:.0f}N/m; "
            f"Q0={result.Q0:.0f}"
        )
        print(
            "covariances:         "
            f"{pcov[0, 0]:.2e} ; "
            f"{pcov[1, 1]:.0f} ; "
            f"{pcov[2, 2]:.0f}"
        )
        print(
            "standard deviations: "
            f"{perr[0]:.2e} ; "
            f"{perr[1]:.0f} ; "
            f"{perr[2]:.0f}"
        )

    return result