# src/qafm/fm/inversion.py

import numpy as np
import warnings
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter
from typing import Literal

from .conversions import df_to_ktscap
from ..numerics.utils import _as_array
from ..averaging import wcup, wcap

from qafm.oscillator import OscillatorParameters



def Feven_deconv(
    zk: ArrayLike,
    ktscap: ArrayLike,
    A0: float,
    sgwin: int = 51,
    sgdegree: int = 3,
    tozero: int | bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate conservative force from cap-averaged force gradient.

    This function performs the Sader-Jarvis force deconvolution [1] from the
    cap-averaged force gradient `<k_ts>_cap`.
    It uses partial integration for the first and last terms according to [2].

    A Savitzky-Golay filter is provided for data smoothing. 

    Parameters
    ----------
    zk:
        Distance axis in m.
    ktscap:
        Cap-averaged force gradient in N/m.
    A0:
        Oscillation amplitude in m.
    sgwin:
        Savitzky-Golay filter window length. Must be odd and larger than
        `sgdegree`. If invalid or <= 1, a numerical gradient is used instead.
    sgdegree:
        Savitzky-Golay polynomial order.
    tozero:
        If larger than zero, subtract the mean of the last `tozero` ktscap
        values before deconvolution.

    Returns
    -------
    zFeven:
        Distance axis for the reconstructed conservative force in m.
    Feven:
        Reconstructed conservative force in N.

    Raises
    ------
    ValueError:
        if length of zk and ktscap differ
        if zk or ktscap are not 1D arrays
        if len(zk)<3
        if A0 is negative (must be positive)
        if tozero is larger than len(zk)

    References
    ----------
    [1] Sader, J. E. and Jarvis, S. P., 
        Accurate formulas for interaction force and energy in frequency modulation force spectroscopy
        Applied Physics Letters 84, 1801 (2004), DOI: 10.1063/1.1667267
    [2] Jarvis, S. P., Mathematica notebook for force conversion
        https://www.nanofunction.org/s/fmafm_Sader.nb
    
    """

    zk = _as_array(zk)
    ktscap = _as_array(ktscap)

    N = len(zk)

    if N != len(ktscap):
        raise ValueError(
            f"Lengths of zk (len {N}) and ktscap (len {len(ktscap)}) do not match."
        )

    if zk.ndim != 1:
        raise ValueError("zk and ktscap must be one-dimensional arrays.")

    if N < 3:
        raise ValueError("At least three data points are required.")

    if A0 <= 0:
        raise ValueError("A0 must be positive.")

    if np.any(np.diff(zk) <= 0):
        warnings.warn(
            "zk must be strictly increasing. Performing sort.",
            RuntimeWarning,
            stacklevel=2,
        )
        sortidx = np.argsort(zk)
        zk = zk[sortidx]
        ktscap = ktscap[sortidx]
        

    if isinstance(tozero, bool):
        tozero_n = 0
    else:
        tozero_n = int(tozero)

    if tozero_n > 0:
        if tozero_n > N:
            raise ValueError("tozero must not be larger than the data length.")
        ktscap = ktscap - np.mean(ktscap[-tozero_n:])

    if (
        sgwin > 1
        and sgdegree >= 1
        and sgwin > sgdegree
        and sgwin % 2 == 1
        and sgwin <= N
    ):
        ddfdz = savgol_filter(
            ktscap,
            window_length=sgwin,
            polyorder=sgdegree,
            deriv=1,
            delta=zk[1] - zk[0],
        )
    else:
        ddfdz = np.gradient(ktscap, zk)

    ddz = np.gradient(zk)

    Feven = np.zeros(N - 2, dtype=float)

    for j in range(N - 2):
        dz_rel = zk[j + 1 :] - zk[j]

        integrand = (
            (
                1
                + np.sqrt(A0 / (64 * np.pi * dz_rel))
            )
            * ktscap[j + 1 :]
            - np.sqrt(A0**3 / (2 * dz_rel)) * ddfdz[j + 1 :]
        )

        int_val = -np.trapezoid(integrand, zk[j + 1 :])

        corr0 = -ktscap[j] * abs(ddz[j])
        corr1 = -np.sqrt(A0 / (16 * np.pi)) * ktscap[j] * np.sqrt(abs(ddz[j]))
        corr2 = np.sqrt(4 * A0**3 / 2) * ddfdz[j] * np.sqrt(abs(ddz[j]))

        Feven[j] = int_val + corr0 + corr1 + corr2

    zFeven = zk[:-2]

    return zFeven, Feven


# TODO: implement the potential recovery here
def Ueven_deconv():
    pass





def Feven_matrix_deconv(
    zk: ArrayLike,
    ktscap: ArrayLike,
    A0: float,
    tozero: int | bool = False,
    spacing_tolerance: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Reconstruct the even force using the matrix method published in [1]

    Parameters
    ----------
    zk:
        Distance axis in m (needs to be equidistant)
    ktscap:
        Cap-averaged force gradient in N/m.
    A0:
        Oscillation amplitude in m.
    tozero:
        If an integer > 0, subtract the mean of the last `tozero`
        ktscap values before deconvolution.
        If False, no offset correction is applied.
    spacing_tolerance:
        Maximum allowed relative deviation from equidistant sampling.

    Returns
    -------
    zFeven:
        Distance axis in m.
    Feven:
        Reconstructed even force in N.

    Raises
    ------
    ValueError:
        if length of zk and ktscap differ
        if zk or ktscap are not 1D arrays
        if len(zk)<3
        if A0 is negative or infinite
        if zk or ktscap contain infinite numbers
        if zk is not equidistant. 
        if discretisation of oscillation range [-A0,A0] fails
        if tozero is larger than len(zk)

    References
    ----------
    [1] Giessibl, F. J., 
        A direct method to calculate tip-sample forces from frequency shifts in frequency-modulation atomic force microscopy,
        Applied Physics Letters, 78, 123 (2001), DOI: 10.1063/1.1335546

    Notes
    -----
    (1) This method has an error showing a divergence at 3times A0 ??
    """

    zk = _as_array(zk)
    ktscap = _as_array(ktscap)

    N = len(zk)

    # check if all iputs are valid
    if zk.ndim != 1 or ktscap.ndim != 1:
        raise ValueError("zk and ktscap must be one-dimensional arrays.")

    n = zk.size

    if n != ktscap.size:
        raise ValueError(
            f"Lengths do not match: len(zk)={n}, len(ktscap)={ktscap.size}."
        )

    if n < 3:
        raise ValueError("At least three data points are required.")

    if not np.isfinite(A0) or A0 <= 0:
        raise ValueError("A0 must be a finite positive number.")

    if not np.all(np.isfinite(zk)):
        raise ValueError("zk contains non-finite values.")

    if not np.all(np.isfinite(ktscap)):
        raise ValueError("ktscap contains non-finite values.")

    dz = np.diff(zk)

    if np.any(np.diff(zk) <= 0):
        warnings.warn(
            "zk must be strictly increasing. Performing sort.",
            RuntimeWarning,
            stacklevel=2,
        )
        sortidx = np.argsort(zk)
        zk = zk[sortidx]
        ktscap = ktscap[sortidx]

    delta = float(np.mean(dz))

    relative_spacing_error = np.max(np.abs(dz - delta)) / abs(delta)

    if relative_spacing_error > spacing_tolerance:
        raise ValueError(
            "zk must be equidistant. "
            f"Maximum relative spacing deviation is "
            f"{relative_spacing_error:.3e}."
        )

    # Optional zero-level correction
    if isinstance(tozero, bool):
        tozero_n = 0
    else:
        tozero_n = int(tozero)

    if tozero_n < 0:
        raise ValueError("tozero must be non-negative.")

    if tozero_n > n:
        raise ValueError("tozero must not exceed the data length.")

    ktscap_work = ktscap.copy()

    if tozero_n > 0:
        offset = np.mean(ktscap_work[-tozero_n:])
        ktscap_work -= offset

    # Amplitude as integer number of grid steps
    alpha = int(np.rint(A0 / delta))

    if alpha < 1:
        raise ValueError(
            "A0 is smaller than approximately one distance step."
        )

    effective_amplitude = alpha * delta
    relative_amplitude_error = abs(effective_amplitude - A0) / A0

    if relative_amplitude_error > 0.01:
        import warnings

        warnings.warn(
            "Amplitude discretization error exceeds 1%. "
            f"A0={A0:.6e} m, "
            f"A_eff={effective_amplitude:.6e} m, "
            f"relative error={relative_amplitude_error:.2%}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Reverse order to match the original matrix algorithm
    ktscap_reversed = ktscap_work[::-1]

    # initalize matrix W
    W = np.zeros((n, n), dtype=float)

    prefactor = (
        2.0 / (np.pi * effective_amplitude)
        * 2.0 / (2 * alpha + 1)
    )

    # calculate elements of matrix W, Eq.(6) in F. J. Giessibl "A Direct
    # Method to Calculate Tip-Sample Forces from Frequency Shifts in
    # Frequency-Modulation Atomic Force Microscopy"
    # Applied Physics Letters 78, 123-125 (2001)
    for i in range(n):
        first_index = max(i - 2 * alpha, 0)

        for j in range(first_index, i + 1):
            m = i - j

            root1_argument = (
                (2 * alpha + 1) * (m + 1)
                - (m + 1) ** 2
            )

            root0_argument = (
                (2 * alpha + 1) * m
                - m**2
            )

            root1_argument = max(root1_argument, 0.0)
            root0_argument = max(root0_argument, 0.0)

            W[i, j] = prefactor * (
                np.sqrt(root1_argument)
                - np.sqrt(root0_argument)
            )

    # Solve K @ F = -ktscap
    Feven_reversed = np.linalg.solve(
        W,
        -ktscap_reversed,
    )

    Feven = Feven_reversed[::-1]
    zFeven = zk.copy()

    return zFeven, Feven


def df_to_force(
    z: ArrayLike,
    df: ArrayLike,
    sensor: OscillatorParameters = OscillatorParameters(),
    *,
    algorithm: Literal["sj", "matrix"] = "sj",
    sgwin: int = 51,
    sgdegree: int = 3,
    tozero: int | bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert frequency-shift data to a conservative force curve.

    This function performs the workflow:

        df(z) -> <k_ts>_cap(z) -> F_even(z)

    First, the frequency shift is converted to the cap-averaged force
    gradient using `qafm.quant.conversions.df_to_ktscap`. Then the
    Sader-Jarvis deconvolution is applied using `Feven_deconv`.

    Parameters
    ----------
    z:
        Distance axis in m.
    df:
        Frequency shift in Hz.
    sensor:
        Sensor parameters containing f0, k0, Q0, and A0.
        The amplitude `sensor.A0` is used for deconvolution.
    algorithm:
        Selects between 
        'sj' : Sader/Jarvis via Feven_deconv
        'matrix' : Matrix method via Feven_matrix_deconv
    sgwin:
        Savitzky-Golay filter window length passed to `Feven_deconv`.
    sgdegree:
        Savitzky-Golay polynomial order passed to `Feven_deconv`.
    tozero:
        If larger than zero, subtract the mean of the last `tozero`
        ktscap values before deconvolution.

    Returns
    -------
    zFeven:
        Distance axis for the reconstructed conservative force in m.
    Feven:
        Reconstructed conservative force in N.
    """

    z = _as_array(z)
    df = _as_array(df)

    if z.shape != df.shape:
        raise ValueError(
            f"z and df must have the same shape. Got {z.shape=} and {df.shape=}."
        )

    # check for valid algorithm value
    algorithms = {"sj", "matrix"}

    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. "
            "Expected 'sj', 'matrix'."
        )

    ktsevencap = df_to_ktscap(df, sensor)

    if algorithm == 'sj':
        # Sader/Jarvis method selected
        zFeven, Feven = Feven_deconv(
            zk=z,
            ktscap=ktsevencap,
            A0=sensor.A0,
            sgwin=sgwin,
            sgdegree=sgdegree,
            tozero=tozero,
        )
    elif algorithm == 'matrix':
        zFeven, Feven = Feven_matrix_deconv(
            zk=z,
            ktscap=ktsevencap,
            A0=sensor.A0,
            tozero=tozero,
        )
    else:
        raise ValueError(
                    f"Algorithm {algorithm} is unknown. Select from ['sj', 'matrix']."
                )

    return zFeven, Feven


