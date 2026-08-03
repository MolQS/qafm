# src/qafm/fm/observables.py

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import fsolve
from typing import Literal

from qafm.averaging import wcap, wcup
from qafm.numerics.utils import _as_array
from qafm.oscillator import OscillatorParameters
from qafm.parameters import resolve_params


from qafm.parameters import resolve_params


# TODO: check the output of Feven_circ and how to document the input Feven.
# TODO: Allow zc to be an array. Iterate over the array values. 
def Feven_circ(
    Feven,
    zc: float,
    A0: float,
    fexc: float,
    phi: float,
    N: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """ Evaluate the even force along one harmonic oscillation cycle.

    The tip-sample distance and velocity are calculated using the harmonic
    approximation,

    ``zts(t)  = zc + A * cos(2 * pi * fexc * t + phi)``, and
    ``ztsp(t) = -A * 2 * pi * fexc * sin(2 * pi * fexc * t + phi)``

    and the supplied force function Feven is evaluated along this trajectory.

    Parameters
    ----------
    Feven:
        Function (callable) for even force. It must accept the tip-sample distance
        ``zts`` and velocity ``ztsp`` as arguments.
    zc:
        Center position of the oscillation in m.
    A0:
        Oscillation amplitude in m.
    fexc:
        Excitation frequency in Hz.
    phi:
        Phase offset in rad.
    N:
        Number of sample points used for one oscillation cycle.

    Returns
    -------
    zts:
        Tip-sample distance values sampled along one oscillation cycle.
    t:
        Time values over one oscillation period.
    Feven_values:
        Even force values evaluated along the oscillation trajectory.
    """

    t = np.linspace(0, 1 / fexc, N)

    zts = zc + A0 * np.cos(2 * np.pi * fexc * t + phi)
    ztsp = -A0 * 2 * np.pi * fexc * np.sin(2 * np.pi * fexc * t + phi)

    return zts, t, Feven(zts, ztsp)


def Fevencup(
    zc: ArrayLike,
    Fevencirc,
    model_par: OscillatorParameters | object = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ Calculate the cup-averaged even force.

    The function evaluates the cup average of an even force sampled along the
    oscillation trajectory using the oscillation amplitude stored in
    ``model_par``.

    Parameters
    ----------
    zc:
        Center-position axis in m.
    Fevencirc:
        Even force values or callable to be cup-averaged.
    model_par:
        FM parameters containing the oscillation amplitude ``A0``.

    Returns
    -------
    zc_wcup:
        Center-position axis corresponding to the cup-averaged values.
    Fevencup_values:
        Cup-averaged even force values.
    """

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )
    
    A0 = model_par.A0

    zc_wcup, Fevencup_values = wcup(zc, Fevencirc, A0)

    return zc_wcup, Fevencup_values



def solve_approx_smallA(
    z_p: ArrayLike,
    ktsfunc,
    model_par: OscillatorParameters | object = None,
    return_coordinate: Literal["zc", "zp", "zts"] = "zc",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculate FM-AFM observable <kts>_cap using the small-amplitude approximation.

    This approximation neglects convolution over the oscillation cycle, ignores
    the static deflection (``q_s=0``), and assumes that the frequency shift is 
    small compared to the free resonance frequency f_0. 
    The harmonic approximation is used. 
    The approximation reads:

    ``<k_ts>^\circ_cap(z) ≈ k_ts(z)``.

    It further assumes no dissipation, i.e. ``\gamma_ts = 0``. 

    Parameters
    ----------
    z_p:
        Piezo-position axis in m.
    ktsfunc:
        Tip-sample force-gradient function or precomputed force-gradient values.
        If callable, it is evaluated at ``z_p``. If array-like, it must describe
        the force gradient on the same axis as ``z_p``.
    model_par:
        AFM parameters containing the sensor and oscillation parameters.
    return_coordinate:
        Coordinate returned as the first result:

        - ``"zc"``: center-position axis ``zc = zp + qs``;
        - ``"zp"``: input piezo-position axis;
        - ``"zts"``: lower turning-point axis ``zts = zc - A0``.

        The default is ``"zc"``.

    Returns
    -------
    z:
        z-koordinate. Default is the center-position axis ``zc``.
    ktscap:
        Approximate cap-averaged force gradient in N/m.

    """

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    z_p = _as_array(z_p)

    # check for valid return_coordinate value
    valid_z_coordinates = {"zc", "zp", "zts"}

    if return_coordinate not in valid_z_coordinates:
        raise ValueError(
            f"Unknown return_coordinate {return_coordinate!r}. "
            "Expected 'zc', 'zp', or 'zts'."
        )

    if callable(ktsfunc):
        ktscapapprox = ktsfunc(z_p)
    else:
        ktscapapprox = _as_array(ktsfunc)

    zc = z_p

    # get amplitude (in m)
    A0 = model_par.A0

    # Select the requested output coordinate.
    if return_coordinate == "zc":
        z = zc
    elif return_coordinate == "zp":
        z = z_p
    else:  # return_coordinate == "zts"
        z = zc - A0

    return z, ktscapapprox


def solve_approx(
    z_p: ArrayLike,
    ktsfunc,
    model_par: OscillatorParameters | object = None,
    return_coordinate: Literal["zc", "zp", "zts"] = "zc",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r""" Approximate FM-AFM observable <kts>_cap from a given force gradient function.

    The function calculates the cap-averaged force gradient using the harmonic
    approximation by assuming zero static deflection, ``q_s = 0``, as well
    as zero damping, ``\gamma_ts = 0``. 
    The AFM observables are calculated with respect to the piezo position z_p.
    As ``q_s=0``, this position is identical to z_ts and z_tip with ``\delta z_0=0``

    Parameters
    ----------
    z_p:
        Piezo-position axis in m.
    ktsfunc:
        Tip-sample force-gradient function or precomputed force-gradient values.
        If callable, it is passed to the averaging routine. If it is ArrayLike, 
        it is interpreted as force-gradient values sampled at ``z_p``.
    model_par:
        OscillatorParameters containing the oscillation amplitude ``A0``.
    return_coordinate:
        Coordinate returned as the first result:

        - ``"zc"``: center-position axis ``zc = zp``;
        - ``"zp"``: input piezo-position axis;
        - ``"zts"``: lower turning-point axis ``zts = zc - A0``.

        The default is ``"zc"``.

    Returns
    -------
    z:
        z-koordinate. Default is the center-position axis ``zc``.
    ktscap:
        Cap-averaged force gradient in N/m.

    """

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    # check for valid return_coordinate value
    valid_z_coordinates = {"zc", "zp", "zts"}

    if return_coordinate not in valid_z_coordinates:
        raise ValueError(
            f"Unknown return_coordinate {return_coordinate!r}. "
            "Expected 'zc', 'zp', or 'zts'."
        )

    z_p = _as_array(z_p)

    # get amplitude (in m)
    A0 = model_par.A0

    # Sort z_p array
    sortidx = np.argsort(z_p)
    z_p = z_p[sortidx]

    if isinstance(ktsfunc, np.ndarray):
        ktsfunc = ktsfunc[sortidx]

    z_c = z_p

    # call general cap averaging
    zc, ktscap = wcap(z_c, ktsfunc, A0)


    # Select the requested output coordinate.
    if return_coordinate == "zc":
        z = zc
    elif return_coordinate == "zp":
        z = z_p
    else:  # return_coordinate == "zts"
        z = zc - A0

    return z, ktscap


def solve_iter(
    z_p: ArrayLike,
    Fts,
    kts,
    gammats,
    model_par: OscillatorParameters | object = None,
    maxIter: int = 10,
    debug: bool = False,
    return_coordinate: Literal["zc", "zp", "zts"] = "zc",
):
    """ Solve the q-afm FM-AFM equations iteratively.

    This solver evaluates the FM case with fixed oscillation amplitude
    ``A = A0`` and fixed phase ``phi = -pi / 2``. 
    It iteratively updates the static deflection ``q_s`` from 
    the cup-averaged even force, then recalculates the center-position axis 
    and the cap-averaged force-gradient and damping terms.

    Parameters
    ----------
    z_p:
        Piezo-position axis in m.
    Fts:
        Tip-sample force function or values used for the cup-averaged even
        force.
    kts:
        Tip-sample force-gradient function or values used for the cap-averaged
        force gradient.
    gammats:
        Tip-sample damping function or values used for the cap-averaged damping
        coefficient.
    model_par:
        OscillatorParameters containing ``f0``, ``k0``, ``Q0``, and ``A0``.
    maxIter:
        Maximum number of iterations.
    debug:
        If ``True``, prints iteration diagnostics and includes additional debug
        data in the returned tuple.
    return_coordinate:
        Coordinate returned as the first result:

        - ``"zc"``: center-position axis ``zc = zp + qs``;
        - ``"zp"``: input piezo-position axis;
        - ``"zts"``: lower turning-point axis ``zts = zc - A0``.

        The default is ``"zc"``.

    Returns
    -------
    z:
        z-koordinate. Default is the center-position axis ``zc``.
    Fevencup:
        Cup-averaged even force in N.
    ktscap:
        Cap-averaged force gradient in N/m.
    gammatscap:
        Cap-averaged damping coefficient.
    debug_data:
        Only returned if ``debug=True``. List of dictionaries containing
        intermediate arrays and tolerance estimates for each iteration.

    Warns
    -----
    RuntimeWarning
        Emitted if the averaging routines return arrays with lengths different
        from the current center-position axis.
    """

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

    z_p = _as_array(z_p)

    # check for valid return_coordinate value
    valid_z_coordinates = {"zc", "zp", "zts"}

    if return_coordinate not in valid_z_coordinates:
        raise ValueError(
            f"Unknown return_coordinate {return_coordinate!r}. "
            "Expected 'zc', 'zp', or 'zts'."
        )

    # arrays for results
    zc = z_p * 0.0
    qs = z_p * 0.0
    ktscapprev = z_p * 0.0
    gammatscapprev = z_p * 0.0

    if debug:
        debug_data = []

    # oscillation amplitude zero-peak, in m
    A0 = model_par.A0

    # z0 shift
    z0 = 0.0

    # Iteration counter
    i = 0

    while True:
        if debug:
            print(f"iteration {i}/{maxIter}...")

        if i >= maxIter:
            if debug:
                print("maximum number of iterations reached.")
            break

        # 1. get the z centre position axis using the static deflection
        zc = z0 + z_p + qs

        # 2. calculate the cup-averaged force
        zc_wcup, Fevencup = wcup(zc, Fts, A0)

        # Ensure that zc_wcup and zc have the same size
        if len(zc_wcup) != len(zc):
            warnings.warn(
                "Different length returned by wcup for Feven: "
                f"{len(zc_wcup)}. len(zc)={len(zc)}.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # deviation to the round before
        tolerrqs = np.sqrt(np.mean(np.pow(Fevencup - qs*model_par.k0, 2)))

        # static deflection
        qs = Fevencup / model_par.k0

        # and re-define the z axis for centre position z_c
        zc = z0 + z_p + qs

        # 3. calculate the average of the even force gradient
        zc_wcap, ktscap = wcap(zc, kts, A0)

        # Ensure that zc_wcap and zc have the same size
        if len(zc_wcap) != len(zc):
            warnings.warn(
                "Different length returned by wcap for kts: "
                f"{len(zc_wcap)}. len(zc)={len(zc)}.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # deviation to the round before
        tolerrkts = np.sqrt(np.mean(np.pow(ktscapprev - ktscap, 2)))

        # 4. calculate the average damping
        zc_wcap, gammatscap = wcap(zc, gammats, A0)

        # Ensure that zc_wcap and zc have the same size
        if len(zc_wcap) != len(zc):
            warnings.warn(
                "Different length returned by wcap for gammats: "
                f"{len(zc_wcap)}. len(zc)={len(zc)}.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # deviation to the round before
        tolerrgts = np.sqrt(np.mean(np.pow(gammatscapprev - gammatscap, 2)))

        # store for next run
        ktscapprev = ktscap
        gammatscapprev = gammatscap

        if debug:
            print(
                f"round i={i}, "
                f"Tolqs={tolerrqs:.2e}, "
                f"Tolkts={tolerrkts:.2e}, "
                f"Tolgts={tolerrgts:.2e}"
            )

        # TODO: Could include an abort condition here depending on the Tolqs, Tolkts and Tolgts values.

        if debug:
            debug_data.append(
                {
                    "i": i,
                    "zc": zc,
                    "zc_wcup": zc_wcup,
                    "Fevencup": Fevencup,
                    "qs": qs,
                    "tolerrqs": tolerrqs,
                    "tolerrkts": tolerrkts,
                    "zc_wcap": zc_wcap,
                    "ktscap": ktscap,
                    "gammatscap": gammatscap,
                    "tolerrgts": tolerrgts,
                }
            )

        i += 1

    # calculate the final zc, Fevencup, ktscap, gammatscap values
    zc = z0 + z_p + qs

    zc_wcup, Fevencup = wcup(zc, Fts, A0)
    zc_wcap, ktscap = wcap(zc, kts, A0)
    _, gammatscap = wcap(zc, gammats, A0)

    # Select the requested output coordinate.
    if return_coordinate == "zc":
        z = zc
    elif return_coordinate == "zp":
        z = z_p
    else:  # return_coordinate == "zts"
        z = zc - A0

    if debug:
        return z, Fevencup, ktscap, gammatscap, debug_data

    return z, Fevencup, ktscap, gammatscap


# TODO: Rethink the parameter z_order: Is one case redundant?
def solve_full(
    z_p: ArrayLike,
    Fts,
    kts,
    gammats,
    model_par: OscillatorParameters | object = None,
    z_order = 'zp',
    debug: bool = False,
    return_coordinate: Literal["zc", "zp", "zts"] = "zc",
) -> tuple[NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64],
    ]:
    """ Solve the FM-AFM equations pointwise.

    This function solves the constant-amplitude, constant-phase
    FM-AFM equations numerically at each piezo position, starting at the largest z. 
    The solution at the previous, larger tip-sample distance is used as 
    the initial guess for the next point. 
    The solved quantities are the
       - excitation frequency (fexc), 
       - static deflection (qs), and
       - excitation force amplitude (F0). 
    
    (original code by Hagen Söngen for 'mode 4' in the QAFM paper 
    [Söngen et al., J. Phys. Cond. Matter 29, 274001 (2017)].
    Original function name:
    solve_constant_A_phi(zp, qs_guess, F0_guess, fexc_guess).

    Parameters
    ----------
    z_p:
        Piezo-position axis in m.
    Fts:
        Tip-sample force function used for the cup-averaged even force.
    kts:
        Tip-sample force-gradient function used for the cap-averaged force
        gradient.
    gammats:
        Tip-sample damping function used for the cap-averaged damping
        coefficient.
    model_par:
        OscillatorParameters containing ``f0``, ``k0``, ``Q0``, and ``A0``.
    z_order:
        If set to 'keep', solve_full sorts values during the calculation, but
            restores the original order before returning the values. 
            (This is the default case)
        If set to 'order', solve_full will order both z_p and z_c incrementally. 
        If set to 'zp', it will order z_p but will not reorder the resulting z_c.
    debug:
        If ``True``, print progress information during the pointwise solve.

    return_coordinate:
        Coordinate returned as the first result:

        - ``"zc"``: center-position axis ``zc = zp + qs``;
        - ``"zp"``: input piezo-position axis;
        - ``"zts"``: lower turning-point axis ``zts = zc - A0``.

        The default is ``"zc"``.

    Returns
    -------
    z:
        z-koordinate. Default is the center-position axis ``zc``.
    Fevencup:
        Cup-averaged even force in N.
    ktscap:
        Cap-averaged force gradient in N/m.
    gammatscap:
        Cap-averaged damping coefficient.
    debug_data:
        Only returned if ``debug=True``. List of dictionaries containing
        intermediate arrays and tolerance estimates for each iteration.

    Warns
    -----
    RuntimeWarning
        Returned if z_order is not well defined
    """

    model_par = resolve_params(
        model_par,
        expected_type=OscillatorParameters,
        default=OscillatorParameters(),
    )

	# check for valid return_coordinate value
    valid_z_coordinates = {"zc", "zp", "zts"}

    if return_coordinate not in valid_z_coordinates:
        raise ValueError(
            f"Unknown return_coordinate {return_coordinate!r}. "
            "Expected 'zc', 'zp', or 'zts'."
        )

    # get fixed sensor parameters
    f0 = model_par.f0
    k0 = model_par.k0
    Q0 = model_par.Q0
    A0 = model_par.A0

    # rescale gamma_0
    gamma0 = k0 / (2.0 * np.pi * f0 * Q0)
    phi0 = -np.pi / 2.0

    # ensure we're dealing with a numpy array
    z_p = _as_array(z_p)

    # sort input array. Fts, kts, gammats are callables evaluated pointwise
    # inside sseq_A_phi (via wcap/wcup), so only z_p needs sorting here.
    sortidx = np.argsort(z_p)
    z_p = z_p[sortidx]

    def gain_function(F0, fexc, k, gamma):
        ''' Direct access to the amplitude of the cantilever response G_ho 
            in the disturbed case, meaning 
            with effective force gradient k = k_0 + k_ts
            and with effective damping gamma = gamma_0 + gamma_ts

            gain_func = F_0*|G_ho|
        '''
        m0 = k0 / (2.0 * np.pi * f0) ** 2

        return F0 / np.sqrt(
            (k - m0 * (2.0 * np.pi * fexc) ** 2) ** 2
            + (2.0 * np.pi * fexc * gamma) ** 2
        )

    def phase_function(fexc, k, gamma):
        ''' Direct access to the phase of the cantilever response G_ho 
            in the disturbed case, see also comment for gain_function
        '''
        m0 = k0 / (2.0 * np.pi * f0) ** 2

        return np.arctan2(
            -2.0 * np.pi * fexc * gamma,
            k - m0 * np.pow(2.0 * np.pi * fexc, 2),
        )

    def sseq_A_phi(p, zp, Fts, kts, gammats):
        """ function to find the zeros.
            first argument p holds the optimisation paramters:
            p = [qs, F0, fexc]

            other arguments are the present variables.
            Note: This function expects scalar values for zp etc.!

            Idea is to find the solution to the AFM equations for a
            single position zp
        """

        # optimisation parameters, solution to AFM equations
        qs, F0, fexc = p

        # zc position, shift by initial q_s
        zc = zp + qs

        zckts, avgkts = wcap(zc, kts, A0)
        zcgts, avggts = wcap(zc, gammats, A0)
        zcFts, avgFeven = wcup(zc, Fts, A0)

        # z axes not used in the following
        _ = zckts, zcgts, zcFts

        # calculate the residuals
        res1 = A0 - gain_function(F0, fexc, k0 - avgkts, gamma0 + avggts)
        res2 = (k0 * qs) - avgFeven
        res3 = phi0 - phase_function(fexc, k0 - avgkts, gamma0 + avggts)

        return (res1[0], res2[0], res3[0])

    # initial values for fsolve
    qs_guess = 0.0
    fexc_guess = f0
    F0_guess = A0 / gain_function(1.0, fexc_guess, k0, gamma0)

    # result arrays
    qs_result = z_p * 0.0
    F0_result = z_p * 0.0
    fexc_result = z_p * 0.0

    # iterate over the z axis and solve for each z
    # start at largest z to re-use result in next iteration
    for i, zi in enumerate(reversed(z_p)):
        if debug:
            print(f" . z_index {i}/{len(z_p)}")

        # fsolve arg 1: 
        # "A function that takes at least one (possibly vector) argument, 
        # and returns a value of the same length."
        qs_i, F0_i, fexc_i = fsolve(
            sseq_A_phi,
            x0=(qs_guess, F0_guess, fexc_guess),
            args=(zi, Fts, kts, gammats),
            xtol=1e-12,
        )

        # store in result array
        qs_result[i] = qs_i
        F0_result[i] = F0_i
        fexc_result[i] = fexc_i

        # define starting values for next iteration
        qs_guess = qs_i
        F0_guess = F0_i
        fexc_guess = fexc_i

    # reverse arrays to bring to original ascending order
    qs_result = qs_result[::-1]
    F0_result = F0_result[::-1]
    fexc_result = fexc_result[::-1]

    # static deflection changes the z axis to zc
    zc = z_p + qs_result

    if z_order == 'keep':
        # restore the caller's original z_p order
        unsortidx = np.argsort(sortidx)
        zts = zts[unsortidx]
        qs_result = qs_result[unsortidx]
        F0_result = F0_result[unsortidx]
        fexc_result = fexc_result[unsortidx]
    elif z_order == 'order':
        # static deflection might cause disorder, so reorder zts
        sortidx = np.argsort(zts)
        zts = zts[sortidx]
        qs_result = qs_result[sortidx]
        F0_result = F0_result[sortidx]
        fexc_result = fexc_result[sortidx]
    elif z_order == 'zp':
        # keep z_p ordered
        pass
    else:
        warnings.warn(
                f"z_order  {z_order} is unkown."
                "Using 'zp'.",
                RuntimeWarning,
                stacklevel=2,
        )


    # Convert the solved FM observables to averaged interaction quantities.

    # <F_even>_U = k0 * qs
    Fevencup = k0 * qs_result

    # At phi = -pi/2:
    # ktscap = k0 * (1 - (fexc/f0)**2)
    ktscap = k0 * (
        1.0 - np.square(fexc_result / f0)
    )

    # gammatscap = F0/(2*pi*fexc*A0) - gamma0
    gammatscap = (
        F0_result
        / (2.0 * np.pi * fexc_result * A0)
        - gamma0
    )

    # Select the requested output coordinate.
    if return_coordinate == "zc":
        z = zc
    elif return_coordinate == "zp":
        z = z_p
    else:  # return_coordinate == "zts"
        z = zc - A0

    return z, Fevencup, ktscap, gammatscap
