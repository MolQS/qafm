# src/qafm/averaging.py

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from .numerics.utils import (
    _as_array,
    _is_equidistant,
    _num_positional_args
)


def _wcap_int(z0: float, z1: float, A0: float) -> float:
    """Evaluate an interval integral of the cap-weighting kernel.

    Calculates

    ``integral(sqrt(A0**2 - z**2), z=z0..z1)``.

    This helper is used to approximate the partially sampled boundary
    intervals in the array-based cap-averaging implementations.

    Parameters
    ----------
    z0:
        Lower integration boundary in m.
    z1:
        Upper integration boundary in m.
    A0:
        Oscillation amplitude in m.

    Returns
    -------
    integral:
        Value of the definite integral.
    """

    def stammfkt(z: float, A0: float) -> float:
        # Make sure no negative values are within the sqrt
        sqrtterm = np.sqrt(np.clip(np.pow(A0, 2) - np.pow(z, 2), 0.0, None))

        # arctan2 handles the case sqrtterm=0 intrinsically
        arctanterm = np.arctan2(z, sqrtterm)

        # Final result
        res = 1.0 / 2.0 * (z * sqrtterm + np.pow(A0, 2) * arctanterm)

        return float(res)

    return stammfkt(z1, A0) - stammfkt(z0, A0)


def _wcup_int(z0: float, z1: float, A0: float) -> float:
    """NOT SURE IF REQUIERED.
    
    Evaluate an interval integral of the cup-weighting kernel.

    Calculates

    ``integral(1 / sqrt(A0**2 - z**2), z=z0..z1)``.

    Parameters
    ----------
    z0:
        Lower integration boundary in m.
    z1:
        Upper integration boundary in m.
    A0:
        Oscillation amplitude in m.

    Returns
    -------
    integral:
        Value of the definite integral.
    """

    def stammfkt_wcup(z: float, A0: float) -> float:
        # Make sure no negative values are within the sqrt
        sqrtterm = np.sqrt(np.clip(np.pow(A0, 2) - np.pow(z, 2), 0.0, None))

        # arctan2 handles the case sqrtterm=0 intrinsically
        arctanterm = np.arctan2(z, sqrtterm)

        # Final result
        res = arctanterm

        return float(res)

    return stammfkt_wcup(z1, A0) - stammfkt_wcup(z0, A0)


def wcap(
    z_axis: ArrayLike,
    feq,
    A0: float,
    debug: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the cap-weighted average of a quantity.

    The appropriate averaging implementation is selected automatically based
    on the type of ``feq`` and the spacing of ``z_axis``. Callable inputs are
    integrated with ``scipy.integrate.quad``. Array inputs are processed with
    either an equidistant or non-equidistant numerical implementation.

    Parameters
    ----------
    z_axis:
        Position axis in m. For callable input, this represents the oscillation
        center positions. For array input, it represents the positions at which
        ``feq`` is sampled.
    feq:
        Quantity to average. May be a callable or a NumPy array containing
        sampled values.
    A0:
        Oscillation amplitude in m.
    debug:
        If ``True``, print information about the selected averaging method.

    Returns
    -------
    zc:
        Oscillation-center positions corresponding to the averaged values.
        For array input, the returned axis may be shorter than ``z_axis``.
    averaged:
        Cap-weighted average of ``feq``.

    Raises
    ------
    TypeError
        If ``feq`` is neither callable nor a NumPy array.
    """

    z_axis = _as_array(z_axis)

    # check if feq is numpy array or a function handle
    if callable(feq):
        if debug:
            print("wcap averaging. feq is a callable.")

        # perform analytical integration for each z_axis value.
        # This keeps z_axis at identical size.
        return _wcap_callable(z_axis, feq, A0)

    if isinstance(feq, np.ndarray):
        feq = _as_array(feq)

        # perform numerical integration if feq is an array.
        # check if z_axis is equidistant
        equidistant = _is_equidistant(z_axis)

        if equidistant:
            if debug:
                print("wcap averaging. z_axis is equidistant.")
            return _wcap_equiz(z_axis, feq, A0)

        if debug:
            print("wcap averaging. z_axis is NOT equidistant.")
        return _wcap_nonequiz(z_axis, feq, A0)

    raise TypeError(f"type of feq is unknown: {type(feq)}")

def wcup(
    z_axis: ArrayLike,
    feq,
    A0: float,
    debug: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the cup-weighted average of a quantity.

    The appropriate averaging implementation is selected automatically based
    on the type of ``feq`` and the spacing of ``z_axis``. Callable inputs are
    integrated with ``scipy.integrate.quad``. Array inputs are processed with
    either an equidistant or non-equidistant numerical implementation.

    Parameters
    ----------
    z_axis:
        Position axis in m. For callable input, this represents the oscillation
        center positions. For array input, it represents the positions at which
        ``feq`` is sampled.
    feq:
        Quantity to average. May be a callable or a NumPy array containing
        sampled values.
    A0:
        Oscillation amplitude in m.
    debug:
        If ``True``, print information about the selected averaging method.

    Returns
    -------
    zc:
        Oscillation-center positions corresponding to the averaged values.
        For array input, the returned axis may be shorter than ``z_axis``.
    averaged:
        Cup-weighted average of ``feq``.

    Raises
    ------
    TypeError
        If ``feq`` is neither callable nor a NumPy array.
    """

    z_axis = _as_array(z_axis)

    # check if feq is numpy array or a function handle
    if callable(feq):
        if debug:
            print("wcup averaging. feq is a callable.")

        # perform analytical integration for each z_axis value.
        # This keeps z_axis at identical size.
        return _wcup_callable(z_axis, feq, A0)

    if isinstance(feq, np.ndarray):
        feq = _as_array(feq)

        # perform numerical integration if feq is an array.
        # check if z_axis is equidistant
        equidistant = _is_equidistant(z_axis)

        if equidistant:
            if debug:
                print("wcup averaging. z_axis is equidistant.")
            return _wcup_equiz(z_axis, feq, A0)

        if debug:
            print("wcup averaging. z_axis is NOT equidistant.")
        return _wcup_nonequiz(z_axis, feq, A0)

    raise TypeError(f"type of feq is unknown: {type(feq)}")


def wcap_weighting(
    z: ArrayLike,
    A: float,
) -> NDArray[np.float64]:
    """Evaluate the cap-weighting function.

    The weighting function is nonzero only within the oscillation interval
    ``-A < z < A``.

    Parameters
    ----------
    z:
        Positions relative to the oscillation center in m.
    A:
        Oscillation amplitude in m.

    Returns
    -------
    weights:
        Cap-weighting values evaluated at ``z``.
    """

    z = _as_array(z)

    innersqrt = np.pow(A, 2) - np.pow(z, 2)

    return np.where(
        innersqrt > 0,
        2 / (np.pi * np.pow(A, 2)) * np.sqrt(innersqrt),
        0.0,
    )


def wcup_weighting(
    z: ArrayLike,
    A: float,
) -> NDArray[np.float64]:
    """Evaluate the cup-weighting function.

    The weighting function is nonzero only within the oscillation interval
    ``-A < z < A``. It has integrable singularities at ``z = -A`` and
    ``z = A``.

    Parameters
    ----------
    z:
        Positions relative to the oscillation center in m.
    A:
        Oscillation amplitude in m.

    Returns
    -------
    weights:
        Cup-weighting values evaluated at ``z``.
    """

    z = _as_array(z)

    innersqrt = np.pow(A, 2) - np.pow(z, 2)

    wcup_values = np.where(
        innersqrt > 0,
        1 / (np.pi * np.sqrt(innersqrt)),
        0.0,
    )

    return wcup_values


def _wcap_callable(
    z_axis: ArrayLike,
    feq,
    A0: float,
    fexc: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate cap-weighted averages for a callable quantity.

    For each oscillation-center position in ``z_axis``, the callable is
    integrated over one spatial oscillation interval from ``-A0`` to ``A0``.

    Callables accepting only the tip-sample position are fully supported.
    Velocity-dependent callables accepting two positional arguments are
    detected, but their averaging is not yet implemented.

    Parameters
    ----------
    z_axis:
        Oscillation-center positions in m.
    feq:
        Callable quantity to average. It should accept the tip-sample position
        as its first argument.
    A0:
        Oscillation amplitude in m.
    fexc:
        Excitation frequency in Hz. Currently only relevant for the planned
        velocity-dependent implementation.

    Returns
    -------
    zc:
        Oscillation-center positions. This has the same size as ``z_axis``.
    averaged:
        Cap-weighted values of ``feq``.

    Notes
    -----
    Velocity-dependent averaging is currently not implemented. IMPORTANT NOTE:
    For F_even (and k_even) fw and bw direction have to be identical!
    """

    z_axis = _as_array(z_axis)

    if z_axis.ndim == 0:
        z_axis = np.array([float(z_axis)])

    n = len(z_axis)

    # check number of arguments
    zts_only = True
    if _num_positional_args(feq) == 2:
        zts_only = False
        print("wcap_callable: function is velocity dependent.\
              NOT YET IMPLEMENTED!")

    fwc = z_axis * 0.0

    for i in range(0, n):
        quadres = 0.0

        # wcap integrates z in -A ... +A
        wcap_weight = lambda z: (
            2
            / (np.pi * np.pow(A0, 2))
            * np.sqrt(np.abs(np.pow(A0, 2) - np.pow(z, 2)))
        )

        if zts_only:
            ifunc = lambda z: feq(z + z_axis[i]) * wcap_weight(z)
            quadres = quad(ifunc, -A0, A0)
        else:
            ztsfunc = lambda z: -2 * np.pi * fexc * np.sqrt(
                1 - np.pow((z - z_axis[i]) / A0, 2)
            )
            _ = ztsfunc
            # TODO: velocity-dependent cap averaging
            pass

        fwc[i] = quadres[0]

    return z_axis, fwc


def _wcup_callable(
    z_axis: ArrayLike,
    feq,
    A0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate cup-weighted averages for a callable quantity.

    For each oscillation-center position in ``z_axis``, the callable is
    integrated over one spatial oscillation interval from ``-A0`` to ``A0``
    using the cup-weighting kernel.

    Parameters
    ----------
    z_axis:
        Oscillation-center positions in m.
    feq:
        Callable quantity to average as a function of tip-sample position.
    A0:
        Oscillation amplitude in m.

    Returns
    -------
    zc:
        Oscillation-center positions. This has the same size as ``z_axis``.
    averaged:
        Cup-weighted values of ``feq``.
    """

    z_axis = _as_array(z_axis)

    if z_axis.ndim == 0:
        z_axis = np.array([float(z_axis)])

    n = len(z_axis)

    fwc = z_axis * 0.0

    for i in range(0, n):
        # wcup integrates z in -A ... +A
        wcup_weight = lambda z: 1.0 / (
            np.pi * np.sqrt(np.abs(np.pow(A0, 2) - np.pow(z, 2)))
        )

        ifunc = lambda z: feq(z + z_axis[i]) * wcup_weight(z)

        quadres = quad(ifunc, -A0, A0)
        fwc[i] = quadres[0]

    return z_axis, fwc


def _wcap_equiz(
    z_axis: ArrayLike,
    feq: ArrayLike,
    A0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate cap-weighted averages for equidistant sampled data.

    The input quantity is assumed to be sampled along the tip-sample position
    axis. For each valid oscillation center, the values within approximately
    ``-A0`` to ``A0`` are integrated using the trapezoidal rule. Partially
    sampled boundary intervals are treated separately.

    Parameters
    ----------
    z_axis:
        Equidistant position axis in m.
    feq:
        Sampled quantity defined on ``z_axis``.
    A0:
        Oscillation amplitude in m.

    Returns
    -------
    zc:
        Oscillation-center positions for which a complete averaging interval
        is available.
    averaged:
        Cap-weighted average of ``feq``.

    Raises
    ------
    ValueError
        If ``z_axis`` is too short to cover a full oscillation interval of
        approximately ``2 * A0``.

    Notes
    -----
    (1) the resulting vector zc is shorter by the length ``2 * A0``.

    (2) int() truncates towards zero: int(3.9)=3, int(-3.9)=3.
    """

    z_axis = _as_array(z_axis)
    feq = _as_array(feq)

    dz = np.mean(np.diff(z_axis))

    # Note: int() truncates towards zero: int(3.9)=3, int(-3.9)=3
    NArange = int(round(A0 / dz)) + 1
    N2Arange = 2 * NArange
    _ = N2Arange

    # This is the difference between the exact amplitude and the sampled value
    Aerr = A0 - dz * (NArange - 1)

    if Aerr > dz:
        print(f"Something is wrong, {Aerr=} larger than {dz=}.")

    # calculate precise values via np.linspace for the Amplitude range at given
    # z sample values. These are the exact z positions for the integration.
    q_range = np.linspace(-A0 + Aerr, A0 - Aerr, 2 * NArange - 1)

    # length of new z axis (center position zc)
    zc_len = len(z_axis) - 2 * NArange

    if zc_len <= 0:
        raise ValueError(
            "z_axis is too short for the chosen amplitude A0. "
            "Need enough points to cover at least 2*A0."
        )

    # resulting data array, shorter by 2*A than z_axis
    wc = np.zeros(zc_len)
    zc = np.zeros(zc_len)

    # iterate over all tip-sample distances within the suitable range
    # loop runs from NArange ... len(z_axis)-NArange-1
    for i in range(NArange, len(z_axis) - NArange):
        # Indices run from (i-NArange) ... (i+NArange)
        t_feq = feq[(i - NArange) : (i + NArange + 1)]
        tz_axis = z_axis[(i - NArange) : (i + NArange + 1)]

        # Note: len(wcapsqrt) is len(t_feq)-2
        wcapsqrt = np.pow(A0, 2) - np.pow(q_range, 2)

        # avoid complex values by setting negative values to zero although
        # this should not happen.
        idx = np.where(wcapsqrt < 0)

        if len(idx[0]) > 0:
            print(f"_wcap_equiz:values <0 ({len(idx[0])=}):\
                  {idx[0]}, {wcapsqrt[idx[0]]}")

        wcapsqrt[wcapsqrt < 0] = 0

        # Approximate the first and last (part) segment by separate integration
        feq1 = (t_feq[0] * Aerr / dz + t_feq[1] * (dz - Aerr) / dz) / 2.0
        feq2 = (t_feq[-2] * (dz - Aerr) / dz + t_feq[-1] * Aerr / dz) / 2.0

        term1 = feq1 * _wcap_int(-A0, -A0 + Aerr, A0)
        term2 = feq2 * _wcap_int(A0 - Aerr, A0, A0)

        # calculate the integral itself.
        res = np.trapezoid(t_feq[1:-1] * np.sqrt(wcapsqrt), x=tz_axis[1:-1])

        wc[i - NArange] = 2.0 / (np.pi * (A0) ** 2) * (res + term1 + term2)
        zc[i - NArange] = z_axis[i]

    return zc, wc


def _wcup_equiz(
    z_axis: ArrayLike,
    feq: ArrayLike,
    A0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate cup-weighted averages for equidistant sampled data.

    The input quantity is sampled on an equidistant position axis. For each
    valid oscillation center, the transformed sample positions are evaluated
    by linear interpolation on ``z_axis``. The integral over ``theta`` is then
    approximated using the trapezoidal rule. Only center positions for which the
    complete interval ``[zc - A0, zc + A0]`` lies within the input axis are
    returned.

    Parameters
    ----------
    z_axis:
        Equidistant position axis in m.
    feq:
        Sampled quantity defined on ``z_axis``.
    A0:
        Oscillation amplitude in m.

    Returns
    -------
    zc:
        Oscillation-center positions for which a complete averaging interval
        is available.
    averaged:
        Cup-weighted average of ``feq``.

    Raises
    ------
    ValueError
        If ``z_axis`` is too short to cover a full oscillation interval of
        approximately ``2 * A0``.

    Notes
    -----
    (1) The singular cup-weighting kernel is removed by applying the
    substitution

    ``q = A0 * sin(theta)``.

    This transforms the weighted average

    ``1 / pi * integral(
        feq(zc + q) / sqrt(A0**2 - q**2),
        q=-A0..A0
    )``

    into the nonsingular integral

    ``1 / pi * integral(
        feq(zc + A0 * sin(theta)),
        theta=-pi/2..pi/2
    )``.
    """

    z_axis = _as_array(z_axis)
    feq = _as_array(feq)

    dz_values = np.diff(z_axis)
    dz = float(np.mean(dz_values))

    # Only retain actual sampled positions that can serve as centers.
    tolerance = 10.0 * np.finfo(float).eps * max(
        1.0,
        np.max(np.abs(z_axis)),
        abs(A0),
    )

    valid = (
        (z_axis - A0 >= z_axis[0] - tolerance)
        & (z_axis + A0 <= z_axis[-1] + tolerance)
    )

    zc = z_axis[valid]

    if zc.size == 0:
        raise ValueError(
            "z_axis is too short for the chosen amplitude A0. "
            "No complete interval [zc-A0, zc+A0] is available."
        )

    # Use enough theta points to resolve the original spatial sampling.
    # The factor 4 gives several integration points per original grid interval.
    n_theta = max(
        501,
        int(np.ceil(4.0 * np.pi * A0 / dz)) + 1,
    )

    # An odd number includes theta = 0 exactly.
    if n_theta % 2 == 0:
        n_theta += 1

    theta = np.linspace(
        -np.pi / 2.0,
        np.pi / 2.0,
        n_theta,
    )

    offsets = A0 * np.sin(theta)

    wc = np.empty(zc.size, dtype=float)

    for i, center in enumerate(zc):
        sample_positions = center + offsets

        # Protect against tiny floating-point excursions past the boundaries.
        sample_positions = np.clip(
            sample_positions,
            z_axis[0],
            z_axis[-1],
        )

        sample_values = np.interp(
            sample_positions,
            z_axis,
            feq,
        )

        wc[i] = np.trapezoid(
            sample_values,
            x=theta,
        ) / np.pi

    return zc, wc


def _wcap_nonequiz(
    z_axis: ArrayLike,
    feq: ArrayLike,
    A0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate cap-weighted averages for non-equidistant sampled data.

    The input axis and values are sorted in ascending position order. For each
    valid oscillation center, the sampled values within the amplitude interval
    are integrated using the trapezoidal rule. Partially sampled boundary
    segments are approximated separately.

    Parameters
    ----------
    z_axis:
        Non-equidistant position axis in m.
    feq:
        Sampled quantity defined on ``z_axis``.
    A0:
        Oscillation amplitude in m.

    Returns
    -------
    zc:
        Oscillation-center positions for which a complete averaging interval
        is available.
    averaged:
        Cap-weighted average of ``feq``.
    """

    z_axis = _as_array(z_axis).copy()
    feq = _as_array(feq).copy()

    # make sure z_axis is in ascending order
    orderidx = np.argsort(z_axis)
    z_axis = z_axis[orderidx]
    feq = feq[orderidx]

    # set smallest z (temporarily) to zero. This makes the code a bit shorter.
    # zzero will be added again to zc axis before returning.
    zzero = z_axis[0]
    z_axis = z_axis - zzero

    # Total number of data points
    Ndata = len(feq)

    # get the first index where z>A0
    icentre = np.searchsorted(z_axis, A0, side="right")

    # Output data arrays. Size is larger than required, precise size will be set
    # at the end after iterating through the array.
    zc = np.zeros_like(z_axis)
    wc = np.zeros_like(z_axis)

    # index boundaries for the output data arrays
    istart = icentre
    iend = len(z_axis)

    # iterate along z_axis until hitting the upper limit.
    while True:
        if icentre >= Ndata - 1:
            iend = icentre
            break

        # find lower bound of oscillation range for current centre position at
        # icentre
        ilower = np.searchsorted(
            z_axis[:icentre], z_axis[icentre] - A0, side="left")

        if ilower <= 0:
            icentre += 1
            continue

        # find upper bound of oscillation range for current centre position at
        # icentre
        iupper = np.searchsorted(
            z_axis[icentre:], z_axis[icentre] + A0, side="left")
        iupper = iupper + icentre

        if iupper >= Ndata:
            # we've hit the upper limit or are even beyond it. Stop here.
            iend = icentre
            break

        # range shifted to z_axis[icentre]; this corresponds to the positions
        # along the oscillation
        q_range = z_axis[ilower:iupper] - z_axis[icentre]

        # deviation from exact amplitude range. note that q_range[0] is a
        # negative number.
        Aerrs = [A0 + q_range[0], A0 - q_range[-1]]

        # difference along z for the points in- and outside of the +-A0 ranges
        dzs = [
            z_axis[ilower] - z_axis[ilower - 1],
            z_axis[iupper] - z_axis[iupper - 1],
        ]

        # cap averaging function for q_range
        wcapsqrt = np.pow(A0, 2) - np.pow(q_range, 2)
        wcapsqrt[wcapsqrt < 0] = 0

        # perform integration
        # Approximate the first and last (part) segment by separate integration
        feq1 = (
            feq[ilower - 1] * Aerrs[0] / dzs[0]
            + feq[ilower] * (dzs[0] - Aerrs[0]) / dzs[0]
        ) / 2.0

        feq2 = (
            feq[iupper - 1] * (dzs[1] - Aerrs[1]) / dzs[1]
            + feq[iupper] * Aerrs[1] / dzs[1]
        ) / 2.0

        term1 = feq1 * _wcap_int(-A0, -A0 + Aerrs[0], A0)
        term2 = feq2 * _wcap_int(A0 - Aerrs[1], A0, A0)

        # calculate the integral itself.
        # note that the range ilower ... iupper-1 is integrated.
        res = np.trapezoid(
            feq[ilower:iupper] * np.sqrt(wcapsqrt),
            x=z_axis[ilower:iupper],
        )

        wc[icentre] = 2.0 / (np.pi * (A0) ** 2) * (res + term1 + term2)
        zc[icentre] = z_axis[icentre]

        # advance to next index
        icentre += 1

    return zc[istart:iend] + zzero, wc[istart:iend]


def _wcup_nonequiz(
    z_axis: ArrayLike,
    feq: ArrayLike,
    A0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate cup-weighted averages for non-equidistant sampled data.

    For each valid interval spanning approximately ``2 * A0``, the cup integral
    is split into two boundary contributions and one interior contribution.

    Parameters
    ----------
    z_axis:
        Non-equidistant position axis in m.
    feq:
        Sampled quantity defined on ``z_axis``.
    A0:
        Oscillation amplitude in m.

    Returns
    -------
    zc:
        Oscillation-center positions for which the averaging interval is
        available.
    averaged:
        Cup-weighted average of ``feq``.

    Notes
    -----
    This implementation is currently marked as untested.
    """

    print("wcup_nonequiz: TODO: This function is untested so far.")

    z_axis = _as_array(z_axis)
    feq = _as_array(feq)

    n = len(feq)
    wc = np.zeros((n, 1))

    i = 0

    while True:
        # get the amplitude range for this tip-sample distance
        ri = np.arange(i, n)

        A_max_id_x = np.where(z_axis[ri] - z_axis[i] <= 2 * A0)[0][-1]
        A_max_id_x = i + A_max_id_x

        # if hitting the end, abort
        if A_max_id_x >= n - 1:
            break

        # z positions for the amplitude are:
        A_range = z_axis[i:A_max_id_x] - z_axis[i] - A0

        # integral (1)
        if i >= len(z_axis) - 1:
            tdz = 0
        else:
            tdz = z_axis[i + 1] - z_axis[i]

        I1 = feq[i] / np.pi * np.arccos(1 - tdz / A0)

        # integral (3)
        tdz = z_axis[i + len(A_range) - 1] - z_axis[i + len(A_range) - 2]
        I3 = feq[i + len(A_range) - 1] / np.pi * np.arccos(1 - tdz / A0)

        # integral (2)
        tf = feq[range(i + 1, i + len(A_range) - 1)]
        tz = z_axis[range(i + 1, i + len(A_range) - 1)]

        I2 = np.trapezoid(
            tf / (np.pi * np.sqrt(np.abs(A0**2 - A_range[1:-1] ** 2))),
            x=tz,
        )

        # sum all parts for result
        wc[i] = I1 + I2 + I3

        i += 1

    wc = wc[:i]
    zc = z_axis[:i] + A0

    return zc.flatten(), wc.flatten()