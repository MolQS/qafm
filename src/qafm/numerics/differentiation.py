# src/qafm/numerics/differentiation.py

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import _as_array


# TODO: What is "stencil size"?
def numdiff(
    xin: ArrayLike,
    yin: ArrayLike,
    order: int = 1,
    method: str = "diff",
    degree: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Numerically differentiate y(x).

    The derivative of ``yin`` with respect to ``xin`` is calculated using one of
    several numerical differentiation methods. Depending on the selected method,
    the returned axis may be shorter than the original input axis.

    Parameters
    ----------
    xin:
        x-axis.
    yin:
        y-values.
    order:
        Derivative order. Currently mainly order 1 and 2 are useful.
    method:
        Differentiation method. Supported: "diff", "gradient", "fdm", "lanczos".
    degree:
        Degree/stencil size for "fdm" and "lanczos".

    Returns
    -------
    x:
        x-axis of derivative.
    y:
        differentiated values.
    
    Raises
    ------
    ValueError
        If ``xin`` and ``yin`` have different shapes, are not one-dimensional,
        contain fewer than two points, or if ``order`` is less than one.
    ValueError
        If an unsupported method or degree is selected. ValueError If
        ``fdm`` or ``lanczos`` is used with non-equidistant input data.
    """

    x = _as_array(xin)
    y = _as_array(yin)

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have same shape. Got {x.shape=} and {y.shape=}.")

    if x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")

    if len(x) < 2:
        raise ValueError("Need at least two data points.")

    if order < 1:
        raise ValueError("order must be >= 1.")

    method = method.lower().strip()

    if method == "diff":
        return _numdiff_diff(x, y, order)

    if method == "gradient":
        return _numdiff_gradient(x, y, order)

    if method == "fdm":
        if order != 1:
            raise ValueError("method='fdm' currently supports only order=1.")
        return _numdiff_fdm_first_order(x, y, degree)

    if method == "lanczos":
        if order != 1:
            raise ValueError(
                "method='lanczos' currently supports only order=1.")
        return _numdiff_lanczos_first_order(x, y, degree)

    raise ValueError(
        f"Unknown differentiation method {method!r}. "
        "Supported methods are 'diff', 'gradient', 'fdm', and 'lanczos'."
    )


def _numdiff_diff(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Differentiate data using repeated first-order finite differences.
    
    Each differentiation step reduces the length of the output arrays by one.
    The returned axis uses the left points of the corresponding
    finite-difference intervals.
    
    Parameters
    ----------
    x:
        One-dimensional axis values.
    y:
        Values sampled at ``x``.
    order:
        Number of times the finite-difference operation is applied.

    Returns
    -------
    x_out:
        Axis values corresponding to the derivative.
    y_out:
        Numerical derivative values.
    """

    x_out = x.copy()
    y_out = y.copy()

    for _ in range(order):
        y_out = np.diff(y_out) / np.diff(x_out)
        x_out = x_out[:-1]

    return x_out, y_out


def _numdiff_gradient(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Differentiate data using repeated applications of ``numpy.gradient``.
    
    Unlike simple finite differences, this method preserves the original number
    of data points.
    
    Parameters
    ----------
    x:
        One-dimensional axis values.
    y:
        Values sampled at ``x``.
    order:
        Number of times ``numpy.gradient`` is applied.

    Returns
    -------
    x_out:
        Copy of the original axis.
    y_out:
        Numerical derivative values with the same shape as ``y``.
    """

    y_out = y.copy()

    for _ in range(order):
        y_out = np.gradient(y_out, x)

    return x.copy(), y_out


# TODO: Combine with utils.is_equidistant ? 
def _check_equidistant(x: NDArray[np.float64], rtol: float = 1e-6) -> float:
    """Validate that an axis is approximately equidistant.
    
    The mean spacing is compared with all individual spacings using
    ``numpy.allclose``.
    
    Parameters
    ----------
    x:
        One-dimensional axis values.
    rtol:
        Relative tolerance used when comparing the individual spacings with
        their mean value.

    Returns
    -------
    dz:
        Mean spacing between adjacent axis values.

    Raises
    ------
    ValueError
        If the axis is not approximately equidistant.
    """

    dx = np.diff(x)
    dz = float(np.mean(dx))

    if not np.allclose(dx, dz, rtol=rtol, atol=0.0):
        raise ValueError(
            "This method requires an approximately equidistant x-axis. "
            "Use method='diff' or method='gradient' for non-equidistant data."
        )

    return dz


# stencil degree ?
def _numdiff_fdm_first_order(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degree: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate a first derivative using centered finite differences.
    
    The method requires an approximately equidistant axis. Higher stencil
    degrees generally provide a higher-order approximation but remove more
    points near the boundaries.
    
    Parameters
    ----------
    x:
        One-dimensional, approximately equidistant axis values.
    y:
        Values sampled at ``x``.
    degree:
        Finite-difference stencil degree. Supported values are ``2``, ``4``
        ``6``, and ``8``.

    Returns
    -------
    x_out:
        Interior axis values for which the centered finite-difference stencil is
        defined.
    y_out:
        First derivative calculated using the selected stencil degree.

    Raises
    ------
    ValueError
        If ``x`` is not approximately equidistant.
    ValueError
        If ``degree`` is unsupported or too few data points are available for
        the selected stencil.
    """

    dz = _check_equidistant(x)
    n = len(y)

    if degree == 2:
        if n < 3:
            raise ValueError("degree=2 requires at least 3 points.")
        y_out = (-0.5 * y[0 : n - 2] + 0.5 * y[2:n]) / dz
        x_out = x[1 : n - 1]

    elif degree == 4:
        if n < 5:
            raise ValueError("degree=4 requires at least 5 points.")
        y_out = (
            1 / 12 * y[0 : n - 4]
            - 2 / 3 * y[1 : n - 3]
            + 2 / 3 * y[3 : n - 1]
            - 1 / 12 * y[4:n]
        ) / dz
        x_out = x[2 : n - 2]

    elif degree == 6:
        if n < 7:
            raise ValueError("degree=6 requires at least 7 points.")
        y_out = (
            -1 / 60 * y[0 : n - 6]
            + 3 / 20 * y[1 : n - 5]
            - 3 / 4 * y[2 : n - 4]
            + 3 / 4 * y[4 : n - 2]
            - 3 / 20 * y[5 : n - 1]
            + 1 / 60 * y[6:n]
        ) / dz
        x_out = x[3 : n - 3]

    elif degree == 8:
        if n < 9:
            raise ValueError("degree=8 requires at least 9 points.")
        y_out = (
            1 / 280 * y[0 : n - 8]
            - 4 / 105 * y[1 : n - 7]
            + 1 / 5 * y[2 : n - 6]
            - 4 / 5 * y[3 : n - 5]
            + 4 / 5 * y[5 : n - 3]
            - 1 / 5 * y[6 : n - 2]
            + 4 / 105 * y[7 : n - 1]
            - 1 / 280 * y[8:n]
        ) / dz
        x_out = x[4 : n - 4]

    else:
        raise ValueError("fdm degree must be one of 2, 4, 6, or 8.")

    return x_out, y_out


def _numdiff_lanczos_first_order(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degree: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate a first derivative using a Lanczos-type differentiator.
    
    The method uses symmetric weighted differences and requires an approximately
    equidistant axis. Larger stencil degrees use more neighboring points and
    reduce the length of the returned arrays.
      
    Parameters
    ----------
    x:
        One-dimensional, approximately equidistant axis values.
    y:
        Values sampled at ``x``.
    degree:
        Lanczos degree. Supported values are ``0``, ``5``, ``7``,
        ``9``, and ``11``. Degree ``0`` uses a simple first difference.

    Returns
    -------
    x_out:
        Axis values corresponding to the calculated derivative.
    y_out:
        First derivative calculated with the selected Lanczos.

    Raises
    ------
    ValueError
        If ``x`` is not approximately equidistant.
    ValueError
        If ``degree`` is unsupported or too few data points are available for
        the selected stencil.
    """

    dz = _check_equidistant(x)
    n = len(y)

    if degree == 0:
        y_out = np.diff(y) / np.diff(x)
        x_out = x[1:n]

    elif degree == 5:
        if n < 5:
            raise ValueError("degree=5 requires at least 5 points.")
        y_out = (
            1 * (y[3 : n - 1] - y[1 : n - 3])
            + 2 * (y[4:n] - y[0 : n - 4])
        ) / (10 * dz)
        x_out = x[2 : n - 2]

    elif degree == 7:
        if n < 7:
            raise ValueError("degree=7 requires at least 7 points.")
        y_out = (
            1 * (y[4 : n - 2] - y[2 : n - 4])
            + 2 * (y[5 : n - 1] - y[1 : n - 5])
            + 3 * (y[6:n] - y[0 : n - 6])
        ) / (28 * dz)
        x_out = x[3 : n - 3]

    elif degree == 9:
        if n < 9:
            raise ValueError("degree=9 requires at least 9 points.")
        y_out = (
            1 * (y[5 : n - 3] - y[3 : n - 5])
            + 2 * (y[6 : n - 2] - y[2 : n - 6])
            + 3 * (y[7 : n - 1] - y[1 : n - 7])
            + 4 * (y[8:n] - y[0 : n - 8])
        ) / (60 * dz)
        x_out = x[4 : n - 4]

    elif degree == 11:
        if n < 11:
            raise ValueError("degree=11 requires at least 11 points.")
        y_out = (
            1 * (y[6 : n - 4] - y[4 : n - 6])
            + 2 * (y[7 : n - 3] - y[3 : n - 7])
            + 3 * (y[8 : n - 2] - y[2 : n - 8])
            + 4 * (y[9 : n - 1] - y[1 : n - 9])
            + 5 * (y[10:n] - y[0 : n - 10])
        ) / (110 * dz)
        x_out = x[5 : n - 5]

    else:
        raise ValueError("lanczos degree must be one of 0, 5, 7, 9, or 11.")

    return x_out, y_out