# src/qafm/quant/utils.py

import inspect

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_array(x: ArrayLike) -> NDArray[np.float64]:
    """Convert input to a NumPy float array.

    Parameters
    ----------
    x : ArrayLike
        Input data to convert.

    Returns
    -------
    NDArray[np.float64]
        Input converted to a NumPy array with float dtype.

    Examples
    --------
    >>> _as_array([1, 2, 3])
    array([1., 2., 3.])

    >>> _as_array(np.array([1.0, 2.5]))
    array([1. , 2.5])
    """

    return np.asarray(x, dtype=float)


def _is_equidistant(
    z: ArrayLike,
    atol: float = 1e-12,
    rtol: float = 0.0,
) -> bool:
    """Return whether values in z are approximately equidistant.

    Parameters
    ----------
    z : ArrayLike
        Input 1-D sequence of numerical values.
    atol : float, optional
        Absolute tolerance for comparing successive differences (default
        1e-12).
    rtol : float, optional
        Relative tolerance for comparing successive differences (default
        0.0).

    Returns
    -------
    bool
        True if the spacing between successive elements of ``z`` is
        approximately constant within the given tolerances.
    """

    z = _as_array(z)

    if z.size < 3:
        # arrays with 0, 1, or 2 points are by definition equidistant
        return True

    d = np.diff(z)

    return bool(np.allclose(d, d[0], atol=atol, rtol=rtol))


def _num_positional_args(func) -> int | None:
    """Return the number of declared positional arguments a callable accepts.

    Parameters
    ----------
    func : callable
        The callable to inspect.

    Returns
    -------
    int or None
        The number of declared positional parameters (POSITIONAL_ONLY and
        POSITIONAL_OR_KEYWORD). Returns ``None`` if the signature cannot be
        obtained (e.g., for some builtins).

    Notes
    -----
    Variadic parameters (*args, **kwargs) are ignored.
    """

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        # builtins sometimes fail; assume unknown
        return None

    n = 0

    for parameter in sig.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            n += 1

    return n


def _find_approx_intersect(
    z1: ArrayLike,
    z2: ArrayLike,
    tol: float = 1e-12,
) -> tuple[NDArray[np.float64], list[int], list[int]]:
    """Find approximate intersection of two numeric arrays.

    Parameters
    ----------
    z1, z2 : ArrayLike
        Input 1-D numeric arrays to compare.
    tol : float, optional
        Absolute tolerance used to decide whether two values are equal
        (default 1e-12).

    Returns
    -------
    tuple
        A tuple ``(common, idx1, idx2)`` where ``common`` is a NumPy array
        of values from ``z1`` that have a close match in ``z2`` within
        ``tol``, and ``idx1`` and ``idx2`` are lists of corresponding
        indices in ``z1`` and ``z2`` respectively.

    Notes
    -----
    This behaves like a floating-point aware equivalent of
    ``np.intersect1d(..., return_indices=True)`` using nearest-neighbor
    matching from ``z1`` into ``z2``.
    """

    z1 = _as_array(z1)
    z2 = _as_array(z2)

    common = []
    idx1 = []
    idx2 = []

    for i, value in enumerate(z1):
        j = np.abs(z2 - value).argmin()

        if abs(z2[j] - value) < tol:
            common.append(value)
            idx1.append(i)
            idx2.append(j)

    return np.array(common), idx1, idx2