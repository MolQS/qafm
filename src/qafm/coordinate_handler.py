from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


ZCoordinate = Literal["zp", "zc", "zts"]


def convert_z_axis(
    z: ArrayLike,
    input_coordinate: ZCoordinate,
    output_coordinate: ZCoordinate,
    *,
    A0: float,
    qs: ArrayLike | None = None,
    z0: float = 0.0,
) -> NDArray[np.float64]:
    """Convert between different AFM z-coordinate axes.

    The following coordinate definitions are used:

        zc  = zp + z0 + qs
        zts = zc - A0

    If ``qs`` is not supplied, zero static displacement is assumed.

    Parameters
    ----------
    z:
        One-dimensional input z-axis in m.
    input_coordinate:
        Coordinate represented by ``z``:

        - ``"zp"``: piezo-position axis;
        - ``"zc"``: oscillation-center axis;
        - ``"zts"``: lower turning-point tip-sample axis.

    output_coordinate:
        Coordinate that should be returned. The supported values are
        ``"zp"``, ``"zc"``, and ``"zts"``.
    A0:
        Oscillation amplitude in m. Must be finite and greater than zero.
    qs:
        Optional static-displacement axis in m. It must be one-dimensional
        and have the same shape as ``z``. If ``None``, ``qs = 0`` is used.
    z0:
        Optional constant position offset in m. The default is zero.

    Returns
    -------
    z_output:
        Converted one-dimensional z-axis in m.

    Raises
    ------
    ValueError:
        If an unknown coordinate is requested, input values are invalid,
        or ``qs`` does not have the same shape as ``z``.
    """

    valid_coordinates = {"zp", "zc", "zts"}

    if input_coordinate not in valid_coordinates:
        raise ValueError(
            f"Unknown input_coordinate {input_coordinate!r}. "
            f"Expected one of {sorted(valid_coordinates)}."
        )

    if output_coordinate not in valid_coordinates:
        raise ValueError(
            f"Unknown output_coordinate {output_coordinate!r}. "
            f"Expected one of {sorted(valid_coordinates)}."
        )

    # Validate input z-axis.
    try:
        z_axis = np.asarray(z, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "z must contain numeric values."
        ) from error

    if z_axis.ndim != 1:
        raise ValueError(
            "z must be a one-dimensional axis."
        )

    if z_axis.size == 0:
        raise ValueError(
            "z must not be empty."
        )

    if not np.all(np.isfinite(z_axis)):
        raise ValueError(
            "z contains non-finite values."
        )

    # Validate amplitude.
    try:
        A0 = float(A0)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "A0 must be a numeric scalar."
        ) from error

    if not np.isfinite(A0) or A0 <= 0.0:
        raise ValueError(
            "A0 must be finite and greater than zero."
        )

    # Validate position offset.
    try:
        z0 = float(z0)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "z0 must be a numeric scalar."
        ) from error

    if not np.isfinite(z0):
        raise ValueError(
            "z0 must be finite."
        )

    # Validate or generate static-displacement axis.
    if qs is None:
        qs_axis = np.zeros_like(z_axis)
    else:
        try:
            qs_axis = np.asarray(qs, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "qs must contain numeric values."
            ) from error

        if qs_axis.ndim != 1:
            raise ValueError(
                "qs must be a one-dimensional axis."
            )

        if qs_axis.shape != z_axis.shape:
            raise ValueError(
                "qs must have the same shape as z: "
                f"z.shape={z_axis.shape}, "
                f"qs.shape={qs_axis.shape}."
            )

        if not np.all(np.isfinite(qs_axis)):
            raise ValueError(
                "qs contains non-finite values."
            )

    # Convert the input coordinate to zc.
    if input_coordinate == "zc":
        zc = z_axis.copy()

    elif input_coordinate == "zp":
        zc = z_axis + z0 + qs_axis

    else:  # input_coordinate == "zts"
        zc = z_axis + A0

    # Convert zc to the requested output coordinate.
    if output_coordinate == "zc":
        return zc

    if output_coordinate == "zp":
        return zc - z0 - qs_axis

    # output_coordinate == "zts"
    return zc - A0