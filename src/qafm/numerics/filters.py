# src/qafm/numerics/filters.py

from collections.abc import Sequence
from math import factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray

import scipy.signal as signal

FloatArray = NDArray[np.float64]


def _as_1d_float_array(data: ArrayLike, *, name: str = "data") -> FloatArray:
    """Return *data* as a one-dimensional float array."""

    array = np.asarray(data, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    return array


def _validate_window(window_size: int, data_length: int, *, name: str) -> int:
    """Validate an odd positive window length for filters that require one."""

    window_size = int(window_size)

    if window_size < 1:
        raise ValueError(f"{name} must be at least 1.")

    if window_size % 2 == 0:
        window_size += 1

    if window_size > data_length:
        raise ValueError(
            f"{name} must not be larger than the data length "
            f"({window_size} > {data_length})."
        )

    return window_size


def _validate_savgol_parameters(
    window_size: int,
    poly_order: int,
    data_length: int,
) -> tuple[int, int]:
    """Validate Savitzky-Golay parameters."""

    window_size = _validate_window(
        window_size,
        data_length,
        name="window_size",
    )
    poly_order = int(poly_order)

    if poly_order < 0:
        raise ValueError("poly_order must be non-negative.")

    if poly_order >= window_size:
        raise ValueError("poly_order must be smaller than window_size.")

    return window_size, poly_order


def _transition_weights(
    n_points: int,
    transition_points: Sequence[float],
    transition_steepness: float,
) -> list[FloatArray]:
    """Create smooth logistic weights for blending several filtered curves."""

    if n_points < 1:
        raise ValueError("n_points must be at least 1.")

    if transition_steepness <= 0:
        raise ValueError("transition_steepness must be positive.")

    if any(point <= 0 or point >= 1 for point in transition_points):
        raise ValueError("All transition points must lie between 0 and 1.")

    if list(transition_points) != sorted(transition_points):
        raise ValueError("transition_points must be sorted in ascending order.")

    t = np.linspace(0.0, 1.0, n_points)
    cumulative = [
        1.0 / (1.0 + np.exp((t - point) * transition_steepness))
        for point in transition_points
    ]

    weights: list[FloatArray] = [np.clip(cumulative[0], 0.0, 1.0)]

    for previous, current in zip(cumulative, cumulative[1:]):
        weights.append(np.clip(previous - current, 0.0, 1.0))

    weights.append(np.clip(1.0 - cumulative[-1], 0.0, 1.0))

    total = np.sum(weights, axis=0)
    total[total == 0.0] = 1.0

    return [weight / total for weight in weights]


def savitzky_golay_smooth(
    data: ArrayLike,
    window_size: int,
    poly_order: int,
) -> FloatArray:
    """Smooth one curve with a Savitzky-Golay filter.

    Parameters
    ----------
    data:
        One-dimensional measurement data.
    window_size:
        Filter window length. Even values are increased by one because SciPy
        requires an odd window length.
    poly_order:
        Polynomial order used by the Savitzky-Golay filter.

    Returns
    -------
    numpy.ndarray
        Smoothed data with the same length as the input.
    """

    array = _as_1d_float_array(data)
    window_size, poly_order = _validate_savgol_parameters(
        window_size,
        poly_order,
        array.size,
    )

    return np.asarray(
        signal.savgol_filter(
            array,
            window_length=window_size,
            polyorder=poly_order,
        ),
        dtype=np.float64,
    )


def median_smooth(data: ArrayLike, kernel_size: int) -> FloatArray:
    """Smooth one curve with a median filter.

    Even kernel sizes are increased by one because median filters require an odd
    kernel length.
    """

    array = _as_1d_float_array(data)
    kernel_size = _validate_window(
        kernel_size,
        array.size,
        name="kernel_size",
    )

    return np.asarray(signal.medfilt(array, kernel_size=kernel_size), dtype=np.float64)


def moving_average_smooth(data: ArrayLike, window_size: int) -> FloatArray:
    """Smooth one curve with a centered moving average.

    The output has the same length as the input. Edge values are padded with the
    nearest input value before convolution.
    """

    array = _as_1d_float_array(data)
    window_size = int(window_size)

    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    if window_size == 1:
        return array.copy()

    left_pad = window_size // 2
    right_pad = window_size - 1 - left_pad
    padded = np.pad(array, (left_pad, right_pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / window_size

    return np.convolve(padded, kernel, mode="valid")


def kalman_filter_exponential(
    data: ArrayLike,
    decay_rate: float = 0.01,
    process_variance: float = 1.0e-5,
    measurement_variance: float = 1.0e-2,
    estimate_error: float = 1.0e-2,
    initial_index: int = 0,
) -> FloatArray:
    """Filter exponentially decaying data with a scalar Kalman filter.

    Parameters
    ----------
    data:
        One-dimensional measurement data.
    decay_rate:
        Exponential decay applied during the prediction step.
    process_variance:
        Estimated process variance.
    measurement_variance:
        Estimated measurement variance.
    estimate_error:
        Initial estimate uncertainty.
    initial_index:
        Index used for the initial state estimate.

    Returns
    -------
    numpy.ndarray
        Filtered data with the same length as the input.
    """

    array = _as_1d_float_array(data)

    if decay_rate < 0:
        raise ValueError("decay_rate must be non-negative.")

    if process_variance < 0:
        raise ValueError("process_variance must be non-negative.")

    if measurement_variance <= 0:
        raise ValueError("measurement_variance must be positive.")

    if estimate_error <= 0:
        raise ValueError("estimate_error must be positive.")

    if initial_index < 0 or initial_index >= array.size:
        raise ValueError("initial_index is outside the data range.")

    current_estimate = float(array[initial_index])
    current_error = float(estimate_error)
    filtered = np.zeros_like(array, dtype=np.float64)
    decay_factor = float(np.exp(-decay_rate))

    for index, value in enumerate(array):
        predicted_estimate = current_estimate * decay_factor
        predicted_error = current_error + process_variance

        kalman_gain = predicted_error / (predicted_error + measurement_variance)
        current_estimate = predicted_estimate + kalman_gain * (
            value - predicted_estimate
        )
        current_error = (1.0 - kalman_gain) * predicted_error

        filtered[index] = current_estimate

    return filtered


def blended_three_filter(
    data: ArrayLike,
    *,
    savgol_window_size: int = 51,
    savgol_poly_order: int = 3,
    moving_average_window_size: int = 51,
    decay_rate: float = 1.0e-4,
    transition_points: Sequence[float] = (0.1, 0.5),
    transition_steepness: float = 50.0,
) -> tuple[FloatArray, tuple[FloatArray, FloatArray, FloatArray]]:
    """Blend Savitzky-Golay, moving-average, and Kalman-filtered curves.

    The first filter dominates the beginning of the curve, the moving average
    dominates the middle region, and the Kalman filter dominates the end. The
    transition between filters is controlled by logistic weights.

    Returns
    -------
    filtered:
        Weighted combination of the three filtered curves.
    weights:
        Tuple containing the Savitzky-Golay, moving-average, and Kalman weights.
    """

    if len(transition_points) != 2:
        raise ValueError("transition_points must contain exactly two values.")

    savgol_curve = savitzky_golay_smooth(
        data,
        window_size=savgol_window_size,
        poly_order=savgol_poly_order,
    )
    moving_average_curve = moving_average_smooth(
        data,
        window_size=moving_average_window_size,
    )
    kalman_curve = kalman_filter_exponential(data, decay_rate=decay_rate)
    weights = _transition_weights(
        savgol_curve.size,
        transition_points,
        transition_steepness,
    )

    filtered = (
        weights[0] * savgol_curve
        + weights[1] * moving_average_curve
        + weights[2] * kalman_curve
    )

    return filtered, (weights[0], weights[1], weights[2])


def blended_savgol_filter(
    data: ArrayLike,
    *,
    window_sizes: Sequence[int] = (11, 91, 301),
    poly_orders: Sequence[int] = (3, 1, 0),
    transition_points: Sequence[float] = (0.05, 0.4),
    transition_steepness: float = 10.0,
) -> tuple[FloatArray, tuple[FloatArray, ...]]:
    """Blend several Savitzky-Golay filters along one curve.

    This is useful when different parts of a measurement curve need different
    smoothing strengths. For example, a small window can preserve sharp features
    near the beginning of a curve, while a larger window can strongly smooth a
    noisy far-field region.

    Parameters
    ----------
    data:
        One-dimensional measurement data.
    window_sizes:
        Savitzky-Golay window sizes for each region.
    poly_orders:
        Polynomial orders matching ``window_sizes``.
    transition_points:
        Normalized positions between 0 and 1. The number of transition points
        must be one less than the number of filters.
    transition_steepness:
        Larger values create sharper transitions between filters.

    Returns
    -------
    filtered:
        Weighted combination of all Savitzky-Golay filtered curves.
    weights:
        Tuple of weight arrays, one per Savitzky-Golay curve.
    """

    if len(window_sizes) != len(poly_orders):
        raise ValueError("window_sizes and poly_orders must have the same length.")

    if len(transition_points) != len(window_sizes) - 1:
        raise ValueError(
            "transition_points must contain one fewer value than window_sizes."
        )

    curves = [
        savitzky_golay_smooth(data, window_size=window_size, poly_order=poly_order)
        for window_size, poly_order in zip(window_sizes, poly_orders)
    ]
    weights = _transition_weights(
        curves[0].size,
        transition_points,
        transition_steepness,
    )

    filtered = np.zeros_like(curves[0], dtype=np.float64)

    for weight, curve in zip(weights, curves):
        filtered += weight * curve

    return filtered, tuple(weights)


def filter_curves_three_filter(
    data_array: ArrayLike,
    *,
    savgol_window_size: int = 51,
    savgol_poly_order: int = 3,
    moving_average_window_size: int = 51,
    decay_rate: float = 1.0e-4,
    transition_points: Sequence[float] = (0.1, 0.5),
    transition_steepness: float = 50.0,
) -> tuple[FloatArray, tuple[FloatArray, FloatArray, FloatArray]]:
    """Apply :func:`blended_three_filter` to multiple curves.

    Parameters
    ----------
    data_array:
        Two-dimensional array-like object with shape ``(n_curves, n_points)``.

    Returns
    -------
    filtered_curves:
        Filtered curves with shape ``(n_curves, n_points)``.
    weights:
        Three arrays containing the weights for each curve.
    """

    curves = np.asarray(data_array, dtype=np.float64)

    if curves.ndim != 2:
        raise ValueError("data_array must be two-dimensional.")

    filtered_curves = []
    weights_0 = []
    weights_1 = []
    weights_2 = []

    for curve in curves:
        filtered, weights = blended_three_filter(
            curve,
            savgol_window_size=savgol_window_size,
            savgol_poly_order=savgol_poly_order,
            moving_average_window_size=moving_average_window_size,
            decay_rate=decay_rate,
            transition_points=transition_points,
            transition_steepness=transition_steepness,
        )
        filtered_curves.append(filtered)
        weights_0.append(weights[0])
        weights_1.append(weights[1])
        weights_2.append(weights[2])

    return (
        np.asarray(filtered_curves, dtype=np.float64),
        (
            np.asarray(weights_0, dtype=np.float64),
            np.asarray(weights_1, dtype=np.float64),
            np.asarray(weights_2, dtype=np.float64),
        ),
    )


def filter_curves_savgol(
    data_array: ArrayLike,
    *,
    window_sizes: Sequence[int] = (7, 71, 31),
    poly_orders: Sequence[int] = (1, 1, 1),
    transition_points: Sequence[float] = (0.07, 0.35),
    transition_steepness: float = 10.0,
) -> tuple[FloatArray, FloatArray]:
    """Apply :func:`blended_savgol_filter` to multiple curves."""

    curves = np.asarray(data_array, dtype=np.float64)

    if curves.ndim != 2:
        raise ValueError("data_array must be two-dimensional.")

    filtered_curves = []
    all_weights = []

    for curve in curves:
        filtered, weights = blended_savgol_filter(
            curve,
            window_sizes=window_sizes,
            poly_orders=poly_orders,
            transition_points=transition_points,
            transition_steepness=transition_steepness,
        )
        filtered_curves.append(filtered)
        all_weights.append(weights)

    return (
        np.asarray(filtered_curves, dtype=np.float64),
        np.asarray(all_weights, dtype=np.float64),
    )


def smooth_curve(data: ArrayLike, window_size: int) -> FloatArray:
    """Backward-compatible alias for :func:`moving_average_smooth`."""

    return moving_average_smooth(data, window_size)


def custom_filter_3(
    data: ArrayLike,
    savgol_window_size: int = 51,
    savgol_poly_order: int = 3,
    median_kernel_size: int = 51,
    decay_rate: float = 1.0e-4,
    transition_point1: float = 0.1,
    transition_point2: float = 0.5,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Backward-compatible wrapper for the original three-filter blend.

    The parameter name ``median_kernel_size`` is kept for compatibility with old
    scripts. Internally it controls the moving-average window, matching the
    behavior of the provided reference implementation.
    """

    filtered, weights = blended_three_filter(
        data,
        savgol_window_size=savgol_window_size,
        savgol_poly_order=savgol_poly_order,
        moving_average_window_size=median_kernel_size,
        decay_rate=decay_rate,
        transition_points=(transition_point1, transition_point2),
    )

    return filtered, weights[0], weights[1], weights[2]


def custom_savgol_filter(
    data: ArrayLike,
    savgol_window_sizes: Sequence[int] = (11, 91, 301),
    savgol_poly_orders: Sequence[int] = (3, 1, 0),
    transition_points: Sequence[float] = (0.05, 0.4),
    filter_decay: float = 10.0,
) -> tuple[FloatArray, tuple[FloatArray, ...]]:
    """Backward-compatible wrapper for :func:`blended_savgol_filter`."""

    return blended_savgol_filter(
        data,
        window_sizes=savgol_window_sizes,
        poly_orders=savgol_poly_orders,
        transition_points=transition_points,
        transition_steepness=filter_decay,
    )


def apply_custom_filter_3(
    data_array: ArrayLike,
    savgol_window_size: int = 51,
    savgol_poly_order: int = 3,
    median_kernel_size: int = 51,
    decay_rate: float = 1.0e-4,
    transition_point1: float = 0.1,
    transition_point2: float = 0.5,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Backward-compatible wrapper for filtering multiple curves."""

    filtered, weights = filter_curves_three_filter(
        data_array,
        savgol_window_size=savgol_window_size,
        savgol_poly_order=savgol_poly_order,
        moving_average_window_size=median_kernel_size,
        decay_rate=decay_rate,
        transition_points=(transition_point1, transition_point2),
    )

    return filtered, weights[0], weights[1], weights[2]


def apply_custom_savgol_filter(
    data_array: ArrayLike,
    savgol_window_sizes: Sequence[int] = (7, 71, 31),
    savgol_poly_orders: Sequence[int] = (1, 1, 1),
    transition_points: Sequence[float] = (0.07, 0.35),
    filter_decay: float = 10.0,
) -> tuple[FloatArray, FloatArray]:
    """Backward-compatible wrapper for filtering multiple curves."""

    return filter_curves_savgol(
        data_array,
        window_sizes=savgol_window_sizes,
        poly_orders=savgol_poly_orders,
        transition_points=transition_points,
        transition_steepness=filter_decay,
    )




# TODO: --- taken from -- AG-Rahe(private)/code/analysisafm_toolbox/savgolfilter.py --

class SavGolFilter:
    """Savitzky-Golay filter for equidistant and non-equidistant x data.

    For equidistant x values, this class delegates to
    ``scipy.signal.savgol_filter``. For non-equidistant x values, it computes
    local least-squares coefficients for each window and applies them directly.

    The implementation follows the Savitzky-Golay idea of fitting a local
    polynomial and evaluating either the smoothed value or a derivative at the
    window center.

    References
    ----------
    A. Savitzky and M. J. E. Golay, Analytical Chemistry 36, 1627 (1964).
    DOI: 10.1021/ac60214a047.
    """

    @staticmethod
    def savgolfilter(
        x_val: ArrayLike,
        y_val: ArrayLike,
        fullWindowSize: int,
        derivativeOrder: int,
        degreeSmoothingPoly: int,
    ) -> tuple[FloatArray, FloatArray]:
        """Apply a Savitzky-Golay filter to x/y data.

        Parameters
        ----------
        x_val:
            One-dimensional x-axis values.
        y_val:
            One-dimensional measurement values. Must have the same length as
            ``x_val``.
        fullWindowSize:
            Full symmetric window size. The value must be odd.
        derivativeOrder:
            Derivative order to evaluate. Use 0 for smoothing.
        degreeSmoothingPoly:
            Degree of the local smoothing polynomial.

        Returns
        -------
        x_filtered:
            Copy of the input x-axis.
        y_filtered:
            Smoothed data or derivative values evaluated at ``x_filtered``.
        """

        x_array = _as_1d_float_array(x_val, name="x_val")
        y_array = _as_1d_float_array(y_val, name="y_val")

        if x_array.size != y_array.size:
            raise ValueError("x_val and y_val must have the same length.")

        if np.any(np.diff(x_array) <= 0):
            raise ValueError("x_val must be strictly increasing.")

        full_window_size = _validate_window(
            fullWindowSize,
            x_array.size,
            name="fullWindowSize",
        )

        if full_window_size != int(fullWindowSize):
            raise ValueError("fullWindowSize must be odd.")

        derivative_order = int(derivativeOrder)
        degree = int(degreeSmoothingPoly)

        if derivative_order < 0:
            raise ValueError("derivativeOrder must be non-negative.")

        if degree < 0:
            raise ValueError("degreeSmoothingPoly must be non-negative.")

        if derivative_order > degree:
            raise ValueError("derivativeOrder must not exceed degreeSmoothingPoly.")

        if degree >= full_window_size:
            raise ValueError("degreeSmoothingPoly must be smaller than fullWindowSize.")

        dx = np.diff(x_array)

        if SavGolFilter._is_equidistant(x_array):
            y_filtered = signal.savgol_filter(
                y_array,
                window_length=full_window_size,
                polyorder=degree,
                deriv=derivative_order,
                delta=float(dx[0]),
            )
            return x_array.copy(), np.asarray(y_filtered, dtype=np.float64)

        half_window_size = (full_window_size - 1) // 2
        y_filtered = np.zeros_like(y_array, dtype=np.float64)
        n_points = x_array.size

        for index in range(half_window_size):
            left_x = x_array[:full_window_size]
            left_coeff = SavGolFilter._savgolcoeff_nonequidist(
                left_x,
                index,
                derivative_order,
                degree,
            )
            y_filtered[index] = np.dot(left_coeff, y_array[:full_window_size])

            right_slice = slice(n_points - full_window_size, n_points)
            right_x = x_array[right_slice]
            right_center = full_window_size - 1 - index
            right_coeff = SavGolFilter._savgolcoeff_nonequidist(
                right_x,
                right_center,
                derivative_order,
                degree,
            )
            y_filtered[n_points - index - 1] = np.dot(
                right_coeff,
                y_array[right_slice],
            )

        for index in range(half_window_size, n_points - half_window_size):
            window_slice = slice(
                index - half_window_size,
                index + half_window_size + 1,
            )
            x_window = x_array[window_slice]
            coeff = SavGolFilter._savgolcoeff_nonequidist(
                x_window,
                half_window_size,
                derivative_order,
                degree,
            )
            y_filtered[index] = np.dot(coeff, y_array[window_slice])

        return x_array.copy(), y_filtered

    @staticmethod
    def savgolcoeff(
        win_left: int,
        win_right: int,
        derivativeOrder: int,
        degreeSmoothingPoly: int,
    ) -> FloatArray:
        """Calculate Savitzky-Golay coefficients for equidistant spacing.

        Parameters
        ----------
        win_left:
            Number of points used to the left of the target point.
        win_right:
            Number of points used to the right of the target point.
        derivativeOrder:
            Derivative order to evaluate.
        degreeSmoothingPoly:
            Degree of the smoothing polynomial.

        Returns
        -------
        numpy.ndarray
            Savitzky-Golay convolution coefficients.
        """

        win_left = int(win_left)
        win_right = int(win_right)
        derivative_order = int(derivativeOrder)
        degree = int(degreeSmoothingPoly)

        if (
            win_left < 0
            or win_right < 0
            or derivative_order < 0
            or degree < 0
            or derivative_order > degree
            or win_left + win_right < degree
        ):
            raise ValueError("Invalid Savitzky-Golay coefficient parameters.")

        n_coefficients = win_left + win_right + 1
        a_matrix = np.zeros((degree + 1, degree + 1), dtype=np.float64)
        b_vector = np.zeros(degree + 1, dtype=np.float64)
        coefficients = np.zeros(n_coefficients, dtype=np.float64)

        for power in range(2 * degree + 1):
            lsf_sum = 1.0 if power == 0 else 0.0

            for offset in range(1, win_right + 1):
                lsf_sum += np.power(offset, power)

            for offset in range(1, win_left + 1):
                lsf_sum += np.power(-offset, power)

            span = min(power, 2 * degree - power)

            for matrix_offset in range(-span, span + 1, 2):
                row = (power + matrix_offset) // 2
                column = (power - matrix_offset) // 2
                a_matrix[row, column] = lsf_sum

        b_vector[derivative_order] = 1.0
        b_vector = np.linalg.solve(a_matrix, b_vector)

        for offset in range(-win_left, win_right + 1):
            lsf_sum = b_vector[0]
            factor = 1.0

            for poly_index in range(degree):
                factor *= offset
                lsf_sum += b_vector[poly_index + 1] * factor

            coefficients[offset + win_left] = lsf_sum

        if derivative_order > 1:
            coefficients *= factorial(derivative_order)

        return coefficients

    @staticmethod
    def _savgolcoeff_nonequidist(
        x_window: ArrayLike,
        center_idx: int,
        derivativeOrder: int,
        degreeSmoothingPoly: int,
    ) -> FloatArray:
        """Compute local Savitzky-Golay coefficients for arbitrary x spacing."""

        x_array = _as_1d_float_array(x_window, name="x_window")
        center_idx = int(center_idx)
        derivative_order = int(derivativeOrder)
        degree = int(degreeSmoothingPoly)

        if center_idx < 0 or center_idx >= x_array.size:
            raise ValueError("center_idx is outside the x_window range.")

        if derivative_order > degree:
            raise ValueError("derivativeOrder must not exceed degreeSmoothingPoly.")

        offsets = x_array - x_array[center_idx]
        vandermonde = np.vander(offsets, degree + 1, increasing=True)
        pseudo_inverse = np.linalg.pinv(vandermonde)

        return pseudo_inverse[derivative_order, :] * factorial(derivative_order)

    # TODO: Already in utils ?
    @staticmethod
    def _is_equidistant(x_val: ArrayLike, atol: float = 1.0e-12, rtol: float = 0.0) -> bool:
        """Return whether x values are approximately equidistant."""

        x_array = _as_1d_float_array(x_val, name="x_val")

        if x_array.size < 3:
            return True

        dx = np.diff(x_array)

        return bool(np.allclose(dx, dx[0], atol=atol, rtol=rtol))


def custom_savgol_xy_filter(
    x: ArrayLike,
    y: ArrayLike,
    window_size: int,
    derivative_order: int = 0,
    poly_order: int = 3,
) -> tuple[FloatArray, FloatArray]:
    """Apply the custom Savitzky-Golay filter to x/y data.

    This convenience wrapper exposes :class:`SavGolFilter` with snake_case
    argument names.
    """

    return SavGolFilter.savgolfilter(
        x,
        y,
        fullWindowSize=window_size,
        derivativeOrder=derivative_order,
        degreeSmoothingPoly=poly_order,
    )