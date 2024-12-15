from typing import Dict

import numpy as np


def adjust_values(values, lower_bound=None, upper_bound=None, min_delta=None):
    """
    Adjust values to ensure they are strictly increasing and within bounds.
    Raises ValueError if input is decreasing.

    Note as this is possibly confusing: Original R code also had such 'nudges', but they:
    - were applied to the Gamma distribution (and others), whereas we only need this for the flexible
        distributions. We fit 2-parameter distributions cleanly without any massaging (multi-guess
        approach).
    - were larger, and worked differently


    :param values: Input list or array of values
    :param lower_bound: Minimum allowed value (optional)
    :param upper_bound: Maximum allowed value (optional)
    :param min_delta: Minimum difference between consecutive values (optional)
    """
    if np.any(np.diff(values) < 0):
        raise ValueError("Input must be non-decreasing")

    # Apply bounds if provided
    if lower_bound is not None:
        values = np.maximum(values, lower_bound)
    if upper_bound is not None:
        values = np.minimum(values, upper_bound)

    # Ensure strict monotonicity
    for i in range(1, len(values)):
        next_value = np.nextafter(values[i - 1], np.inf)
        if min_delta is not None:
            next_value = max(next_value, values[i - 1] + min_delta)
        if values[i] <= values[i - 1]:
            values[i] = next_value

    # Check if we've reached the upper bound and need to adjust backwards
    if upper_bound is not None and values[-1] > upper_bound:
        values[-1] = upper_bound
        for i in range(len(values) - 2, -1, -1):
            prev_value = np.nextafter(values[i + 1], -np.inf)
            if min_delta is not None:
                prev_value = min(prev_value, values[i + 1] - min_delta)
            if values[i] >= values[i + 1]:
                values[i] = prev_value

    return values


def adjust_x(xs):
    """
    Adjust x values to ensure they are strictly increasing.
    """
    xrange = max(xs) - min(xs)

    # This is especially ad-hoc, and makes me sad
    min_delta = max(1e-6, xrange / 1000)

    return adjust_values(xs, min_delta=min_delta)


def adjust_p(ps):
    """
    Adjust p values to ensure they are strictly increasing and within [0, 1].
    """
    min_delta = 1e-5
    return adjust_values(ps, lower_bound=0, upper_bound=1, min_delta=min_delta)


def add_implicit_minimum(quantiles: Dict[float, float]):
    """
    Add implicit minimum (0 probability) at 0 years. This is because a Gamma
    (with loc=0) is left-bounded by 0, but this isn't automatically the case for a flexible
    distribution.
    """
    # Judgement call: if 0. is already in the probabilities, don't add it.
    # Note the asymmetry: we don't check for 0 in the years. Otherwise, you could get
    # substantial (0.1 or 0.5) probability mass to the left of 0, which would be too different
    # from the Gamma distribution.
    if 0.0 not in quantiles:

        # More nasty data-massaging to avoid data that is too extreme for the numerical
        # code in flexible distributions.
        smallest_x = min(quantiles.values())
        largest_x = max(quantiles.values())

        # Especially ugly to have to use a negative number here: if we don't do this, we'll
        # re-introduce the non-increasingness that we just fixed in adjust_x.
        # Previously we handled this by modifying ``lower_bound`` in ``adjust_x``, which I think
        # was even worse (because it modifies more input data, which is harder to reason about, e.g.
        # when comparing loss function values between different fits).
        if smallest_x == 0:
            implicit_x = -1e-6
        else:
            xrange = largest_x - smallest_x
            if smallest_x > 1000 * xrange:
                implicit_x = smallest_x - xrange / 100
            else:
                implicit_x = 0.0

        quantiles[0.0] = implicit_x
    return quantiles
