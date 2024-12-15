import numpy as np


def probs_to_odds(ps: np.ndarray):
    """Probs to odds while handling infinities"""
    ones_mask = ps == 1
    zeroes_mask = ps == 0

    odds = ps / (1 - ps)

    odds[ones_mask] = np.inf
    odds[zeroes_mask] = 0

    return odds


def odds_to_probs(odds: np.ndarray) -> np.ndarray:
    """Odds to probs while handling infinities"""
    inf_mask = odds == np.inf
    zeros_mask = odds == 0

    probs = odds / (1 + odds)

    probs[inf_mask] = 1
    probs[zeros_mask] = 0

    return probs


def robust_prod(arr: np.ndarray):
    assert np.all(arr >= 0)

    has_zeros = np.any(arr == 0)
    has_infs = np.any(arr == np.inf)

    if has_zeros and has_infs:
        return np.nan
    elif has_zeros:
        return 0
    elif has_infs:
        return np.inf

    return np.exp(np.sum(np.log(arr)))


def geomean_odds(ps: np.ndarray):
    """
    Geometric mean of odds robust to infinities: only returns nan if *both* zero and one are
    among the probabilities to be aggregated
    """
    odds = probs_to_odds(ps)
    agg_odds = robust_prod(odds) ** (1 / len(ps))

    # TODO: make the functions accept a float instead of this
    agg_odds = np.array([agg_odds])

    agg_odds = odds_to_probs(agg_odds)

    print(agg_odds)
    return agg_odds.item()
