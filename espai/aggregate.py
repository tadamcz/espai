import enum
from typing import List

import pandas as pd
import numpy as np
import logging

from espai.robust_geomean import geomean_odds

logger = logging.getLogger(__name__)


class AggMethod(enum.Enum):
    """
    Potentially relevant background reading about aggregation:
    - https://www.researchgate.net/profile/Yael-Grushka-Cockayne/publication/256020452_Is_It_Better_to_Average_Probabilities_or_Quantiles/links/5b2a44aba6fdcc72db4c8c4e/Is-It-Better-to-Average-Probabilities-or-Quantiles.pdf
    - https://sci-hub.se/https://pubsonline.informs.org/doi/10.1287/deca.2013.0282
    - https://learnmoore.org/papers/QES1v12.pdf
    """

    ARITH_MEAN_PROBS = enum.auto()
    MEDIAN_PROBS = enum.auto()
    GEO_MEAN_ODDS = enum.auto()

    GEO_MEAN_ODDS_WINSORIZED = enum.auto()

    ARITH_MEAN_YEARS = enum.auto()

    @property
    def vertical(self):
        return self in [
            AggMethod.ARITH_MEAN_PROBS,
            AggMethod.MEDIAN_PROBS,
            AggMethod.GEO_MEAN_ODDS,
            AggMethod.GEO_MEAN_ODDS_WINSORIZED,
        ]

    @property
    def horizontal(self):
        return self in [AggMethod.ARITH_MEAN_YEARS]


def summarize_dists(
    data: pd.DataFrame,
    evaluate_at: np.ndarray,
    dist_col: str,
    method: AggMethod = AggMethod.ARITH_MEAN_PROBS,
    dists_quantiles: List[float] = (),
) -> pd.DataFrame:
    def calc_cdf(row):
        dist = row[dist_col]
        return dist.cdf(evaluate_at)

    def calc_ppf(row):
        dist = row[dist_col]
        return dist.ppf(evaluate_at)

    if method == AggMethod.ARITH_MEAN_PROBS:
        cdfs = data.apply(calc_cdf, axis=1, result_type="expand")
        metrics = {
            "aggregated": np.mean(cdfs, axis=0),
        }
        for p in dists_quantiles:
            metrics[p] = np.quantile(cdfs, p, axis=0)

    elif method == AggMethod.MEDIAN_PROBS:
        cdfs = data.apply(calc_cdf, axis=1, result_type="expand")
        metrics = {
            "aggregated": np.median(cdfs, axis=0),
        }
        for p in dists_quantiles:
            metrics[p] = np.quantile(cdfs, p, axis=0)

    elif method == AggMethod.GEO_MEAN_ODDS:
        cdfs = data.apply(calc_cdf, axis=1, result_type="expand")

        # Manually do each column here instead of axis=1
        # (to keep robust_geomean.py more general purpose)
        aggregated = []
        for col in range(len(evaluate_at)):
            aggregated.append(geomean_odds(cdfs.iloc[:, col]))

        metrics = {
            "aggregated": aggregated,
        }
        for p in dists_quantiles:
            metrics[p] = np.quantile(cdfs, p, axis=0)

    elif method == AggMethod.GEO_MEAN_ODDS_WINSORIZED:
        cdfs = data.apply(calc_cdf, axis=1, result_type="expand")
        odds = cdfs / (1 - cdfs)
        # Geometric mean of odds is unworkable with our data, because many
        # respondents give probabilities of 0 or 1, and even more give responses that
        # imply CDFs that are 0 or 1 over a significant portion of the range of interest.
        # So we can winsorize, but the problem is that the cutoff is arbitrary and enormously
        # affects the results. It's a tough problem.
        odds_max = 100  # arbitrary, but greatly affects results
        odds_min = 1 / odds_max
        odds_win = np.clip(odds, odds_min, odds_max)
        count_win = np.sum(np.sum(odds != odds_win, axis=1))
        logger.info(f"Winsorized {count_win} / {cdfs.size} CDF values across {len(data)} responses")
        geo_mean_odds = np.exp(np.mean(np.log(odds_win), axis=0))
        metrics = {
            "aggregated": geo_mean_odds / (1 + geo_mean_odds),
        }
        for p in dists_quantiles:
            metrics[p] = np.quantile(cdfs, p, axis=0)

    elif method == AggMethod.ARITH_MEAN_YEARS:
        ppfs = data.apply(calc_ppf, axis=1, result_type="expand")
        metrics = {
            "aggregated": np.mean(ppfs, axis=0),
        }
        for p in dists_quantiles:
            metrics[p] = np.quantile(ppfs, p, axis=0)
    else:
        raise ValueError(f"Invalid aggregation method: {method}")

    return pd.DataFrame(metrics)
