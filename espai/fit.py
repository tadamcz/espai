import enum
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from scipy.optimize import Bounds, least_squares, minimize
from scipy.stats import linregress
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from espai import FIELDS, Framing, cache, PROJECT_ROOT, Question
from espai.adjust import adjust_x, adjust_p, add_implicit_minimum

from make_distribution.client import JSONClient, SciPyClient
from requests_cache import CachedSession


logger = logging.getLogger(__name__)


class CustomJsonClient(JSONClient):
    def __init__(self, token=None, version="v0"):
        super().__init__(token, version)
        # MakeDistribution.com offers proprietary 'flexible' distributions that can match
        # arbitrary CDF data exactly. This is a paid service. We store the responses in this repo
        # so that this analysis can be run without paying for the service. To run a new analysis
        # involving these proprietary distributions, you'll need to pay for the service.
        #
        # If you  have any questions, please contact Tom Adamczewski (tadamcz.com), the author of
        # this commit.
        self.session = CachedSession(
            cache_name=f"{PROJECT_ROOT}/makedistribution_responses",
            backend="filesystem",
            # This makes the files human-readable and avoids committing binary files to git.
            # The downside is that large numbers of files are created.
            serializer="json",
            # Correct for our non-standard usage to include POST
            allowable_methods=["GET", "POST"],
            match_headers=False,
            allowable_codes=(200, 201, 204),
        )
        if token is not None:
            self.session.headers.update({"Authorization": f"Token {token}"})


load_dotenv()
client = CustomJsonClient(token=os.environ.get("MAKE_DISTRIBUTION_API_TOKEN"))
client = SciPyClient(client)


class DistFamily(enum.Enum):
    """
    This is an Enum rather than a more typical class for a good reason: because SciPy's
    optimizers strongly expect you to use the non-frozen distribution objects.
    """

    GAMMA = "gamma"
    WEIBULL = "weibull"
    GEN_GAMMA = "gen_gamma"
    cinterp5 = "cinterp5_01"
    cinterp3 = "cinterp3_01"

    def readable_kwarg_names(self):
        match self:
            case DistFamily.GAMMA:
                return ["shape", "scale"]
            case DistFamily.WEIBULL:
                return ["shape", "scale"]
            case DistFamily.GEN_GAMMA:
                return ["a", "c", "scale"]
            case DistFamily.cinterp5:
                return []
            case DistFamily.cinterp3:
                return []

    @property
    def nargs(self):
        return len(self.readable_kwarg_names())

    def to_scipy_positional(self, **kwargs):
        match self:
            case DistFamily.GAMMA:
                assert kwargs.keys() == set(self.readable_kwarg_names())
                a = kwargs["shape"]
                scale = kwargs["scale"]
                return a, scale
            case DistFamily.WEIBULL:
                assert kwargs.keys() == set(self.readable_kwarg_names())
                c = kwargs["shape"]
                scale = kwargs["scale"]
                return c, scale
            case DistFamily.GEN_GAMMA:
                assert kwargs.keys() == set(self.readable_kwarg_names())
                a = kwargs["a"]
                c = kwargs["c"]
                scale = kwargs["scale"]
                return a, c, scale

    def to_readable_kwargs(self, *args):
        assert len(args) == self.nargs
        return dict(zip(self.readable_kwarg_names(), args))

    def cdf_positional(self, x, *args):
        match self:
            case DistFamily.GAMMA:
                assert len(args) == self.nargs
                loc = 0
                a, scale = args
                return stats.gamma.cdf(x, a, loc=loc, scale=scale)
            case DistFamily.WEIBULL:
                assert len(args) == self.nargs
                loc = 0
                c, scale = args
                return stats.weibull_min.cdf(x, c, loc=loc, scale=scale)
            case DistFamily.GEN_GAMMA:
                assert len(args) == self.nargs
                loc = 0
                a, c, scale = args
                return stats.gengamma.cdf(x, a, c, loc=loc, scale=scale)

    def ppf_positional(self, p, *args):
        match self:
            case DistFamily.GAMMA:
                assert len(args) == self.nargs
                loc = 0
                a, scale = args
                return stats.gamma.ppf(p, a, loc=loc, scale=scale)
            case DistFamily.WEIBULL:
                assert len(args) == self.nargs
                loc = 0
                c, scale = args
                return stats.weibull_min.ppf(p, c, loc=loc, scale=scale)
            case DistFamily.GEN_GAMMA:
                assert len(args) == self.nargs
                loc = 0
                a, c, scale = args
                return stats.gengamma.ppf(p, a, c, loc=loc, scale=scale)

    def cdf(self, x, **kwargs):
        assert kwargs.keys() == set(self.readable_kwarg_names())
        return self.cdf_positional(x, *self.to_scipy_positional(**kwargs))

    def ppf(self, p, **kwargs):
        assert kwargs.keys() == set(self.readable_kwarg_names())
        return self.ppf_positional(p, *self.to_scipy_positional(**kwargs))

    def freeze(self, **kwargs):
        """Nomenclature comes from SciPy"""
        match self:
            case DistFamily.GAMMA:
                assert kwargs.keys() == set(self.readable_kwarg_names())
                a = kwargs["shape"]
                scale = kwargs["scale"]
                return stats.gamma(a, scale=scale)
            case DistFamily.WEIBULL:
                assert kwargs.keys() == set(self.readable_kwarg_names())
                c = kwargs["shape"]
                scale = kwargs["scale"]
                return stats.weibull_min(c, scale=scale)
            case DistFamily.GEN_GAMMA:
                assert kwargs.keys() == set(self.readable_kwarg_names())
                a = kwargs["a"]
                c = kwargs["c"]
                scale = kwargs["scale"]
                return stats.gengamma(a, c, scale=scale)


class LossFunction(enum.Enum):
    CDF_MSE_PROBS = "CDF_MSE_PROBS"
    CDF_MSE_YEARS = "CDF_MSE_YEARS"

    # Where p is the true probability given by the respondent, and q is the CDF's prediction at
    # that point.
    #
    # Loss function inspired by the KL divergence between discrete distributions:
    # L_KL(p, q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
    #            = p * log(p) - p * log(q) + (1-p) * log(1-p) - (1-p) * log(1-q)
    #
    # Binary cross-entropy loss (or simply log loss):
    # L_BCE(p, q) = -[p * log(q) + (1-p) * log(1-q)]
    #
    # The KL divergence loss and binary cross-entropy loss are equivalent (have the same minimum)
    # because they differ only by a constant term that does not depend on q.
    # TODO: optimization necessarily fails for any response that contains p=0 or p=1. I don't see
    #     a principled way to solve this. The problem is analogous to the problem with the geometric
    #     mean of odds.
    LOG_LOSS = "LOG_LOSS"


def log_loss(xs, ps, qs):
    loss = 0
    for x, p, q in zip(xs, ps, qs):
        loss += -(p * np.log(q) + (1 - p) * np.log1p(-q))
    return loss


def _fit_local(xs, ps, init: Dict[str, float], dist: DistFamily, loss_function: LossFunction):
    """
    Local optimization of the distribution parameters
    """
    p0 = dist.to_scipy_positional(**init)

    # Define bounds based on the distribution
    if dist == DistFamily.GAMMA or dist == DistFamily.WEIBULL:
        bounds = Bounds([0, 0], [np.inf, np.inf])
    elif dist == DistFamily.GEN_GAMMA:
        # a>0, c!=0, scale>0
        # TODO: hacky approach for handling the discontinuity in 'c':
        #      only look on the side of the discontinuity where the initial guess is.
        #      I don't like how this is implicit (instead of explicit) in the API of this function.
        if init["c"] > 0:
            c_lower, c_upper = 0, np.inf
        else:
            c_lower, c_upper = -np.inf, 0
        bounds = Bounds([0, c_lower, 0], [np.inf, c_upper, np.inf])
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

    if loss_function in [LossFunction.CDF_MSE_PROBS, LossFunction.CDF_MSE_YEARS]:
        if loss_function == LossFunction.CDF_MSE_PROBS:

            def residuals(params):
                return dist.cdf_positional(xs, *params) - ps

        elif loss_function == LossFunction.CDF_MSE_YEARS:

            def residuals(params):
                return dist.ppf_positional(ps, *params) - xs

        # Directly use ``least_squares`` instead of ``curve_fit``
        # to have greater control over the results object
        result = least_squares(
            residuals,
            p0,
            bounds=bounds,
            method="trf",
        )
        if not result.success:
            # TODO: previous code would raise here, and some downstream code still exists to handle
            #  this. We lose some potentially useful information here.
            #  But these results are in fact often pretty good, and I currently prefer this approach to
            #  using higher values for ftol, gtol and xtol.
            logger.warning(
                f"Optimization failed: {result.message} after {result.nfev} evaluations. Using last values reached during optimization."
            )
    elif loss_function == LossFunction.LOG_LOSS:

        def loss(params):
            qs = dist.cdf_positional(xs, *params)
            return log_loss(xs, ps, qs)

        result = minimize(
            loss,
            p0,
            bounds=bounds,
            method="L-BFGS-B",
        )
        if not result.success:
            logger.warning(
                f"Optimization failed: {result.message} after {result.nfev} evaluations. Using last values reached during optimization."
            )
    else:
        raise NotImplementedError(f"{loss_function} is not implemented")

    fitted_positional = result.x
    fitted = dist.to_readable_kwargs(*fitted_positional)
    logger.debug(
        f"xs={xs}, ps={ps}, init={init}, fitted={fitted}, message={result.message}, nfev={result.nfev}"
    )
    return fitted


@cache.memoize()
def fit_local(xs, ps, init: Dict[str, float], dist: DistFamily, loss_function: LossFunction):
    """Return NaNs instead of raising, so that values can be cached."""
    try:
        return _fit_local(xs, ps, init, dist, loss_function)
    except (ValueError, RuntimeError):
        return {k: np.nan for k in dist.readable_kwarg_names()}


def cdf_mean_sqerr_probs(fitted_dist: rv_continuous_frozen, xs, ps):
    """Mean squared error between the predicted and actual CDF values."""
    predicted_ps = fitted_dist.cdf(xs)
    return np.mean((ps - predicted_ps) ** 2)


def cdf_mean_sqerr_years(fitted_dist: rv_continuous_frozen, xs, ps):
    """Mean squared error between the predicted and actual years."""
    predicted_xs = fitted_dist.ppf(ps)
    return np.mean((xs - predicted_xs) ** 2)


def fit_makedistribution(quantiles: Dict[float, float], dist: DistFamily):
    data = {
        "family": {"requested": dist.value},
        "arguments": {
            "quantiles": [
                {
                    "p": k,
                    "x": v,
                }
                for k, v in quantiles.items()
            ]
        },
    }
    return client.post("1d/dists/", json=data)


def fit(xs, ps, dist: DistFamily, loss_function: LossFunction):
    """
    Pseudo-global optimization by setting multiple initial guesses for local optimization.
    """
    if dist in [DistFamily.GAMMA, DistFamily.WEIBULL, DistFamily.GEN_GAMMA]:
        # Silence warnings about invalid floating point operations
        # when generating initial guesses
        with np.errstate(all="ignore"):
            if dist == DistFamily.GAMMA:
                guesses = [
                    gamma_moments_guess(xs, ps),
                    {"shape": 1, "scale": 1},
                    {"shape": 1, "scale": 100},
                ]
                guesses.extend(leave_one_out_guesses(xs, ps, dist, loss_function))
            elif dist == DistFamily.WEIBULL:
                guesses = [
                    {"shape": 1, "scale": 1},
                    {"shape": 1, "scale": 100},
                ]
                try:
                    # TODO: is this ever the winning guess?
                    weibull_linearization_based_guess(ps, xs),
                except ValueError:
                    pass
                guesses.extend(leave_one_out_guesses(xs, ps, dist, loss_function))
            elif dist == DistFamily.GEN_GAMMA:
                # TODO: this is arbitrary and messy
                a_guesses = [1 / 100, 100]
                c_guesses = [-2, -1 / 2, 1 / 2, 2]
                scale_guesses = [1, 1000]

                ac_guesses = [
                    {"a": a, "c": c, "scale": scale}
                    for a in a_guesses
                    for c in c_guesses
                    for scale in scale_guesses
                ]

                guesses = [
                    *gen_gamma_guesses(xs, ps),
                    {"a": 1, "c": 1, "scale": 1},
                    {"a": 1, "c": -1, "scale": 1},
                    *ac_guesses,
                ]

    elif dist in [DistFamily.cinterp5, DistFamily.cinterp3]:
        xs_adj = adjust_x(xs)
        ps_adj = adjust_p(ps)

        if any(xs_adj != xs) or any(ps_adj != ps):
            quantiles_unadjusted = {p: x for p, x in zip(ps, xs)}
            quantiles_adjusted = {p: x for p, x in zip(ps_adj, xs_adj)}

            # For readability of logs only: change np.float64 to float
            quantiles_adjusted = {float(k): float(v) for k, v in quantiles_adjusted.items()}
            quantiles_unadjusted = {float(k): float(v) for k, v in quantiles_unadjusted.items()}

            logger.debug(f"Adjusted quantiles from {quantiles_unadjusted} to {quantiles_adjusted}")

        quantiles = {p: x for p, x in zip(ps_adj, xs_adj)}
        quantiles = add_implicit_minimum(quantiles)

        fitted_dist = fit_makedistribution(quantiles, dist)
        if fitted_dist.data["fit"]["status"] != "success":
            raise ValueError(f"Failed to fit distribution: {fitted_dist.data['fit']}")
        fitted_params = {}
        return fitted_params, fitted_dist
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    results = []

    for guess in guesses:
        fitted_params = fit_local(xs, ps, guess, dist, loss_function)
        fitted_dist = dist.freeze(**fitted_params)
        if loss_function == LossFunction.CDF_MSE_PROBS:
            loss = cdf_mean_sqerr_probs(fitted_dist, xs, ps)
        elif loss_function == LossFunction.CDF_MSE_YEARS:
            loss = cdf_mean_sqerr_years(fitted_dist, xs, ps)
        elif loss_function == LossFunction.LOG_LOSS:
            qs = fitted_dist.cdf(xs)
            loss = log_loss(xs, ps, qs)
        results.append(
            {
                "guess": guess,
                "fitted_params": fitted_params,
                "fitted_dist": fitted_dist,
                "loss": loss,
            }
        )

    best_result = {"loss": np.inf}
    for result in results:
        if result["loss"] < best_result["loss"]:
            best_result = result

    if best_result["loss"] == np.inf:
        raise ValueError(f"All {len(guesses)} initial guesses failed.")

    return best_result["fitted_params"], best_result["fitted_dist"]


def get_x_p(row, question, framing):
    # TODO: Hacky special-casing of HLMI_FAOL
    if question == Question.HLMI_FAOL:
        if row["question"] == Question.HLMI.value:
            return get_x_p(row, Question.HLMI, framing)
        elif row["question"] == Question.FAOL.value:
            return get_x_p(row, Question.FAOL, framing)
        else:
            raise ValueError(f"Unknown question: {row['question']}")

    if framing == Framing.FY:
        x = np.array([years for years in FIELDS[question][Framing.FY]], dtype=float)
        p = np.array(row[framing.human_columns], dtype=float)
        # Probabilities expressed as percentages (0-100) in the data
        p = p / 100
    elif framing == Framing.FP:
        x = np.array(row[framing.human_columns], dtype=float)
        p = np.array([prob for prob in FIELDS[question][Framing.FP]], dtype=float)
    else:
        raise ValueError(f"Unknown framing: {framing}")

    return x, p


def df_calc_err(
    data: pd.DataFrame,
    fitted_col: str,
    question: Question,
    framing: Framing,
):
    def calc_err(row):
        x, p = get_x_p(row, question, framing)
        fitted_dist = row[fitted_col]
        return cdf_mean_sqerr_probs(fitted_dist, x, p)

    return data.apply(calc_err, axis=1)


def df_fitted_values(
    data: pd.DataFrame,
    kwargs_cols: Dict[str, str],
    question: Question,
    framing: Framing,
    dist: DistFamily,
):
    """
    should add columns [f"{h}_fitted" for h in framing.human_columns]
    """

    def fitted_ps(row):
        x, _ = get_x_p(row, question, framing)
        fitted_params = {param: row[colname] for param, colname in kwargs_cols.items()}
        cdf = dist.cdf(x, **fitted_params)
        return pd.Series(cdf, index=[f"{h}_fitted" for h in framing.human_columns])

    def fitted_xs(row):
        _, p = get_x_p(row, question, framing)
        fitted_params = {param: row[colname] for param, colname in kwargs_cols.items()}
        ppf = dist.ppf(p, **fitted_params)
        return pd.Series(ppf, index=[f"{h}_fitted" for h in framing.human_columns])

    if framing == Framing.FY:
        func = fitted_ps
        out_cols = [f"{h}_fitted" for h in framing.human_columns]
    elif framing == Framing.FP:
        func = fitted_xs
        out_cols = [f"{h}_fitted" for h in framing.human_columns]

    data.loc[:, out_cols] = data.apply(func, axis=1)
    return data


def df_fit(
    data: pd.DataFrame,
    question: Question,
    framing: Framing | None,
    dist: DistFamily,
    loss_function: LossFunction | None = None,
):

    if dist in [DistFamily.cinterp5, DistFamily.cinterp3]:
        if loss_function is not None:
            raise ValueError("loss_function should be None for flexible distributions")
    else:
        if loss_function is None:
            raise ValueError("Please provide a loss function")

    assert data.index.name == "ResponseId"

    if framing is None:
        data_fy = data[data["treat_assign"] == Framing.FY.value]
        data_fy = df_fit(data_fy, question, Framing.FY, dist, loss_function)

        data_fp = data[data["treat_assign"] == Framing.FP.value]
        data_fp = df_fit(data_fp, question, Framing.FP, dist, loss_function)
        return pd.concat([data_fy, data_fp])

    def fit_row(row):
        x, p = get_x_p(row, question, framing)

        if np.any(np.isnan(x)) or np.any(np.isnan(p)):
            nans = {k: np.nan for k in dist.readable_kwarg_names()}
            return pd.Series({**nans, "fitted_dist": None, "fit_msg": "data_NaNs"})

        try:
            fitted_params, fitted_dist = fit(x, p, dist, loss_function)
            fit_msg = "success"
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Failed to fit row {row.name} with xs={x}, ps={p}. Error: {e}")
            fitted_params = {k: np.nan for k in dist.readable_kwarg_names()}
            fitted_dist = None
            fit_msg = str(e)

        return pd.Series({**fitted_params, "fitted_dist": fitted_dist, "fit_msg": fit_msg})

    new_cols = dist.readable_kwarg_names() + ["fitted_dist", "fit_msg"]
    data.loc[:, new_cols] = data.apply(fit_row, axis=1)
    return data


@cache.memoize()
def gamma_moments_guess(xs, ps):
    # Approximate the PDF
    pdf = np.diff(ps, prepend=0)
    # Approximate the mean
    mean = np.sum(xs * pdf)
    # Approximate the variance
    var = np.sum((xs - mean) ** 2 * pdf)

    # Using the following properties of the gamma distribution:
    # mean == shape * scale
    # var == shape * scale ** 2
    shape = mean**2 / var
    scale = var / mean
    return {"shape": shape, "scale": scale}


@cache.memoize()
def weibull_two_quantiles_guess(p1, x1, p2, x2):
    """
    Estimate Weibull parameters using two quantiles from CDF data.

    Args:
    p1: first cumulative probability
    x1: first quantile value
    p2: second cumulative probability
    x2: second quantile value

    Returns:
    dict with 'shape' and 'scale' parameters
    """
    # Ensure x2 > x1 and p2 > p1
    if x1 > x2:
        x1, x2 = x2, x1
        p1, p2 = p2, p1

    # Calculate shape parameter
    shape = np.log(np.log(1 - p1) / np.log(1 - p2)) / np.log(x1 / x2)
    # Calculate scale parameter
    scale = x1 / (-np.log(1 - p1)) ** (1 / shape)

    return {"shape": shape, "scale": scale}


@cache.memoize()
def weibull_linearization_based_guess(ps, xs):
    # Ensure ps and xs are sorted and remove any p values equal to 1
    sorted_indices = np.argsort(ps)
    ps = np.array(ps)[sorted_indices]
    xs = np.array(xs)[sorted_indices]
    mask = ps < 1
    ps = ps[mask]
    xs = xs[mask]

    if len(ps) < 2 or len(xs) < 2:
        raise ValueError("At least two quantiles are needed to estimate the Weibull parameters.")

    # Linearize the CDF
    y = np.log(-np.log(1 - ps))
    x = np.log(xs)

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(x, y)

    shape_estimate = slope
    scale_estimate = np.exp(-intercept / shape_estimate)

    return {"shape": shape_estimate, "scale": scale_estimate}


@cache.memoize()
def gen_gamma_guesses(xs, ps):
    # Start with gamma guess
    gamma_guess = gamma_moments_guess(xs, ps)

    # Use gamma parameters as initial guess for a and scale
    a_guess = gamma_guess["shape"]
    scale_guess = gamma_guess["scale"]

    # TODO: this is arbitrary and messy
    cs = [1, 1 / 2, 2]
    cs = cs + [-c for c in cs]

    return [{"a": a_guess, "c": c, "scale": scale_guess} for c in cs]


@cache.memoize()
def leave_one_out_guesses(xs, ps, dist: DistFamily, loss_function: LossFunction):
    guesses = []
    for i in range(len(xs)):
        x = np.delete(xs, i)
        p = np.delete(ps, i)

        # After leaving one out, all others are the same
        if np.all(x[0] == x) or np.all(p[0] == p):
            continue

        if dist == DistFamily.GAMMA:
            init = gamma_moments_guess(x, p)
        elif dist == DistFamily.WEIBULL:
            p1, x1 = p[0], x[0]
            p2, x2 = p[1], x[1]
            init = weibull_two_quantiles_guess(p1, x1, p2, x2)
        else:
            raise NotImplementedError(dist)

        fitted = fit_local(x, p, init, dist, loss_function)
        guesses.append(fitted)
    return guesses
