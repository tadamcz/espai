import numpy as np
import pytest

from espai import Framing, FIELDS, Question
from espai.fit import df_fit, DistFamily
from espai.preprocess import merged, filter_rename


@pytest.fixture(params=[Question.HLMI, Question.FAOL], scope="module")
def question(request):
    return request.param


@pytest.fixture(params=[Framing.FY, Framing.FP], scope="module")
def framing(request):
    return request.param


@pytest.fixture(scope="module", params=[DistFamily.cinterp5])
def family(request):
    return request.param


@pytest.fixture(scope="module")
def data(question, framing, family):
    data = merged(question)
    data = filter_rename(data, question, framing)

    # Implicitly tests that this even runs without raising
    data = df_fit(data, question, framing, family)

    return data


def test_no_nan(data, question, framing):
    # Check we have values for all rows where the previous analysis had successfully fitted
    # gammas
    nans = data[data["fitted_dist"].isna()]

    want_nans = 0
    if question == Question.FAOL and framing == Framing.FY:
        # TODO: couldn't get it all the way down to zero despite extensive massaging.
        #     But 1 is pretty good.
        want_nans = 1

    assert len(nans) == want_nans


def test_flexible_is_exact(data, question, framing):
    """
    Flexible distribution does what it advertises: fits the CDF data exactly.
    """
    data = data.dropna(subset=["fitted_dist"])

    # Use the ppf for fixed years, and cdf for fixed probabilities
    # This means we use two different loss functions here, but it doesn't matter
    # since we only assert (approximate) equality to zero, not the value of the loss.
    # This also explains why I do not want to use ``df_calc_err`` or ``df_fitted_values`` here.
    def get_fitted_years(dist, probabilities):
        return [dist.ppf(p) for p in probabilities]

    def get_fitted_ps(dist, years):
        return [dist.cdf(y) for y in years]

    # TODO: Tolerances are only used for degenerate rows where two or more data points
    #    have the same value. The more correct way would be: if two (or three) values are the same,
    #    we assert only that _one_ of the corresponding fitted values is an exact match.
    #    This would target the loosening of the test to only the narrowest possible surface area:
    #    individual CDF points (not even triples!) where the user gave impossible data.
    #    This dirty version is OK for now.
    if framing == Framing.FP:
        ps = FIELDS[question][framing].keys()
        fitted_years = data["fitted_dist"].apply(get_fitted_years, args=(ps,))
        fitted_years = np.array(fitted_years.tolist())
        data.loc[:, ["fitted_x1", "fitted_x2", "fitted_x3"]] = fitted_years

        abstol = 1 / 10  # 10% of a year
        rtol = 1 / 1000
        failed_rows = data[
            ~np.isclose(data["fitted_x1"], data["x1"], atol=abstol, rtol=rtol)
            | ~np.isclose(data["fitted_x2"], data["x2"], atol=abstol, rtol=rtol)
            | ~np.isclose(data["fitted_x3"], data["x3"], atol=abstol, rtol=rtol)
        ]
        assert len(failed_rows) == 0, failed_rows.index

    elif framing == Framing.FY:
        xs = FIELDS[question][framing].keys()
        fitted_ps = data["fitted_dist"].apply(get_fitted_ps, args=(xs,))
        fitted_ps = np.array(fitted_ps.tolist())
        data.loc[:, ["fitted_p1", "fitted_p2", "fitted_p3"]] = fitted_ps

        abstol = 1e-4
        failed_rows = data[
            # Divide by 100: source data expresses probabilities as percentages (0-100)
            ~np.isclose(data["fitted_p1"], data["p1"] / 100, atol=abstol)
            | ~np.isclose(data["fitted_p2"], data["p2"] / 100, atol=abstol)
            | ~np.isclose(data["fitted_p3"], data["p3"] / 100, atol=abstol)
        ]
        assert len(failed_rows) == 0, failed_rows.index
