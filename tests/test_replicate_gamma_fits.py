import numpy as np
import pandas as pd
import pytest

from espai import Framing, Question
from espai.fit import df_calc_err, df_fit, DistFamily, LossFunction
from espai.preprocess import merged, filter_rename


@pytest.fixture(params=[Question.HLMI, Question.FAOL], scope="module")
def question(request):
    return request.param


@pytest.fixture(params=[Framing.FY, Framing.FP], scope="module")
def framing(request):
    return request.param


@pytest.fixture(scope="module")
def data(question, framing):
    data = merged(question)
    data = filter_rename(data, question, framing)
    data = df_fit(data, question, framing, DistFamily.GAMMA, LossFunction.CDF_MSE_PROBS)

    def attach_prev_fitted_dist(row):
        return DistFamily.GAMMA.freeze(shape=row["prev_shape"], scale=row["prev_scale"])

    data.loc[:, "mean_sq_err"] = df_calc_err(
        data,
        "fitted_dist",
        question,
        framing,
    )

    data.loc[:, "prev_fitted_dist"] = data.apply(attach_prev_fitted_dist, axis=1)

    data.loc[:, "prev_mean_sq_err"] = df_calc_err(
        data,
        "prev_fitted_dist",
        question,
        framing,
    )

    return data


def test_no_nan(data):
    # Check we actually have values for all rows where the previous analysis had values
    nans = data[data["shape"].isna() | data["scale"].isna()]
    assert len(nans) == 0


def test_no_error_increase(data):
    """
    This is the strictest and most important test in this file.

    It checks that _all_ our fits (about 1700) are at least as good as the previous fit for that
    row. It's rare to be able to make such an assertion for literally every row in a real-world
    dataset.
    """
    err_diff = data["mean_sq_err"] - data["prev_mean_sq_err"]
    eps = 1e-10
    n_error_increase = sum(err_diff > eps)
    assert n_error_increase == 0


def test_params_same(data, question, framing):
    # More than this fraction of the rows should have the same shape and scale.
    # Deviations are OK (as long as the error doesn't degrade, see other test),
    # because multiple sets of parameters can fit the data equally well.
    if framing == Framing.FP:
        want_same = 0.90
    elif framing == Framing.FY:
        if question == Question.HLMI:
            want_same = 0.80
        elif question == Question.FAOL:
            want_same = 0.50

    rtol = 1 / 100
    n_same_shape = np.isclose(data["shape"], data["prev_shape"], rtol=rtol).sum()
    n_same_scale = np.isclose(data["scale"], data["prev_scale"], rtol=rtol).sum()

    assert n_same_shape / len(data) >= want_same
    assert n_same_scale / len(data) >= want_same
