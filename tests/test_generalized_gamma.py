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
    data = df_fit(data, question, framing, DistFamily.GEN_GAMMA, LossFunction.CDF_MSE_PROBS)

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
    nans = data[data["a"].isna() | data["c"].isna() | data["scale"].isna()]
    assert len(nans) == 0


def test_no_error_increase(data):
    """
    Generalized gamma always does at least as well as Gamma.
    TODO: could check it beats Weibull as well, which I would hope is the case
    """
    err_diff = data["mean_sq_err"] - data["prev_mean_sq_err"]
    eps = 1e-10
    n_error_increase = sum(err_diff > eps)

    # TODO: we should be able to get this down to zero
    allow_error_increase = 2

    assert n_error_increase <= allow_error_increase


def test_fits_good(data):
    # TODO: I would like to be more ambitious here
    thresh = 1e-5
    n_close_fits = sum(data["mean_sq_err"] < thresh)
    assert n_close_fits > 0.75 * len(data)
