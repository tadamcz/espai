import pytest

from espai import Framing, Question

from espai.fit import df_fit, DistFamily, LossFunction
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

    # This implicitly checks that the Weibull fitting code runs without raising, which is one of the
    # few things we can usefully test for Weibull.
    data = df_fit(data, question, framing, DistFamily.WEIBULL, LossFunction.CDF_MSE_PROBS)

    return data


def test_no_nan(data):
    # Check we have values for all rows where the previous analysis had successfully fitted
    # gammas
    nans = data[data["shape"].isna() | data["scale"].isna()]
    assert len(nans) == 0
