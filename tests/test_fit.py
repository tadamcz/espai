import pandas as pd
import pytest

from espai import PROJECT_ROOT, FIELDS
from espai.fit import df_fit
from espai.preprocess import filter_rename
from espai import Framing
from espai.fit import DistFamily, LossFunction


from espai import Question


@pytest.fixture(params=Question)
def question(request):
    return request.param


@pytest.fixture(params=[Framing.FY, Framing.FP, None])
def framing(request):
    return request.param


@pytest.fixture(params=DistFamily)
def family(request):
    return request.param


@pytest.fixture(params=LossFunction)
def loss_function(request, family):
    if family in [DistFamily.cinterp3, DistFamily.cinterp5]:
        return None
    return request.param


@pytest.fixture
def data(question, framing):
    data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")
    data = filter_rename(data, question, framing)
    return data


def test_no_raise(data, question, framing, family, loss_function):
    # Select a subset, to make this test run in a manageable time
    # This is far from ideal, as some bugs might only appear on some input data
    if family == DistFamily.GEN_GAMMA:
        # Generalized gamma is very slow
        n = 1
    else:
        n = 50
    data = data.sample(n=n, random_state=111111)

    data = df_fit(data, question, framing, family, loss_function)
