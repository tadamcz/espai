import numpy as np
import pandas as pd
import pytest

from espai import PROJECT_ROOT, Question
from espai.fit import df_fit, DistFamily, LossFunction
from espai.preprocess import filter_rename
from espai.aggregate import summarize_dists, AggMethod


@pytest.fixture
def fitted_data():
    data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")

    # Just pick some values, it shouldn't matter which
    question = Question.HLMI
    framing = None
    family = DistFamily.GAMMA
    loss_function = LossFunction.CDF_MSE_PROBS

    data = filter_rename(data, question, framing)
    data = df_fit(data, question, framing, family, loss_function)
    data = data.dropna(subset=["fitted_dist"])
    return data


@pytest.fixture(params=AggMethod)
def method(request):
    return request.param


def test_no_raise(fitted_data, method):
    if method.vertical:
        evaluate_at = np.linspace(0, 300)
    elif method.horizontal:
        evaluate_at = np.linspace(0, 1)

    fitted_col = "fitted_dist"
    summarize_dists(fitted_data, evaluate_at, fitted_col, method)["aggregated"]
