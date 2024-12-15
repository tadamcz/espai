import pytest
from matplotlib import pyplot as plt

from espai import FIELDS, Framing
from espai.aggregate import AggMethod
from espai.analyses import (
    individual_replication,
    agg_replication_plots,
    prev_fits_bias_scatter,
    prev_fits_bias_hist,
    plot_aggregated_cdf,
)
from espai.config import Config
from espai.fit import DistFamily, LossFunction


@pytest.mark.parametrize(
    "func",
    [
        individual_replication,
        agg_replication_plots,
        prev_fits_bias_scatter,
        prev_fits_bias_hist,
    ],
)
def test_no_raise(func):
    """
    Script functions run without raising
    """
    func()


from espai import Question


@pytest.fixture(params=[Question.HLMI, Question.FAOL, Question.SURGEON, Question.TRUCK_DRIVER])
def question(request):
    return request.param


@pytest.fixture(
    params=[
        # The full cartesian product (every possible config) would be way too much
        # so just kind of arbitrarily pick a few
        Config(
            framing=Framing.FY,
            family=DistFamily.GAMMA,
            loss_function=LossFunction.CDF_MSE_PROBS,
            aggregation=AggMethod.ARITH_MEAN_PROBS,
        ),
        Config(
            framing=Framing.FP,
            family=DistFamily.cinterp5,
            loss_function=None,
            aggregation=AggMethod.MEDIAN_PROBS,
        ),
        Config(
            framing=Framing.FY,
            family=DistFamily.GAMMA,
            loss_function=LossFunction.LOG_LOSS,
            aggregation=AggMethod.MEDIAN_PROBS,
        ),
    ]
)
def config(request):
    return request.param


def test_no_raise_configs(config, question):
    fig, ax = plt.subplots()
    plot_aggregated_cdf(config, question, ax)
