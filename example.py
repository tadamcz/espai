import logging

import matplotlib.pyplot as plt

from espai import project_logger, Question, Framing
from espai.aggregate import AggMethod
from espai.analyses import compare_configurations
from espai.config import Config
from espai.fit import DistFamily, LossFunction

project_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s | %(name)s] %(message)s"))
project_logger.addHandler(handler)

configs = [
    Config(
        framing=Framing.FP,
        family=DistFamily.GAMMA,
        loss_function=LossFunction.CDF_MSE_PROBS,
        aggregation=AggMethod.MEDIAN_PROBS,
    ),
    Config(
        framing=Framing.FY,
        family=DistFamily.GAMMA,
        loss_function=LossFunction.CDF_MSE_PROBS,
        aggregation=AggMethod.MEDIAN_PROBS,
    ),
]

fig, ax = plt.subplots()
compare_configurations(configs, Question.FAOL, ax)
fig.show()
