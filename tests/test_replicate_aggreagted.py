import numpy as np
import pandas as pd
import pytest

from espai import PROJECT_ROOT, Question
from espai.aggregate import summarize_dists, AggMethod
from espai.fit import df_fit, DistFamily, LossFunction
from espai import Framing
from espai.preprocess import filter_rename


@pytest.mark.parametrize("question", [Question.HLMI, Question.FAOL])
def test(question):
    data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")
    data = filter_rename(data, question, None)
    data = df_fit(data, question, None, DistFamily.GAMMA, LossFunction.CDF_MSE_PROBS)

    # Check all NaNs are for legitimate reasons
    nans = data[data["shape"].isna() | data["scale"].isna()]
    assert (nans["fit_msg"] == "data_NaNs").all()

    # Then drop NaNs
    data = data.dropna(subset=["shape", "scale"])

    previous_aggregated = pd.read_csv(
        PROJECT_ROOT / "data" / "aggregated" / f"{question.name}_2023.csv"
    )

    # Significant differences can be expected because:
    #    - we may have found different shape/scale parameters (that improve the fit,
    #      see ``test_no_error_increase``)
    #    - the previous analysis may have dropped data that we don't drop
    #      (TODO: it may be possible to investigate this by inspecting the responseIDs, though
    #        it's probably quite painstaking to guess the reasons why 1000+ rows were dropped)
    # These differences tend to appear in the tails (because they're often due to weird responses)
    # So, we don't check the tails, and we use a tolerance.
    rows_notail = previous_aggregated[previous_aggregated["y"].between(0.05, 0.95)]
    if question == Question.HLMI:
        rtol = 2 / 100
    elif question == Question.FAOL:
        # TODO: We need a larger tolerance for FAOL (though still low enough for the replication
        #  to be meaningful). Investigate this if/when we get access to the previous gamma fits
        #  for FAOL, it's pretty hopeless to debug the aggregated CDFs without the individual fits.
        rtol = 4 / 100

    years = rows_notail["x"]
    mean_cdf = summarize_dists(
        data, years, dist_col="fitted_dist", method=AggMethod.ARITH_MEAN_PROBS
    )["aggregated"]

    # Avoid indices mismatch for comparison
    mean_cdf = np.array(mean_cdf)

    assert mean_cdf == pytest.approx(rows_notail["y"], rel=rtol)
