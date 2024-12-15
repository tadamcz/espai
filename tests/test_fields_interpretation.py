import pandas as pd
import pytest

from espai import FIELDS, Framing
from espai import PROJECT_ROOT


from espai import Question

# Exclude HLMI_FAOL which isn't a real question present in the data but our own
# hack for aggregating HLMI and FAOL
data_questions = [q for q in Question if q != Question.HLMI_FAOL]


@pytest.mark.parametrize("question", data_questions)
def test_a_b_fields(question):
    """
    `treat_assign` can be either "Fixed probabilities" or "Fixed years".

    I'm inferring from the data that fields with 'a' (e.g. `hb_a_1`, `hb_a_2`, `hb_a_3`) correspond
    to the "Fixed probabilities" framing, and fields with 'b' (e.g. `hb_b_1`, `hb_b_2`, `hb_b_3`)
    correspond to the "Fixed years" framing.
    """

    data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")

    fy = data[data["treat_assign"] == Framing.FY.value]
    fp = data[data["treat_assign"] == Framing.FP.value]

    fy_fields = list(FIELDS[question][Framing.FY].values())
    fp_fields = list(FIELDS[question][Framing.FP].values())

    def count_numeric(series):
        return series.apply(pd.to_numeric, errors="coerce").notna().sum()

    MIN_NUMERIC_ROWS = 50

    for fy_field in fy_fields:
        # fp is all NaN for the FY fields
        assert fp[fy_field].isna().all()
        # fy has some numeric values for the FY fields
        assert count_numeric(fy[fy_field]) >= MIN_NUMERIC_ROWS

    for fp_field in fp_fields:
        # fy is all NaN for the FP fields
        assert fy[fp_field].isna().all()
        # fp has some numeric values for the FP fields
        assert count_numeric(fp[fp_field]) >= MIN_NUMERIC_ROWS


@pytest.mark.parametrize("question", data_questions)
def test_probabilities_0_100(question):
    """The source data expresses probabilities as values between 0 and 100."""

    data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")

    probability_fields = list(FIELDS[question][Framing.FY].values())

    # All the percentages is between 0 and 100
    for field in probability_fields:
        coldata = data[field].dropna()
        # There is one wrong data point in each of 'Atari games novice',
        # 'ML study replication', 'Website creation'
        MAX_WRONG = 1
        assert (~coldata.between(0, 100)).sum() <= MAX_WRONG

    # Less than 90% of the percentages are between 0 and 1
    for field in probability_fields:
        coldata = data[field].dropna()
        assert coldata.between(0, 1).sum() < 0.90 * len(coldata)


@pytest.mark.parametrize("question", data_questions)
def test_years(question):
    data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")

    year_fields = list(FIELDS[question][Framing.FP].values())

    # All the years are positive
    for field in year_fields:
        coldata = data[field].dropna()
        assert (coldata < 0).sum() == 0

    # Less than 90% of the years are between 0 and 1
    for field in year_fields:
        coldata = data[field].dropna()
        assert coldata.between(0, 1).sum() < 0.90 * len(coldata)
