import logging

import pandas as pd
from espai import PROJECT_ROOT
from espai import Framing, FIELDS, Question

logger = logging.getLogger(__name__)


def merged(question: Question):
    """Returns merge of the cleaned data with the previously fitted gamma curves."""

    # The ID column is `ResponseId`
    data = pd.read_csv(f"{PROJECT_ROOT}/data/combined-cleaned-personal-anon.csv")

    if question == Question.HLMI:
        # The ID column is `response.id`
        prev_fitted_gamma = pd.read_csv(
            f"{PROJECT_ROOT}/data/gamma_fitted/all_2023_hlmi_gamma_curves.csv"
        )
    elif question == Question.FAOL:
        # The ID column is `response.id`
        prev_fitted_gamma = pd.read_csv(
            f"{PROJECT_ROOT}/data/gamma_fitted/all_2023_faol_gamma_curves.csv"
        )
    else:
        raise ValueError(f"Question {question.name} does not have previous gamma distributions.")

    # Only retain these columns
    prev_fitted_gamma_cols = ["response.id", "shape", "scale", "error", "convergence"]
    prev_fitted_gamma = prev_fitted_gamma[prev_fitted_gamma_cols]

    # Match on the ID column with an indicator
    merged_data = data.merge(
        prev_fitted_gamma, left_on="ResponseId", right_on="response.id", how="outer", indicator=True
    )

    # Check for unmatched rows in both datasets
    data_unmatched = merged_data[merged_data["_merge"] == "left_only"]
    gammas_unmatched = merged_data[merged_data["_merge"] == "right_only"]

    if not data_unmatched.empty:
        logger.debug(
            f"{question}: {len(data_unmatched)}/{len(data)} rows did not have a previous gamma distribution."
        )

    if not gammas_unmatched.empty:
        raise ValueError(
            f"{question}: {len(gammas_unmatched)}/{len(prev_fitted_gamma)} gamma distributions did not have source data."
        )

    # Keep only the matched rows
    data = merged_data[merged_data["_merge"] == "both"].copy()

    # Remove the indicator column
    data = data.drop("_merge", axis=1)

    data = data.rename(
        columns={
            "shape": "prev_shape",
            "scale": "prev_scale",
            "error": "prev_error",
            "convergence": "prev_convergence",
        }
    )

    return data


def filter_rename(data: pd.DataFrame, question: Question, framing: Framing | None):
    data.index, data.index.name = data["ResponseId"], "ResponseId"

    # TODO: Our very hacky way to aggregate HLMI and FAOL is to
    #       put responses to both questions into our DataFrame, but
    #       to resample from FAOL responses so that their number
    #       matches the number of HLMI responses. This ensures that both
    #       ways of asking the question receive equal weight the aggregation,
    #       while preserving information about the distribution of responses.
    if question == Question.HLMI_FAOL:
        data_faol = filter_rename(data, Question.FAOL, framing)
        data_hlmi = filter_rename(data, Question.HLMI, framing)
        assert len(data_hlmi) > len(data_faol)

        additional_samples_needed = len(data_hlmi) - len(data_faol)
        logger.warning(
            f"For {Question.HLMI_FAOL}: Using all {len(data_faol)} original FAOL responses "
            f"plus {additional_samples_needed} 'fake' resampled responses to match {len(data_hlmi)} HLMI responses"
        )
        extra_faol = data_faol.sample(n=additional_samples_needed, random_state=12345, replace=True)

        # Add unique suffixes to resampled ResponseIds
        extra_faol.index = [f"{idx}_resampled_{i}" for i, idx in enumerate(extra_faol.index)]
        extra_faol.index.name = "ResponseId"  # slight hack to repeat this twice

        data_faol = pd.concat([data_faol, extra_faol])
        data_hlmi["question"] = Question.HLMI.value
        data_faol["question"] = Question.FAOL.value

        return pd.concat([data_hlmi, data_faol])

    if framing is None:
        data_fy = filter_rename(data, question, Framing.FY)
        data_fp = filter_rename(data, question, Framing.FP)
        data_fy["treat_assign"] = Framing.FY.value
        data_fp["treat_assign"] = Framing.FP.value
        return pd.concat([data_fy, data_fp])

    data = data[data["treat_assign"] == framing.value]

    columns = []

    # TODO: this isn't the cleanest place to do this:
    #  - readability
    #  - these will be included even when irrelevant (e.g. not fitting gammas)
    #  See ``df_calc_gamma_err``.
    if "prev_shape" in data.columns and "prev_scale" in data.columns:
        columns.extend(["prev_shape", "prev_scale"])

    framing_cols_in = FIELDS[question][framing].values()
    data = data.dropna(subset=framing_cols_in)

    rename_map = {k: v for k, v in zip(framing_cols_in, framing.human_columns)}
    data = data.rename(columns=rename_map)
    columns.extend(framing.human_columns)
    data = data[columns]
    return data
