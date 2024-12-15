import pandas as pd
import pytest
from espai import PROJECT_ROOT, Question, Framing
from espai.preprocess import filter_rename


@pytest.fixture
def data():
    return pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")


def test_hlmi_faol_size_matches_hlmi(data):
    """The combined dataset should have twice the number of HLMI responses"""
    hlmi = filter_rename(data, Question.HLMI, Framing.FY)
    combined = filter_rename(data, Question.HLMI_FAOL, Framing.FY)
    assert len(combined) == 2 * len(hlmi)


def test_hlmi_faol_all_original_responses_included(data):
    """All original FAOL responses should be present in the combined dataset"""
    faol = filter_rename(data, Question.FAOL, Framing.FY)
    combined = filter_rename(data, Question.HLMI_FAOL, Framing.FY)

    original_faol_ids = set(faol.index)
    combined_faol_ids = {
        idx.replace("_resampled", "")
        for idx in combined[combined["question"] == Question.FAOL.value].index
    }

    assert original_faol_ids.issubset(combined_faol_ids)


def test_hlmi_faol_question_labels(data):
    """Each row should be labeled with its source question"""
    combined = filter_rename(data, Question.HLMI_FAOL, Framing.FY)

    # Check we have both question types
    questions = combined["question"].unique()
    assert len(questions) == 2
    assert Question.HLMI.value in questions
    assert Question.FAOL.value in questions

    # Check counts match
    hlmi_count = (combined["question"] == Question.HLMI.value).sum()
    faol_count = (combined["question"] == Question.FAOL.value).sum()
    assert hlmi_count == faol_count


def test_hlmi_faol_resampled_ids(data):
    """Resampled rows should have modified ResponseIds"""
    combined = filter_rename(data, Question.HLMI_FAOL, Framing.FY)

    faol_rows = combined[combined["question"] == Question.FAOL.value]
    resampled = faol_rows[faol_rows.index.str.contains("_resampled")]
    original = faol_rows[~faol_rows.index.str.contains("_resampled")]

    # Should have some resampled rows
    assert len(resampled) > 0
    # Original rows should match original FAOL data
    assert len(original) == len(filter_rename(data, Question.FAOL, Framing.FY))
    # All indices should be unique
    assert len(combined.index) == len(combined.index.unique())


def test_hlmi_faol_reproducibility(data):
    """The resampling should be reproducible"""
    result1 = filter_rename(data, Question.HLMI_FAOL, Framing.FY)
    result2 = filter_rename(data, Question.HLMI_FAOL, Framing.FY)
    pd.testing.assert_frame_equal(result1, result2)


def test_hlmi_faol_both_framings(data):
    data_fy = filter_rename(data, Question.HLMI_FAOL, Framing.FY)
    data_fp = filter_rename(data, Question.HLMI_FAOL, Framing.FP)
    data_all = filter_rename(data, Question.HLMI_FAOL, None)

    assert len(data_all) == len(data_fy) + len(data_fp)
