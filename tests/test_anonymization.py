"""This file also serves as a record of the anonymization process."""
import re

import pandas as pd
from espai import PROJECT_ROOT


def get_redacted_cols():
    # Load the public spreadsheet
    df = pd.read_csv(PROJECT_ROOT / "data" / "2023 ESPAI - Cleaned and Anonymized Responses.csv")

    # Identify columns that consist entirely of the string "redacted"
    return df.columns[df.apply(lambda col: (col == "redacted").all())]


def standardize(col: str) -> str:
    special_chars = ["_", ".", " "]

    for char in special_chars:
        col = col.replace(char, "")

    return col.lower()


def pattern_drop(col: str) -> bool:
    """
    Drop columns if they meet these manually specified patterns.
    """
    contains = [
        "comment",
        "consider",
        "First.Click",
        "Last.Click",
        "Click.Count",
        "Page.Submit",
    ]

    suffixes = [
        "Click",
        "Submit",
        "Count",
        "final",
        "TEXT",
    ]
    prefixes = [
        "ms_",
        "dem_",
    ]

    exact = [
        "longest_area",
        "time_in_area",
        "hh_area",
    ]

    for pattern in contains:
        if standardize(pattern) in standardize(col):
            return True

    for pattern in suffixes:
        if standardize(col).endswith(standardize(pattern)):
            return True

    for pattern in prefixes:
        if standardize(col).startswith(standardize(pattern)):
            return True

    for pattern in exact:
        if standardize(col) == standardize(pattern):
            return True

    # Match capital Q followed by numbers or special characters
    pattern = re.compile(r"Q[\d\W]+")
    if pattern.match(col):
        return True

    return False


def test():
    data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")

    redacted_cols = set(get_redacted_cols())

    # We need the ResponseId even though it was redacted in the public spreadsheet
    redacted_cols.remove("ResponseId")

    for data_col in data.columns:
        for redacted_col in redacted_cols:
            # Fail if any column meets this loose sense of equality.
            # Needed because the data sources are inconsistent in their handling of special characters.
            assert not standardize(redacted_col) == standardize(data_col)

        # Fail if any column meets these manually specified patterns.
        assert not pattern_drop(data_col)

    assert "ResponseId" in data.columns
