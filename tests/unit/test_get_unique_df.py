import pytest
import pandas as pd
import numpy as np

from panthera.utils.get_unique_df import get_unique_df


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture
def base_df() -> pd.DataFrame:
    """A standard DataFrame for baseline testing."""
    return pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})


@pytest.fixture
def exact_duplicate_df() -> pd.DataFrame:
    """An exact duplicate of base_df, created independently."""
    return pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})


@pytest.fixture
def diff_data_df() -> pd.DataFrame:
    """DataFrame with the same shape/columns, but different data."""
    return pd.DataFrame({"A": [1, 99, 3], "B": ["x", "y", "z"]})


@pytest.fixture
def diff_index_df() -> pd.DataFrame:
    """DataFrame with identical data, but a shifted index."""
    return pd.DataFrame(
        {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=pd.Index([10, 20, 30])
    )


@pytest.fixture
def diff_columns_df() -> pd.DataFrame:
    """DataFrame with identical data, but different column names."""
    return pd.DataFrame({"C": [1, 2, 3], "D": ["x", "y", "z"]})


@pytest.fixture
def nan_df_1() -> pd.DataFrame:
    """DataFrame containing NaN values."""
    return pd.DataFrame({"A": [1, np.nan, 3], "B": ["x", None, "z"]})


@pytest.fixture
def nan_df_2() -> pd.DataFrame:
    """Duplicate of nan_df_1 to ensure NaNs hash deterministically."""
    return pd.DataFrame({"A": [1, np.nan, 3], "B": ["x", None, "z"]})


# ==========================================
# TEST CASES
# ==========================================


def test_empty_list():
    """Edge Case: Passing an empty list should return an empty list."""
    assert get_unique_df([]) == []


def test_single_dataframe(base_df):
    """Edge Case: Passing a single DataFrame should return that DataFrame."""
    result = get_unique_df([base_df])
    assert len(result) == 1
    pd.testing.assert_frame_equal(result[0], base_df)


def test_all_duplicates(base_df, exact_duplicate_df):
    """Happy Path: List of identical DataFrames should collapse to one."""
    input_list = [base_df, exact_duplicate_df, exact_duplicate_df]
    result = get_unique_df(input_list)

    assert len(result) == 1
    # Verify the returned DataFrame is structurally identical to the input
    pd.testing.assert_frame_equal(result[0], base_df)


def test_mixed_uniques(base_df, exact_duplicate_df, diff_data_df):
    """Happy Path: Mix of duplicates and unique DataFrames."""
    input_list = [base_df, diff_data_df, exact_duplicate_df]
    result = get_unique_df(input_list)

    assert len(result) == 2
    pd.testing.assert_frame_equal(result[0], base_df)
    pd.testing.assert_frame_equal(result[1], diff_data_df)


def test_index_sensitivity(base_df, diff_index_df):
    """Edge Case: Same data but different index MUST be treated as unique."""
    input_list = [base_df, diff_index_df]
    result = get_unique_df(input_list)

    assert len(result) == 2


def test_column_name_sensitivity(base_df, diff_columns_df):
    """Edge Case: Same data but different column names MUST be treated as unique."""
    input_list = [base_df, diff_columns_df]
    result = get_unique_df(input_list)

    assert len(result) == 2


def test_nan_handling(nan_df_1, nan_df_2):
    """
    Edge Case: DataFrames with NaN/None values.
    NaNs can sometimes break equality checks (np.nan != np.nan).
    Hashing must handle them deterministically.
    """
    input_list = [nan_df_1, nan_df_2]
    result = get_unique_df(input_list)

    assert len(result) == 1
    pd.testing.assert_frame_equal(result[0], nan_df_1)


def test_preserves_original_order(base_df, diff_data_df, diff_index_df):
    """Contract Check: The function must preserve the first-seen insertion order."""
    input_list = [diff_index_df, base_df, diff_index_df, diff_data_df, base_df]
    result = get_unique_df(input_list)

    assert len(result) == 3
    # Ensure they appear in the exact order they were first introduced
    pd.testing.assert_frame_equal(result[0], diff_index_df)
    pd.testing.assert_frame_equal(result[1], base_df)
    pd.testing.assert_frame_equal(result[2], diff_data_df)


def test_different_dtypes():
    """Edge Case: Same numeric values, but different datatypes (int vs float)."""
    df_int = pd.DataFrame({"A": [1, 2, 3]})
    df_float = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

    result = get_unique_df([df_int, df_float])

    # Pandas hashing differentiates between int and float types
    assert len(result) == 2
