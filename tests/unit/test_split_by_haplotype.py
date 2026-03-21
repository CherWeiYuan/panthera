import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from pandera.typing import DataFrame
from typing import cast
from unittest.mock import patch

from panthera.core.input import VariantSchema
from panthera.core.split import split_by_haplotype


# ---------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------


@pytest.fixture
def standard_vdf():
    """Fixture providing a DataFrame that strictly matches VariantSchema."""
    data = {
        "chrom": ["1", "2", "3", "4", "5", "6", "22"],
        "pos": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "ref": ["A", "C", "G", "T", "A", "C", "G"],
        "alt": ["T", "G", "A", "C", "T", "G", "A"],
        "genotype": ["1|0", "0|1", "1|1", "0|0", "0/1", "1/1", "missing"],
        "genetic_background": ["EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS"],
    }
    # Validate the mock data against the schema right away to ensure test integrity
    df = pd.DataFrame(data)
    return VariantSchema.validate(df)


# ---------------------------------------------------------
# Test Cases
# ---------------------------------------------------------


def test_split_by_haplotype_standard(standard_vdf):
    """Test that haplotypes are split accurately while preserving schema and original indices."""
    hap_a, hap_b = split_by_haplotype(standard_vdf)

    # Expected Haplotype A: Indices 0 ('1|0') and 2 ('1|1')
    expected_a = standard_vdf.iloc[[0, 2]].copy()

    # Expected Haplotype B: Indices 1 ('0|1') and 2 ('1|1')
    expected_b = standard_vdf.iloc[[1, 2]].copy()

    assert_frame_equal(hap_a, expected_a)
    assert_frame_equal(hap_b, expected_b)


def test_invalid_input_type():
    """Test that a TypeError is raised if the input is not a pandas DataFrame."""
    invalid_input = "not_a_dataframe"

    # Add 'type: ignore' because we are intentionally violating the type hint
    with pytest.raises(TypeError, match="Expected pandas DataFrame, got str"):
        split_by_haplotype(invalid_input)  # type: ignore


@patch("panthera.core.split.logger.error")
def test_missing_genotype_column_and_logger(mock_logger_error):
    """Test that missing 'genotype' raises KeyError AND triggers the expected logger error."""
    # Create a DataFrame missing the genotype column
    bad_data = {
        "chrom": ["1"],
        "pos": [1000],
        "ref": ["A"],
        "alt": ["T"],
        "genetic_background": ["EUR"],
    }
    bad_df = pd.DataFrame(bad_data)
    bad_df = cast(DataFrame[VariantSchema], pd.DataFrame(bad_data))

    with pytest.raises(
        KeyError, match="The input DataFrame must contain a 'genotype' column."
    ):
        split_by_haplotype(bad_df)

    # Verify the logger was called exactly once with the specific message
    # The @patch above def creates a mock ("fake") logger.error
    # When split_by_haplotype() crashes, the function talks to the mock logger.error
    # assert_called_once_with() checks if the function did talk to the mock logger
    mock_logger_error.assert_called_once_with(
        "Failed to split haplotypes: 'genotype' column missing."
    )


def test_empty_dataframe():
    """Test behavior with an empty DataFrame that correctly matches the schema."""
    # Generate an empty DataFrame with the correct schema typing
    empty_data = {
        "chrom": pd.Series([], dtype="str"),
        "pos": pd.Series([], dtype="int64"),
        "ref": pd.Series([], dtype="str"),
        "alt": pd.Series([], dtype="str"),
        "genotype": pd.Series([], dtype="str"),
        "genetic_background": pd.Series([], dtype="str"),
    }
    empty_df = VariantSchema.validate(pd.DataFrame(empty_data))

    hap_a, hap_b = split_by_haplotype(empty_df)

    assert_frame_equal(hap_a, empty_df)
    assert_frame_equal(hap_b, empty_df)


def test_no_valid_genotypes(standard_vdf):
    """Test when the DataFrame has no valid phased genotypes to extract."""
    # Filter the standard fixture to only include unphased/hom-ref genotypes
    no_valid_df = standard_vdf[
        standard_vdf["genotype"].isin(["0|0", "0/1", "1/1", "missing"])
    ].copy()

    hap_a, hap_b = split_by_haplotype(no_valid_df)

    # Expected output is empty DataFrames maintaining the schema structure
    expected_empty = no_valid_df.iloc[0:0].copy()

    assert_frame_equal(hap_a, expected_empty)
    assert_frame_equal(hap_b, expected_empty)
