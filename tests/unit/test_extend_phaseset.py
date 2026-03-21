from typing import Any, cast

import pandas as pd
from pandera.typing import DataFrame

from panthera.core.extend_phaseset import extend_phaseset
from panthera.core.io import VariantSchema

# -------------------------------------------------------------------
# Fixtures & Helpers
# -------------------------------------------------------------------


def create_vdf(data: list[dict[str, Any]]) -> DataFrame[VariantSchema]:
    """
    Helper to create a typed DataFrame from a list of dictionaries.
    This resolves the 'DataFrame is not assignable to DataFrame[VariantSchema]' error
    and provides default values for required columns not specified in test cases.
    """
    defaults = {
        "chrom": "chr1",
        "ref": "A",
        "alt": "T",
        "genetic_background": "HG001",
        "phase_set": None,
    }

    # Merge defaults with provided test data
    processed_data = []
    for entry in data:
        row = defaults.copy()
        row.update(entry)
        processed_data.append(row)

    columns = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "genotype",
        "genetic_background",
        "phase_set",
    ]

    # Cast columns to Any to satisfy Pyright's strict 'Axes' requirement
    df = pd.DataFrame(processed_data, columns=cast(Any, columns))

    # Cast to the Pandera Type so Pyright is happy passing this to the function
    return cast(DataFrame[VariantSchema], df)


# -------------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------------


def test_no_extension_possible():
    """Test when no adjacent variants exist within the extension length."""
    vdf = create_vdf(
        [
            {"pos": 100, "genotype": "0|1", "phase_set": "PS1"},
            {"pos": 500, "genotype": "1|1", "phase_set": None},  # Too far
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=100)

    assert len(result) == 1
    assert result.iloc[0]["pos"] == 100
    assert result.iloc[0]["phase_set"] == "PS1"  # No extension occurred


def test_forward_extension_homozygous():
    """Test extending to a contiguous homozygous variant downstream."""
    vdf = create_vdf(
        [
            {"pos": 100, "genotype": "0|1", "phase_set": "PS1"},
            {"pos": 110, "genotype": "1|1", "phase_set": None},  # Should be included
            {"pos": 150, "genotype": "1/1", "phase_set": None},  # Too far from 110
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=20)

    assert len(result) == 2
    assert list(result["pos"]) == [100, 110]
    # Check that phase_set tag was updated due to extension
    assert all(result["phase_set"] == "PS1EXT")


def test_backward_extension_homozygous():
    """Test extending to a contiguous homozygous variant upstream."""
    vdf = create_vdf(
        [
            {"pos": 90, "genotype": "1/1", "phase_set": None},  # Should be included
            {"pos": 100, "genotype": "0|1", "phase_set": "PS1"},
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=15)

    assert len(result) == 2
    assert list(result["pos"]) == [90, 100]
    assert all(result["phase_set"] == "PS1EXT")


def test_bidirectional_extension():
    """Test extending in both directions simultaneously."""
    vdf = create_vdf(
        [
            {"pos": 85, "genotype": "1|1", "phase_set": None},  # Included
            {"pos": 100, "genotype": "0|1", "phase_set": "PS1"},
            {"pos": 115, "genotype": "1/1", "phase_set": None},  # Included
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=20)

    assert len(result) == 3
    assert list(result["pos"]) == [85, 100, 115]
    assert all(result["phase_set"] == "PS1EXT")


def test_stop_extension_at_phased_heterozygous():
    """Test that extension strictly stops at the first heterozygous variant."""
    vdf = create_vdf(
        [
            {"pos": 100, "genotype": "0|1", "phase_set": "PS1"},
            {
                "pos": 110,
                "genotype": "0|1",
                "phase_set": None,
            },  # Heterozygous (Blocks extension)
            {
                "pos": 115,
                "genotype": "1|1",
                "phase_set": None,
            },  # Hom, but blocked by 110
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=20)

    # Should only return the original phase set
    assert len(result) == 1
    assert result.iloc[0]["pos"] == 100
    assert result.iloc[0]["phase_set"] == "PS1"


def test_stop_extension_at_unphased_heterozygous():
    """Test that extension strictly stops at the first heterozygous variant."""
    vdf = create_vdf(
        [
            {"pos": 100, "genotype": "0|1", "phase_set": "PS1"},
            {
                "pos": 110,
                "genotype": "0/1",
                "phase_set": None,
            },  # Heterozygous (Blocks extension)
            {
                "pos": 115,
                "genotype": "1|1",
                "phase_set": None,
            },  # Hom, but blocked by 110
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=20)

    # Should only return the original phase set
    assert len(result) == 1
    assert result.iloc[0]["pos"] == 100
    assert result.iloc[0]["phase_set"] == "PS1"


def test_stop_extension_at_distance_limit():
    """Test that extension respects the exact extension length limit."""
    vdf = create_vdf(
        [
            {"pos": 100, "genotype": "0|1", "phase_set": "PS1"},
            {
                "pos": 121,
                "genotype": "1|1",
                "phase_set": None,
            },  # Distance is 21 (limit is 20)
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=20)

    assert len(result) == 1
    assert result.iloc[0]["pos"] == 100


def test_multiple_contiguous_extensions():
    """Test iterative extension across multiple valid homozygous variants."""
    vdf = create_vdf(
        [
            {"pos": 100, "genotype": "0|1", "phase_set": "PS1"},
            {"pos": 110, "genotype": "1|1", "phase_set": None},  # Dist 10 from PS1
            {"pos": 120, "genotype": "1/1", "phase_set": None},  # Dist 20 from PS1
            {"pos": 130, "genotype": "1|1", "phase_set": None},  # Dist 30 from PS1
            {
                "pos": 140,
                "genotype": "1|1",
                "phase_set": None,
            },  # Dist 40 from PS1; ignored
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=30)

    assert len(result) == 4
    assert list(result["pos"]) == [100, 110, 120, 130]


def test_ignore_other_chromosomes():
    """Test that variants on other chromosomes are strictly ignored."""
    vdf = create_vdf(
        [
            {"chrom": "chr1", "pos": 100, "genotype": "0|1", "phase_set": "PS1"},
            {
                "chrom": "chr2",
                "pos": 105,
                "genotype": "1|1",
                "phase_set": None,
            },  # Different chrom
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=20)

    assert len(result) == 1
    assert result.iloc[0]["pos"] == 100


def test_phase_set_not_found():
    """Test behavior when the target phase set does not exist."""
    vdf = create_vdf(
        [
            {"pos": 100, "genotype": "0|1", "phase_set": "PS_OTHER"},
        ]
    )

    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=20)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_empty_dataframe():
    """Test behavior with an empty valid dataframe."""
    # Create empty df but with valid schema columns
    empty_df = pd.DataFrame(
        columns=cast(
            Any,
            (
                "chrom",
                "pos",
                "ref",
                "alt",
                "genotype",
                "genetic_background",
                "phase_set",
            ),
        )
    )
    empty_df = empty_df.astype({"pos": "int64"})  # enforce type for Pandera
    empty_df = cast(DataFrame[VariantSchema], empty_df)

    vdf = VariantSchema.validate(empty_df)
    result = extend_phaseset(vdf, chrom="chr1", ps_id="PS1", ext_len=20)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
