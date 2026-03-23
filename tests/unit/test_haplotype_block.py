import pytest
import pandas as pd
import pandera.errors
from pandera.typing import DataFrame
from typing import cast

from panthera.core.bio.blocks import (
    HaplotypeBlock,
    VariantSchema,
    TARGET_VARIANTS,
    BACKGROUND_VARIANTS,
)
from panthera.utils.exceptions import (
    BackgroundConflictError,
    NonUniqueChromError,
    NonUniquePhaseSetTagError,
)

# ==========================================
# FIXTURES
# ==========================================


@pytest.fixture
def valid_variants_df():
    """Provides a standard variants DataFrame with two SNPs."""
    data = {
        "chrom": ["chr1", "chr1"],
        "pos": [1000, 2000],
        "ref": ["A", "C"],
        "alt": ["G", "T"],
        "genotype": ["1|0", "0|1"],
        "phase_set": ["PS1", "PS1"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def valid_background_df():
    """Provides a background DataFrame that does NOT conflict."""
    data = {
        "chrom": ["chr1", "chr1"],
        "pos": [1500, 2500],
        "ref": ["T", "G"],
        "alt": ["C", "A"],
        "genotype": ["1|1", "1|1"],
        "phase_set": ["PS1", "PS1"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def deletion_variants_df():
    """Provides a variants DataFrame with a Deletion."""
    data = {
        "chrom": ["chr1"],
        "pos": [100],
        "ref": ["ATCG"],  # Length = 4
        "alt": ["A"],  # Length = 1. Span is 100 to 103.
        "genotype": ["1|0"],
        "phase_set": ["PS1"],
    }
    return pd.DataFrame(data)


# ==========================================
# INITIALIZATION & SCHEMA TESTS
# ==========================================


def test_initialization_success(valid_variants_df):
    """Test successful initialization and metadata extraction."""
    block = HaplotypeBlock(valid_variants_df)

    assert block.chrom == "chr1"
    assert block.phaseset_tag == "PS1"
    # Ensure background column was added and assigned correctly
    assert all(block.vdf["background"] == TARGET_VARIANTS)


def test_initialization_non_unique_chrom(valid_variants_df):
    """Test that multiple chromosomes raise an error."""
    df = valid_variants_df.copy()
    df.loc[1, "chrom"] = "chr2"

    with pytest.raises(NonUniqueChromError, match="Expected one chrom"):
        HaplotypeBlock(df)


def test_initialization_non_unique_phaseset(valid_variants_df):
    """Test that multiple phase sets raise an error."""
    df = valid_variants_df.copy()
    df.loc[1, "phase_set"] = "PS2"

    with pytest.raises(NonUniquePhaseSetTagError, match="Expected one PS tag"):
        HaplotypeBlock(df)


def test_pandera_schema_validation_fails():
    """Test that bad data types or missing columns trigger Pandera schema errors."""
    bad_data = {
        "chrom": ["chr1"],
        # Missing 'pos', 'ref', 'alt'
    }
    df = pd.DataFrame(bad_data)

    with pytest.raises(pandera.errors.SchemaError):
        VariantSchema.validate(df)


# ==========================================
# NAMING TESTS
# ==========================================


@pytest.fixture
def base_variants():
    """Initial set of 2 variants."""
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [1000, 2000],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "phase_set": ["PS1", "PS1"],
        }
    )


@pytest.fixture
def bg_variants():
    """Background variant at a different position."""
    return pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [1500],
            "ref": ["T"],
            "alt": ["A"],
            "phase_set": ["PS1"],
        }
    )


def test_name_updates_after_adding_background(base_variants, bg_variants):
    """
    Ensures that calling add_background_variants automatically
    updates the name property.
    """
    block = HaplotypeBlock(base_variants)
    initial_name = block.name
    assert initial_name == "chr1-1000-A-G.chr1-2000-C-T"

    # Add background variants
    block.add_background_variants(
        cast(DataFrame[VariantSchema], bg_variants),
        "EAS",
        "HG001",
        "A",
        "WT",
        resolve_conflicts=True,
    )

    # The name should now include the new variant (sorted by position)
    expected_new_name = "chr1-1000-A-G.chr1-1500-T-A.chr1-2000-C-T"
    assert block.name == expected_new_name
    assert block.name != initial_name


def test_name_updates_after_manual_pandas_slicing(base_variants):
    """
    The 'Ultimate Test': If we bypass class methods and modify
    self.vdf directly via pandas, does the name still update?
    """
    block = HaplotypeBlock(base_variants)
    assert "chr1-2000-C-T" in block.name

    # Manually drop the second variant using standard pandas
    block.vdf = block.vdf.iloc[[0]]

    # The name should update immediately because it's a dynamic property
    assert block.name == "chr1-1000-A-G"
    assert "chr1-2000-C-T" not in block.name


def test_name_sorting_consistency(base_variants):
    """
    Ensures the name is deterministic regardless of row order
    in the input DataFrame.
    """
    # Reverse the rows
    reversed_df = base_variants.iloc[::-1].copy()
    block = HaplotypeBlock(reversed_df)

    # Name should still be sorted by position (1000 before 2000)
    assert block.name == "chr1-1000-A-G.chr1-2000-C-T"


def test_name_empty_after_clearing_vdf(base_variants):
    """Ensures name becomes empty string if data is cleared."""
    block = HaplotypeBlock(base_variants)
    assert block.name != ""

    # Clear the dataframe
    block.vdf = block.vdf.iloc[0:0]

    assert block.name == ""


# ==========================================
# CONFLICT RESOLUTION TESTS
# ==========================================


def test_add_background_no_conflict(valid_variants_df, valid_background_df):
    """Test adding background variants with no overlapping positions."""
    block = HaplotypeBlock(valid_variants_df)

    block.add_background_variants(
        cast(DataFrame[VariantSchema], valid_background_df),
        "EAS",
        "HG00512",
        "A",
        "WT",
        resolve_conflicts=False,
    )

    # vdf should now contain both target (2) and background (2) variants
    assert len(block.vdf) == 4
    assert block.population == "EAS"


def test_add_background_exact_conflict_raises_error(valid_variants_df):
    """Test that exact SNP overlaps raise BackgroundConflictError."""
    block = HaplotypeBlock(valid_variants_df)

    # Create a background df with a conflicting position (1000)
    conflict_df = valid_variants_df.copy()
    conflict_df.loc[0, "ref"] = "A"
    conflict_df.loc[0, "alt"] = "C"  # Different alt, but same position

    with pytest.raises(BackgroundConflictError, match="conflicting background variant"):
        block.add_background_variants(
            cast(DataFrame[VariantSchema], conflict_df),
            "EAS",
            "HG00512",
            "A",
            "WT",
            resolve_conflicts=False,
        )


def test_add_background_exact_conflict_resolved(valid_variants_df):
    """Test that resolve_conflicts=True drops the background variant."""
    block = HaplotypeBlock(valid_variants_df)

    # Exact duplicate position
    conflict_df = valid_variants_df.copy()

    block.add_background_variants(
        cast(DataFrame[VariantSchema], conflict_df),
        "EAS",
        "HG00512",
        "A",
        "WT",
        resolve_conflicts=True,
    )

    # The background variants should have been dropped.
    # Only the original 2 TARGET_VARIANTS should remain.
    assert len(block.vdf) == 2
    assert all(block.vdf["background"] == TARGET_VARIANTS)


def test_deletion_overlap_conflict_raises_error(deletion_variants_df):
    """
    Test that a background SNP falling INSIDE a deletion interval raises an error.
    Deletion at 100, span is 3 (100 to 103).
    Background SNP at 102 should conflict.
    """
    block = HaplotypeBlock(deletion_variants_df)

    bg_data = {
        "chrom": ["chr1", "chr1"],
        "pos": [102, 105],  # 102 conflicts, 105 is safe
        "ref": ["G", "A"],
        "alt": ["C", "T"],
        "phase_set": ["PS1", "PS1"],
    }
    bg_df = pd.DataFrame(bg_data)

    with pytest.raises(BackgroundConflictError):
        block.add_background_variants(
            cast(DataFrame[VariantSchema], bg_df),
            "EAS",
            "HG00512",
            "A",
            "WT",
            resolve_conflicts=False,
        )


def test_deletion_overlap_conflict_resolved(deletion_variants_df):
    """
    Test that a background SNP falling INSIDE a deletion interval is dropped,
    while safe background SNPs are kept.
    """
    block = HaplotypeBlock(deletion_variants_df)

    bg_data = {
        "chrom": ["chr1", "chr1"],
        "pos": [102, 105],  # 102 conflicts, 105 is safe
        "ref": ["G", "A"],
        "alt": ["C", "T"],
        "phase_set": ["PS1", "PS1"],
    }
    bg_df = pd.DataFrame(bg_data)

    block.add_background_variants(
        cast(DataFrame[VariantSchema], bg_df),
        "EAS",
        "HG00512",
        "A",
        "WT",
        resolve_conflicts=True,
    )

    # 1 target variant + 1 safe background variant = 2 total
    assert len(block.vdf) == 2

    # Ensure the remaining background variant is the safe one at pos 105
    remaining_bg = block.vdf[block.vdf["background"] == BACKGROUND_VARIANTS]
    assert remaining_bg.iloc[0]["pos"] == 105
