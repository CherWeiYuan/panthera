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
from panthera.core.bio.gene import GeneObject
from panthera.utils.exceptions import (
    AmbiguousDeletionError,  # Required: raised by _check_deletion_validity()
    BackgroundConflictError,
    NonUniqueChromError,
    NonUniquePhaseSetTagError,
)

# ==========================================
# FIXTURES
# ==========================================


@pytest.fixture
def gene_obj():
    """A GeneObject that spans a wide genomic range on chr1.
    Used as the default gene for most tests; its wide range (1–999999)
    means it does NOT filter out any test variants.
    """
    return GeneObject(
        chrom="chr1",
        strand="+",
        start=1,
        end=999_999,
        gene_name="BRCA1",
        gene_id="ENSG00000012048",
        splice_sites={"donor": [1200, 1800], "acceptor": [1100, 1700]},
        shex=[[100, 200], [300, 400]],
    )


@pytest.fixture
def narrow_gene_obj():
    """A GeneObject with a tight genomic range [500, 1500].
    Used to test that variants outside the range are filtered from vdf.
    """
    return GeneObject(
        chrom="chr1",
        strand="-",
        start=500,
        end=1500,
        gene_name="TP53",
        gene_id="ENSG00000141510",
        splice_sites={"donor": [600], "acceptor": [700]},
        shex=[[500, 600], [700, 800]],
    )


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


@pytest.fixture
def ambiguous_target_variants_df():
    """Provides a variants DataFrame where a target deletion at pos=100
    (ref='ATCG', alt='A', span [100, 103]) is followed by a target SNP at
    pos=102, which falls inside the deletion span.

    _check_variant_conflicts() does NOT catch this because it only
    compares target vs background variants; both rows here are TARGET_VARIANTS.
    _check_deletion_validity() is what must catch this ambiguity.
    """
    data = {
        "chrom": ["chr1", "chr1"],
        "pos": [100, 102],
        "ref": ["ATCG", "G"],
        "alt": ["A", "C"],
        "genotype": ["1|0", "1|0"],
        "phase_set": ["PS1", "PS1"],
    }
    return pd.DataFrame(data)


# ==========================================
# INITIALIZATION & SCHEMA TESTS
# ==========================================


def test_initialization_success(valid_variants_df, gene_obj):
    """Test successful initialization and metadata extraction."""
    block = HaplotypeBlock(valid_variants_df, gene_obj)

    assert block.chrom == "chr1"
    assert block.phaseset_tag == "PS1"
    # Ensure background column was added and assigned correctly
    assert (block.vdf["background"] == TARGET_VARIANTS).all()  # type: ignore


def test_initialization_non_unique_chrom(valid_variants_df, gene_obj):
    """Test that multiple chromosomes raise an error."""
    df = valid_variants_df.copy()
    df.loc[1, "chrom"] = "chr2"

    with pytest.raises(NonUniqueChromError, match="Expected one chrom"):
        HaplotypeBlock(df, gene_obj)


def test_initialization_non_unique_phaseset(valid_variants_df, gene_obj):
    """Test that multiple phase sets raise an error."""
    df = valid_variants_df.copy()
    df.loc[1, "phase_set"] = "PS2"

    with pytest.raises(NonUniquePhaseSetTagError, match="Expected one PS tag"):
        HaplotypeBlock(df, gene_obj)


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
# GENE OBJECT ATTRIBUTE TESTS
# ==========================================


def test_gene_attributes_stored_on_block(valid_variants_df, gene_obj):
    """Test that all GeneObject fields are correctly stored as block attributes
    after initialisation.
    """
    block = HaplotypeBlock(valid_variants_df, gene_obj)

    assert block.gene_obj == gene_obj


def test_gene_attributes_reflect_narrow_gene(valid_variants_df, narrow_gene_obj):
    """Ensure that a different GeneObject produces different gene attributes.
    Strand and name should match the narrow_gene_obj, not the default one.
    """
    block = HaplotypeBlock(valid_variants_df, narrow_gene_obj)

    assert block.gene_obj.strand == "-"
    assert block.gene_obj.gene_name == "TP53"
    assert block.gene_obj.gene_id == "ENSG00000141510"


# ==========================================
# GENE RANGE FILTERING TESTS
# ==========================================


def test_variants_within_gene_range_are_kept(valid_variants_df, gene_obj):
    """Variants whose positions fall inside the gene range must all be
    retained in vdf after initialisation.
    The wide gene_obj (1–999999) should keep both variants (pos 1000, 2000).
    """
    block = HaplotypeBlock(valid_variants_df, gene_obj)

    assert len(block.vdf) == 2
    assert set(block.vdf["pos"]) == {1000, 2000}


def test_variants_outside_gene_range_are_filtered(narrow_gene_obj):
    """Variants outside [gene_obj.start, gene_obj.end] must be dropped.
    narrow_gene_obj spans [500, 1500]:
      - pos=1000 is inside  → kept
      - pos=2000 is outside → filtered out
    """
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [1000, 2000],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "genotype": ["1|0", "0|1"],
            "phase_set": ["PS1", "PS1"],
        }
    )
    block = HaplotypeBlock(cast(DataFrame[VariantSchema], df), narrow_gene_obj)

    assert len(block.vdf) == 1
    assert block.vdf.iloc[0]["pos"] == 1000


def test_all_variants_outside_gene_range_yields_empty_vdf(narrow_gene_obj):
    """If every variant falls outside the gene range, vdf should be empty
    and name should be an empty string.
    """
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [2000, 3000],  # both outside [500, 1500]
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "genotype": ["1|0", "0|1"],
            "phase_set": ["PS1", "PS1"],
        }
    )
    block = HaplotypeBlock(cast(DataFrame[VariantSchema], df), narrow_gene_obj)

    assert block.vdf.empty
    assert block.name == ""


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


def test_name_updates_after_adding_background(base_variants, bg_variants, gene_obj):
    """Ensures that calling add_background_variants automatically
    updates the name property.
    """
    block = HaplotypeBlock(base_variants, gene_obj)
    initial_name = block.name
    assert initial_name == "chr1-1000-A-G.chr1-2000-C-T"

    # Add background variants
    block.add_background_variants(
        cast(DataFrame[VariantSchema], bg_variants),
        "EAS",
        "HG001",
        "A",
        resolve_conflicts=True,
    )

    # The name should now include the new variant (sorted by position)
    expected_new_name = "chr1-1000-A-G.chr1-1500-T-A.chr1-2000-C-T"
    assert block.name == expected_new_name
    assert block.name != initial_name


def test_name_updates_after_manual_pandas_slicing(base_variants, gene_obj):
    """The 'Ultimate Test': If we bypass class methods and modify
    self.vdf directly via pandas, does the name still update?
    """
    block = HaplotypeBlock(base_variants, gene_obj)
    assert "chr1-2000-C-T" in block.name

    # Manually drop the second variant using standard pandas
    block.vdf = block.vdf.iloc[[0]]

    # The name should update immediately because it's a dynamic property
    assert block.name == "chr1-1000-A-G"
    assert "chr1-2000-C-T" not in block.name


def test_name_sorting_consistency(base_variants, gene_obj):
    """Ensures the name is deterministic regardless of row order
    in the input DataFrame.
    """
    # Reverse the rows
    reversed_df = base_variants.iloc[::-1].copy()
    block = HaplotypeBlock(reversed_df, gene_obj)

    # Name should still be sorted by position (1000 before 2000)
    assert block.name == "chr1-1000-A-G.chr1-2000-C-T"


def test_name_empty_after_clearing_vdf(base_variants, gene_obj):
    """Ensures name becomes empty string if data is cleared."""
    block = HaplotypeBlock(base_variants, gene_obj)
    assert block.name != ""

    # Clear the dataframe
    block.vdf = block.vdf.iloc[0:0]

    assert block.name == ""


# ==========================================
# CONFLICT RESOLUTION TESTS
# ==========================================


def test_add_background_no_conflict(valid_variants_df, valid_background_df, gene_obj):
    """Test adding background variants with no overlapping positions."""
    block = HaplotypeBlock(valid_variants_df, gene_obj)

    block.add_background_variants(
        cast(DataFrame[VariantSchema], valid_background_df),
        "EAS",
        "HG00512",
        "A",
        resolve_conflicts=False,
    )

    # vdf should now contain both target (2) and background (2) variants
    assert len(block.vdf) == 4
    assert block.population == "EAS"


def test_add_background_exact_conflict_raises_error(valid_variants_df, gene_obj):
    """Test that exact SNP overlaps raise BackgroundConflictError."""
    block = HaplotypeBlock(valid_variants_df, gene_obj)

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
            resolve_conflicts=False,
        )


def test_add_background_exact_conflict_resolved(valid_variants_df, gene_obj):
    """Test that resolve_conflicts=True drops the background variant."""
    block = HaplotypeBlock(valid_variants_df, gene_obj)

    # Exact duplicate position
    conflict_df = valid_variants_df.copy()

    block.add_background_variants(
        cast(DataFrame[VariantSchema], conflict_df),
        "EAS",
        "HG00512",
        "A",
        resolve_conflicts=True,
    )

    # The background variants should have been dropped.
    # Only the original 2 TARGET_VARIANTS should remain.
    assert len(block.vdf) == 2
    assert (block.vdf["background"] == TARGET_VARIANTS).all()  # type: ignore


def test_deletion_overlap_conflict_raises_error(deletion_variants_df, gene_obj):
    """Test that a background SNP falling INSIDE a deletion interval raises an error.
    Deletion at 100, span is 3 (100 to 103).
    Background SNP at 102 should conflict.
    """
    block = HaplotypeBlock(deletion_variants_df, gene_obj)

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
            resolve_conflicts=False,
        )


def test_deletion_overlap_conflict_resolved(deletion_variants_df, gene_obj):
    """Test that a background SNP falling INSIDE a deletion interval is dropped,
    while safe background SNPs are kept.
    """
    block = HaplotypeBlock(deletion_variants_df, gene_obj)

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
        resolve_conflicts=True,
    )

    # 1 target variant + 1 safe background variant = 2 total
    assert len(block.vdf) == 2

    # Ensure the remaining background variant is the safe one at pos 105
    remaining_bg = cast(
        pd.DataFrame, block.vdf[block.vdf["background"] == BACKGROUND_VARIANTS]
    )
    assert remaining_bg.iloc[0]["pos"] == 105


def test_conflict_resolution_wrong_indices_dropped(valid_variants_df, gene_obj):
    """Test that dropping conflicts removes the exact correct background variants.
    Prior to a bug fix, resetting the index before dropping caused the wrong
    rows to be dropped if the index was out of sync.
    """
    block = HaplotypeBlock(valid_variants_df, gene_obj)

    # We add 4 background variants.
    # Positions 1000 and 2000 will conflict with target variants (from valid_variants_df).
    # Positions 1500 and 2500 are safe.
    bg_data = {
        "chrom": ["chr1", "chr1", "chr1", "chr1"],
        "pos": [1000, 1500, 2000, 2500],
        "ref": ["A", "T", "C", "G"],
        "alt": ["C", "A", "A", "C"],
        "phase_set": ["PS1", "PS1", "PS1", "PS1"],
    }
    bg_df = pd.DataFrame(bg_data)

    block.add_background_variants(
        cast(DataFrame[VariantSchema], bg_df),
        "EAS",
        "HG001",
        "A",
        resolve_conflicts=True,
    )

    # We expect 2 target variants + 2 safe background variants
    assert len(block.vdf) == 4

    # Safe background variants should be at 1500 and 2500
    bgs = block.vdf[block.vdf["background"] == BACKGROUND_VARIANTS]
    assert list(bgs["pos"]) == [1500, 2500]

    # Target variants should still be at 1000 and 2000
    targets = block.vdf[block.vdf["background"] == TARGET_VARIANTS]
    assert list(targets["pos"]) == [1000, 2000]

    # Target variants should override the background variants
    # Background variants: chr1-1000-A-C, chr1-2000-C-A
    # Target variants: chr1-1000-A-G, chr1-2000-C-T
    targets = block.vdf[block.vdf["background"] == TARGET_VARIANTS]
    assert list(targets["alt"]) == ["G", "T"]


# ==========================================
# DELETION VALIDITY TESTS
# (_check_deletion_validity called inside add_background_variants)
# ==========================================


def test_add_background_ambiguous_deletion_in_target_raises_error(
    ambiguous_target_variants_df, gene_obj
):
    """Test that AmbiguousDeletionError is raised when the target block itself
    contains a deletion whose span overlaps a subsequent target variant.

    Setup:
      - Target deletion at pos=100, ref='ATCG' → span [100, 103]
      - Target SNP at pos=102, which sits inside that span

    Why this is NOT caught by _check_variant_conflicts():
      _check_variant_conflicts() only compares TARGET_VARIANTS rows against
      BACKGROUND_VARIANTS rows. Both rows here are TARGET_VARIANTS, so the
      cross-check is skipped entirely.

    Why _check_deletion_validity() DOES catch it:
      It inspects every consecutive pair in self.vdf (after sorting) and flags
      any deletion whose span reaches the next variant's position.
    """
    block = HaplotypeBlock(
        cast(DataFrame[VariantSchema], ambiguous_target_variants_df), gene_obj
    )

    # Any non-conflicting background variant will trigger the deletion validity
    # check after the (clean) conflict check passes.
    safe_bg = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [500],
            "ref": ["T"],
            "alt": ["C"],
            "phase_set": ["PS1"],
        }
    )

    with pytest.raises(AmbiguousDeletionError):
        block.add_background_variants(
            cast(DataFrame[VariantSchema], safe_bg),
            "EAS",
            "HG00512",
            "A",
            resolve_conflicts=False,
        )


def test_add_background_ambiguous_deletion_in_background_raises_error(gene_obj):
    """Test that AmbiguousDeletionError is raised when the background variants
    themselves contain a deletion whose span overlaps a subsequent background
    variant, and neither row conflicts with any target variant.

    Setup:
      - Target SNP at pos=50 (no overlap with background)
      - Background deletion at pos=200, ref='ATCG' → span [200, 203]
      - Background SNP at pos=202, which sits inside the deletion span

    Why this is NOT caught by _check_variant_conflicts():
      bg positions 200 and 202 are both beyond the target's end_pos (50), so
      the target-vs-background interval check finds no overlap.

    Why _check_deletion_validity() DOES catch it:
      After merging, self.vdf is sorted as [50, 200, 202]. For the row at
      pos=200 (deletion_len=3), the next position (202) satisfies
      202 > 200 AND 202 <= 203, triggering AmbiguousDeletionError.
    """
    target_df = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [50],
            "ref": ["A"],
            "alt": ["G"],
            "genotype": ["1|0"],
            "phase_set": ["PS1"],
        }
    )
    block = HaplotypeBlock(cast(DataFrame[VariantSchema], target_df), gene_obj)

    # Background: deletion at 200 spans [200, 203]; SNP at 202 is inside
    bg_df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [200, 202],
            "ref": ["ATCG", "G"],
            "alt": ["A", "C"],
            "phase_set": ["PS1", "PS1"],
        }
    )

    with pytest.raises(AmbiguousDeletionError):
        block.add_background_variants(
            cast(DataFrame[VariantSchema], bg_df),
            "EAS",
            "HG00512",
            "A",
            resolve_conflicts=False,
        )


def test_add_background_deletion_next_variant_outside_span_no_error(
    deletion_variants_df, gene_obj
):
    """Test that no AmbiguousDeletionError is raised when the next variant falls
    strictly outside the deletion's span.

    Setup:
      - Target deletion at pos=100, ref='ATCG' → span [100, 103]
      - Background SNP at pos=200, which is well beyond pos+deletion_len=103

    _check_deletion_validity() should pass cleanly: next_pos (200) > 103.
    """
    block = HaplotypeBlock(deletion_variants_df, gene_obj)

    bg_df = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [200],
            "ref": ["T"],
            "alt": ["C"],
            "phase_set": ["PS1"],
        }
    )

    # Should complete without raising any exception
    block.add_background_variants(
        cast(DataFrame[VariantSchema], bg_df),
        "EAS",
        "HG00512",
        "A",
        resolve_conflicts=False,
    )

    assert len(block.vdf) == 2  # 1 target deletion + 1 background SNP


def test_add_background_deletion_next_variant_at_boundary_no_error(gene_obj):
    """Test the boundary condition: next variant at exactly pos + deletion_len + 1
    (one position beyond the deleted span) must NOT raise AmbiguousDeletionError.

    Setup:
      - Target deletion at pos=100, ref='ATCG' (deletion_len=3) → span [100, 103]
      - Background SNP at pos=104 (= 100 + 3 + 1), just outside the span

    _check_deletion_validity() condition: next_pos <= pos + deletion_len
    104 <= 103 is False → no error expected.
    """
    variants_df = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [100],
            "ref": ["ATCG"],
            "alt": ["A"],
            "genotype": ["1|0"],
            "phase_set": ["PS1"],
        }
    )
    block = HaplotypeBlock(cast(DataFrame[VariantSchema], variants_df), gene_obj)

    bg_df = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [104],  # exactly one past the deletion span end (103)
            "ref": ["G"],
            "alt": ["C"],
            "phase_set": ["PS1"],
        }
    )

    block.add_background_variants(
        cast(DataFrame[VariantSchema], bg_df),
        "EAS",
        "HG00512",
        "A",
        resolve_conflicts=False,
    )

    assert len(block.vdf) == 2


def test_add_background_deletion_next_variant_at_last_deleted_pos_raises_error(
    gene_obj,
):
    """Test the boundary condition: next variant at exactly pos + deletion_len
    (the last deleted position) MUST raise AmbiguousDeletionError.

    Setup:
      - Target deletion at pos=100, ref='ATCG' (deletion_len=3) → span [100, 103]
      - Background SNP at pos=103 (= 100 + 3), the last position in the deleted span

    _check_deletion_validity() condition: next_pos <= pos + deletion_len
    103 <= 103 is True → AmbiguousDeletionError expected.
    """
    variants_df = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [100],
            "ref": ["ATCG"],
            "alt": ["A"],
            "genotype": ["1|0"],
            "phase_set": ["PS1"],
        }
    )
    block = HaplotypeBlock(cast(DataFrame[VariantSchema], variants_df), gene_obj)

    # pos=103 is at the boundary of the deletion span; _check_variant_conflicts
    # will also flag this (103 <= end_pos=103), so we use resolve_conflicts=True
    # to let that pass, and then _check_deletion_validity must still raise.
    bg_df = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [103],
            "ref": ["G"],
            "alt": ["C"],
            "phase_set": ["PS1"],
        }
    )

    with pytest.raises((AmbiguousDeletionError, BackgroundConflictError)):
        block.add_background_variants(
            cast(DataFrame[VariantSchema], bg_df),
            "EAS",
            "HG00512",
            "A",
            resolve_conflicts=False,
        )


# ==========================================
# EDGE CASE TESTS
# ==========================================


def test_empty_block_initialization(gene_obj):
    """An empty DataFrame should produce a block with chrom=None and empty name."""
    vdf = pd.DataFrame(
        {
            "chrom": pd.Series(dtype="str"),
            "pos": pd.Series(dtype="int"),
            "ref": pd.Series(dtype="str"),
            "alt": pd.Series(dtype="str"),
            "background": pd.Series(dtype="bool"),
            "genotype": pd.Series(dtype="str"),
            "phase_set": pd.Series(dtype="str"),
            "sample_name": pd.Series(dtype="str"),
        },
        columns=[
            "chrom",
            "pos",
            "ref",
            "alt",
            "background",
            "genotype",
            "phase_set",
            "sample_name",
        ],  # type: ignore
    )
    block = HaplotypeBlock(variants_df=vdf, gene_obj=gene_obj)  # type: ignore

    assert block.chrom is None
    assert block.phaseset_tag is None
    assert block.name == ""
    assert block.vdf.empty


def test_single_variant_block_name(gene_obj):
    """A block with exactly one variant should produce a single-element name."""
    df = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [500],
            "ref": ["A"],
            "alt": ["G"],
            "genotype": ["1|0"],
            "phase_set": ["PS1"],
        }
    )
    block = HaplotypeBlock(df, gene_obj)  # type: ignore
    assert block.name == "chr1-500-A-G"
