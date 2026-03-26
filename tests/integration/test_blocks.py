import pytest
import pandas as pd

# Assuming your module is named panthera.blocks
from panthera.core.bio.blocks import HaplotypeBlock, VariantSchema
from panthera.core.bio.gene import GeneObject
from panthera.utils.exceptions import BackgroundConflictError

# --- Fixtures ---


@pytest.fixture
def target_variants():
    """Valid target variants dataframe."""
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [10, 20],  # 1-based positions
            "ref": ["A", "A"],
            "alt": ["G", "AT"],  # SNP at 10, Insertion at 20
            "genotype": ["1/1", "0/1"],
            "phase_set": ["PS1", "PS1"],
            "sample_name": ["sample1", "sample1"],
        }
    )


@pytest.fixture
def background_variants():
    """Valid background variants that do not conflict."""
    return pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [30],
            "ref": ["A"],
            "alt": ["T"],  # SNP at 30
            "genotype": ["1/1"],
            "phase_set": ["PS1"],
            "sample_name": ["HG00512"],
        }
    )


@pytest.fixture
def conflicting_background():
    """Background variants that overlap with target variants."""
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [10, 20],  # Exact overlaps with target_variants
            "ref": ["A", "A"],
            "alt": ["T", "AG"],
        }
    )


@pytest.fixture
def base_chromosome_seq():
    """A dummy sequence of 50 'A's to act as the reference chromosome."""
    return "A" * 50


@pytest.fixture
def gene_obj():
    """
    A GeneObject that spans a wide genomic range on chr1.
    Used as the default gene for most tests; its wide range (1–999999)
    means it does NOT filter out any test variants.
    """
    return GeneObject(
        chrom="chr1",
        strand="+",
        start=1,
        end=999_999,
        name="BRCA1",
        gene_id="ENSG00000012048",
        splice_sites={"donor": [1200, 1800], "acceptor": [1100, 1700]},
        shex=[[100, 200], [300, 400]],
    )


# --- Integration Tests ---


def test_haplotype_block_initialization_and_schema(target_variants, gene_obj):
    """
    INTEGRATION: Tests that Pandera schema correctly validates and coerces
    data upon entering the HaplotypeBlock.
    """
    # Act
    # Validate through schema to ensure coerce=True triggers
    validated_df = VariantSchema.validate(target_variants)
    block = HaplotypeBlock(validated_df, gene_obj)

    # Assert
    assert block.chrom == "chr1"
    assert block.phaseset_tag == "PS1"
    # Target variants should be assigned background = 0
    assert all(block.vdf["background"] == 0)
    # Check the dynamic naming property
    assert block.name == "chr1-10-A-G.chr1-20-A-AT"


def test_add_background_variants_success(
    target_variants, background_variants, gene_obj
):
    """
    INTEGRATION: Tests merging of target and background dataframes
    and ensures no false-positive conflicts are detected.
    """
    # Setup
    block = HaplotypeBlock(VariantSchema.validate(target_variants), gene_obj)
    validated_bg = VariantSchema.validate(background_variants)

    # Act
    block.add_background_variants(
        background_df=validated_bg,
        population="EAS",
        background_id="HG00512",
        haplotype_id="A",
        resolve_conflicts=False,
    )

    # Assert
    # Total variants should be 3 (2 target + 1 background)
    assert len(block.vdf) == 3
    # Check metadata assignment
    assert block.population == "EAS"
    assert block.background_id == "HG00512"
    # Ensure background flag was set correctly for the new variant
    bg_mask = block.vdf["pos"] == 30
    assert block.vdf.loc[bg_mask, "background"].iloc[0] == 1


def test_conflict_resolution_raises_error(
    target_variants, conflicting_background, gene_obj
):
    """
    INTEGRATION: Verifies that the numpy interval logic correctly identifies
    overlapping coordinates and raises the custom exception.
    """
    block = HaplotypeBlock(VariantSchema.validate(target_variants), gene_obj)
    validated_conflict = VariantSchema.validate(conflicting_background)

    # Act & Assert
    with pytest.raises(
        BackgroundConflictError, match="Found 2 conflicting background variant"
    ):
        block.add_background_variants(
            background_df=validated_conflict,
            population="EAS",
            background_id="HG00512",
            haplotype_id="A",
            resolve_conflicts=False,  # Should raise error
        )


def test_conflict_resolution_drops_background(
    target_variants, conflicting_background, gene_obj
):
    """
    INTEGRATION: Verifies that resolve_conflicts=True successfully mutates
    the internal dataframe to remove ONLY the conflicting background variants.
    """
    block = HaplotypeBlock(VariantSchema.validate(target_variants), gene_obj)
    validated_conflict = VariantSchema.validate(conflicting_background)

    # Act
    block.add_background_variants(
        background_df=validated_conflict,
        population="EAS",
        background_id="HG00512",
        haplotype_id="A",
        resolve_conflicts=True,  # Should silently drop
    )

    # Assert
    # The 2 background variants should have been dropped,
    # leaving only the 2 targets
    assert len(block.vdf) == 2
    assert all(block.vdf["background"] == 0)


def test_sequence_extraction_integration(
    target_variants, background_variants, base_chromosome_seq, gene_obj
):
    """
    INTEGRATION: Tests the full flow from dataframe merging to
    coordinate shifting and sequence string manipulation.

    Note: This relies on your imported mutation functions (snp_mutation, etc.)
    working correctly.
    """
    block = HaplotypeBlock(VariantSchema.validate(target_variants), gene_obj)
    validated_bg = VariantSchema.validate(background_variants)

    block.add_background_variants(
        background_df=validated_bg,
        population="EAS",
        background_id="HG00512",
        haplotype_id="A",
        resolve_conflicts=True,
    )

    # Act
    # Context len of 5. Min pos is 10, max pos is 30.
    # Bounds should be max(1, 10-5)=5 to 30+5+net_shift.
    wt_seq, mt_seq = block.extract_seqs(chrom_seq=base_chromosome_seq, context_len=5)

    # Assert
    # We aren't testing the exact string (since that's the job of the mutation
    # unit tests), but we ARE testing the integration contract:
    # Both sequences must be returned and MUST be equal in length.
    assert isinstance(wt_seq, str)
    assert isinstance(mt_seq, str)
    assert len(wt_seq) == len(mt_seq)
    assert len(wt_seq) > 0
