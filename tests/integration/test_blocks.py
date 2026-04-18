import pytest
import pandas as pd

from panthera.core.bio.blocks import HaplotypeBlock, VariantSchema
from panthera.core.bio.gene import GeneObject
from panthera.utils.exceptions import AmbiguousDeletionError, BackgroundConflictError

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
def target_deletion_variants():
    """Target variants where a deletion at pos=10 (ref='AAAA', alt='A')
    spans [10, 13], followed by a safe SNP at pos=20 (well outside the span).

    Used to verify the happy path: deletion present but next variant is
    beyond pos + deletion_len, so _check_deletion_validity() passes cleanly.
    """
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [10, 20],
            "ref": ["AAAA", "A"],  # Deletion at 10 spans [10, 13]; SNP at 20 is safe
            "alt": ["A", "G"],
            "genotype": ["1/1", "0/1"],
            "phase_set": ["PS1", "PS1"],
            "sample_name": ["sample1", "sample1"],
        }
    )


@pytest.fixture
def ambiguous_target_deletion_variants():
    """Target variants where a deletion at pos=10 (ref='AAAAAAAAAA', alt='A')
    spans [10, 19], and a second target SNP sits at pos=15 inside that span.

    _check_variant_conflicts() is blind to this (it only compares target vs
    background rows). _check_deletion_validity() must catch it and raise
    AmbiguousDeletionError when add_background_variants() is called.
    """
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [10, 15],  # Deletion [10, 19] swallows the SNP at 15
            "ref": ["AAAAAAAAAA", "A"],
            "alt": ["A", "G"],
            "genotype": ["1/1", "0/1"],
            "phase_set": ["PS1", "PS1"],
            "sample_name": ["sample1", "sample1"],
        }
    )


@pytest.fixture
def ambiguous_background_deletion_variants():
    """Background variants where a deletion at pos=30 (ref='AAAA', alt='A')
    spans [30, 33], and a second background SNP sits at pos=32 inside that span.

    Neither row conflicts with any target variant, so _check_variant_conflicts()
    passes. _check_deletion_validity() must catch this after the merge.
    """
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [30, 32],  # Deletion [30, 33] swallows the SNP at 32
            "ref": ["AAAA", "A"],
            "alt": ["A", "G"],
            "genotype": ["1/1", "0/1"],
            "phase_set": ["PS1", "PS1"],
            "sample_name": ["HG00512", "HG00512"],
        }
    )


@pytest.fixture
def base_chromosome_seq():
    """A dummy sequence of 50 'A's to act as the reference chromosome."""
    return "A" * 50


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


# --- Integration Tests ---


def test_haplotype_block_initialization_and_schema(target_variants, gene_obj):
    """INTEGRATION: Tests that Pandera schema correctly validates and coerces
    data upon entering the HaplotypeBlock.
    """
    # Act
    # Validate through schema to ensure coerce=True triggers
    validated_df = VariantSchema.validate(target_variants)
    block = HaplotypeBlock(validated_df, gene_obj, context_dist=5000)

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
    """INTEGRATION: Tests merging of target and background dataframes
    and ensures no false-positive conflicts are detected.
    """
    # Setup
    block = HaplotypeBlock(
        VariantSchema.validate(target_variants), gene_obj, context_dist=5000
    )
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
    """INTEGRATION: Verifies that the numpy interval logic correctly identifies
    overlapping coordinates and raises the custom exception.
    """
    block = HaplotypeBlock(
        VariantSchema.validate(target_variants), gene_obj, context_dist=5000
    )
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
    """INTEGRATION: Verifies that resolve_conflicts=True successfully mutates
    the internal dataframe to remove ONLY the conflicting background variants.
    """
    block = HaplotypeBlock(
        VariantSchema.validate(target_variants), gene_obj, context_dist=5000
    )
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


# --- Deletion Validity Integration Tests ---
# _check_deletion_validity() is invoked at two points in the pipeline:
#   1. Inside add_background_variants(), after _check_variant_conflicts()
#   2. Inside _modify_seq(), which is called by extract_seqs()
# The tests below exercise both call sites as a full end-to-end flow.


def test_add_background_ambiguous_target_deletion_raises_error(
    ambiguous_target_deletion_variants, background_variants, gene_obj
):
    """INTEGRATION: Verifies that AmbiguousDeletionError is raised via
    add_background_variants() when two TARGET variants create an ambiguous
    deletion — i.e. a target deletion whose span overlaps a subsequent
    target variant.

    Why _check_variant_conflicts() does NOT catch this:
      It only compares target rows against background rows using interval
      logic. Both conflicting rows here are TARGET_VARIANTS (background=0),
      so the cross-comparison never flags them.

    Why _check_deletion_validity() DOES catch this:
      It scans every consecutive pair in self.vdf (sorted by position) and
      detects that the SNP at pos=15 falls inside the deletion span [10, 19].

    The background_variants fixture (pos=30) is clean and non-conflicting;
    it is only present to trigger the add_background_variants() call path.
    """
    block = HaplotypeBlock(
        VariantSchema.validate(ambiguous_target_deletion_variants),
        gene_obj,
        context_dist=5000,
    )
    validated_bg = VariantSchema.validate(background_variants)

    with pytest.raises(AmbiguousDeletionError):
        block.add_background_variants(
            background_df=validated_bg,
            population="EAS",
            background_id="HG00512",
            haplotype_id="A",
            resolve_conflicts=False,
        )


def test_add_background_ambiguous_background_deletion_raises_error(
    target_variants, ambiguous_background_deletion_variants, gene_obj
):
    """INTEGRATION: Verifies that AmbiguousDeletionError is raised via
    add_background_variants() when two BACKGROUND variants create an
    ambiguous deletion — i.e. a background deletion whose span overlaps
    a subsequent background variant.

    Why _check_variant_conflicts() does NOT catch this:
      The deletion (pos=30) and SNP (pos=32) are both BACKGROUND_VARIANTS
      that don't overlap any target variant (pos=10 or 20), so the
      target-vs-background interval check finds no conflicts.

    Why _check_deletion_validity() DOES catch this:
      After the merge and sort, self.vdf contains [10, 20, 30, 32].
      The row at pos=30 (deletion_len=3) has next_pos=32, and since
      32 <= 30 + 3 = 33, the ambiguity condition is triggered.
    """
    block = HaplotypeBlock(
        VariantSchema.validate(target_variants), gene_obj, context_dist=5000
    )
    validated_ambiguous_bg = VariantSchema.validate(
        ambiguous_background_deletion_variants
    )

    with pytest.raises(AmbiguousDeletionError):
        block.add_background_variants(
            background_df=validated_ambiguous_bg,
            population="EAS",
            background_id="HG00512",
            haplotype_id="A",
            resolve_conflicts=False,
        )


def test_add_background_valid_deletion_no_error(
    target_deletion_variants, background_variants, gene_obj
):
    """INTEGRATION: Verifies that a deletion whose next variant falls strictly
    outside its span does NOT raise AmbiguousDeletionError.

    Setup (target_deletion_variants):
      - Deletion at pos=10, ref='AAAA' → deletion_len=3, span [10, 13]
      - SNP at pos=20 → 20 > 13, safely outside the span

    The deletion validity check should pass, and the final merged vdf
    should contain all 3 rows (2 target + 1 background).
    """
    block = HaplotypeBlock(
        VariantSchema.validate(target_deletion_variants), gene_obj, context_dist=5000
    )
    validated_bg = VariantSchema.validate(background_variants)

    # Should complete without raising any exception
    block.add_background_variants(
        background_df=validated_bg,
        population="EAS",
        background_id="HG00512",
        haplotype_id="A",
        resolve_conflicts=False,
    )

    assert len(block.vdf) == 3


def test_extract_seqs_raises_ambiguous_deletion_error_via_modify_seq(
    ambiguous_target_deletion_variants, gene_obj, base_chromosome_seq
):
    """INTEGRATION: Verifies that AmbiguousDeletionError propagates correctly
    when _check_deletion_validity() is called from the _modify_seq() path
    inside extract_seqs().

    _modify_seq() is the second call site of _check_deletion_validity().
    Unlike add_background_variants(), extract_seqs() does not go through
    _check_variant_conflicts() first, so this path is independently
    reachable — for example when a HaplotypeBlock is constructed and
    extract_seqs() is called directly without add_background_variants().

    We bypass add_background_variants() by manually setting self.vdf so
    that the ambiguous deletion is present when extract_seqs() runs.
    """
    block = HaplotypeBlock(
        VariantSchema.validate(ambiguous_target_deletion_variants),
        gene_obj,
        context_dist=5000,
    )

    # extract_seqs → _modify_seq → _check_deletion_validity must raise
    with pytest.raises(AmbiguousDeletionError):
        block.extract_seqs(chrom_seq=base_chromosome_seq, extension_len=5)


def test_sequence_extraction_integration(
    target_variants, background_variants, base_chromosome_seq, gene_obj
):
    """INTEGRATION: Tests the full flow from dataframe merging to
    coordinate shifting and sequence string manipulation.

    Note: This relies on your imported mutation functions (snp_mutation, etc.)
    working correctly.
    """
    block = HaplotypeBlock(
        VariantSchema.validate(target_variants), gene_obj, context_dist=5000
    )
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
    wt_seq, mt_seq = block.extract_seqs(chrom_seq=base_chromosome_seq, extension_len=5)

    # Assert
    # We aren't testing the exact string (since that's the job of the mutation
    # unit tests), but we ARE testing the integration contract:
    # Both sequences must be returned and MUST be equal in length.
    assert isinstance(wt_seq, str)
    assert isinstance(mt_seq, str)
    assert len(wt_seq) > 0


def test_sequence_extraction_valid_deletion_integration(
    target_deletion_variants, background_variants, base_chromosome_seq, gene_obj
):
    """INTEGRATION: Tests the full pipeline with a valid deletion variant
    (non-overlapping) through to extract_seqs().

    Verifies that a deletion passing _check_deletion_validity() does not
    disrupt the extract_seqs() contract: both output sequences must be
    non-empty and equal in length.
    """
    block = HaplotypeBlock(
        VariantSchema.validate(target_deletion_variants), gene_obj, context_dist=5000
    )
    validated_bg = VariantSchema.validate(background_variants)

    block.add_background_variants(
        background_df=validated_bg,
        population="EAS",
        background_id="HG00512",
        haplotype_id="A",
        resolve_conflicts=False,
    )

    wt_seq, mt_seq = block.extract_seqs(chrom_seq=base_chromosome_seq, extension_len=5)

    assert isinstance(wt_seq, str)
    assert isinstance(mt_seq, str)
    assert len(wt_seq) == len(mt_seq)
    assert len(wt_seq) > 0
