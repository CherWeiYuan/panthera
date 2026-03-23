import pytest
import pandas as pd
from unittest.mock import patch

from panthera.core.bio.blocks import (
    HaplotypeBlock,
    TARGET_VARIANTS,
    BACKGROUND_VARIANTS,
)

from panthera.utils.exceptions import AmbiguousDeletionError


# --- Test Fixtures ---
@pytest.fixture
def empty_vdf():
    return pd.DataFrame(
        columns=pd.Index(
            [
                "chrom",
                "pos",
                "ref",
                "alt",
                "genotype",
                "background",
                "phase_set",
                "sample_name",
            ]
        )
    )


@pytest.fixture
def standard_vdf():
    """Returns a basic VDF with one background SNP and one target SNP."""
    return pd.DataFrame(
        [
            {
                "chrom": "chr1",
                "pos": 10,
                "ref": "A",
                "alt": "G",
                "background": BACKGROUND_VARIANTS,
                "genotype": "1/1",
                "phase_set": "1",
                "sample_name": "sample1",
            },
            {
                "chrom": "chr1",
                "pos": 20,
                "ref": "T",
                "alt": "C",
                "background": TARGET_VARIANTS,
                "genotype": "1/1",
                "phase_set": "1",
                "sample_name": "sample1",
            },
        ]
    )


@pytest.fixture
def insertion_vdf():
    """
    Returns a VDF with a target insertion to test net_shift and buffer logic.
    """
    return pd.DataFrame(
        [
            {
                "chrom": "chr1",
                "pos": 10,
                "ref": "A",
                "alt": "ATGC",
                "background": TARGET_VARIANTS,
                "genotype": "1/1",
                "phase_set": "1",
                "sample_name": "sample1",
            }
        ]
    )


@pytest.fixture
def sequence_block(standard_vdf):
    """Returns an instantiated haplotype block object."""
    # We patch the methods onto our dummy HaplotypeBlock class dynamically
    # for the test environment.
    return HaplotypeBlock(variants_df=standard_vdf)


# --- Tests for _check_deletion_validity ---


def test_deletion_validity_safe(sequence_block):
    """Test that non-overlapping variants pass the validity check."""
    vdf = pd.DataFrame(
        [
            # 2bp deletion (deletes bases 11, 12)
            {"pos": 10, "ref": "AGC", "alt": "A"},
            # Next variant safely at 13
            {"pos": 13, "ref": "T", "alt": "C"},
        ]
    )
    # Should not raise any error
    # We are telling this function, even though we initialized HaplotypeBlock
    # with standard_vdf, but for this specific check, I want you to look at
    # vdf instead
    sequence_block._check_deletion_validity(vdf)


def test_deletion_validity_ambiguous(sequence_block):
    """Test that an overlapping deletion raises AmbiguousDeletionError."""
    vdf = pd.DataFrame(
        [
            # 2bp deletion (deletes bases 11, 12)
            {"pos": 10, "ref": "AGC", "alt": "A"},
            # Overlaps with the deleted region
            {"pos": 12, "ref": "G", "alt": "C"},
        ]
    )
    with pytest.raises(AmbiguousDeletionError):
        sequence_block._check_deletion_validity(vdf)


# --- Tests for _modify_seq ---


def test_modify_seq_empty_df(sequence_block, empty_vdf):
    """Test that an empty DataFrame returns the untouched sequence."""
    seq, mt_vdf = sequence_block._modify_seq(
        vdf=empty_vdf, seq="ATGC", in_char="}", del_char="{", mutation_class="WT"
    )
    assert seq == "ATGC"
    assert mt_vdf.empty
    assert list(mt_vdf.columns) == list(empty_vdf.columns)


def test_modify_seq_wt_pass(sequence_block, standard_vdf):
    """
    WT PASS LOGIC:
    - Background variant (pos 10) SHOULD be mutated.
    - Target variant (pos 20) SHOULD NOT be mutated, but saved to mt_vdf.
    """
    seq, mt_vdf = sequence_block._modify_seq(
        vdf=standard_vdf, seq="A" * 30, in_char="}", del_char="{", mutation_class="WT"
    )

    # Assert mutated sequence
    assert seq == "A" * 9 + "G" + "A" * 20

    # Assert mt_vdf contains ONLY the target variant
    assert len(mt_vdf) == 1
    assert mt_vdf.iloc[0]["pos"] == 20
    assert mt_vdf.iloc[0]["background"] == TARGET_VARIANTS


def test_modify_seq_coordinate_shifting(sequence_block):
    """
    Test that an insertion correctly shifts the downstream coordinates
    recorded in the mt_vdf output for the WT pass.
    """
    vdf = pd.DataFrame(
        [
            {
                "chrom": "chr1",
                "pos": 10,
                "ref": "A",
                "alt": "AT",
                "background": BACKGROUND_VARIANTS,
            },
            {
                "chrom": "chr1",
                "pos": 20,
                "ref": "C",
                "alt": "G",
                "background": TARGET_VARIANTS,
            },
        ]
    )

    # We don't care about the returned string here,
    # just the shift variable math
    _, mt_vdf = sequence_block._modify_seq(
        vdf=vdf, seq="A" * 30, in_char="}", del_char="{", mutation_class="WT"
    )

    # ref="A" (1), alt="AT" (2). len(alt) - 1 = 1.
    # Shift should be 2 * (1) = 2.
    # Downstream target variant at pos 20 should now be at 22.
    assert mt_vdf.iloc[0]["pos"] == 22


def test_modify_seq_invalid_class(sequence_block, standard_vdf):
    """Test that passing a bad mutation class raises a ValueError."""
    with pytest.raises(ValueError, match="Expected mutation class"):
        sequence_block._modify_seq(
            vdf=standard_vdf,
            seq="ATGC",
            in_char="}",
            del_char="{",
            mutation_class="INVALID",
        )


# --- Tests for extract_seqs ---


def test_extract_seqs_empty_vdf(sequence_block, empty_vdf):
    """If vdf is empty, it should return two empty strings."""
    sequence_block.vdf = empty_vdf
    wt_seq, mt_seq = sequence_block.extract_seqs(chrom_seq="ATGC" * 10, context_len=5)
    assert wt_seq == ""
    assert mt_seq == ""


@patch.object(HaplotypeBlock, "_modify_seq")
def test_extract_seqs_slicing_math(mock_modify, sequence_block, standard_vdf):
    """
    Test that the base_seq is sliced from the chromosome accurately using
    context_len, and that relative coordinates are correctly calculated before
    passing to _modify_seq.
    """
    # min_pos is 10, max_pos is 20. context_len = 5.
    # Start bound: max(1, 10 - 5) = 5
    # End bound: 20 + 5 + 0 (no insertions) = 25
    # Slice expected: [4:25] -> length 21

    chrom_seq = "N" * 50  # 50 nucleotide fake chromosome

    # Mock return values for _modify_seq to avoid complex string matching
    # in this test
    # Returns (seq, mt_vdf_seed)
    mock_modify.side_effect = [
        ("WT_SEQ_MOCK", standard_vdf),  # WT pass return
        ("MT_SEQ_MOCK", None),  # MT pass return
    ]

    wt_seq, mt_seq = sequence_block.extract_seqs(chrom_seq=chrom_seq, context_len=5)

    # Verify _modify_seq was called twice
    assert mock_modify.call_count == 2

    # Verify WT pass was called with the correctly sliced string
    wt_call_kwargs = mock_modify.call_args_list[0].kwargs
    assert len(wt_call_kwargs["seq"]) == 21  # 25 - 5 + 1

    # Verify relative coordinate shift
    # pos 10 - start_bound(5) + 1 = 6
    # pos 20 - start_bound(5) + 1 = 16
    passed_vdf = wt_call_kwargs["vdf"]
    assert passed_vdf.iloc[0]["pos"] == 6
    assert passed_vdf.iloc[1]["pos"] == 16


@patch.object(HaplotypeBlock, "_modify_seq")
def test_extract_seqs_net_shift_expansion(mock_modify, sequence_block, insertion_vdf):
    """Test that net_shift expands the end_bound for insertion variants."""
    sequence_block.vdf = insertion_vdf

    # pos = 10. len(ref)=1, len(alt)=4 -> len_change = 3.
    # net_shift = 3 * 2 = 6.
    # context_len = 5.
    # end_bound = 10 + 5 + 6 = 21. start_bound = 5.
    # base_seq len expected = 21 - 5 + 1 = 17.

    mock_modify.side_effect = [("W" * 17, pd.DataFrame()), ("M" * 17, None)]

    sequence_block.extract_seqs(chrom_seq="N" * 50, context_len=5)

    wt_call_kwargs = mock_modify.call_args_list[0].kwargs
    assert len(wt_call_kwargs["seq"]) == 17
