import pytest
import numpy as np
import numpy.typing as npt

from panthera.core.ssp.calc_delta import SSPScorer

# --- Test Utilities ---


def create_arrays(*args: list[float]) -> list[npt.NDArray[np.float32]]:
    """Helper to quickly convert lists to float32 numpy arrays."""
    return [np.array(arr, dtype=np.float32) for arr in args]


# --- Integration Tests ---


def test_pipeline_snps_only():
    """
    Integration Test 1: Standard sequence with no INDELs.
    Validates end-to-end pipeline with standard point mutations.
    """
    wt_acc, wt_dnr = create_arrays(
        [0.1, 0.9, 0.2, 0.1],  # Pos: 100, 101, 102, 103
        [0.1, 0.1, 0.8, 0.1],
    )
    mt_acc, mt_dnr = create_arrays(
        [0.5, 0.5, 0.2, 0.1],  # 100 (unk) increases, 101 (known) decreases
        [0.1, 0.1, 0.9, 0.1],  # 102 (known) increases
    )

    scorer = SSPScorer(
        chrom_start=100,
        splice_sites={"acc": [101], "dnr": [102]},
        wt_seq="ATCG",
        mt_seq="ATCG",
        wt_acc=wt_acc,
        wt_dnr=wt_dnr,
        mt_acc=mt_acc,
        mt_dnr=mt_dnr,
    )

    # 1. Align
    scorer.align_prob()
    assert scorer.reference_pos == ["100", "101", "102", "103"]

    # 2. Raw Delta
    # Max raw should be |0.9 - 0.5| = 0.4 (Acceptor at 101)
    # OR |0.1 - 0.5| = 0.4 (Acceptor at 100)
    raw_max = scorer.calc_raw_delta()
    assert raw_max == pytest.approx(0.4, rel=1e-5)

    # 3. Masked Delta
    # Acc 100 (Unk): 0.5 > 0.1 -> Keep 0.4
    # Acc 101 (Known): 0.5 < 0.9 -> Keep 0.4
    # Dnr 102 (Known): 0.9 > 0.8 -> Mask to 0.0 (Increase at known site)
    masked_max = scorer.calc_masked_delta()

    assert masked_max == pytest.approx(0.4, rel=1e-5)

    # Both position 100 and 101 have a masked delta of 0.4
    # The output should be sorted and semicolon-separated
    assert scorer.max_mds_loc == "100;101"


def test_pipeline_insertion():
    """
    Integration Test 2: Sequence with an insertion mutation ('>').
    Validates pointer advancement and zero-padding of WT probabilities.
    """
    # WT sequence is 3 bases long
    wt_acc, wt_dnr = create_arrays([0.1, 0.9], [0.1, 0.1])
    # MT sequence has an insertion ('>'), meaning the array has 3 probabilities
    # (the literal base following '>' is skipped by the ignore_counter)
    mt_acc, mt_dnr = create_arrays(
        [0.1, 0.5, 0.2],  # The 0.5 belongs to the inserted base
        [0.1, 0.1, 0.8],
    )

    scorer = SSPScorer(
        chrom_start=100,
        splice_sites={"acc": [101], "dnr": []},
        wt_seq="AT",
        mt_seq="AT>C",
        wt_acc=wt_acc,
        wt_dnr=wt_dnr,
        mt_acc=mt_acc,
        mt_dnr=mt_dnr,
    )

    scorer.align_prob()

    # Check that alignment inserted the 'p' coordinate correctly
    assert scorer.reference_pos == ["100", "101", "101p1"]

    # Raw Max:
    # At 101p1, WT is zero-padded (0.0). MT is 0.8. Diff = 0.8.
    raw_max = scorer.calc_raw_delta()
    assert raw_max == pytest.approx(0.8, rel=1e-5)

    # Masked Max:
    # 101p1 is an unknown site. 0.8 > 0.0 -> Keep 0.8.
    masked_max = scorer.calc_masked_delta()
    assert masked_max == pytest.approx(0.8, rel=1e-5)
    assert scorer.max_mds_loc == "101p1"


def test_pipeline_deletion():
    """
    Integration Test 3: Sequence with a deletion mutation ('<').
    Validates pointer management and zero-padding of MT probabilities.
    """
    # WT sequence is 4 bases long
    wt_acc, wt_dnr = create_arrays([0.1, 0.9, 0.5, 0.1], [0.1, 0.1, 0.1, 0.1])
    # MT sequence deletes one base ('<'), so it only provides 3 probabilities
    mt_acc, mt_dnr = create_arrays([0.1, 0.9, 0.1], [0.1, 0.1, 0.1])

    scorer = SSPScorer(
        chrom_start=100,
        splice_sites={"acc": [102], "dnr": []},
        wt_seq="ATCG",
        mt_seq="AT<G",
        wt_acc=wt_acc,
        wt_dnr=wt_dnr,
        mt_acc=mt_acc,
        mt_dnr=mt_dnr,
    )

    scorer.align_prob()

    # Deletions do not create new coordinate names, they just match up to the WT
    assert scorer.reference_pos == ["100", "101", "102", "103"]

    # At pos 102 (the deletion): WT = 0.5. MT = 0.0 (zero-padded). Diff = 0.5
    raw_max = scorer.calc_raw_delta()
    assert raw_max == pytest.approx(0.5, rel=1e-5)

    # Masked Max:
    # Pos 102 is a known site. MT (0.0) < WT (0.5) ->
    # Decrease at known site -> Keep 0.5
    masked_max = scorer.calc_masked_delta()
    assert masked_max == pytest.approx(0.5, rel=1e-5)
    assert scorer.max_mds_loc == "102"


def test_pipeline_background_indels():
    """
    Integration Test 4: Handling background INDEL placeholders ('{' and '}').
    Validates that the translation table correctly strips them before
    processing.
    """
    (arr,) = create_arrays([0.1, 0.2])

    # Background indels should be completely ignored by the alignment logic
    scorer = SSPScorer(
        chrom_start=1,
        splice_sites={"acc": [], "dnr": []},
        wt_seq="{A}T",
        mt_seq="{A}T",
        wt_acc=arr,
        wt_dnr=arr,
        mt_acc=arr,
        mt_dnr=arr,
    )

    scorer.align_prob()
    assert scorer.reference_pos == ["1", "2"]

    # Identical arrays should result in exactly 0.0 deltas
    assert scorer.calc_raw_delta() == 0.0
    assert scorer.calc_masked_delta() == 0.0
    assert scorer.max_mds_loc == ""


def test_pipeline_validation_guards():
    """
    Integration Test 5: Tests the fail-fast enterprise validation gates.
    """
    arr_valid, arr_invalid = create_arrays([0.1, 0.5], [0.1, 1.5])

    # 1. Init Validator: Probability out of bounds
    with pytest.raises(ValueError, match="contains values outside"):
        SSPScorer(
            chrom_start=1,
            splice_sites={},
            wt_seq="AT",
            mt_seq="AT",
            wt_acc=arr_invalid,
            wt_dnr=arr_valid,  # Invalid array here
            mt_acc=arr_valid,
            mt_dnr=arr_valid,
        )

    # 2. State Validator: Calling calculation before alignment
    scorer = SSPScorer(
        chrom_start=1,
        splice_sites={"acc": [], "dnr": []},
        wt_seq="AT",
        mt_seq="AT",
        wt_acc=arr_valid,
        wt_dnr=arr_valid,
        mt_acc=arr_valid,
        mt_dnr=arr_valid,
    )

    with pytest.raises(RuntimeError, match="Must call align_prob"):
        scorer.calc_raw_delta()

    with pytest.raises(RuntimeError, match="Must call align_prob"):
        scorer.calc_masked_delta()
