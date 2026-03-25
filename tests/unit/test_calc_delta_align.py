import pytest
import numpy as np
import numpy.typing as npt

from panthera.core.ssp.calc_delta import SSPScorer 

# --- Fixtures to reduce boilerplate ---

@pytest.fixture
def base_wt_acc() -> npt.NDArray[np.float32]:
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

@pytest.fixture
def base_wt_dnr() -> npt.NDArray[np.float32]:
    return np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)

# --- Tests ---

def test_align_prob_wildtype(base_wt_acc, base_wt_dnr):
    """Test standard sequence with no INDELs (perfect match)."""
    scorer = SSPScorer(
        chrom_start=100,
        splice_sites={"acc": [], "dnr": []},
        wt_seq="ATCGN",
        mt_seq="ATCGN",
        wt_acc=base_wt_acc,
        wt_dnr=base_wt_dnr,
        mt_acc=base_wt_acc.copy(), # Identical for wildtype
        mt_dnr=base_wt_dnr.copy(),
    )
    
    scorer.align_prob()
    
    assert scorer.reference_pos == ["100", "101", "102", "103", "104"]
    np.testing.assert_array_equal(scorer.aligned_prob[0], base_wt_acc) # wt_acc
    np.testing.assert_array_equal(scorer.aligned_prob[2], base_wt_acc) # mt_acc


def test_align_prob_insertion(base_wt_acc, base_wt_dnr):
    """Test sequence with an insertion (mt_seq has '>', wt_acc should be 0)."""
    # MT sequence: 'AT>CGN' -> The '>' represents the insertion of 'C'
    # Therefore MT prob arrays only need 5 elements, because the literal 'C' is skipped
    mt_acc = np.array([0.1, 0.2, 0.99, 0.4, 0.5], dtype=np.float32)
    mt_dnr = np.array([0.9, 0.8, 0.99, 0.6, 0.5], dtype=np.float32)

    scorer = SSPScorer(
        chrom_start=100,
        splice_sites={"acc": [], "dnr": []},
        wt_seq="ATGN", # Length 4
        mt_seq="AT>CGN", # Clean length 6, '>' removed length 5
        wt_acc=np.array([0.1, 0.2, 0.4, 0.5], dtype=np.float32),
        wt_dnr=np.array([0.9, 0.8, 0.6, 0.5], dtype=np.float32),
        mt_acc=mt_acc,
        mt_dnr=mt_dnr,
    )
    
    scorer.align_prob()
    
    # Expected Reference Positions: 100, 101, 102p1 (for >), 102, 103
    assert scorer.reference_pos == ["100", "101", "102p1", "102", "103"]
    
    # WT should have a 0.0 inserted at index 2
    expected_wt_acc = np.array([0.1, 0.2, 0.0, 0.4, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(scorer.aligned_prob[0], expected_wt_acc)
    
    # MT should exactly match the input mt_acc
    np.testing.assert_array_equal(scorer.aligned_prob[2], mt_acc)


def test_align_prob_deletion(base_wt_acc, base_wt_dnr):
    """Test sequence with a deletion (mt_seq has '<', mt_acc should be 0)."""
    # MT sequence: 'AT<GN' -> The '<' represents deletion of 'C'
    # mt_acc only has 4 actual probabilities
    mt_acc = np.array([0.1, 0.2, 0.4, 0.5], dtype=np.float32)
    mt_dnr = np.array([0.9, 0.8, 0.6, 0.5], dtype=np.float32)

    scorer = SSPScorer(
        chrom_start=100,
        splice_sites={"acc": [], "dnr": []},
        wt_seq="ATCGN", # Length 5
        mt_seq="AT<GN", # Clean length 5, '>' removed length 5
        wt_acc=base_wt_acc,
        wt_dnr=base_wt_dnr,
        mt_acc=mt_acc,
        mt_dnr=mt_dnr,
    )
    
    scorer.align_prob()
    
    # Expected Reference Positions
    assert scorer.reference_pos == ["100", "101", "102", "103", "104"]
    
    # MT should have a 0.0 inserted at index 2
    expected_mt_acc = np.array([0.1, 0.2, 0.0, 0.4, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(scorer.aligned_prob[2], expected_mt_acc)


def test_align_prob_background_indels_and_exceptions():
    """Test that background indels {} are removed and invalid chars raise errors."""
    wt_acc = np.array([0.1], dtype=np.float32)
    mt_acc = np.array([0.1], dtype=np.float32)

    # Valid background removal
    scorer = SSPScorer(
        chrom_start=1, splice_sites={}, wt_seq="{A}", mt_seq="{A}",
        wt_acc=wt_acc, wt_dnr=wt_acc, mt_acc=mt_acc, mt_dnr=mt_acc
    )
    scorer.align_prob()
    assert scorer.reference_pos == ["1"]

    # Invalid character 'X'
    scorer.mt_seq = "X"
    with pytest.raises(ValueError, match="Expected characters are"):
        scorer.align_prob()


def test_align_prob_index_error_fail_fast():
    """Test that mismatched array lengths are caught dynamically."""
    # Provide an mt_seq that requires 3 probabilities, but only pass 2
    wt_acc = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mt_acc = np.array([0.1, 0.2], dtype=np.float32) # Too short!

    scorer = SSPScorer(
        chrom_start=1, splice_sites={}, wt_seq="ATC", mt_seq="ATC",
        wt_acc=wt_acc, wt_dnr=wt_acc, mt_acc=mt_acc, mt_dnr=mt_acc
    )
    
    with pytest.raises(IndexError, match="Probability array length mismatch"):
        scorer.align_prob()