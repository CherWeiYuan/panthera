import pytest
import numpy as np

from panthera.core.ssp.calc_delta import SSPScorer

# --- Fixtures ---

@pytest.fixture
def masked_scorer() -> SSPScorer:
    """Provides a scorer pre-configured with known splice sites and reference positions."""
    dummy_arr = np.array([0.0], dtype=np.float32)
    scorer = SSPScorer(
        chrom_start=100,
        splice_sites={"acc": [101, 104], "dnr": [201]},  # Known genomic coordinates
        wt_seq="ATCGN",
        mt_seq="ATCGN",
        wt_acc=dummy_arr,
        wt_dnr=dummy_arr,
        mt_acc=dummy_arr,
        mt_dnr=dummy_arr,
    )

    # Mock the internal state that would normally be set by align_prob()
    scorer.reference_pos = ["100", "101", "102p1", "103", "104"]
    return scorer


# --- Tests for _masked_delta_helper ---

def test_masked_delta_helper_logic(masked_scorer):
    """
    Test the biological masking rules:
    - Known sites (101, 104): Keep delta ONLY if MT < WT (disruption).
    - Unknown sites (100, 102p1, 103): Keep delta ONLY if MT > WT (cryptic creation).
    """
    wt_acc = np.array([0.1, 0.9, 0.2, 0.5, 0.8], dtype=np.float32)
    mt_acc = np.array([0.5, 0.5, 0.1, 0.5, 0.9], dtype=np.float32)

    # Pos "100" (Unknown): 0.5 > 0.1 (Increase) -> KEEP (0.4)
    # Pos "101" (Known):   0.5 < 0.9 (Decrease) -> KEEP (0.4)
    # Pos "102p1" (Unk):   0.1 < 0.2 (Decrease) -> MASK (0.0)
    # Pos "103" (Unknown): 0.5 == 0.5 (No change) -> MASK (0.0)
    # Pos "104" (Known):   0.9 > 0.8 (Increase) -> MASK (0.0)

    expected_deltas = np.array([0.4, 0.4, 0.0, 0.0, 0.0], dtype=np.float32)
    result = masked_scorer._masked_delta_helper(wt_acc, mt_acc, ss_type="acc")

    np.testing.assert_array_almost_equal(result, expected_deltas, decimal=5)


def test_masked_delta_helper_missing_ref_pos(masked_scorer):
    """Edge Case: Test failure when reference_pos is None."""
    masked_scorer.reference_pos = None
    dummy = np.array([0.1], dtype=np.float32)
    
    with pytest.raises(RuntimeError, match="Reference positions unavailable"):
        masked_scorer._masked_delta_helper(dummy, dummy, ss_type="acc")


def test_masked_delta_helper_exact_zero_diff(masked_scorer):
    """Edge Case: Test logic handles identical WT and MT arrays (zero diff)."""
    probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    expected_deltas = np.zeros(5, dtype=np.float32)
    
    result = masked_scorer._masked_delta_helper(probs, probs, ss_type="acc")
    np.testing.assert_array_equal(result, expected_deltas)


# --- Tests for _find_max_delta_locations ---

def test_find_max_delta_locations_single(masked_scorer):
    """Test finding a single maximum location."""
    max_deltas = np.array([0.1, 0.8, 0.0, 0.2, 0.0], dtype=np.float32)
    max_val = 0.8

    loc_str = masked_scorer._find_max_delta_locations(max_deltas, max_val)

    # Max is at index 1, which corresponds to reference_pos "101"
    assert loc_str == "101"


def test_find_max_delta_locations_ties(masked_scorer):
    """Test when the maximum delta occurs in multiple places."""
    # Max value of 0.6 at index 0 and index 3
    max_deltas = np.array([0.6, 0.1, 0.0, 0.6, 0.0], dtype=np.float32)
    max_val = 0.6

    loc_str = masked_scorer._find_max_delta_locations(max_deltas, max_val)

    # Indices 0 and 3 correspond to "100" and "103"
    assert loc_str == "100;103"


def test_find_max_delta_locations_zero(masked_scorer):
    """Test behavior when the maximum delta is <= 0.0 (returns early)."""
    zeros = np.zeros(5, dtype=np.float32)
    loc_str = masked_scorer._find_max_delta_locations(zeros, max_val=0.0)
    assert loc_str == ""


def test_find_max_delta_locations_numpy_ref_pos(masked_scorer):
    """Edge Case: Test selection logic when reference_pos is an ndarray instead of a list."""
    masked_scorer.reference_pos = np.array(["100", "101", "102p1", "103", "104"])
    max_deltas = np.array([0.0, 0.9, 0.0, 0.9, 0.0], dtype=np.float32)
    
    loc_str = masked_scorer._find_max_delta_locations(max_deltas, max_val=0.9)
    assert loc_str == "101;103"


def test_find_max_delta_locations_missing_ref_pos(masked_scorer):
    """Edge Case: Test failure when reference_pos is None."""
    masked_scorer.reference_pos = None
    max_deltas = np.array([0.5], dtype=np.float32)
    
    with pytest.raises(RuntimeError, match="Reference positions are unavailable"):
        masked_scorer._find_max_delta_locations(max_deltas, 0.5)


# --- Tests for calc_masked_deltas ---

def test_calc_masked_deltas_integration(masked_scorer):
    """End-to-end integration test of the main calculation method returning an element-wise max array."""
    # Inject aligned probabilities
    wt_acc = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    mt_acc = np.array([0.8, 0.9, 0.2], dtype=np.float32)  
    # ACC Diff: [0.7 (unk inc -> KEEP), 0, 0] -> [0.7, 0, 0]

    wt_dnr = np.array([0.2, 0.5, 0.9], dtype=np.float32)
    mt_dnr = np.array([0.2, 0.5, 0.1], dtype=np.float32)  
    # DNR Diff: [0, 0, -0.8 (unk dec -> MASK!)] -> [0, 0, 0]

    masked_scorer.reference_pos = ["100", "101", "101p1"]
    masked_scorer.aligned_prob = (wt_acc, wt_dnr, mt_acc, mt_dnr)

    # Expected element-wise max between ACC and DNR masked arrays
    expected_max_array = np.array([0.7, 0.0, 0.0], dtype=np.float32)
    
    result = masked_scorer.calc_masked_deltas()

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, expected_max_array, decimal=5)


def test_calc_masked_deltas_unaligned_error(masked_scorer):
    """Test that the fail-fast RuntimeError triggers if alignment is missing."""
    masked_scorer.aligned_prob = None
    with pytest.raises(RuntimeError, match="Must call align_prob"):
        masked_scorer.calc_masked_deltas()