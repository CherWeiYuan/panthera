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
    # Setup Probabilities
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


# --- Tests for _find_max_mds_locations ---


def test_find_max_mds_locations_single(masked_scorer):
    """Test finding a single maximum location."""
    acc_deltas = np.array([0.1, 0.8, 0.0, 0.2, 0.0], dtype=np.float32)
    dnr_deltas = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    max_val = 0.8

    loc_str = masked_scorer._find_max_mds_locations(acc_deltas, dnr_deltas, max_val)

    # Max is at index 1, which corresponds to reference_pos "101"
    assert loc_str == "101"


def test_find_max_mds_locations_ties_across_arrays(masked_scorer):
    """Test when the maximum delta occurs in multiple places and across both ACC and DNR."""
    # Max value of 0.6 at index 0 (acc), index 3 (acc), and index 0 (dnr)
    acc_deltas = np.array([0.6, 0.1, 0.0, 0.6, 0.0], dtype=np.float32)
    dnr_deltas = np.array([0.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    max_val = 0.6

    loc_str = masked_scorer._find_max_mds_locations(acc_deltas, dnr_deltas, max_val)

    # Indices 0 and 3 correspond to "100" and "103"
    assert loc_str == "100;103"


def test_find_max_mds_locations_zero(masked_scorer):
    """Test behavior when the maximum delta is 0.0 (all biological noise masked out)."""
    zeros = np.zeros(5, dtype=np.float32)
    loc_str = masked_scorer._find_max_mds_locations(zeros, zeros, max_val=0.0)

    assert loc_str == ""


# --- Tests for calc_masked_delta ---


def test_calc_masked_delta_integration(masked_scorer):
    """End-to-end integration test of the main calculation method."""
    # Inject aligned probabilities
    wt_acc = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    mt_acc = np.array([0.8, 0.9, 0.2], dtype=np.float32)  # Diff: [0.7 (keep), 0, 0]

    wt_dnr = np.array([0.2, 0.5, 0.9], dtype=np.float32)
    mt_dnr = np.array(
        [0.2, 0.5, 0.1], dtype=np.float32
    )  # Diff: [0, 0, -0.8 (mask! 102p1 is unk)]

    masked_scorer.reference_pos = ["100", "101", "101p1"]
    masked_scorer.aligned_prob = (wt_acc, wt_dnr, mt_acc, mt_dnr)

    # Expected max is 0.7 from ACC at index 0 ("100")
    # DNR's 0.8 difference is masked to 0.0 because it's a decrease at an unknown site
    result = masked_scorer.calc_masked_delta()

    assert isinstance(result, float)
    assert result == pytest.approx(0.7, rel=1e-5)
    assert masked_scorer.max_masked_delta == pytest.approx(0.7, rel=1e-5)
    assert masked_scorer.max_mds_loc == "100"


def test_calc_masked_delta_unaligned_error(masked_scorer):
    """Test that the fail-fast RuntimeError triggers if alignment is missing."""
    masked_scorer.aligned_prob = None
    with pytest.raises(RuntimeError, match="Must call align_prob"):
        masked_scorer.calc_masked_delta()
