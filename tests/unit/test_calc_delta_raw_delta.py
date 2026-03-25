import pytest
import numpy as np

from panthera.core.ssp.calc_delta import SSPScorer

# --- Fixtures ---


@pytest.fixture
def base_scorer() -> SSPScorer:
    """Provides a baseline SSPScorer instance to avoid __init__ boilerplate."""
    dummy_arr = np.array([0.1, 0.2], dtype=np.float32)
    return SSPScorer(
        chrom_start=100,
        splice_sites={"acc": [], "dnr": []},
        wt_seq="AT",
        mt_seq="AT",
        wt_acc=dummy_arr,
        wt_dnr=dummy_arr,
        mt_acc=dummy_arr,
        mt_dnr=dummy_arr,
    )


# --- Tests ---


def test_calc_raw_delta_normal(base_scorer):
    """Test standard calculation where max delta is found in acceptor array."""
    # Setup mock aligned probabilities
    wt_acc = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    wt_dnr = np.array([0.2, 0.4, 0.6], dtype=np.float32)

    # Acceptor max diff: |0.5 - 0.1| = 0.4
    mt_acc = np.array([0.2, 0.1, 0.9], dtype=np.float32)
    # Donor max diff: |0.6 - 0.9| = 0.3
    mt_dnr = np.array([0.1, 0.4, 0.9], dtype=np.float32)

    # Inject directly into state
    base_scorer.aligned_prob = (wt_acc, wt_dnr, mt_acc, mt_dnr)

    result = base_scorer.calc_raw_delta()

    # Assert correct return type (crucial for downstream pipelines)
    assert isinstance(result, float)

    # Use pytest.approx to handle standard floating point arithmetic imprecision
    assert result == pytest.approx(0.4, rel=1e-5)

    # Assert internal state was updated correctly
    assert base_scorer.max_raw_delta == result


def test_calc_raw_delta_donor_max(base_scorer):
    """Test standard calculation where max delta is found in donor array."""
    wt_acc = np.array([0.5, 0.5], dtype=np.float32)
    wt_dnr = np.array([0.1, 0.9], dtype=np.float32)

    mt_acc = np.array([0.5, 0.5], dtype=np.float32)  # Diff: 0.0
    mt_dnr = np.array([0.8, 0.1], dtype=np.float32)  # Diff: 0.7 and 0.8

    base_scorer.aligned_prob = (wt_acc, wt_dnr, mt_acc, mt_dnr)

    result = base_scorer.calc_raw_delta()
    assert result == pytest.approx(0.8, rel=1e-5)


def test_calc_raw_delta_zero_diff(base_scorer):
    """Test behavior when WT and MT probabilities are identical."""
    arr = np.array([0.2, 0.4, 0.6], dtype=np.float32)

    base_scorer.aligned_prob = (arr, arr, arr, arr)

    result = base_scorer.calc_raw_delta()
    assert result == 0.0


def test_calc_raw_delta_unaligned_error(base_scorer):
    """Test that the fail-fast RuntimeError triggers if alignment is missing."""
    # base_scorer.aligned_prob is None upon initialization
    with pytest.raises(RuntimeError, match="Must call align_prob"):
        base_scorer.calc_raw_delta()
