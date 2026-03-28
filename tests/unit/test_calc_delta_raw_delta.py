import pytest
import numpy as np
import numpy.testing as nptest

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


def test_calc_raw_deltas_normal(base_scorer):
    """Test standard calculation for element-wise maximum deltas."""
    # Setup mock aligned probabilities
    wt_acc = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    wt_dnr = np.array([0.2, 0.4, 0.6], dtype=np.float32)

    mt_acc = np.array([0.2, 0.1, 0.9], dtype=np.float32)
    mt_dnr = np.array([0.1, 0.4, 0.9], dtype=np.float32)

    # Acceptor diffs: |[0.1, 0.5, 0.9] - [0.2, 0.1, 0.9]| = [0.1, 0.4, 0.0]
    # Donor diffs:    |[0.2, 0.4, 0.6] - [0.1, 0.4, 0.9]| = [0.1, 0.0, 0.3]
    # Expected max:   [max(0.1,0.1), max(0.4,0.0), max(0.0,0.3)] = [0.1, 0.4, 0.3]
    expected = np.array([0.1, 0.4, 0.3], dtype=np.float32)

    # Inject directly into state
    base_scorer.aligned_prob = (wt_acc, wt_dnr, mt_acc, mt_dnr)
    result = base_scorer.calc_raw_deltas()

    # Assert correct return type
    assert isinstance(result, np.ndarray)

    # Use numpy testing for array comparisons with floating point tolerance
    nptest.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_calc_raw_deltas_donor_max(base_scorer):
    """Test calculation where max deltas are predominantly in the donor array."""
    wt_acc = np.array([0.5, 0.5], dtype=np.float32)
    wt_dnr = np.array([0.1, 0.9], dtype=np.float32)

    mt_acc = np.array([0.5, 0.5], dtype=np.float32)  # Diffs: [0.0, 0.0]
    mt_dnr = np.array([0.8, 0.1], dtype=np.float32)  # Diffs: [0.7, 0.8]

    expected = np.array([0.7, 0.8], dtype=np.float32)

    base_scorer.aligned_prob = (wt_acc, wt_dnr, mt_acc, mt_dnr)
    result = base_scorer.calc_raw_deltas()

    nptest.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_calc_raw_deltas_zero_diff(base_scorer):
    """Test behavior when WT and MT probabilities are identical."""
    arr = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    expected = np.zeros(3, dtype=np.float32)

    base_scorer.aligned_prob = (arr, arr, arr, arr)
    result = base_scorer.calc_raw_deltas()

    nptest.assert_allclose(result, expected)


def test_calc_raw_deltas_unaligned_error(base_scorer):
    """Test that the fail-fast RuntimeError triggers if alignment is missing."""
    # base_scorer.aligned_prob is None upon initialization
    with pytest.raises(
        RuntimeError, match=r"Must call align_prob\(\) before calc_raw_delta\(\)"
    ):
        base_scorer.calc_raw_deltas()


def test_probability_boundary_validation():
    """Test that input probabilities outside [0.0, 1.0] are rejected."""
    dummy_arr = np.array([0.5, 0.5], dtype=np.float32)
    invalid_arr = np.array([1.5, 0.5], dtype=np.float32)  # 1.5 is > 1.0

    with pytest.raises(ValueError, match=r"outside \[0\.0, 1\.0\]"):
        SSPScorer(
            chrom_start=100,
            splice_sites={"acc": [], "dnr": []},
            wt_seq="AT",
            mt_seq="AT",
            wt_acc=invalid_arr,
            wt_dnr=dummy_arr,
            mt_acc=dummy_arr,
            mt_dnr=dummy_arr,
        )


# --- New Edge Cases ---


def test_calc_raw_deltas_empty_arrays(base_scorer):
    """Test behavior with empty probability arrays (e.g., zero-length sequence)."""
    empty_arr = np.array([], dtype=np.float32)

    base_scorer.aligned_prob = (empty_arr, empty_arr, empty_arr, empty_arr)
    result = base_scorer.calc_raw_deltas()

    assert isinstance(result, np.ndarray)
    assert len(result) == 0
    nptest.assert_array_equal(result, empty_arr)


def test_calc_raw_deltas_extreme_boundaries(base_scorer):
    """Test calculations at the extreme boundaries of 0.0 and 1.0."""
    wt_acc = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    wt_dnr = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    mt_acc = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mt_dnr = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    # acc diffs: [1.0, 1.0, 0.0]
    # dnr diffs: [1.0, 1.0, 0.0]
    expected = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    base_scorer.aligned_prob = (wt_acc, wt_dnr, mt_acc, mt_dnr)
    result = base_scorer.calc_raw_deltas()

    nptest.assert_allclose(result, expected)
