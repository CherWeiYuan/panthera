"""Tests for SSPManager.predict_ssp().

Covers:
- Cache miss: predictions are computed and stored
- Cache hit: cached results are returned without re-running the model
- Mixed cache hit/miss: only uncached sequences are sent to the model
- LRU eviction: oldest entry is dropped when max_cache_size is exceeded
- LRU recency update: a cache hit moves the entry to the "recently used" end
- reverse_output: output arrays are reversed element-wise
- Empty input: returns two empty lists immediately
- Model dispatch: both "modelp" and "spliceai" branches are exercised
"""

from typing import Literal
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from panthera.core.ssp.ssp_manager import SSPManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arrays(n: int, length: int = 4):
    """Return (acceptors, donors) as lists of float32 arrays of given length."""
    acc = [np.array([i * 0.1] * length, dtype=np.float32) for i in range(n)]
    dnr = [np.array([i * 0.2] * length, dtype=np.float32) for i in range(n)]
    return acc, dnr


def _make_manager(
    model_name: Literal["modelp", "spliceai"] = "modelp", max_cache_size: int = 5
) -> SSPManager:
    """Construct an SSPManager with the heavy initialisation mocked out.
    load_frozen_graph is patched so no real model file is needed.
    """
    with patch("panthera.core.ssp.ssp_manager.load_frozen_graph") as mock_lgf:
        mock_lgf.return_value = MagicMock(name="frozen_graph_fn")
        manager = SSPManager(
            model_name=model_name,
            batch_size=2,
            max_cache_size=max_cache_size,
        )
    return manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def modelp_manager():
    return _make_manager(model_name="modelp")


@pytest.fixture()
def spliceai_manager():
    return _make_manager(model_name="spliceai")


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_returns_two_empty_lists(self, modelp_manager):
        acc, dnr = modelp_manager.predict_ssp([])
        assert acc == []
        assert dnr == []

    def test_model_is_not_called_on_empty_input(self, modelp_manager):
        with patch("panthera.core.ssp.ssp_manager.modelp_predict") as mock_pred:
            modelp_manager.predict_ssp([])
        mock_pred.assert_not_called()


# ---------------------------------------------------------------------------
# Cache miss — first prediction
# ---------------------------------------------------------------------------


class TestCacheMiss:
    def test_predictions_are_returned(self, modelp_manager):
        seqs = ["ACGT", "TGCA"]
        expected_acc, expected_dnr = _make_arrays(2)

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(expected_acc, expected_dnr),
        ):
            acc, dnr = modelp_manager.predict_ssp(seqs)

        np.testing.assert_array_equal(acc[0], expected_acc[0])
        np.testing.assert_array_equal(acc[1], expected_acc[1])
        np.testing.assert_array_equal(dnr[0], expected_dnr[0])
        np.testing.assert_array_equal(dnr[1], expected_dnr[1])

    def test_model_is_called_once_with_all_seqs(self, modelp_manager):
        seqs = ["ACGT", "TGCA"]
        mock_acc, mock_dnr = _make_arrays(2)

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ) as mock_pred:
            modelp_manager.predict_ssp(seqs)

        mock_pred.assert_called_once()
        _, kwargs = mock_pred.call_args
        assert kwargs["seqs"] == seqs

    def test_results_are_stored_in_cache(self, modelp_manager):
        seqs = ["ACGT"]
        mock_acc, mock_dnr = _make_arrays(1)

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ):
            modelp_manager.predict_ssp(seqs)

        assert "ACGT" in modelp_manager._cache


# ---------------------------------------------------------------------------
# Cache hit — repeated prediction
# ---------------------------------------------------------------------------


class TestCacheHit:
    def test_model_not_called_on_second_request(self, modelp_manager):
        seqs = ["ACGT"]
        mock_acc, mock_dnr = _make_arrays(1)

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ) as mock_pred:
            modelp_manager.predict_ssp(seqs)  # populates cache
            modelp_manager.predict_ssp(seqs)  # should hit cache

        assert mock_pred.call_count == 1

    def test_cached_values_are_returned_correctly(self, modelp_manager):
        seqs = ["ACGT"]
        mock_acc, mock_dnr = _make_arrays(1)

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ):
            modelp_manager.predict_ssp(seqs)
            acc2, dnr2 = modelp_manager.predict_ssp(seqs)

        np.testing.assert_array_equal(acc2[0], mock_acc[0])
        np.testing.assert_array_equal(dnr2[0], mock_dnr[0])


# ---------------------------------------------------------------------------
# Mixed cache hit / miss
# ---------------------------------------------------------------------------


class TestMixedCacheHitMiss:
    def test_only_uncached_seqs_sent_to_model(self, modelp_manager):
        warm_seq = "AAAA"
        cold_seq = "CCCC"

        warm_acc, warm_dnr = _make_arrays(1)
        cold_acc, cold_dnr = _make_arrays(1)

        # Warm the cache for warm_seq
        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(warm_acc, warm_dnr),
        ):
            modelp_manager.predict_ssp([warm_seq])

        # Now request both; only cold_seq should reach the model
        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(cold_acc, cold_dnr),
        ) as mock_pred:
            modelp_manager.predict_ssp([warm_seq, cold_seq])

        _, kwargs = mock_pred.call_args
        assert kwargs["seqs"] == [cold_seq]

    def test_output_order_matches_input_order(self, modelp_manager):
        warm_seq, cold_seq = "AAAA", "CCCC"
        warm_acc, warm_dnr = _make_arrays(1, length=4)
        cold_acc = [np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)]
        cold_dnr = [np.array([8.0, 8.0, 8.0, 8.0], dtype=np.float32)]

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(warm_acc, warm_dnr),
        ):
            modelp_manager.predict_ssp([warm_seq])

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(cold_acc, cold_dnr),
        ):
            # Input order: [warm, cold]
            acc, dnr = modelp_manager.predict_ssp([warm_seq, cold_seq])

        # First position → warm_seq's result
        np.testing.assert_array_equal(acc[0], warm_acc[0])
        # Second position → cold_seq's result
        np.testing.assert_array_equal(acc[1], cold_acc[0])


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_oldest_entry_evicted_when_cache_full(self):
        manager = _make_manager(max_cache_size=3)
        seqs = ["S1", "S2", "S3"]
        mock_acc, mock_dnr = _make_arrays(3)

        # Fill the cache to capacity
        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ):
            manager.predict_ssp(seqs)

        # Adding one more sequence must evict "S1" (the oldest)
        extra_acc, extra_dnr = _make_arrays(1)
        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(extra_acc, extra_dnr),
        ):
            manager.predict_ssp(["S4"])

        assert "S1" not in manager._cache
        assert "S4" in manager._cache

    def test_cache_does_not_exceed_max_size(self):
        max_size = 3
        manager = _make_manager(max_cache_size=max_size)

        for i in range(max_size + 2):
            mock_acc, mock_dnr = _make_arrays(1)
            with patch(
                "panthera.core.ssp.ssp_manager.modelp_predict",
                return_value=(mock_acc, mock_dnr),
            ):
                manager.predict_ssp([f"SEQ{i}"])

        assert len(manager._cache) <= max_size


# ---------------------------------------------------------------------------
# LRU recency update on cache hit
# ---------------------------------------------------------------------------


class TestLRURecencyUpdate:
    def test_cache_hit_promotes_entry_to_most_recent(self):
        """Fill cache to capacity, re-access the oldest entry, then insert a new
        sequence. The re-accessed entry must survive; the *second* oldest must
        be evicted instead.
        """
        manager = _make_manager(max_cache_size=3)
        seqs = ["S1", "S2", "S3"]
        mock_acc, mock_dnr = _make_arrays(3)

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ):
            manager.predict_ssp(seqs)

        # Re-access "S1" — it should move to the "recently used" end
        manager.predict_ssp(["S1"])

        # Now add a new sequence, which must evict "S2" (now the true oldest)
        extra_acc, extra_dnr = _make_arrays(1)
        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(extra_acc, extra_dnr),
        ):
            manager.predict_ssp(["S4"])

        assert "S1" in manager._cache, "S1 should survive after being re-accessed"
        assert "S2" not in manager._cache, "S2 should be evicted as the true oldest"


# ---------------------------------------------------------------------------
# reverse_output
# ---------------------------------------------------------------------------


class TestReverseOutput:
    def test_arrays_are_reversed_when_flag_is_true(self, modelp_manager):
        seqs = ["ACGT"]
        mock_acc = [np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        mock_dnr = [np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)]

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ):
            acc, dnr = modelp_manager.predict_ssp(seqs, reverse_output=True)

        np.testing.assert_array_equal(acc[0], mock_acc[0][::-1])
        np.testing.assert_array_equal(dnr[0], mock_dnr[0][::-1])

    def test_arrays_are_not_reversed_by_default(self, modelp_manager):
        seqs = ["ACGT"]
        mock_acc = [np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        mock_dnr = [np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)]

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ):
            acc, dnr = modelp_manager.predict_ssp(seqs)

        np.testing.assert_array_equal(acc[0], mock_acc[0])
        np.testing.assert_array_equal(dnr[0], mock_dnr[0])

    def test_reverse_applied_to_cached_results(self, modelp_manager):
        """reverse_output must also apply when all sequences are cache hits."""
        seqs = ["ACGT"]
        mock_acc = [np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        mock_dnr = [np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)]

        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ):
            modelp_manager.predict_ssp(seqs)  # populate cache

        # Second call hits cache; reverse_output should still apply
        acc, dnr = modelp_manager.predict_ssp(seqs, reverse_output=True)
        np.testing.assert_array_equal(acc[0], mock_acc[0][::-1])
        np.testing.assert_array_equal(dnr[0], mock_dnr[0][::-1])


# ---------------------------------------------------------------------------
# Model dispatch (modelp vs spliceai)
# ---------------------------------------------------------------------------


class TestModelDispatch:
    def test_modelp_branch_calls_modelp_predict(self, modelp_manager):
        mock_acc, mock_dnr = _make_arrays(1)
        with (
            patch(
                "panthera.core.ssp.ssp_manager.modelp_predict",
                return_value=(mock_acc, mock_dnr),
            ) as mp,
            patch("panthera.core.ssp.ssp_manager.spliceai_predict") as sp,
        ):
            modelp_manager.predict_ssp(["ACGT"])

        mp.assert_called_once()
        sp.assert_not_called()

    def test_spliceai_branch_calls_spliceai_predict(self, spliceai_manager):
        mock_acc, mock_dnr = _make_arrays(1)
        with (
            patch(
                "panthera.core.ssp.ssp_manager.spliceai_predict",
                return_value=(mock_acc, mock_dnr),
            ) as sp,
            patch("panthera.core.ssp.ssp_manager.modelp_predict") as mp,
        ):
            spliceai_manager.predict_ssp(["ACGT"])

        sp.assert_called_once()
        mp.assert_not_called()

    def test_batch_size_forwarded_to_predict_fn(self, modelp_manager):
        mock_acc, mock_dnr = _make_arrays(1)
        with patch(
            "panthera.core.ssp.ssp_manager.modelp_predict",
            return_value=(mock_acc, mock_dnr),
        ) as mock_pred:
            modelp_manager.predict_ssp(["ACGT"])

        _, kwargs = mock_pred.call_args
        assert kwargs["batch_size"] == modelp_manager.batch_size
