"""
Unit tests for the Splice Site Probability Manager.
"""

from unittest.mock import patch
import numpy as np
import pytest

from panthera.core.ssp.ssp_manager import SSPManager
from panthera.core.ssp.model_paths import MODELP_MODEL_PATH, SPLICEAI_MODEL_PATH


@pytest.fixture
def dummy_seqs() -> list[str]:
    """Fixture providing standard dummy sequences."""
    return ["ACGTACGT", "GGCCTTAA"]


@pytest.fixture
def mock_predictions() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Fixture providing deterministic mock probability arrays.
    Creates asymmetric arrays to verify the reverse sequence logic.
    """
    acceptor_arrays = [
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        np.array([0.5, 0.6], dtype=np.float32),
    ]
    donor_arrays = [
        np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        np.array([0.4, 0.3], dtype=np.float32),
    ]
    return acceptor_arrays, donor_arrays


class TestSSPManagerInitialization:
    def test_init_loads_correct_model_modelp(self, mocker):
        # 1. Setup: Patch the function INSIDE the manager module
        mock_load = mocker.patch("panthera.core.ssp.ssp_manager.load_frozen_graph")
        mock_fn = mocker.MagicMock()
        mock_load.return_value = mock_fn

        # 2. Act
        manager = SSPManager(model_name="modelp", batch_size=32)

        # 3. Assert
        assert manager.model_fn == mock_fn
        mock_load.assert_called_once_with(MODELP_MODEL_PATH)

    def test_init_loads_correct_model_spliceai(self, mocker):
        mock_load = mocker.patch("panthera.core.ssp.ssp_manager.load_frozen_graph")
        mock_fn = mocker.MagicMock()
        mock_load.return_value = mock_fn

        manager = SSPManager(model_name="spliceai", batch_size=32)

        assert manager.model_fn == mock_fn
        mock_load.assert_called_once_with(SPLICEAI_MODEL_PATH)


class TestSSPManagerPredictions:
    @pytest.fixture
    def mock_preds(self):
        """Deterministic mock data for consistency."""
        return (
            [np.array([0.1], dtype=np.float32)],
            [np.array([0.9], dtype=np.float32)],
        )

    def test_predict_ssp_modelp_forward(self, mocker, mock_preds):
        # Patch load_frozen_graph so __init__ doesn't fail
        mocker.patch("panthera.core.ssp.ssp_manager.load_frozen_graph")

        # Patch the prediction function where it is used (in the manager)
        mock_modelp = mocker.patch("panthera.core.ssp.ssp_manager.modelp_predict")
        mock_modelp.return_value = mock_preds

        manager = SSPManager(model_name="modelp", batch_size=16)
        acc, dnr = manager.predict_ssp(seqs=["ACGT"], reverse_output=False)

        # Check that the mock was actually hit
        mock_modelp.assert_called_once()
        assert acc == mock_preds[0]

    def test_predict_ssp_spliceai_reverse(self, mocker, mock_preds):
        mocker.patch("panthera.core.ssp.ssp_manager.load_frozen_graph")

        # Patch where it's used
        mock_splice = mocker.patch("panthera.core.ssp.ssp_manager.spliceai_predict")
        mock_splice.return_value = mock_preds

        manager = SSPManager(model_name="spliceai", batch_size=16)
        # We expect reversal logic here
        acc, dnr = manager.predict_ssp(seqs=["ACGT"], reverse_output=True)

        mock_splice.assert_called_once()
        # The mock returned [0.1], so [::-1] of [0.1] is still [0.1]
        np.testing.assert_array_equal(acc[0], np.array([0.1], dtype=np.float32))


class TestSSPManagerUtilities:
    """Tests covering sequence string manipulation tools."""

    manager: SSPManager

    @pytest.fixture(autouse=True)
    def setup_manager(self) -> None:
        """Provides a mocked manager instance for utility testing."""
        with patch("panthera.core.ssp.load_model.load_frozen_graph"):
            self.manager = SSPManager(model_name="modelp", batch_size=32)

    def test_remove_indel_markers(self) -> None:
        """Test removal of special formatting characters."""
        input_seqs = ["A>T<C", "G{G}C", ">A<C{G}T>"]
        expected = ["ATC", "GGC", "ACGT"]

        result = self.manager.remove_indel_markers(input_seqs)
        assert result == expected

    def test_reverse_complement(self) -> None:
        """Test accurate reverse complement generation."""
        # A <-> T, C <-> G, reversed.
        input_seqs = ["ATGC", "CGTA", "N"]
        expected = ["GCAT", "TACG", "N"]

        result = self.manager.reverse_complement(input_seqs)
        assert result == expected

    def test_predict_ssp_empty_input(self) -> None:
        """predict_ssp called with empty sequences should return empty output."""
        acc, dnr = self.manager.predict_ssp(seqs=[], reverse_output=False)
        assert acc == []
        assert dnr == []


def test_ssp_manager_invalid_model():
    """Initializing manager with an unknown model raises ValueError."""
    with patch("panthera.core.ssp.ssp_manager.load_frozen_graph"):
        with pytest.raises(ValueError, match="Unexpected model name"):
            SSPManager(model_name="invalid", batch_size=32)  # type: ignore
