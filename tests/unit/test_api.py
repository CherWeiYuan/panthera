import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from panthera.api import load_model, predict
from panthera.core.ssp.ssp_manager import SSPManager


def test_load_model_success():
    """Test that load_model properly initialises an SSPManager with batch_size 1."""
    with patch("panthera.api.SSPManager") as mock_ssp_manager_cls:
        # Arrange
        mock_instance = MagicMock(spec=SSPManager)
        mock_ssp_manager_cls.return_value = mock_instance

        # Act
        manager = load_model("modelp")

        # Assert
        mock_ssp_manager_cls.assert_called_once_with("modelp", batch_size=1)
        assert manager is mock_instance


def test_load_model_invalid_name():
    """Test that load_model propagates exceptions from SSPManager init."""
    with patch("panthera.api.SSPManager") as mock_ssp_manager_cls:
        # Arrange
        mock_ssp_manager_cls.side_effect = ValueError("Invalid model name")

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid model name"):
            load_model("invalid_model")  # type: ignore[reportArgumentType]


def test_predict_success():
    """Test standard valid prediction logic."""
    # Arrange
    mock_model = MagicMock(spec=SSPManager)
    # create arrays simulating numpy arrays returned from SSP predict
    mock_acceptor = [np.array([0.1, 0.2, 0.3], dtype=np.float32)]
    mock_donor = [np.array([0.4, 0.5, 0.6], dtype=np.float32)]
    mock_model.predict_ssp.return_value = (mock_acceptor, mock_donor)

    seq = "ACGT"

    # Act
    acceptor, donor = predict(seq, mock_model)

    # Assert
    mock_model.predict_ssp.assert_called_once_with(["ACGT"], reverse_output=False)
    np.testing.assert_array_equal(acceptor, mock_acceptor[0])
    np.testing.assert_array_equal(donor, mock_donor[0])


def test_predict_empty_sequence_raises_index_error_if_model_returns_empty():
    """Test how predict() handles a model that returns empty arrays (e.g. for invalid seq)."""
    # Arrange
    mock_model = MagicMock(spec=SSPManager)
    mock_model.predict_ssp.return_value = ([], [])

    seq = ""

    # Act & Assert
    with pytest.raises(IndexError):
        predict(seq, mock_model)

    mock_model.predict_ssp.assert_called_once_with([""], reverse_output=False)


def test_predict_model_raises_exception():
    """Test that predict() propagates exceptions from model prediction."""
    # Arrange
    mock_model = MagicMock(spec=SSPManager)
    mock_model.predict_ssp.side_effect = RuntimeError("Prediction failed")

    seq = "ACGT"

    # Act & Assert
    with pytest.raises(RuntimeError, match="Prediction failed"):
        predict(seq, mock_model)


def test_predict_extracts_first_element():
    """Test that predict() unpacks correctly even if multiple batches are somehow returned."""
    # Arrange
    mock_model = MagicMock(spec=SSPManager)
    mock_acceptor = [
        np.array([0.1], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
    ]
    mock_donor = [np.array([0.2], dtype=np.float32), np.array([0.8], dtype=np.float32)]
    mock_model.predict_ssp.return_value = (mock_acceptor, mock_donor)

    seq = "ACGT"

    # Act
    acceptor, donor = predict(seq, mock_model)

    # Assert
    mock_model.predict_ssp.assert_called_once_with([seq], reverse_output=False)
    np.testing.assert_array_equal(acceptor, mock_acceptor[0])
    np.testing.assert_array_equal(donor, mock_donor[0])
    # Ensure it only returned the first array, not a list of arrays
    assert isinstance(acceptor, np.ndarray)
    assert isinstance(donor, np.ndarray)


def test_predict_invalid_seq_type():
    """Test that predict() raises TypeError when a non-string sequence is provided."""
    mock_model = MagicMock(spec=SSPManager)

    with pytest.raises(TypeError, match="Sequence must be a string."):
        predict(123, mock_model)  # type: ignore[reportArgumentType]


def test_predict_invalid_model_type():
    """Test that predict() raises TypeError when a non-SSPManager model is provided."""
    seq = "ACGT"

    with pytest.raises(TypeError, match="Model must be a SSPManager object."):
        predict(seq, "not_a_model")  # type: ignore[reportArgumentType]
