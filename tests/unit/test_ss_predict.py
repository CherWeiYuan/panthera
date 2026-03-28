import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

from panthera.core.ssp.predict import modelp_predict, spliceai_predict

# --- Fixtures for Mocking --- #


@pytest.fixture
def mock_dependencies():
    """
    Mocks the SeqEncoder and EncodingSchema to prevent the tests from
    requiring the actual panthera module or actual encoding logic.
    """
    with (
        patch("panthera.core.ssp.onehotencoder.SeqEncoder") as MockEncoder,
        patch("panthera.core.ssp.onehotencoder.EncodingSchema") as MockSchema,
    ):
        # Make one_hot_encode return a dummy numpy array of shape (len(seq), 4)
        instance = MockEncoder.return_value
        instance.one_hot_encode.side_effect = lambda seq, schema: np.zeros(
            (len(seq), 4)
        )

        yield MockEncoder, MockSchema


# --- SpliceAI Tests --- #


def test_spliceai_predict_empty_seqs():
    """Test that empty sequences return empty lists cleanly."""
    mock_model = MagicMock()
    acc, dnr = spliceai_predict([], batch_size=2, spliceai_fn=mock_model)
    assert acc == []
    assert dnr == []
    mock_model.assert_not_called()


def test_spliceai_predict_valid_input(mock_dependencies):
    seqs = ["ACGT", "A"]  # Lengths 4 and 1. Max length X = 4.

    def mock_spliceai(tensor):
        # Tensor shape is (batch_size, 5000 + max_len + 5000, 4)
        batch_size, padded_len, _ = tensor.shape

        # The model strips the 10,000 context bases and returns max_len
        max_len = padded_len - 10000

        # Returns shape: (batch_size, max_len, 3)
        return (tf.ones((batch_size, max_len, 3), dtype=tf.float32),)

    acc, dnr = spliceai_predict(seqs, batch_size=2, spliceai_fn=mock_spliceai)

    assert len(acc) == 2
    assert len(dnr) == 2
    assert len(acc[0]) == 4  # First sequence was length 4
    assert len(acc[1]) == 1  # Second sequence was length 1
    assert acc[0][0] == 1.0


def test_spliceai_predict_batch_size_adjustment(mock_dependencies):
    """Test that a batch size < 1 is safely forced to 1."""
    seqs = ["ACGT", "TGCA"]

    def mock_model(tensor):
        # Assert batch size is exactly 1 due to the internal fix
        assert tensor.shape[0] == 1
        batch_size, seq_len, _ = tensor.shape
        return (tf.ones((batch_size, seq_len, 3), dtype=tf.float32),)

    spliceai_predict(seqs, batch_size=-5, spliceai_fn=mock_model)


def test_spliceai_predict_model_failure(mock_dependencies):
    """Test that model exceptions are caught and raised as RuntimeErrors."""

    def failing_model(tensor):
        raise ValueError("Simulated OOM or GPU Error")

    with pytest.raises(RuntimeError, match="Model prediction failed: Simulated OOM"):
        spliceai_predict(["ACGT"], 1, failing_model)


def test_spliceai_predict_sequence_loss(mock_dependencies):
    """Test defense against the model returning fewer sequences than submitted."""

    def loss_model(tensor):
        # Simulate returning 1 less sequence than the batch contains
        batch_size, seq_len, _ = tensor.shape
        return (tf.ones((batch_size - 1, seq_len, 3)),)

    with pytest.raises(RuntimeError, match="Sequence loss detected"):
        spliceai_predict(["ACGT", "TGCA"], 2, loss_model)


def test_spliceai_predict_length_mismatch(mock_dependencies):
    """Test defense against the model returning truncated sequence lengths."""

    def mismatch_model(tensor):
        batch_size, padded_len, _ = tensor.shape
        max_len = padded_len - 10000

        # Cut the expected output sequence length short by 5 bases
        return (tf.ones((batch_size, max_len - 5, 3)),)

    # Sequence of 50 -> padded to 10050.
    # Mock returns 45.
    # Slicing [:50] on an array of 45 will result in an array of 45.
    # The check (seq_len == len(acc)) -> (50 == 45) will fail,
    # triggering the error.
    with pytest.raises(RuntimeError, match="Prediction output length mismatch"):
        spliceai_predict(["A" * 50], 1, mismatch_model)


# --- ModelP Tests --- #


def test_modelp_predict_empty_seqs():
    """Test ModelP handles empty inputs gracefully."""
    mock_model = MagicMock()
    acc, dnr = modelp_predict([], 2, mock_model)
    assert acc == []
    assert dnr == []


def test_modelp_predict_valid_input(mock_dependencies):
    seqs = ["A" * 500, "C" * 1500]

    def mock_modelp(tensor):
        batch_size, input_len, _ = tensor.shape

        # Assert the model is strictly receiving the 3000bp window
        # (1000 context + 1000 target + 1000 context)
        assert input_len == 3000

        # Model strictly returns the middle 1000bp
        return (tf.ones((batch_size, 1000, 2), dtype=tf.float32),)

    acc, dnr = modelp_predict(
        seqs=seqs,
        batch_size=2,
        modelp_fn=mock_modelp,
        crop_len=1000,
        model_input_len=3000,
        model_output_len=1000,
    )

    assert len(acc) == 2
    assert len(dnr) == 2
    assert len(acc[0]) == 500
    assert len(acc[1]) == 1500


def test_modelp_predict_tiny_sequence(mock_dependencies):
    """Test ModelP specifically on sequences smaller than the output window."""
    seqs = ["A" * 10]

    def mock_model(tensor):
        batch_size = tensor.shape[0]
        return (tf.ones((batch_size, 1000, 2), dtype=tf.float32),)

    acc, dnr = modelp_predict(
        seqs=seqs,
        batch_size=1,
        modelp_fn=mock_model,
        crop_len=1000,
        model_input_len=3000,
        model_output_len=1000,
    )

    assert len(acc) == 1
    assert len(acc[0]) == 10  # Result sliced correctly back to 10
