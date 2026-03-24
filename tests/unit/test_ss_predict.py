import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from panthera.core.ssp.predict import spliceai_predict, modelp_predict

# =====================================================================
# 1. Targeted Dependency Mocking (Scenario B)
# =====================================================================
# Create a dummy module to stand in for the one-hot encoder
mock_encoder_module = MagicMock()

# Configure the mocked SeqEncoder to return a valid numpy array shape
mock_instance = MagicMock()
mock_instance.one_hot_encode.side_effect = lambda seq, schema: np.zeros(
    (len(seq), 4), dtype=np.float32
)
mock_encoder_module.SeqEncoder.return_value = mock_instance

# INJECT ONLY THE DEEP DEPENDENCIES:
# We do NOT mock "panthera" or "panthera.core". This allows Python to
# search your hard drive for the actual panthera/core/splice_site_ml folder.
# (Including both paths you mentioned just to be safe)
sys.modules["panthera.core.ssp.onehotencoder"] = mock_encoder_module
sys.modules["panthera.core.ssp.onehotencoder"] = mock_encoder_module


# =====================================================================
# 2. TensorFlow Model Mock Fixtures
# =====================================================================
@pytest.fixture
def mock_spliceai_model():
    """Mocks the SpliceAI TensorFlow model to return predictable tensor outputs."""

    def _model(tensor_batch):
        # SpliceAI model returns shape (batch_size, sequence_length, 3)
        batch_size = tensor_batch.shape[0]
        seq_len = tensor_batch.shape[1]

        mock_preds = tf.constant(
            np.random.rand(batch_size, seq_len, 3), dtype=tf.float32
        )
        return (mock_preds,)

    return _model


@pytest.fixture
def mock_modelp_model():
    """Mocks the ModelP TensorFlow model to return predictable tensor outputs."""

    def _model(tensor_batch):
        # ModelP is assumed to output shape (batch_size, seq_len, 2)
        batch_size = tensor_batch.shape[0]
        seq_len = tensor_batch.shape[1]

        mock_preds = tf.constant(
            np.random.rand(batch_size, seq_len, 2), dtype=tf.float32
        )
        return (mock_preds,)

    return _model


# =====================================================================
# 3. Tests for spliceai_predict
# =====================================================================
class TestSpliceAIPredict:
    def test_empty_input(self, mock_spliceai_model):
        """Test that empty lists return empty lists without failing."""
        acc, dnr = spliceai_predict(
            [], [], batch_size=2, spliceai_model=mock_spliceai_model
        )
        assert acc == []
        assert dnr == []

    def test_mismatched_lengths(self, mock_spliceai_model):
        """
        Test that mismatched sequence and strand lists raise a ValueError.
        """
        with pytest.raises(ValueError, match="Input mismatch"):
            spliceai_predict(["ACGT"], ["+", "-"], 2, mock_spliceai_model)

    def test_invalid_strand(self, mock_spliceai_model):
        """Test that an invalid strand character raises a ValueError."""
        with pytest.raises(ValueError, match="Invalid strand"):
            spliceai_predict(["ACGT"], ["*"], 2, mock_spliceai_model)

    def test_model_failure(self):
        """
        Test that a failing TensorFlow model correctly bubbles up a RuntimeError.
        """

        def broken_model(tensor_batch):
            raise tf.errors.InternalError(None, None, "GPU Out of Memory")

        with pytest.raises(RuntimeError, match="Model prediction failed"):
            spliceai_predict(["ACGT"], ["+"], 2, broken_model)

    @pytest.mark.parametrize(
        "strand, expected_reverse",
        [("+", False), ("plus", False), ("-", True), ("minus", True)],
    )
    def test_strand_handling_and_valid_output(
        self, strand, expected_reverse, mock_spliceai_model
    ):
        """Test both positive and negative strands output the correct shapes."""
        seqs = ["ACGT", "ACGTAA"]
        strands = [strand, strand]

        acc, dnr = spliceai_predict(
            seqs, strands, batch_size=2, spliceai_model=mock_spliceai_model
        )

        # Check lengths match input sequences perfectly
        assert len(acc) == 2
        assert len(acc[0]) == 4
        assert len(acc[1]) == 6
        assert len(dnr[0]) == 4
        assert len(dnr[1]) == 6

    def test_batch_size_correction(self, mock_spliceai_model):
        """Test that batch sizes < 1 are forced to 1 to prevent infinite loops/crashes."""
        seqs = ["ACGT", "CGTA"]
        strands = ["+", "+"]

        # Should not crash, should process as batch_size=1
        acc, dnr = spliceai_predict(
            seqs, strands, batch_size=-5, spliceai_model=mock_spliceai_model
        )
        assert len(acc) == 2


# =====================================================================
# 4. Tests for modelp_predict
# =====================================================================
class TestModelPPredict:
    def test_mismatched_lengths(self, mock_modelp_model):
        """Test that mismatched lists raise a ValueError."""
        with pytest.raises(
            ValueError, match="Sequence and strand lists must be of equal length"
        ):
            modelp_predict(["ACGT"], [], 2, mock_modelp_model)

    def test_invalid_strand(self, mock_modelp_model):
        """Test that an unrecognized strand raises an error."""
        with pytest.raises(ValueError, match="Invalid strand"):
            modelp_predict(["ACGT"], ["bad_strand"], 2, mock_modelp_model)

    @pytest.mark.parametrize("strand", ["+", "-"])
    def test_valid_prediction_standard_length(self, strand, mock_modelp_model):
        """Test end-to-end sliding window prediction for a standard sequence."""
        # Using a sequence long enough to trigger multiple sliding windows
        seqs = ["A" * 2500, "C" * 3500]
        strands = [strand, strand]

        acc, dnr = modelp_predict(
            seqs,
            strands,
            batch_size=2,
            model_fn=mock_modelp_model,
            crop_len=1000,
            model_input_len=3000,
            model_output_len=1000,
        )

        assert len(acc) == 2
        assert len(dnr) == 2
        assert len(acc[0]) == 2500
        assert len(acc[1]) == 3500

    def test_short_sequence(self, mock_modelp_model):
        """
        Test prediction for sequences shorter than the model output length.
        """
        seq = "A" * 500  # Less than model_output_len (1000)

        acc, dnr = modelp_predict(
            [seq],
            ["+"],
            batch_size=1,
            model_fn=mock_modelp_model,
            crop_len=1000,
            model_input_len=3000,
            model_output_len=1000,
        )

        assert len(acc[0]) == 500
        assert len(dnr[0]) == 500
