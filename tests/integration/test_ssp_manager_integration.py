"""
Integration tests for the Splice Site Probability Manager.

These tests load the actual TensorFlow .pb frozen graphs and run real predictions.
They are marked with '@pytest.mark.integration' so they can be run selectively.
"""

import os
import pytest
import numpy as np

# Adjust imports to match your project structure
from panthera.core.ssp.ssp_manager import SSPManager
from panthera.core.ssp.model_paths import MODELP_MODEL_PATH, SPLICEAI_MODEL_PATH

# Check if model files exist locally to prevent CI pipeline crashes
HAS_MODELP = os.path.exists(MODELP_MODEL_PATH)
HAS_SPLICEAI = os.path.exists(SPLICEAI_MODEL_PATH)


@pytest.fixture
def sample_sequences() -> list[str]:
    """
    Provides sequences of varying lengths to rigorously test dynamic
    batch padding and sequence length reconstruction.
    """
    return [
        "A" * 150 + "GT" + "C" * 100 + "AG" + "T" * 150,  # Length: 404
        "G" * 500,  # Length: 500
        "T" * 50 + "ACGT" + "A" * 50,  # Length: 104
    ]


@pytest.mark.integration
class TestSSPManagerIntegration:
    """Integration test suite for SSPManager using real TensorFlow models."""

    @pytest.mark.skipif(
        not HAS_MODELP, reason=f"ModelP weights not found at {MODELP_MODEL_PATH}"
    )
    def test_modelp_real_prediction(self, sample_sequences: list[str]) -> None:
        """Test ModelP loading, batching, and prediction output shapes/bounds."""
        # Use batch_size=2 with 3 sequences to force batch chunking logic
        manager = SSPManager(model_name="modelp", batch_size=2)

        acceptors, donors = manager.predict_ssp(
            seqs=sample_sequences, reverse_output=False
        )

        # 1. Check outer list lengths
        assert len(acceptors) == len(sample_sequences)
        assert len(donors) == len(sample_sequences)

        # 2. Check inner array properties
        for i, seq in enumerate(sample_sequences):
            seq_len = len(seq)

            # Check Shapes: Output array must exactly match input sequence length
            assert acceptors[i].shape == (seq_len,), (
                f"Acceptor shape mismatch at index {i}"
            )
            assert donors[i].shape == (seq_len,), f"Donor shape mismatch at index {i}"

            # Check Types: Must be float32
            assert acceptors[i].dtype == np.float32
            assert donors[i].dtype == np.float32

            # Check Bounds: Probabilities must be between 0.0 and 1.0
            assert np.all((acceptors[i] >= 0.0) & (acceptors[i] <= 1.0))
            assert np.all((donors[i] >= 0.0) & (donors[i] <= 1.0))

    @pytest.mark.skipif(
        not HAS_SPLICEAI, reason=f"SpliceAI weights not found at {SPLICEAI_MODEL_PATH}"
    )
    def test_spliceai_real_prediction(self, sample_sequences: list[str]) -> None:
        """Test SpliceAI loading, batching, and prediction output shapes/bounds."""
        manager = SSPManager(model_name="spliceai", batch_size=2)

        acceptors, donors = manager.predict_ssp(
            seqs=sample_sequences, reverse_output=False
        )

        assert len(acceptors) == len(sample_sequences)

        for i, seq in enumerate(sample_sequences):
            seq_len = len(seq)
            assert acceptors[i].shape == (seq_len,)
            assert donors[i].shape == (seq_len,)
            assert acceptors[i].dtype == np.float32

            # SpliceAI probabilities should also be bounded [0, 1]
            assert np.all((acceptors[i] >= 0.0) & (acceptors[i] <= 1.0))
            assert np.all((donors[i] >= 0.0) & (donors[i] <= 1.0))

    @pytest.mark.skipif(
        not HAS_MODELP, reason="ModelP weights required for reverse test"
    )
    def test_real_reverse_output_logic(self, sample_sequences: list[str]) -> None:
        """
        Verify that reverse_output=True perfectly reverses the real probability
        arrays compared to reverse_output=False.
        """
        manager = SSPManager(model_name="modelp", batch_size=4)

        # Run Forward
        acc_forward, dnr_forward = manager.predict_ssp(
            seqs=sample_sequences, reverse_output=False
        )

        # Run Reverse
        acc_reverse, dnr_reverse = manager.predict_ssp(
            seqs=sample_sequences, reverse_output=True
        )

        for i in range(len(sample_sequences)):
            # The reversed array reversed again [::-1] should perfectly match the forward array
            np.testing.assert_array_equal(acc_forward[i], acc_reverse[i][::-1])
            np.testing.assert_array_equal(dnr_forward[i], dnr_reverse[i][::-1])
