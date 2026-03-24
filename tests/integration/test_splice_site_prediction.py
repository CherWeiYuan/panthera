from pathlib import Path

import numpy as np
import pytest

# =====================================================================
# 1. Real Imports (No Mocks)
# =====================================================================
# We import the real functions and components to test the full pipeline
from panthera.core.ssp.load_model import load_frozen_graph
from panthera.core.ssp.predict import spliceai_predict, modelp_predict

# =====================================================================
# 2. Path Resolution & Setup
# =====================================================================
# Dynamically resolve the absolute path to the models directory based on
# where this test file is located. Adjust the `.parents` level if your
# test folder is nested deeper.
# 1. Start: myproject/tests/integration/test_splice_site_prediction.py
# 2. .parents[0]: tests/integration/
# 3. .parents[1]: tests/
# 4. .parents[2]: myproject/ (The Project Root)
# --- Path Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "src" / "panthera" / "models"
MODELP_PATH = MODELS_DIR / "modelp.pb"
SPLICEAI_PATH = MODELS_DIR / "spliceai.pb"


def test_debug_paths():
    """Run this to verify your structure if tests are skipping."""
    print(f"\nSearching for models in: {MODELS_DIR}")
    assert MODELS_DIR.exists(), f"Directory NOT FOUND: {MODELS_DIR}"
    assert MODELP_PATH.exists(), f"ModelP file NOT FOUND at {MODELP_PATH}"
    assert SPLICEAI_PATH.exists(), f"SpliceAI file NOT FOUND at {SPLICEAI_PATH}"


# =====================================================================
# 3. Model Loading Fixtures
# =====================================================================
# --- Fixtures ---
@pytest.fixture(scope="module")
def real_spliceai_model():
    if not SPLICEAI_PATH.exists():
        pytest.skip(f"SpliceAI model not found at {SPLICEAI_PATH}")
    return load_frozen_graph(str(SPLICEAI_PATH))


@pytest.fixture(scope="module")
def real_modelp_model():
    if not MODELP_PATH.exists():
        pytest.skip(f"ModelP model not found at {MODELP_PATH}")
    return load_frozen_graph(str(MODELP_PATH))


# =====================================================================
# 4. Integration Tests
# =====================================================================
# Skip these tests entirely if the model files aren't physically on the machine
@pytest.mark.skipif(not SPLICEAI_PATH.exists(), 
                    reason="SpliceAI .pb file not found.")
class TestSpliceAIIntegration:
    def test_end_to_end_prediction(self, real_spliceai_model):
        """
        Tests the full SpliceAI pipeline: Encoding -> Model -> Output Parsing.
        """
        # Use a realistic biological sequence length (e.g., 100bp)
        test_seq = "ACGT" * 25
        seqs = [test_seq, test_seq]

        acc, dnr = spliceai_predict(
            seqs=seqs, batch_size=2, 
            spliceai_model=real_spliceai_model
        )

        # 1. Verify exact output shapes match the input sequences
        assert len(acc) == 2
        assert len(dnr) == 2
        assert len(acc[0]) == 100
        assert len(acc[1]) == 100

        # 2. Verify probabilities are bounded between 0 and 1
        # (valid softmax/sigmoid output)
        assert all(0.0 <= p <= 1.0 for p in acc[0])
        assert all(0.0 <= p <= 1.0 for p in dnr[0])


@pytest.mark.skipif(not MODELP_PATH.exists(), 
                    reason="ModelP .pb file not found.")
class TestModelPIntegration:
    def test_end_to_end_sliding_window(self, real_modelp_model):
        """
        Tests the full ModelP pipeline, ensuring the sliding window logic works.
        """
        # Create a sequence longer than the model_output_len (1000)
        # to force the sliding window and tail-stitching logic to execute.
        test_seq = "ACGT" * 600  # 2400 bp
        seqs = [test_seq]

        acc, dnr = modelp_predict(
            seqs=seqs,
            batch_size=1,
            model_fn=real_modelp_model,
            crop_len=1000,
            model_input_len=3000,
            model_output_len=1000,
        )

        # 1. Verify output shape matches exact
        # input length despite batch padding
        assert len(acc) == 1
        assert len(acc[0]) == 2400
        assert len(dnr[0]) == 2400

        # 2. Verify valid probability bounds
        assert all(0.0 <= p <= 1.0 for p in acc[0])

        # 3. Ensure there are no NaN values
        # introduced by padding logic
        assert not np.isnan(acc[0]).any()
        assert not np.isnan(dnr[0]).any()
