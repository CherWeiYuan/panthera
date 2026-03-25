import pytest
from unittest.mock import patch
import logging

from panthera.core.orchestrator import PantheraOrchestrator


@pytest.fixture
def mock_orchestrator():
    """Returns a PantheraOrchestrator instance with mocked initialization."""
    with (
        patch("panthera.core.orchestrator.setup_logging"),
        patch("panthera.core.orchestrator.initialize_runtime") as mock_init,
    ):
        mock_init.return_value = {"device": "CPU", "count": 0}
        return PantheraOrchestrator(
            prefix="test_run", outdir="/tmp", model_type="spliceai", silent=True
        )


def test_orchestrator_initialization():
    """Test that the orchestrator initializes external modules correctly."""
    with (
        patch("panthera.core.orchestrator.setup_logging") as mock_log,
        patch("panthera.core.orchestrator.initialize_runtime") as mock_runtime,
    ):
        mock_runtime.return_value = {"device": "GPU", "count": 1, "details": ["GPU_0"]}

        orch = PantheraOrchestrator(
            prefix="init_test", outdir="./logs", model_type="modelp", silent=False
        )

        mock_log.assert_called_once_with("./logs", "init_test", False)
        mock_runtime.assert_called_once_with(silent=False)

        assert orch.prefix == "init_test"
        assert orch.outdir == "./logs"
        assert orch.model_type == "modelp"
        assert orch.hardware_info == {"device": "GPU", "count": 1, "details": ["GPU_0"]}


def test_run_survey_no_vcf(mock_orchestrator, caplog):
    """Test run_survey issues a warning if no phased_vcf is provided."""
    with caplog.at_level(logging.WARNING):
        mock_orchestrator.run_survey(fasta="ref.fa")

    assert "No VCF provided. Proceeding with GRCh38 reference only." in caplog.text


def test_run_survey_with_vcf(mock_orchestrator, caplog):
    """Test run_survey does not warn about VCF if one is provided."""
    with caplog.at_level(logging.WARNING):
        mock_orchestrator.run_survey(fasta="ref.fa", phased_vcf="variants.vcf.gz")

    assert "No VCF provided" not in caplog.text


def test_run_survey_exception_handling(mock_orchestrator, caplog):
    """Test run_survey catches and re-raises any exception during logic."""
    # We patch a method inside run_survey to simulate an exception
    with patch("panthera.core.orchestrator.logger.debug") as mock_debug:
        mock_debug.side_effect = ValueError("Simulated processing error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="Simulated processing error"):
                mock_orchestrator.run_survey(fasta="ref.fa")

        assert "A fatal error occurred during the survey process." in caplog.text


def test_run_isolate(mock_orchestrator):
    """Test run_isolate placeholder."""
    # Should run without error
    mock_orchestrator.run_isolate()


def test_query_fasta(mock_orchestrator):
    """Test query_fasta placeholder."""
    # Should run without error
    mock_orchestrator.query_fasta("ref.fa")


def test_query_genomic_range(mock_orchestrator):
    """Test query_genomic_range placeholder."""
    # Should run without error
    mock_orchestrator.query_genomic_range("ref.fa")
