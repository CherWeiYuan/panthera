import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from panthera.core.pipelines.query_fasta import run_query_fasta

### --- Test Data & Fixtures --- ###


@pytest.fixture
def valid_inputs():
    return {
        "fasta_file": "dummy.fasta",
        "model_name": "spliceai",
        "outdir": "/dummy/outdir",
        "prefix": "test_output",
    }


### --- Edge Case: Input Validation --- ###


@pytest.mark.parametrize(
    "missing_param, override_val, expected_err",
    [
        ("fasta_file", "", "Fasta file is required."),
        ("fasta_file", None, "Fasta file is required."),
        ("model_name", "", "Model name is required."),
        ("model_name", None, "Model name is required."),
        ("outdir", "", "Output directory is required."),
        ("outdir", None, "Output directory is required."),
        ("prefix", "", "Prefix is required."),
        ("prefix", None, "Prefix is required."),
    ],
)
def test_run_query_fasta_validation_errors(
    valid_inputs, missing_param, override_val, expected_err
):
    inputs = valid_inputs.copy()
    inputs[missing_param] = override_val

    with pytest.raises(ValueError, match=expected_err):
        run_query_fasta(**inputs)


### --- Happy Path: Successful Execution --- ###


# ALL PATCHES NOW POINT TO query_fasta
@patch("panthera.core.pipelines.query_fasta.logger")
@patch("panthera.core.pipelines.query_fasta.write_wig")
@patch("panthera.core.pipelines.query_fasta.prepare_wig_dataframe")
@patch("panthera.core.pipelines.query_fasta.SSPManager")
@patch("panthera.core.pipelines.query_fasta.GenomeParser")
def test_run_query_fasta_success_single_sequence(
    mock_genome_parser,
    mock_ssp_manager,
    mock_prepare_wig,
    mock_write_wig,
    mock_logger,
    valid_inputs,
):
    # 1. Setup Mocks
    mock_dict = {"chr1": "ACGT"}
    mock_genome_parser.return_value.parse_genome.return_value = mock_dict
    mock_genome_parser.parse_genome.return_value = mock_dict

    mock_ssp_instance = MagicMock()
    mock_acc_prob = np.array([0.1, 0.0, 0.8])
    mock_dnr_prob = np.array([0.0, 0.5, 0.0])
    mock_ssp_instance.predict_ssp.return_value = ([mock_acc_prob], [mock_dnr_prob])
    mock_ssp_manager.return_value = mock_ssp_instance

    dummy_df = pd.DataFrame({"pos": [1, 2, 3], "prob": [0.1, -0.5, 0.8]})
    mock_prepare_wig.return_value = dummy_df

    # 2. Execute Function
    run_query_fasta(**valid_inputs)

    # 3. Assertions
    if mock_genome_parser.parse_genome.called:
        mock_genome_parser.parse_genome.assert_called_once_with("dummy.fasta")
    else:
        mock_genome_parser.return_value.parse_genome.assert_called_once_with(
            "dummy.fasta"
        )

    mock_ssp_manager.assert_called_once_with(
        model_name="spliceai", batch_size=1, max_cache_size=1000
    )
    mock_ssp_instance.predict_ssp.assert_called_once_with(
        seqs=["ACGT"], reverse_output=False
    )

    mock_prepare_wig.assert_called_once()
    kwargs = mock_prepare_wig.call_args.kwargs
    assert kwargs["start"] == 0
    np.testing.assert_array_equal(kwargs["acceptor_prob"], mock_acc_prob)
    np.testing.assert_array_equal(kwargs["donor_prob"], mock_dnr_prob)

    mock_write_wig.assert_called_once()
    mock_logger.info.assert_called_once()


@patch("panthera.core.pipelines.query_fasta.write_wig")
@patch("panthera.core.pipelines.query_fasta.prepare_wig_dataframe")
@patch("panthera.core.pipelines.query_fasta.SSPManager")
@patch("panthera.core.pipelines.query_fasta.GenomeParser")
def test_run_query_fasta_multiple_sequences(
    mock_genome_parser, mock_ssp_manager, mock_prepare_wig, mock_write_wig, valid_inputs
):
    mock_dict = {"chr1": "ACGT", "chr2": "TGCA"}
    mock_genome_parser.return_value.parse_genome.return_value = mock_dict
    mock_genome_parser.parse_genome.return_value = mock_dict

    mock_ssp_instance = MagicMock()
    mock_ssp_instance.predict_ssp.return_value = ([np.array([])], [np.array([])])
    mock_ssp_manager.return_value = mock_ssp_instance

    run_query_fasta(**valid_inputs)

    assert mock_ssp_instance.predict_ssp.call_count == 2
    assert mock_prepare_wig.call_count == 2
    assert mock_write_wig.call_count == 2


### --- Edge Case: Collision Exception Handling --- ###


@patch("panthera.core.pipelines.query_fasta.logger")
@patch("panthera.core.pipelines.query_fasta.write_wig")
@patch("panthera.core.pipelines.query_fasta.SSPManager")
@patch("panthera.core.pipelines.query_fasta.GenomeParser")
# Notice: We intentionally do NOT patch prepare_wig_dataframe here!
def test_run_query_fasta_actual_array_collision(
    mock_genome_parser, mock_ssp_manager, mock_write_wig, mock_logger, valid_inputs
):
    """Tests the pipeline's handling of a real array collision using the actual wig module."""
    # 1. Setup Mocks
    mock_dict = {"chr_collision": "ACGT"}
    mock_genome_parser.return_value.parse_genome.return_value = mock_dict
    mock_genome_parser.parse_genome.return_value = mock_dict

    mock_ssp_instance = MagicMock()

    # Create actual numpy arrays where both have a non-zero value at index 1
    mock_acc_prob = np.array([0.0, 0.8, 0.0, 0.0], dtype=np.float32)
    mock_dnr_prob = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)

    mock_ssp_instance.predict_ssp.return_value = ([mock_acc_prob], [mock_dnr_prob])
    mock_ssp_manager.return_value = mock_ssp_instance

    # 2. Execute
    run_query_fasta(**valid_inputs)

    # 3. Assertions
    # Because the REAL prepare_wig_dataframe is running, it will detect the overlapping
    # 0.8 and 0.5 at index 1 and raise a ValueError internally.

    # write_wig should never be reached
    mock_write_wig.assert_not_called()

    # Verify the exception was safely caught by the try/except block and logged
    mock_logger.error.assert_called_once()

    # We can also verify the log message contains the expected formatting
    logged_msg = mock_logger.error.call_args[0][0]
    assert "Error generating WIG file for chr_collision:" in logged_msg


@patch("panthera.core.pipelines.query_fasta.logger")
@patch("panthera.core.pipelines.query_fasta.write_wig")
@patch("panthera.core.pipelines.query_fasta.prepare_wig_dataframe")
@patch("panthera.core.pipelines.query_fasta.SSPManager")
@patch("panthera.core.pipelines.query_fasta.GenomeParser")
def test_run_query_fasta_handles_wig_collision(
    mock_genome_parser,
    mock_ssp_manager,
    mock_prepare_wig,
    mock_write_wig,
    mock_logger,
    valid_inputs,
):
    # Setup mock_dict robustly here as well
    mock_dict = {"chr_bad": "ACGT"}
    mock_genome_parser.return_value.parse_genome.return_value = mock_dict
    mock_genome_parser.parse_genome.return_value = mock_dict

    mock_ssp_instance = MagicMock()
    mock_ssp_instance.predict_ssp.return_value = ([np.array([])], [np.array([])])
    mock_ssp_manager.return_value = mock_ssp_instance

    error_msg = "Collision at positions: [10]"
    mock_prepare_wig.side_effect = ValueError(error_msg)

    run_query_fasta(**valid_inputs)

    mock_write_wig.assert_not_called()
    mock_logger.error.assert_called_once_with(
        f"Error generating WIG file for chr_bad: {error_msg}"
    )
