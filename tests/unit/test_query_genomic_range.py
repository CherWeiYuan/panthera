import re

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Assuming the module is named query_genomic_range.py and is in the Python path
from panthera.core.pipelines.query_genomic_range import (
    run_query_genomic_range,
    TRACK_COLOR,
    ALT_COLOR,
)


### --- Test Data & Fixtures --- ###


@pytest.fixture
def valid_inputs():
    """Provides valid base inputs for the function."""
    return {
        "fasta_file": "dummy.fasta",
        "genomic_range": "chr1:10-20-plus",
        "model_name": "spliceai",
        "outdir": "/dummy/outdir",
        "prefix": "test_output",
    }


@pytest.fixture
def mock_genome_sequence():
    """Returns a dummy genome dictionary for chr1."""
    # 30-base dummy sequence
    # 1-based indexing:
    # 123456789 0123456789 0123456789
    # AAAAAAAAA CCGGTTCCGG TTTTTTTTTT
    return {"chr1": "AAAAAAAAACCGGTTCCGGTTTTTTTTTT"}


### --- Edge Case: Input Validation & Parsing --- ###


@pytest.mark.parametrize(
    "missing_param, override_val, expected_err",
    [
        ("fasta_file", "", "Fasta file is required."),
        ("fasta_file", None, "Fasta file is required."),
        ("genomic_range", "", "Genomic range is required."),
        ("model_name", "", "Model name is required."),
        ("outdir", "", "Output directory is required."),
        ("prefix", "", "Prefix is required."),
    ],
)
def test_run_query_validation_missing_args(
    valid_inputs, missing_param, override_val, expected_err
):
    """Tests that missing or empty string arguments raise ValueErrors."""
    inputs = valid_inputs.copy()
    inputs[missing_param] = override_val

    with pytest.raises(ValueError, match=expected_err):
        run_query_genomic_range(**inputs)


@pytest.mark.parametrize(
    "bad_range, expected_err_match",
    [
        ("chr1", "Genomic range must have at least 2 colon-separated fields"),
        ("chr1:1000", "Genomic range must have at least 3 dash-separated fields"),
        ("chr1:1000-2000", "Genomic range must have at least 3 dash-separated fields"),
        (
            "chr1:start-end-plus",
            "Position field in variant_target is not a valid integer",
        ),
    ],
)
def test_run_query_invalid_genomic_ranges(valid_inputs, bad_range, expected_err_match):
    """Tests proper exception raising for malformed genomic_range strings."""
    inputs = valid_inputs.copy()
    inputs["genomic_range"] = bad_range

    with pytest.raises(ValueError, match=expected_err_match):
        run_query_genomic_range(**inputs)


@patch("panthera.core.pipelines.query_genomic_range.GenomeParser")
def test_run_query_invalid_strand(
    mock_genome_parser, valid_inputs, mock_genome_sequence
):
    """Tests that an invalid strand string throws an error before prediction."""
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_genome.return_value = mock_genome_sequence
    mock_genome_parser.return_value = mock_parser_instance

    inputs = valid_inputs.copy()
    inputs["genomic_range"] = "chr1:10-20-invalidstrand"

    with pytest.raises(ValueError, match="Invalid strand: invalidstrand"):
        run_query_genomic_range(**inputs)


### --- Happy Path: Successful Execution --- ###


@patch("panthera.core.pipelines.query_genomic_range.logger")
@patch("panthera.core.pipelines.query_genomic_range.write_wig")
@patch("panthera.core.pipelines.query_genomic_range.prepare_wig_dataframe")
@patch("panthera.core.pipelines.query_genomic_range.SSPManager")
@patch("panthera.core.pipelines.query_genomic_range.GenomeParser")
def test_run_query_success_plus_strand(
    mock_genome_parser,
    mock_ssp_manager,
    mock_prepare_wig,
    mock_write_wig,
    mock_logger,
    valid_inputs,
    mock_genome_sequence,
):
    """Tests a standard successful run on the positive strand (slicing & flag check)."""
    # 1. Setup Mocks
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_genome.return_value = mock_genome_sequence
    mock_genome_parser.return_value = mock_parser_instance

    mock_ssp_instance = MagicMock()
    mock_acc_prob = np.array([0.1])
    mock_dnr_prob = np.array([0.5])
    mock_ssp_instance.predict_ssp.return_value = ([mock_acc_prob], [mock_dnr_prob])
    mock_ssp_manager.return_value = mock_ssp_instance

    dummy_df = pd.DataFrame({"pos": [10], "prob": [0.1]})
    mock_prepare_wig.return_value = dummy_df

    # 2. Execute
    run_query_genomic_range(**valid_inputs)

    # 3. Assertions
    # Chromosome-specific parse was called
    mock_parser_instance.parse_genome.assert_called_once_with(
        "dummy.fasta", chrom="chr1"
    )

    # Check sequence slicing (chr1:10-20-plus) -> positions 10 to 20 inclusive
    # 0-based index: 9 to 20 -> "CCGGTTCCGGT"
    expected_seq = "CCGGTTCCGGT"

    mock_ssp_instance.predict_ssp.assert_called_once_with(
        seqs=[expected_seq], reverse_output=False
    )

    # Verify WIG DataFrame prep starts at `start` coordinate (10)
    mock_prepare_wig.assert_called_once()
    kwargs = mock_prepare_wig.call_args.kwargs
    assert kwargs["start"] == 10

    # Verify correct header construction
    expected_header = (
        f'track type=wiggle_0 name="test_output" '
        f'description="Probability" color={TRACK_COLOR} altColor={ALT_COLOR}\n'
        f"variableStep chrom=chr1 span=1\n"
    )
    mock_write_wig.assert_called_with(
        df=dummy_df,
        header=expected_header,
        prefix="test_output",
        outdir="/dummy/outdir",
    )


@patch("panthera.core.pipelines.query_genomic_range.write_wig")
@patch("panthera.core.pipelines.query_genomic_range.prepare_wig_dataframe")
@patch("panthera.core.pipelines.query_genomic_range.SSPManager")
@patch("panthera.core.pipelines.query_genomic_range.GenomeParser")
def test_run_query_success_minus_strand(
    mock_genome_parser,
    mock_ssp_manager,
    mock_prepare_wig,
    mock_write_wig,
    valid_inputs,
    mock_genome_sequence,
):
    """Tests run on the negative strand (reverse complement & reverse_output=True)."""
    # Override input to minus strand
    inputs = valid_inputs.copy()
    inputs["genomic_range"] = "chr1:10-20-minus"

    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_genome.return_value = mock_genome_sequence
    mock_genome_parser.return_value = mock_parser_instance

    mock_ssp_instance = MagicMock()
    mock_ssp_instance.predict_ssp.return_value = ([np.array([])], [np.array([])])
    mock_ssp_manager.return_value = mock_ssp_instance

    run_query_genomic_range(**inputs)

    # Slice: CCGGTTCCGGT -> Reverse Complement: ACCGGAACCGG
    expected_rc_seq = "ACCGGAACCGG"

    # Assert that the reverse complemented sequence was passed and reverse_output flag is True
    mock_ssp_instance.predict_ssp.assert_called_once_with(
        seqs=[expected_rc_seq], reverse_output=True
    )


### --- Edge Case: Exception Handling --- ###


@patch("panthera.core.pipelines.query_genomic_range.logger")
@patch("panthera.core.pipelines.query_genomic_range.write_wig")
@patch("panthera.core.pipelines.query_genomic_range.prepare_wig_dataframe")
@patch("panthera.core.pipelines.query_genomic_range.SSPManager")
@patch("panthera.core.pipelines.query_genomic_range.GenomeParser")
def test_run_query_handles_and_raises_wig_collision(
    mock_genome_parser,
    mock_ssp_manager,
    mock_prepare_wig,
    mock_write_wig,
    mock_logger,
    valid_inputs,
    mock_genome_sequence,
):
    """Tests that a ValueError from prepare_wig_dataframe is logged and re-raised."""
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_genome.return_value = mock_genome_sequence
    mock_genome_parser.return_value = mock_parser_instance

    mock_ssp_instance = MagicMock()
    mock_ssp_instance.predict_ssp.return_value = ([np.array([])], [np.array([])])
    mock_ssp_manager.return_value = mock_ssp_instance

    # Force prepare_wig_dataframe to raise a ValueError (simulating a collision)
    error_msg = "Collision at position(s): [15]"
    mock_prepare_wig.side_effect = ValueError(error_msg)

    # Use re.escape to handle the brackets
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        run_query_genomic_range(**valid_inputs)

    # Verify write_wig was not called
    mock_write_wig.assert_not_called()

    # Verify the error was logged before being re-raised
    mock_logger.error.assert_called_once_with(
        f"Error generating WIG file for {valid_inputs['prefix']}: {error_msg}"
    )


@patch("panthera.core.pipelines.query_genomic_range.logger")
@patch("panthera.core.pipelines.query_genomic_range.write_wig")
@patch("panthera.core.pipelines.query_genomic_range.SSPManager")
@patch("panthera.core.pipelines.query_genomic_range.GenomeParser")
def test_run_query_genomic_range_actual_array_collision(
    mock_genome_parser,
    mock_ssp_manager,
    mock_write_wig,
    mock_logger,
    valid_inputs,
    mock_genome_sequence,
):
    """Tests the pipeline's handling of a real array collision using the actual wig module."""
    # 1. Arrange: Setup Data Mocks
    mock_genome_parser.return_value.parse_genome.return_value = mock_genome_sequence
    mock_genome_parser.parse_genome.return_value = mock_genome_sequence

    mock_ssp_instance = MagicMock()

    # Create actual numpy arrays where both have a non-zero value at index 1
    mock_acc_prob = np.array([0.0, 0.8, 0.0, 0.0], dtype=np.float32)
    mock_dnr_prob = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)

    mock_ssp_instance.predict_ssp.return_value = ([mock_acc_prob], [mock_dnr_prob])
    mock_ssp_manager.return_value = mock_ssp_instance

    # 2. Act & Assert: Execute and Expect Exception
    # The fixture uses "chr1:10-20-plus", so start=10.
    # The arrays collide at index 1.
    # The real prepare_wig_dataframe calculates pos = start + index = 10 + 1 = 11.
    expected_error_msg = "Collision at position(s): [11]"

    # query_genomic_range.py re-raises the error after logging it,
    # so we must catch it with pytest.raises
    with pytest.raises(ValueError, match=re.escape(expected_error_msg)):
        run_query_genomic_range(**valid_inputs)

    # 3. Assert: Verify post-exception behavior
    mock_write_wig.assert_not_called()

    # Verify the error was logged before being re-raised
    mock_logger.error.assert_called_once()

    logged_msg = mock_logger.error.call_args[0][0]
    assert f"Error generating WIG file for {valid_inputs['prefix']}:" in logged_msg
    assert expected_error_msg in logged_msg
