from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandera.errors
from pandera.typing import DataFrame
import pytest

from panthera.core.bio.wig import (
    prepare_wig_dataframe,
    write_wig,
    generate_wig,
    WIGSchema,
)


@pytest.fixture
def dummy_probs():
    """Fixture to provide standard wild-type and mutant probabilities."""
    return {
        "wt_acc": np.array([0.0, 0.8, 0.0, 0.1], dtype=np.float32),
        "wt_dnr": np.array([0.9, 0.0, 0.2, 0.0], dtype=np.float32),
        "mt_acc": np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32),
        "mt_dnr": np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32),
    }


### --- Tests for prepare_wig_dataframe --- ###


def test_prepare_wig_dataframe_success():
    """Test standard execution with valid, non-overlapping arrays."""
    start = 10
    acc_prob = np.array([0.0, 0.5, 0.0, 0.8], dtype=np.float32)
    dnr_prob = np.array([0.2, 0.0, 0.9, 0.0], dtype=np.float32)

    df = prepare_wig_dataframe(start, acc_prob, dnr_prob)

    assert len(df) == 4
    # Positions should be correctly offset by start (10)
    assert list(df["pos"]) == [10, 11, 12, 13]
    # Acceptors are positive, Donors are negative
    np.testing.assert_allclose(df["prob"], [-0.2, 0.5, -0.9, 0.8])


def test_prepare_wig_dataframe_empty():
    """Test execution when all probabilities are zero."""
    start = 1
    acc_prob = np.zeros(5, dtype=np.float32)
    dnr_prob = np.zeros(5, dtype=np.float32)

    df = prepare_wig_dataframe(start, acc_prob, dnr_prob)

    assert df.empty
    assert list(df.columns) == ["pos", "prob"]


def test_prepare_wig_dataframe_collision():
    """Test ValueError is raised when acceptor and donor overlap."""
    start = 1
    # Overlap at index 1
    acc_prob = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    dnr_prob = np.array([0.0, 0.5, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match=r"Collision at positions: \[2\]"):
        prepare_wig_dataframe(start, acc_prob, dnr_prob)


def test_prepare_wig_dataframe_invalid_pos_schema():
    """Test Pandera SchemaError is raised for invalid positions (pos < 1)."""
    # Start = 0 means index 0 + 0 = 0 (Violates WIGSchema pos >= 1)
    start = 0
    acc_prob = np.array([0.5], dtype=np.float32)
    dnr_prob = np.array([0.0], dtype=np.float32)

    with pytest.raises(pandera.errors.SchemaError):
        prepare_wig_dataframe(start, acc_prob, dnr_prob)


def test_prepare_wig_dataframe_invalid_prob_schema():
    """Test Pandera SchemaError is raised for probabilities out of bounds."""
    start = 1
    # 1.5 violates WIGSchema prob <= 1.0
    acc_prob = np.array([1.5], dtype=np.float32)
    dnr_prob = np.array([0.0], dtype=np.float32)

    with pytest.raises(pandera.errors.SchemaError):
        prepare_wig_dataframe(start, acc_prob, dnr_prob)


### --- Tests for write_wig --- ###


def test_write_wig(tmp_path):
    """Test WIG file is written correctly to the file system."""
    df = cast(DataFrame[WIGSchema], pd.DataFrame({"pos": [1, 2], "prob": [0.5, -0.5]}))
    header = "track type=wiggle_0\n"
    outdir = tmp_path / "out"
    prefix = "test_track"

    write_wig(df, header, prefix, str(outdir))

    file_path = outdir / "test_track.wig"
    assert file_path.exists()

    # Read and verify contents
    content = file_path.read_text()
    expected_content = "track type=wiggle_0\n1\t0.5\n2\t-0.5\n"
    assert content == expected_content


### --- Tests for generate_wig --- ###


def test_generate_wig_success(tmp_path, dummy_probs):
    """Test end-to-end WIG generation for both WT and MT streams."""
    generate_wig(
        gene_name="BRCA1",
        background_id="HG001",
        haplotype_id="H1",
        chrom="chr17",
        start=100,
        outdir=tmp_path,
        block_id="BLK1",
        **dummy_probs,
    )

    base_out = tmp_path / "wig" / "BRCA1" / "HG001"
    wt_file = base_out / "BRCA1.HG001.H1.HAPLOTYPE.BLK1.WT.wig"
    mt_file = base_out / "BRCA1.HG001.H1.HAPLOTYPE.BLK1.MT.wig"

    assert wt_file.exists()
    assert mt_file.exists()

    # Spot-check WT content
    wt_content = wt_file.read_text()
    print(wt_content)
    assert 'name="BRCA1 HG001 WT H1"' in wt_content
    assert "variableStep chrom=chr17 span=1" in wt_content
    assert "100\t-0.9" in wt_content  # Index 0 donor
    assert "101\t0.8" in wt_content  # Index 1 acceptor


@patch("pathlib.Path.mkdir")
def test_generate_wig_os_error(mock_mkdir, tmp_path, dummy_probs):
    """Test that OS errors during directory creation are handled and raised."""
    mock_mkdir.side_effect = OSError("Permission denied")

    with pytest.raises(OSError, match="Permission denied"):
        generate_wig(
            gene_name="BRCA1",
            background_id="HG001",
            haplotype_id="H1",
            chrom="chr17",
            start=100,
            outdir=tmp_path,
            **dummy_probs,
        )


def test_generate_wig_processing_exception_handling(tmp_path, dummy_probs):
    """Test that exceptions mid-loop (like collisions) are logged and re-raised."""

    # Intentionally induce a collision in the MT arrays
    bad_probs = dummy_probs.copy()
    bad_probs["mt_acc"] = np.array([0.5, 0.0], dtype=np.float32)
    bad_probs["mt_dnr"] = np.array([0.5, 0.0], dtype=np.float32)  # Collision at index 0

    with pytest.raises(ValueError, match="Collision at positions"):
        generate_wig(
            gene_name="BRCA1",
            background_id="HG001",
            haplotype_id="H1",
            chrom="chr17",
            start=100,
            outdir=tmp_path,
            **bad_probs,
        )
