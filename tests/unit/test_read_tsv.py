import pytest
from pathlib import Path

# Import your classes and exceptions (adjust 'variant_parser' to your actual file name)
from panthera.core.bio.io import TsvVariantReader
from panthera.utils.exceptions import MultipleAltError, NoVariantsError


@pytest.fixture
def reader():
    """Provides a fresh instance of the reader for each test."""
    return TsvVariantReader()


def test_read_valid_tsv_happy_path(tmp_path: Path, reader: TsvVariantReader):
    """
    Test that a well-formed TSV is read, cleaned, and formatted correctly.
    """
    # Arrange: Create a temporary TSV file
    tsv_content = "chrom\tpos\tref\talt\nchr1\t1000\tA\tT\nchr2\t2000\tG\tC\n"
    test_file = tmp_path / "valid_variants.tsv"
    test_file.write_text(tsv_content)

    # Act: Process the file
    result_df = reader.read(test_file)

    # Assert: Verify the enterprise formatting
    assert len(result_df) == 2
    assert list(result_df.columns) == [
        "chrom",
        "pos",
        "ref",
        "alt",
        "genotype",
        "phase_set",
        "sample_name",
    ]

    # Check that 'alt' was converted to a list of strings
    assert result_df.loc[0, "ref"] == "A"
    assert result_df.loc[0, "alt"] == "T"

    # Check added defaults
    assert result_df["genotype"].iloc[0] == "1|1"
    assert result_df["phase_set"].iloc[0] == "PST0"
    assert result_df["sample_name"].iloc[0] == "S0"


def test_validate_alleles_raises_multiple_alt_error(
    tmp_path: Path, reader: TsvVariantReader
):
    """
    Test that the code fails fast and loudly when multiple alleles are present.
    """
    # Arrange
    tsv_content = (
        "chrom\tpos\tref\talt\n"
        "chr1\t1000\tA\tT,C\n"  # Multiple alleles here
    )
    test_file = tmp_path / "bad_variants.tsv"
    test_file.write_text(tsv_content)

    # Act & Assert
    with pytest.raises(MultipleAltError, match="multiple ALT alleles"):
        reader.read(test_file)


def test_no_alleles_raises_no_variants_error(tmp_path: Path, reader: TsvVariantReader):
    """
    Test that the code fails fast and loudly when multiple alleles are present.
    """
    # Arrange
    tsv_content = "chrom\tpos\tref\talt\n"
    test_file = tmp_path / "no_variants.tsv"
    test_file.write_text(tsv_content)

    # Act & Assert
    with pytest.raises(NoVariantsError, match="no variants"):
        reader.read(test_file)


def test_clean_data_handles_messy_strings(tmp_path: Path, reader: TsvVariantReader):
    """
    Test the regex cleaning logic against whitespaces, brackets, and quotes.
    """
    # Arrange: Create highly messy data
    tsv_content = (
        "chrom\tpos\tref\talt\n chr 1 \t1000\t a \t['t']\nchr2\t2000\tG\t[C]\n"
    )
    test_file = tmp_path / "messy_variants.tsv"
    test_file.write_text(tsv_content)

    # Act
    result_df = reader.read(test_file)

    # Assert: Verify everything was stripped and capitalized correctly
    assert result_df.loc[0, "chrom"] == "chr1"
    assert result_df.loc[0, "ref"] == "A"
    assert result_df.loc[0, "alt"] == "T"  # Bracket and quote removed, made list

    assert result_df.loc[1, "alt"] == "C"  # Bracket removed, made list


def test_empty_tsv(tmp_path: Path, reader: TsvVariantReader):
    """
    Test the regex cleaning logic against whitespaces, brackets, and quotes.
    """
    # Arrange: Create empty data (only column headers)
    tsv_content = "chrom\tpos\tref\talt\n"
    test_file = tmp_path / "empty_variants.tsv"
    test_file.write_text(tsv_content)

    # Act
    with pytest.raises(NoVariantsError):
        reader.read(test_file)


def test_read_nonexistent_file_raises_error(tmp_path: Path, reader: TsvVariantReader):
    """
    Test that the reader gracefully bubbles up standard OS errors.
    """
    # Arrange
    fake_file = tmp_path / "does_not_exist.tsv"

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        reader.read(fake_file)


def test_deduplication(tmp_path: Path, reader: TsvVariantReader):
    """Test that duplicate generic variants are removed."""
    tsv_content = (
        "chrom\tpos\tref\talt\n"
        "chr1\t1000\tA\tT\n"
        "chr1\t1000\tA\tT\n"  # Exact duplicate
    )
    test_file = tmp_path / "duplicates.tsv"
    test_file.write_text(tsv_content)

    result_df = reader.read(test_file)
    assert len(result_df) == 1


def test_sorting(tmp_path: Path, reader: TsvVariantReader):
    """Test that output is sorted by chrom and pos."""
    tsv_content = (
        "chrom\tpos\tref\talt\nchr2\t500\tG\tC\nchr1\t2000\tA\tT\nchr1\t1000\tC\tG\n"
    )
    test_file = tmp_path / "unsorted.tsv"
    test_file.write_text(tsv_content)

    result_df = reader.read(test_file)
    assert len(result_df) == 3

    chroms = list(result_df["chrom"])
    positions = list(result_df["pos"])

    assert chroms == ["chr1", "chr1", "chr2"]
    assert positions == [1000, 2000, 500]
