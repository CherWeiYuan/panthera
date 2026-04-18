import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# Note: Assuming your module is saved as `io.py`.
# If it is named differently, update the import below.
from panthera.core.bio.io import (
    TsvVariantReader,
    VcfVariantReader,
    VariantReaderFactory,
    read_variants,
    NoVariantsError,
    MultipleAltError,
    MultipleVcfSampleError,
    NoPhaseSetError,
)

# ---------------------------------------------------------
# Test: Base VariantReader (_clean_data)
# ---------------------------------------------------------


class TestVariantReaderCleanData:
    """Tests the _clean_data method inherited by all readers."""

    @pytest.fixture
    def reader(self):
        # Use TsvVariantReader to access the abstract base methods
        return TsvVariantReader()

    def test_clean_data_empty(self, reader):
        """Edge case: ensure empty dataframes do not break cleaning logic."""
        df = pd.DataFrame()
        assert reader._clean_data(df).empty

    def test_clean_data_formatting_and_regex(self, reader):
        """Tests whitespace stripping, capitalization, regex cleaning, and sorting."""
        dirty_data = {
            "chrom": [" chr1 ", "chr2"],
            "pos": [200, 100],  # Out of order to test sorting
            "ref": [" a ", "T"],
            "alt": ["['c']", " g "],
        }
        df = pd.DataFrame(dirty_data)
        cleaned_df = reader._clean_data(df)

        # Check sorting (chr1/200 should be AFTER chr2/100 alphabetically by chrom? No, chr1 comes first)
        # Wait, sorted by chrom then pos. "chr1" < "chr2".
        assert cleaned_df.iloc[0]["chrom"] == "chr1"
        assert cleaned_df.iloc[0]["pos"] == 200

        # Check stripping and casing
        assert cleaned_df.iloc[0]["ref"] == "A"
        assert (
            cleaned_df.iloc[0]["alt"] == "C"
        )  # brackets, quotes, and whitespace removed

        assert cleaned_df.iloc[1]["chrom"] == "chr2"
        assert cleaned_df.iloc[1]["alt"] == "G"

    def test_clean_data_deduplication(self, reader):
        """Tests that duplicate rows are dropped."""
        df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "pos": [100, 100],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
            }
        )
        cleaned_df = reader._clean_data(df)
        assert len(cleaned_df) == 1


# ---------------------------------------------------------
# Test: TsvVariantReader
# ---------------------------------------------------------


class TestTsvVariantReader:
    def test_read_valid_tsv(self, tmp_path):
        """Tests successful reading and formatting of a TSV file."""
        tsv_file = tmp_path / "valid.tsv"
        tsv_file.write_text("chrom\tpos\tref\talt\nchr1\t100\tA\tT\n")

        reader = TsvVariantReader()
        df = reader.read(tsv_file)

        assert len(df) == 1
        assert df.iloc[0]["genotype"] == "1|1"
        assert df.iloc[0]["phase_set"] == "PST0"
        assert df.iloc[0]["sample_name"] == "S0"

    def test_read_empty_tsv_raises_error(self, tmp_path):
        """Edge case: TSV with only headers."""
        tsv_file = tmp_path / "empty.tsv"
        tsv_file.write_text("chrom\tpos\tref\talt\n")

        reader = TsvVariantReader()
        with pytest.raises(NoVariantsError, match="contains no variants"):
            reader.read(tsv_file)

    def test_validate_alleles_multiple_alt_raises_error(self, tmp_path):
        """Edge case: TSV contains multiple alt alleles separated by comma."""
        tsv_file = tmp_path / "multialt.tsv"
        tsv_file.write_text("chrom\tpos\tref\talt\nchr1\t100\tA\tT,G\n")

        reader = TsvVariantReader()
        with pytest.raises(MultipleAltError):
            reader.read(tsv_file)


# ---------------------------------------------------------
# Test: VcfVariantReader
# ---------------------------------------------------------


class TestVcfVariantReader:
    @pytest.fixture
    def mock_vcf_generator(self):
        """Creates a mock cyvcf2.VCF object to bypass actual file I/O."""
        vcf_mock = MagicMock()
        vcf_mock.get_header_type.return_value = {"ID": "PS"}
        vcf_mock.samples = ["Sample1"]

        # Mock a single valid variant
        var = MagicMock()
        var.CHROM = "chr1"
        var.POS = 100
        var.REF = "A"
        var.ALT = ["T"]
        var.format.return_value = [["12345"]]  # Simulate cyvcf2 numpy array return
        var.genotypes = [[0, 1, True]]  # Phased 0|1

        vcf_mock.__iter__.return_value = [var]
        return vcf_mock

    def test_read_valid_vcf(self, mock_vcf_generator):
        """Tests successful reading and parsing of VCF generator."""
        reader = VcfVariantReader()
        with patch.object(
            reader, "_get_vcf_generator", return_value=mock_vcf_generator
        ):
            df = reader.read(Path("dummy.vcf"))

            assert len(df) == 1
            assert df.iloc[0]["genotype"] == "0|1"
            assert df.iloc[0]["phase_set"] == "PS12345_1"
            assert df.iloc[0]["sample_name"] == "Sample1"

    def test_missing_ps_tag_raises_error(self, mock_vcf_generator):
        """Edge case: VCF header lacks Phase Set tag."""
        mock_vcf_generator.get_header_type.return_value = None
        reader = VcfVariantReader()

        with patch.object(
            reader, "_get_vcf_generator", return_value=mock_vcf_generator
        ):
            with pytest.raises(NoPhaseSetError, match="No PS tag found"):
                reader.read(Path("dummy.vcf"))

    def test_multiple_samples_raises_error(self, mock_vcf_generator):
        """Edge case: VCF has more than one sample."""
        mock_vcf_generator.samples = ["Sample1", "Sample2"]
        reader = VcfVariantReader()

        with patch.object(
            reader, "_get_vcf_generator", return_value=mock_vcf_generator
        ):
            with pytest.raises(MultipleVcfSampleError):
                reader.read(Path("dummy.vcf"))

    def test_no_variants_raises_error(self, mock_vcf_generator):
        """Edge case: Valid VCF but empty body/no rows."""
        mock_vcf_generator.__iter__.return_value = []
        reader = VcfVariantReader()

        with patch.object(
            reader, "_get_vcf_generator", return_value=mock_vcf_generator
        ):
            with pytest.raises(NoVariantsError):
                reader.read(Path("dummy.vcf"))

    def test_load_data_edge_cases(self, mock_vcf_generator):
        """Tests parsing edge cases: Multiple ALTs, unphased genotypes, missing formats."""
        var2 = MagicMock()
        var2.CHROM = "chr2"
        var2.POS = 200
        var2.REF = "C"
        var2.ALT = ["G", "T"]  # Multiple ALTs
        var2.format.return_value = (
            None  # Missing PS formatting for this specific variant
        )
        var2.genotypes = [[-1, 1, False]]  # Unphased missing allele: . / 1

        mock_vcf_generator.__iter__.return_value = [var2]
        reader = VcfVariantReader()

        # Test just the data loader to bypass header checks
        df = reader._load_data(mock_vcf_generator)

        assert len(df) == 1
        assert df.iloc[0]["alt"] == "G"  # Should strictly use the first allele
        assert (
            df.iloc[0]["genotype"] == "./1"
        )  # -1 should map to '.', False maps to '/'
        assert df.iloc[0]["phase_set"] == "PSUNKNOWN_1"  # Handles None formatting


# ---------------------------------------------------------
# Test: Factory and Main Entrypoint
# ---------------------------------------------------------


class TestVariantReaderFactory:
    @pytest.mark.parametrize(
        "filename, expected_class",
        [
            ("test.vcf", VcfVariantReader),
            ("test.vcf.gz", VcfVariantReader),
            ("test.tsv", TsvVariantReader),
            ("test.tsv.gz", TsvVariantReader),
            ("test.txt", TsvVariantReader),
        ],
    )
    def test_get_reader_valid_extensions(self, filename, expected_class):
        """Tests the factory correctly routes based on compound extensions."""
        reader = VariantReaderFactory.get_reader(Path(filename))
        assert isinstance(reader, expected_class)

    def test_get_reader_invalid_extension(self):
        """Edge case: Unsupported file format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            VariantReaderFactory.get_reader(Path("test.csv"))


class TestReadVariants:
    @patch("panthera.core.bio.io.VariantSchema.validate")
    @patch("panthera.core.bio.io.VariantReaderFactory.get_reader")
    def test_read_variants_success(self, mock_get_reader, mock_validate):
        """Tests the main orchestration function."""
        mock_df = pd.DataFrame({"dummy": [1]})

        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.read.return_value = mock_df
        mock_get_reader.return_value = mock_reader
        mock_validate.return_value = mock_df

        # Bypass file-exists check
        with patch("panthera.core.bio.io.Path.exists", return_value=True):
            result = read_variants("dummy.tsv")

            assert not result.empty
            mock_get_reader.assert_called_once_with(Path("dummy.tsv"))
            mock_reader.read.assert_called_once_with(Path("dummy.tsv"))
            mock_validate.assert_called_once_with(mock_df)

    def test_read_variants_file_not_found(self):
        """Edge case: Target file does not exist."""
        with pytest.raises(FileNotFoundError):
            read_variants("does_not_exist.tsv")
