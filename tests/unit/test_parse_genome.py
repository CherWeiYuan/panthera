from pathlib import Path

import pytest

from panthera.core.parse_genome import GenomeParser
from panthera.utils.exceptions import SeqNotFoundError, NonUniqueFastaHeader

# Pytest fixtures


@pytest.fixture
def base_fasta(tmp_path: Path) -> Path:
    """
    Creates a real, temporary FASTA file with two chromosomes.
    Notice 'chr1' has lowercase letters to test the uppercase enforcement.
    """
    file_path = tmp_path / "reference.fasta"
    content = ">chr1\natgc\n>chr2\nCGTA\n"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def duplicate_fasta(tmp_path: Path) -> Path:
    """Creates a temporary FASTA file with duplicate headers."""
    file_path = tmp_path / "duplicates.fasta"
    content = ">chr1\nATGC\n>chr1\nCGTA\n"
    file_path.write_text(content)
    return file_path


# Functional tests


class TestGenomeParserFunctional:
    # --- Tests for _read_fasta_to_dict ---

    def test_read_fasta_to_dict_success(self, base_fasta: Path):
        """Reads a physical file and verifies the output dictionary."""
        result = GenomeParser._read_fasta_to_dict(base_fasta)
        assert result == {"chr1": "ATGC", "chr2": "CGTA"}

    def test_read_fasta_to_dict_duplicate_header(self, duplicate_fasta: Path):
        """Reads a physical file with duplicates and catches the raised exception."""
        with pytest.raises(NonUniqueFastaHeader, match="Duplicate header: chr1"):
            GenomeParser._read_fasta_to_dict(duplicate_fasta)

    # --- Tests for _split_genome_by_chromosome ---

    def test_split_genome_by_chromosome(self, base_fasta: Path, tmp_path: Path):
        """
        Splits a physical file, then verifies the new files exist on disk
        and contain the correctly uppercased sequences.
        """
        GenomeParser._split_genome_by_chromosome(base_fasta)

        # Check that the new files were actually created in the temporary directory
        chr1_file = tmp_path / "reference.chr1.fasta"
        chr2_file = tmp_path / "reference.chr2.fasta"

        assert chr1_file.exists(), "chr1 fasta file was not created"
        assert chr2_file.exists(), "chr2 fasta file was not created"

        # Check file contents to ensure data integrity (uppercasing logic)
        assert chr1_file.read_text() == ">chr1\nATGC\n"
        assert chr2_file.read_text() == ">chr2\nCGTA\n"

    # --- Tests for parse_genome ---

    def test_parse_genome_no_chrom(self, base_fasta: Path):
        """Loads the full file when no specific chromosome is requested."""
        result = GenomeParser.parse_genome(str(base_fasta))
        assert result == {"chr1": "ATGC", "chr2": "CGTA"}

    def test_parse_genome_chrom_exists_skips_split(self, tmp_path: Path):
        """
        If the chromosome file already exists, it should load it directly
        without attempting to read or split the parent file.
        """
        # We purposely do NOT create the parent "reference.fasta".
        # We only create the targeted chromosome file.
        chr3_file = tmp_path / "reference.chr3.fasta"
        chr3_file.write_text(">chr3\nGGCC\n")

        parent_path = tmp_path / "reference.fasta"

        result = GenomeParser.parse_genome(str(parent_path), chrom="chr3")
        assert result == {"chr3": "GGCC"}

    def test_parse_genome_chrom_missing_triggers_split(
        self, base_fasta: Path, tmp_path: Path
    ):
        """
        If the chromosome file is missing, it should read the parent file,
        create the split files on disk, and then load the requested one.
        """
        # Request 'chr1' which exists in the base file, but 'reference.chr1.fasta' doesn't exist yet
        result = GenomeParser.parse_genome(str(base_fasta), chrom="chr1")

        # 1. It should return the requested uppercased sequence
        assert result == {"chr1": "ATGC"}

        # 2. It should have successfully generated both split files on disk
        assert (tmp_path / "reference.chr1.fasta").exists()
        assert (tmp_path / "reference.chr2.fasta").exists()

    def test_parse_genome_raises_seq_not_found(self, base_fasta: Path):
        """
        If a chromosome is requested that isn't in the parent file,
        it should split the file, fail to find the requested chromosome, and raise an error.
        """
        with pytest.raises(
            SeqNotFoundError, match="Could not locate sequence for chr99"
        ):
            GenomeParser.parse_genome(str(base_fasta), chrom="chr99")
