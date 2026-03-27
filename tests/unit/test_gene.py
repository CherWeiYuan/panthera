"""
Tests for panthera.core.bio.gene module.

Tests cover:
- GeneObject dataclass construction
- GTFParser._parse_attributes static method
- GTFParser._load_gtf_to_dataframe
- GTFParser.get_gene_sites (splice site calculation logic)
- GTFParser.get_gtf_dict (JSON caching and gene metadata dictionary)
- find_genes_at_pos (coordinate-based gene lookup)
"""

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import pytest

from panthera.core.bio.gene import GeneObject, GTFParser, find_genes_at_pos

# ==============================================================
# FIXTURES
# ==============================================================

# Minimal valid GTF content with a single gene (2 exons, plus strand)
MINIMAL_GTF_PLUS = """\
seqname\tsource\tfeature\tstart\tend\tscore\tstrand\tframe\tattribute
chr1\tensembl\tgene\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1";
chr1\tensembl\ttranscript\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; transcript_support_level "1";
chr1\tensembl\texon\t1000\t1200\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "1"; transcript_support_level "1";
chr1\tensembl\texon\t2000\t2500\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "2"; transcript_support_level "1";
"""

# Minimal valid GTF content with a single gene (2 exons, minus strand)
MINIMAL_GTF_MINUS = """\
seqname\tsource\tfeature\tstart\tend\tscore\tstrand\tframe\tattribute
chr2\tensembl\tgene\t3000\t8000\t.\t-\t.\tgene_id "G002"; gene_name "TP53";
chr2\tensembl\ttranscript\t3000\t8000\t.\t-\t.\tgene_id "G002"; gene_name "TP53"; transcript_id "T002"; transcript_support_level "2";
chr2\tensembl\texon\t3000\t3500\t.\t-\t.\tgene_id "G002"; gene_name "TP53"; transcript_id "T002"; exon_number "1"; transcript_support_level "2";
chr2\tensembl\texon\t5000\t5800\t.\t-\t.\tgene_id "G002"; gene_name "TP53"; transcript_id "T002"; exon_number "2"; transcript_support_level "2";
"""

# GTF with a weak transcript (TSL 5) that should be filtered out
MINIMAL_GTF_WEAK_TSL = """\
seqname\tsource\tfeature\tstart\tend\tscore\tstrand\tframe\tattribute
chr1\tensembl\tgene\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1";
chr1\tensembl\ttranscript\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T_WEAK"; transcript_support_level "5";
chr1\tensembl\texon\t1000\t1200\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T_WEAK"; exon_number "1"; transcript_support_level "5";
chr1\tensembl\texon\t2000\t2500\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T_WEAK"; exon_number "2"; transcript_support_level "5";
"""

# GTF with a single-exon transcript (should be skipped for splice sites)
MINIMAL_GTF_SINGLE_EXON = """\
seqname\tsource\tfeature\tstart\tend\tscore\tstrand\tframe\tattribute
chr1\tensembl\tgene\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1";
chr1\tensembl\ttranscript\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; transcript_support_level "1";
chr1\tensembl\texon\t1000\t1200\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "1"; transcript_support_level "1";
"""

# GTF with 3 exons on plus strand (tests middle exon getting both acc and dnr)
MINIMAL_GTF_THREE_EXONS = """\
seqname\tsource\tfeature\tstart\tend\tscore\tstrand\tframe\tattribute
chr1\tensembl\tgene\t1000\t9000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1";
chr1\tensembl\ttranscript\t1000\t9000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; transcript_support_level "1";
chr1\tensembl\texon\t1000\t1200\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "1"; transcript_support_level "1";
chr1\tensembl\texon\t3000\t3500\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "2"; transcript_support_level "1";
chr1\tensembl\texon\t7000\t9000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "3"; transcript_support_level "1";
"""

# Multi-gene GTF (two genes on different chromosomes)
MULTI_GENE_GTF = """\
seqname\tsource\tfeature\tstart\tend\tscore\tstrand\tframe\tattribute
chr1\tensembl\tgene\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1";
chr1\tensembl\ttranscript\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; transcript_support_level "1";
chr1\tensembl\texon\t1000\t1200\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "1"; transcript_support_level "1";
chr1\tensembl\texon\t2000\t2500\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "2"; transcript_support_level "1";
chr2\tensembl\tgene\t3000\t8000\t.\t-\t.\tgene_id "G002"; gene_name "TP53";
chr2\tensembl\ttranscript\t3000\t8000\t.\t-\t.\tgene_id "G002"; gene_name "TP53"; transcript_id "T002"; transcript_support_level "2";
chr2\tensembl\texon\t3000\t3500\t.\t-\t.\tgene_id "G002"; gene_name "TP53"; transcript_id "T002"; exon_number "1"; transcript_support_level "2";
chr2\tensembl\texon\t5000\t5800\t.\t-\t.\tgene_id "G002"; gene_name "TP53"; transcript_id "T002"; exon_number "2"; transcript_support_level "2";
"""

# GTF with chromosome name lacking 'chr' prefix (tests standardization)
GTF_NO_CHR_PREFIX = """\
seqname\tsource\tfeature\tstart\tend\tscore\tstrand\tframe\tattribute
1\tensembl\tgene\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1";
1\tensembl\ttranscript\t1000\t5000\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; transcript_support_level "1";
1\tensembl\texon\t1000\t1200\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "1"; transcript_support_level "1";
1\tensembl\texon\t2000\t2500\t.\t+\t.\tgene_id "G001"; gene_name "BRCA1"; transcript_id "T001"; exon_number "2"; transcript_support_level "1";
"""


def _write_gtf(tmp_path: Path, content: str, name: str = "test.gtf") -> Path:
    """Helper to write GTF content to a temporary file."""
    gtf_path = tmp_path / name
    gtf_path.write_text(content)
    return gtf_path


@pytest.fixture
def plus_strand_gtf(tmp_path: Path) -> Path:
    return _write_gtf(tmp_path, MINIMAL_GTF_PLUS)


@pytest.fixture
def minus_strand_gtf(tmp_path: Path) -> Path:
    return _write_gtf(tmp_path, MINIMAL_GTF_MINUS)


@pytest.fixture
def weak_tsl_gtf(tmp_path: Path) -> Path:
    return _write_gtf(tmp_path, MINIMAL_GTF_WEAK_TSL)


@pytest.fixture
def single_exon_gtf(tmp_path: Path) -> Path:
    return _write_gtf(tmp_path, MINIMAL_GTF_SINGLE_EXON)


@pytest.fixture
def three_exon_gtf(tmp_path: Path) -> Path:
    return _write_gtf(tmp_path, MINIMAL_GTF_THREE_EXONS)


@pytest.fixture
def multi_gene_gtf(tmp_path: Path) -> Path:
    return _write_gtf(tmp_path, MULTI_GENE_GTF)


@pytest.fixture
def no_chr_prefix_gtf(tmp_path: Path) -> Path:
    return _write_gtf(tmp_path, GTF_NO_CHR_PREFIX)


# ==============================================================
# GeneObject TESTS
# ==============================================================


class TestGeneObject:
    def test_construction(self):
        """Test that GeneObject can be constructed with valid fields."""
        gene = GeneObject(
            chrom="chr1",
            strand="+",
            start=1000,
            end=5000,
            gene_name="BRCA1",
            gene_id="G001",
            splice_sites={"acc": [2000], "dnr": [1200]},
            shex=[[851, 1349], [1851, 2649]],
        )
        assert gene.chrom == "chr1"
        assert gene.strand == "+"
        assert gene.start == 1000
        assert gene.end == 5000
        assert gene.gene_name == "BRCA1"
        assert gene.gene_id == "G001"
        assert gene.splice_sites == {"acc": [2000], "dnr": [1200]}
        assert gene.shex == [[851, 1349], [1851, 2649]]

    def test_equality(self):
        """Dataclass default equality compares all fields."""
        gene_a = GeneObject("chr1", "+", 1000, 5000, "BRCA1", "G001", {}, [])
        gene_b = GeneObject("chr1", "+", 1000, 5000, "BRCA1", "G001", {}, [])
        assert gene_a == gene_b

    def test_inequality(self):
        """Different fields produce unequal objects."""
        gene_a = GeneObject("chr1", "+", 1000, 5000, "BRCA1", "G001", {}, [])
        gene_b = GeneObject("chr2", "+", 1000, 5000, "BRCA1", "G001", {}, [])
        assert gene_a != gene_b


# ==============================================================
# GTFParser._parse_attributes TESTS
# ==============================================================

class TestVectorizedAttributeParser:
    def test_standard_vectorized(self):
        """Test parsing typical GTF attribute strings in a Series."""
        attrs = pd.Series([
            'gene_id "ENSG001"; gene_name "BRCA1"; transcript_id "ENST001"; exon_number "3";',
            'gene_id "ENSG002"; gene_name "TP53"; transcript_id "ENST002"; exon_number "1";'
        ])
        result = GTFParser._parse_attributes(attrs)
        
        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "gene_id"] == "ENSG001"
        assert result.loc[1, "gene_name"] == "TP53"
        assert result.loc[0, "exon_number"] == "3"

    def test_missing_keys_return_nan(self):
        """If a key is missing in a row, it should result in NaN (standard Pandas behavior)."""
        attrs = pd.Series([
            'gene_id "G1"; gene_name "N1";',  # Missing transcript_id, exon_number, TSL
            'transcript_id "T2";'              # Missing everything else
        ])
        result = GTFParser._parse_attributes(attrs)
        
        assert pd.isna(result.loc[0, "transcript_id"])
        assert pd.isna(result.loc[1, "gene_id"])
        assert result.loc[0, "gene_name"] == "N1"

    def test_filters_non_target_keys(self):
        """Only the five target keys from GTFParser should be columns in the result."""
        attrs = pd.Series(['gene_id "G1"; level "2"; tag "basic";'])
        result = GTFParser._parse_attributes(attrs)
        
        expected_columns = {
            "gene_id", "gene_name", "transcript_id", 
            "exon_number", "transcript_support_level"
        }
        assert set(result.columns) == expected_columns
        assert "level" not in result.columns
        assert "tag" not in result.columns

    def test_whitespace_and_semicolon_robustness(self):
        """Test varying whitespace and trailing separators."""
        attrs = pd.Series([
            'gene_id "G1";gene_name "N1"',      # No space after semicolon
            'gene_id   "G2"  ;  gene_name "N2"; ;', # Extra spaces and multiple semicolons
            '   gene_id "G3";'                  # Leading spaces
        ])
        result = GTFParser._parse_attributes(attrs)
        
        assert result.loc[0, "gene_id"] == "G1"
        assert result.loc[1, "gene_id"] == "G2"
        assert result.loc[2, "gene_id"] == "G3"

    def test_partial_match_prevention(self):
        """Regex should not match keys that are substrings of other words."""
        attrs = pd.Series([
            'my_gene_id "BAD"; gene_id "GOOD";',
            'pseudogene_id "BAD2";'
        ])
        result = GTFParser._parse_attributes(attrs)
        
        # Should only capture the exact match 'gene_id'
        assert result.loc[0, "gene_id"] == "GOOD"
        assert pd.isna(result.loc[1, "gene_id"])

    def test_empty_and_null_inputs(self):
        """Test behavior with empty strings and actual NaN values in the series."""
        attrs = pd.Series(["", np.nan, 'gene_id "G1";'])
        result = GTFParser._parse_attributes(attrs)
        
        assert len(result) == 3
        assert pd.isna(result.loc[0, "gene_id"])
        assert pd.isna(result.loc[1, "gene_id"])
        assert result.loc[2, "gene_id"] == "G1"

    def test_special_characters_in_values(self):
        """Values containing dots, dashes, or spaces (common in gene names) should be captured."""
        attrs = pd.Series([
            'gene_name "MSTRG.1234.1"; gene_id "ID-99"; transcript_id "T 1";'
        ])
        result = GTFParser._parse_attributes(attrs)
        
        assert result.loc[0, "gene_name"] == "MSTRG.1234.1"
        assert result.loc[0, "gene_id"] == "ID-99"
        assert result.loc[0, "transcript_id"] == "T 1"


# ==============================================================
# GTFParser._load_gtf_to_dataframe TESTS
# ==============================================================


class TestLoadGtfToDataframe:
    def test_loads_correct_columns(self, plus_strand_gtf: Path):
        """Verify that the loaded DataFrame has the expected columns."""
        parser = GTFParser(str(plus_strand_gtf))
        df = parser._load_gtf_to_dataframe()
        expected_cols = {
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "gene_id",
            "gene_name",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_chromosome_standardization(self, no_chr_prefix_gtf: Path):
        """Chromosomes lacking 'chr' prefix should be standardized."""
        parser = GTFParser(str(no_chr_prefix_gtf))
        df = parser._load_gtf_to_dataframe()
        assert all(df["seqname"].str.startswith("chr"))

    def test_caches_dataframe(self, plus_strand_gtf: Path):
        """Loading twice should return the cached DataFrame (same object)."""
        parser = GTFParser(str(plus_strand_gtf))
        df1 = parser._load_gtf_to_dataframe()
        df2 = parser._load_gtf_to_dataframe()
        assert df1 is df2


# ==============================================================
# GTFParser.get_gene_sites TESTS
# ==============================================================


class TestGetGeneSites:
    def test_plus_strand_splice_sites(self, plus_strand_gtf: Path):
        """
        For a plus strand gene with 2 exons:
        - Exon 1 (first exon): only donor at exon end (1200)
        - Exon 2 (last exon): only acceptor at exon start (2000)
        """
        parser = GTFParser(str(plus_strand_gtf))
        result = parser.get_gene_sites()

        sites = result["G001"]["BRCA1"]
        assert 2000 in sites["acc"], "Last exon start should be an acceptor site"
        assert 1200 in sites["dnr"], "First exon end should be a donor site"

    def test_minus_strand_splice_sites(self, minus_strand_gtf: Path):
        """
        For a minus strand gene with 2 exons:
        - Exon 1 (first exon): only donor at exon start (3000)
        - Exon 2 (last exon): only acceptor at exon end (5800)
        """
        parser = GTFParser(str(minus_strand_gtf))
        result = parser.get_gene_sites()

        sites = result["G002"]["TP53"]
        assert 5800 in sites["acc"], (
            "Last exon end should be an acceptor site (minus strand)"
        )
        assert 3000 in sites["dnr"], (
            "First exon start should be a donor site (minus strand)"
        )

    def test_three_exon_middle_has_both_sites(self, three_exon_gtf: Path):
        """
        For a plus strand gene with 3 exons:
        - Exon 1 (first): donor only (end=1200)
        - Exon 2 (middle): acceptor (start=3000) AND donor (end=3500)
        - Exon 3 (last): acceptor only (start=7000)
        """
        parser = GTFParser(str(three_exon_gtf))
        result = parser.get_gene_sites()

        sites = result["G001"]["BRCA1"]
        assert sorted(sites["acc"]) == [3000, 7000]
        assert sorted(sites["dnr"]) == [1200, 3500]

    def test_shex_intervals_calculated(self, plus_strand_gtf: Path):
        """Shallow intron + exon intervals should be offset by SHALLOW_INTRON_OFFSET."""
        parser = GTFParser(str(plus_strand_gtf))
        result = parser.get_gene_sites()

        shex = result["G001"]["BRCA1"]["shex"]
        offset = GTFParser.SHALLOW_INTRON_OFFSET

        # Exon 1: 1000-1200 → shex: [1000-149, 1200+149] = [851, 1349]
        assert [1000 - offset, 1200 + offset] in shex
        # Exon 2: 2000-2500 → shex: [2000-149, 2500+149] = [1851, 2649]
        assert [2000 - offset, 2500 + offset] in shex

    def test_weak_tsl_filtered_out(self, weak_tsl_gtf: Path):
        """Transcripts with TSL=5 should produce no splice sites."""
        parser = GTFParser(str(weak_tsl_gtf))
        result = parser.get_gene_sites()

        sites = result["G001"]["BRCA1"]
        assert sites["acc"] == []
        assert sites["dnr"] == []

    def test_single_exon_no_splice_sites(self, single_exon_gtf: Path):
        """Single-exon transcripts should not produce any splice sites."""
        parser = GTFParser(str(single_exon_gtf))
        result = parser.get_gene_sites()

        sites = result["G001"]["BRCA1"]
        assert sites["acc"] == []
        assert sites["dnr"] == []

    def test_splice_sites_are_sorted_and_deduplicated(self, plus_strand_gtf: Path):
        """Splice sites should be sorted and unique."""
        parser = GTFParser(str(plus_strand_gtf))
        result = parser.get_gene_sites()

        sites = result["G001"]["BRCA1"]
        assert sites["acc"] == sorted(set(sites["acc"]))
        assert sites["dnr"] == sorted(set(sites["dnr"]))


# ==============================================================
# GTFParser.get_gtf_dict TESTS
# ==============================================================


class TestGetGtfDict:
    def test_returns_dict_keyed_by_chrom(self, multi_gene_gtf: Path):
        """Output should be a dictionary with chromosome keys."""
        parser = GTFParser(str(multi_gene_gtf))
        result = parser.get_gtf_dict()

        assert "chr1" in result
        assert "chr2" in result

    def test_gene_entries_contain_expected_fields(self, plus_strand_gtf: Path):
        """Each gene entry should have 9 elements."""
        parser = GTFParser(str(plus_strand_gtf))
        result = parser.get_gtf_dict()

        entries = result["chr1"]
        assert len(entries) >= 1
        entry = entries[0]
        # Entry format: [index, seqname, start, end, strand, gene_name, gene_id, splice_sites, shex]
        assert len(entry) == 9

    def test_json_cache_created(self, plus_strand_gtf: Path, tmp_path: Path):
        """After first call, a JSON cache file should exist on disk."""
        parser = GTFParser(str(plus_strand_gtf))
        parser.get_gtf_dict()

        cache_path = plus_strand_gtf.with_suffix(".json")
        assert cache_path.exists(), "JSON cache file should be created"

    def test_json_cache_loaded_on_subsequent_call(
        self, plus_strand_gtf: Path, tmp_path: Path
    ):
        """Second call should load from JSON cache, not reparse."""
        parser = GTFParser(str(plus_strand_gtf))
        result1 = parser.get_gtf_dict()

        # Create a new parser instance (simulates fresh startup)
        parser2 = GTFParser(str(plus_strand_gtf))
        result2 = parser2.get_gtf_dict()

        assert result1 == result2

    def test_gene_coordinates_correct(self, plus_strand_gtf: Path):
        """Gene start/end in the output should match the GTF gene feature."""
        parser = GTFParser(str(plus_strand_gtf))
        result = parser.get_gtf_dict()

        entry = result["chr1"][0]
        start, end = entry[2], entry[3]
        assert start == 1000
        assert end == 5000

    def test_chromosome_standardization_in_output(self, no_chr_prefix_gtf: Path):
        """Chromosomes lacking 'chr' prefix should be standardized in the output."""
        parser = GTFParser(str(no_chr_prefix_gtf))
        result = parser.get_gtf_dict()

        assert "chr1" in result
        assert "1" not in result


# ==============================================================
# find_genes_at_pos TESTS
# ==============================================================


class TestFindGenesAtPos:
    @pytest.fixture
    def sample_gtf_dict(self) -> Dict[str, List[Any]]:
        """Builds a mock gtf_dict matching the output format of get_gtf_dict."""
        return {
            "chr1": [
                [
                    0,
                    "chr1",
                    1000,
                    5000,
                    "+",
                    "BRCA1",
                    "G001",
                    {"acc": [2000], "dnr": [1200]},
                    [[851, 1349], [1851, 2649]],
                ],
                [
                    1,
                    "chr1",
                    6000,
                    9000,
                    "+",
                    "FAKEGENE",
                    "G003",
                    {"acc": [7000], "dnr": [6500]},
                    [[5851, 6649]],
                ],
            ],
            "chr2": [
                [
                    2,
                    "chr2",
                    3000,
                    8000,
                    "-",
                    "TP53",
                    "G002",
                    {"acc": [5800], "dnr": [3000]},
                    [[2851, 3649], [4851, 5949]],
                ],
            ],
        }

    def test_finds_gene_at_position_inside(self, sample_gtf_dict):
        """Position inside gene range should return the gene."""
        result = find_genes_at_pos("chr1", 2500, sample_gtf_dict, [])
        names = [g.gene_name for g in result]
        assert "BRCA1" in names

    def test_finds_gene_at_boundary_start(self, sample_gtf_dict):
        """Position at the exact start boundary should match."""
        result = find_genes_at_pos("chr1", 1000, sample_gtf_dict, [])
        names = [g.gene_name for g in result]
        assert "BRCA1" in names

    def test_finds_gene_at_boundary_end(self, sample_gtf_dict):
        """Position at the exact end boundary should match."""
        result = find_genes_at_pos("chr1", 5000, sample_gtf_dict, [])
        names = [g.gene_name for g in result]
        assert "BRCA1" in names

    def test_no_match_outside_range(self, sample_gtf_dict):
        """Position outside all gene ranges should return empty list."""
        result = find_genes_at_pos("chr1", 999, sample_gtf_dict, [])
        assert result == []

        result = find_genes_at_pos("chr1", 5001, sample_gtf_dict, [])
        # Should not match BRCA1 (ends at 5000) but also not FAKEGENE (starts at 6000)
        assert all(g.gene_name != "BRCA1" for g in result)

    def test_no_match_wrong_chromosome(self, sample_gtf_dict):
        """Position on an unlisted chromosome should return empty list."""
        result = find_genes_at_pos("chrX", 2500, sample_gtf_dict, [])
        assert result == []

    def test_excludes_existing_genes(self, sample_gtf_dict):
        """Genes already in existing_genes should be skipped."""
        existing = [GeneObject("chr1", "+", 1000, 5000, "BRCA1", "G001", {}, [])]
        result = find_genes_at_pos("chr1", 2500, sample_gtf_dict, existing)
        names = [g.gene_name for g in result]
        assert "BRCA1" not in names

    def test_multiple_overlapping_genes(self, sample_gtf_dict):
        """When position overlaps two genes, both should be returned."""
        # Add an overlapping gene to chr1 at the same range as BRCA1
        sample_gtf_dict["chr1"].append(
            [3, "chr1", 1500, 3000, "+", "OVERLAP1", "G004", {"acc": [], "dnr": []}, []]
        )
        result = find_genes_at_pos("chr1", 2500, sample_gtf_dict, [])
        names = [g.gene_name for g in result]
        assert "BRCA1" in names
        assert "OVERLAP1" in names

    def test_returned_gene_fields_correct(self, sample_gtf_dict):
        """Verify that all fields on the returned GeneObject are correctly populated."""
        result = find_genes_at_pos("chr2", 5000, sample_gtf_dict, [])
        assert len(result) == 1
        gene = result[0]
        assert gene.chrom == "chr2"
        assert gene.strand == "-"
        assert gene.start == 3000
        assert gene.end == 8000
        assert gene.gene_name == "TP53"
        assert gene.gene_id == "G002"
        assert gene.splice_sites == {"acc": [5800], "dnr": [3000]}
        assert gene.shex == [[2851, 3649], [4851, 5949]]

    def test_empty_gtf_dict(self):
        """Empty gtf_dict should return empty result."""
        result = find_genes_at_pos("chr1", 1000, {}, [])
        assert result == []


# ==============================================================
# INTEGRATION: GTFParser → find_genes_at_pos
# ==============================================================


class TestGTFParserIntegration:
    def test_end_to_end_plus_strand(self, plus_strand_gtf: Path):
        """Full pipeline: parse GTF → build dict → find gene at position."""
        parser = GTFParser(str(plus_strand_gtf))
        gtf_dict = parser.get_gtf_dict()

        result = find_genes_at_pos("chr1", 1500, gtf_dict, [])
        assert len(result) == 1
        assert result[0].gene_name == "BRCA1"

    def test_end_to_end_no_match(self, plus_strand_gtf: Path):
        """Position outside gene range returns nothing."""
        parser = GTFParser(str(plus_strand_gtf))
        gtf_dict = parser.get_gtf_dict()

        result = find_genes_at_pos("chr1", 999, gtf_dict, [])
        assert result == []

    def test_end_to_end_multi_gene(self, multi_gene_gtf: Path):
        """Multi-gene GTF returns correct genes per chromosome."""
        parser = GTFParser(str(multi_gene_gtf))
        gtf_dict = parser.get_gtf_dict()

        # chr1 query
        result_chr1 = find_genes_at_pos("chr1", 2000, gtf_dict, [])
        assert any(g.gene_name == "BRCA1" for g in result_chr1)

        # chr2 query
        result_chr2 = find_genes_at_pos("chr2", 5000, gtf_dict, [])
        assert any(g.gene_name == "TP53" for g in result_chr2)

    def test_end_to_end_chromosome_standardization(self, no_chr_prefix_gtf: Path):
        """GTF with bare chromosome numbers should still be queryable with 'chr' prefix."""
        parser = GTFParser(str(no_chr_prefix_gtf))
        gtf_dict = parser.get_gtf_dict()

        result = find_genes_at_pos("chr1", 1100, gtf_dict, [])
        assert len(result) == 1
        assert result[0].gene_name == "BRCA1"
