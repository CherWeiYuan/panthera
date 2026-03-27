import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from pandera.errors import SchemaError

# Assuming the provided code is saved as parse_bg_vcf.py
from panthera.core.bio.parse_bg_vcf import (
    GenomicRegion,
    RegionVcfSchema,
    RegionVcfReader,
    read_vcf_region,
    VCFCoordinates,
    BgVcfManager,
)
from panthera.utils.exceptions import (
    DataResolutionError,
    MultipleVcfSampleError,
    NoVariantsError,
)

# ==========================================
# Fixtures & Mocks
# ==========================================

class MockVariant:
    """A mock cyvcf2 Variant object for testing."""
    def __init__(self, chrom, pos, ref, alt, genotypes):
        self.CHROM = chrom
        self.POS = pos
        self.REF = ref
        self.ALT = alt
        self.genotypes = genotypes

@pytest.fixture
def valid_vcf_df():
    """Returns a DataFrame that perfectly matches the RegionVcfSchema."""
    return pd.DataFrame({
        "chrom": ["1", "22"],
        "pos": [1000, 2000],
        "ref": ["A", "G"],
        "alt": ["T", "C"],
        "genotype": ["0/1", "1|1"],
        "genetic_background": ["SAMPLE", "SAMPLE"]
    })

# ==========================================
# Tests: GenomicRegion
# ==========================================

def test_genomic_region_standard():
    region = GenomicRegion(chrom="chr1", start=100, end=200)
    assert region.chrom == "chr1"
    assert region.start == 100
    assert region.end == 200
    assert region.to_region_string() == "chr1:100-200"

def test_genomic_region_swap_coordinates():
    # Edge Case: Caller supplies end before start
    region = GenomicRegion(chrom="chr2", start=500, end=100)
    assert region.start == 100  # Should be swapped
    assert region.end == 500
    assert region.to_region_string() == "chr2:100-500"

# ==========================================
# Tests: RegionVcfSchema
# ==========================================

def test_schema_valid_dataframe(valid_vcf_df):
    # Should not raise an error
    validated = RegionVcfSchema.validate(valid_vcf_df)
    assert not validated.empty

def test_schema_invalid_pos():
    # Edge Case: pos < 1
    df = pd.DataFrame({
        "chrom": ["1"], "pos": [0], "ref": ["A"], "alt": ["T"],
        "genotype": ["0/1"], "genetic_background": ["SAMPLE"]
    })
    with pytest.raises(SchemaError):
        RegionVcfSchema.validate(df)

def test_schema_coerces_chrom():
    df = pd.DataFrame({
        "chrom": [22], "pos": [100], "ref": ["A"], "alt": ["T"], # Integer chrom
        "genotype": ["0/1"], "genetic_background": ["SAMPLE"]
    })
    validated = RegionVcfSchema.validate(df)
    assert validated["chrom"].dtype == "object" # Pandas string/object type
    assert validated["chrom"].iloc[0] == "22"

# ==========================================
# Tests: RegionVcfReader
# ==========================================

def test_get_sample_name_success():
    reader = RegionVcfReader()
    mock_generator = MagicMock()
    mock_generator.samples = ["SAMPLE1"]
    assert reader._get_sample_name(mock_generator) == "SAMPLE1"

def test_get_sample_name_multiple_raises():
    reader = RegionVcfReader()
    mock_generator = MagicMock()
    mock_generator.samples = ["SAMPLE1", "SAMPLE2"]
    with pytest.raises(MultipleVcfSampleError, match="Expected one sample"):
        reader._get_sample_name(mock_generator)

def test_load_data_logic():
    reader = RegionVcfReader()
    records = [
        # Heterozygous unphased
        MockVariant("chr1", 100, "A", ["T"], [[1, 0, False]]), 
        # Homozygous Ref (Should be dropped silently)
        MockVariant("chr1", 200, "G", ["C"], [[0, 0, False]]), 
        # Homozygous Alt, Phased, Multiple Alts (takes first)
        MockVariant("chr1", 300, "T", ["C", "G"], [[1, 1, True]]), 
        # Missing genotype
        MockVariant("chr1", 400, "C", ["A"], [[-1, -1, False]]) 
    ]
    
    df = reader._load_data(records, sample_name="TEST_SAMPLE")
    
    # Assert row count (1 homozygous ref dropped)
    assert len(df) == 3 
    
    # Assert Genotypes
    assert df.iloc[0]["genotype"] == "1/0"
    assert df.iloc[1]["genotype"] == "1|1"
    assert df.iloc[2]["genotype"] == "./."
    
    # Assert Multiple Alt fallback
    assert df.iloc[1]["alt"] == "C" 

def test_clean_data_regex_and_sort():
    reader = RegionVcfReader()
    dirty_df = pd.DataFrame({
        "chrom": [" chr2", "chr1\t"],
        "pos": [200, 100],
        "ref": ["A ", " c"],
        "alt": ["['G']", "[T] "],
        "genotype": ["0/1", "1/1"],
        "sample_name": ["S1", "S1"]
    })
    
    clean_df = reader._clean_data(dirty_df)
    
    # Check sorting
    assert clean_df.iloc[0]["chrom"] == "chr1"
    assert clean_df.iloc[0]["pos"] == 100
    
    # Check regex cleaning
    assert clean_df.iloc[0]["ref"] == "C" # Capitalized and trimmed
    assert clean_df.iloc[0]["alt"] == "T" # Brackets, spaces removed
    assert clean_df.iloc[1]["alt"] == "G" # Quotes removed

@patch.object(RegionVcfReader, '_get_vcf_generator')
@patch.object(RegionVcfReader, '_load_data')
def test_read_empty_variants_raises(mock_load, mock_get_gen, tmp_path):
    mock_load.return_value = pd.DataFrame() # Simulate empty load
    mock_get_gen.return_value.samples = ["S1"]
    
    reader = RegionVcfReader()
    filepath = tmp_path / "test.vcf"
    
    with pytest.warns(UserWarning, match="contains no callable variants"):
        reader.read(filepath)

# ==========================================
# Tests: read_vcf_region Function
# ==========================================

def test_read_vcf_region_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_vcf_region("does_not_exist.vcf")

def test_read_vcf_region_partial_args(tmp_path):
    # Edge Case: Providing only start, missing end and chrom
    filepath = tmp_path / "test.vcf"
    filepath.touch()
    
    with pytest.raises(ValueError, match="requires all three arguments"):
        read_vcf_region(filepath, start=100)
        
    with pytest.raises(ValueError):
        read_vcf_region(filepath, chrom="chr1", start=100)

@patch('panthera.core.bio.parse_bg_vcf.RegionVcfReader.read')
def test_read_vcf_region_success(mock_read, valid_vcf_df, tmp_path):
    mock_read.return_value = valid_vcf_df
    filepath = tmp_path / "test.vcf"
    filepath.touch()
    
    result = read_vcf_region(filepath)
    assert not result.empty
    mock_read.assert_called_once_with(filepath)

# ==========================================
# Tests: BgVcfManager
# ==========================================

def test_manager_missing_external_file(tmp_path):
    manager = BgVcfManager(external_dir=tmp_path)
    coords = VCFCoordinates("chr1", 100, 200)
    
    with pytest.raises(DataResolutionError, match="User-provided file not found"):
        manager.fetch_region("missing_sample", coords)

def test_manager_missing_tbi_index(tmp_path):
    manager = BgVcfManager(external_dir=tmp_path)
    
    # Create fake VCF without an index
    vcf_file = tmp_path / "sample.vcf.gz"
    vcf_file.touch()
    
    coords = VCFCoordinates("chr1", 100, 200)
    
    with pytest.raises(DataResolutionError, match="VCF index missing"):
        manager.fetch_region("sample", coords)

@patch('panthera.core.bio.parse_bg_vcf.read_vcf_region')
def test_manager_fetch_region_success(mock_read_vcf, valid_vcf_df, tmp_path):
    # Mocking read_vcf_region to bypass actual cyvcf2/file I/O parsing
    mock_read_vcf.return_value = valid_vcf_df
    
    manager = BgVcfManager(external_dir=tmp_path)
    
    # Create fake VCF and index
    vcf_file = tmp_path / "sample.vcf.gz"
    vcf_file.touch()
    tbi_file = tmp_path / "sample.vcf.gz.tbi"
    tbi_file.touch()
    
    coords = VCFCoordinates("chr1", 100, 200)
    result = manager.fetch_region("sample", coords)
    
    assert not result.empty
    mock_read_vcf.assert_called_once()
    assert mock_read_vcf.call_args[1]["chrom"] == "chr1"