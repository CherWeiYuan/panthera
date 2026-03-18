import pytest
import pandas as pd
from pathlib import Path

# Import your classes and exceptions
from panthera.core.input import VcfVariantReader
from panthera.utils.exceptions import NoPhaseSetError, MultipleVcfSampleError


# ---------------------------------------------------------
# Pure Python Fakes
# ---------------------------------------------------------
class FakeVariant:
    """A lightweight stub to mimic a cyvcf2 Variant object."""

    def __init__(
        self, chrom="chr1", pos=1, ref="A", alt=["T"], formats=None, genotypes=None
    ):
        self.CHROM = chrom
        self.POS = pos
        self.REF = ref
        self.ALT = alt
        self._formats = formats or {}
        self.genotypes = genotypes or [[]]

    def format(self, tag: str):
        # cyvcf2 returns numpy arrays for formats, so we wrap the value in a nested list
        val = self._formats.get(tag)
        return [[val]] if val is not None else None


class FakeGenerator:
    """A lightweight stub to mimic the cyvcf2 VCF reader."""

    def __init__(self, formats=None, samples=None, variants=None):
        # Store 'formats' as a list of strings mimicking the ID in the header
        self._format_ids = formats or []
        self.samples = samples or ["SAMPLE_01"]
        self.variants = variants or []

    def get_header_type(self, tag: str):
        """Mimics cyvcf2's header lookup."""
        if tag in self._format_ids:
            return {"ID": tag, "Type": "Integer"}  # Mimic a valid header dict
        return None

    def __iter__(self):
        return iter(self.variants)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------
@pytest.fixture
def reader():
    return VcfVariantReader()


@pytest.fixture
def simple_vcf_content():
    """Provides a minimal, valid VCF v4.2 string with WhatsHap phasing."""
    return (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=248956422>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase Set">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE_01\n"
        "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:PS\t0/0:1000\n"
        "chr1\t2000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t0/1:1000\n"
        "chr1\t3000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t1/0:1000\n"
        "chr1\t4000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t1/1:1000\n"
        "chr1\t5000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t0|0:1000\n"
        "chr1\t6000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t0|1:1000\n"
        "chr1\t7000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t1|0:1000\n"
        "chr1\t8000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t1|1:1000\n"
        "chr1\t9000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t0/.:1000\n"
        "chr1\t10000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t./0:1000\n"
        "chr1\t11000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t1/.:1000\n"
        "chr1\t12000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t./1:1000\n"
        "chr1\t13000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t./.:1000\n"
        "chr1\t14000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t0|.:1000\n"
        "chr1\t15000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t.|0:1000\n"
        "chr1\t16000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t1|.:1000\n"
        "chr1\t17000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t.|1:1000\n"
        "chr1\t18000\t.\tG\tC\t.\tPASS\t.\tGT:PS\t.|.:1000\n"
    )


# ---------------------------------------------------------
# 1. Integration Test (Real File I/O)
# ---------------------------------------------------------
def test_read_valid_vcf_integration(
    tmp_path: Path, reader: VcfVariantReader, simple_vcf_content: str
):
    """E2E test: Writes a real VCF to disk, parses it, and validates the DataFrame."""
    vcf_file = tmp_path / "test_phased.vcf"
    vcf_file.write_text(simple_vcf_content)

    df = reader.read(vcf_file)

    assert len(df) == 18
    assert list(df.columns) == [
        "chrom",
        "pos",
        "ref",
        "alt",
        "genotype",
        "phase_set",
        "sample_name",
    ]
    assert df.loc[0, "phase_set"] == "PS1000_1"
    assert df.loc[0, "genotype"] == "0/0"
    assert df.loc[1, "genotype"] == "0/1"
    assert df.loc[2, "genotype"] == "1/0"
    assert df.loc[3, "genotype"] == "1/1"
    assert df.loc[4, "genotype"] == "0|0"
    assert df.loc[5, "genotype"] == "0|1"
    assert df.loc[6, "genotype"] == "1|0"
    assert df.loc[7, "genotype"] == "1|1"
    assert df.loc[8, "genotype"] == "0/."
    assert df.loc[9, "genotype"] == "./0"
    assert df.loc[10, "genotype"] == "1/."
    assert df.loc[11, "genotype"] == "./1"
    assert df.loc[12, "genotype"] == "./."
    assert df.loc[13, "genotype"] == "0|."
    assert df.loc[14, "genotype"] == ".|0"
    assert df.loc[15, "genotype"] == "1|."
    assert df.loc[16, "genotype"] == ".|1"
    assert df.loc[17, "genotype"] == ".|."


# ---------------------------------------------------------
# 2. Unit Tests via Native Python Fakes (No unittest)
# ---------------------------------------------------------
def test_check_phaseset_tag_raises_error(reader: VcfVariantReader):
    """Test that missing PS tags are caught immediately."""
    # Arrange: Pass a fake generator with no 'PS' tag
    fake_generator = FakeGenerator(formats=["GT", "GQ", "DP"])

    # Act & Assert
    with pytest.raises(NoPhaseSetError, match="No PS tag found"):
        reader._check_phaseset_tag(fake_generator)


def test_get_sample_name_raises_multiple_sample_error(reader: VcfVariantReader):
    """Test that multi-sample VCFs are rejected."""
    # Arrange: Pass a fake generator with two samples
    fake_generator = FakeGenerator(samples=["SAMPLE_1", "SAMPLE_2"])

    # Act & Assert
    with pytest.raises(MultipleVcfSampleError, match="Expected one sample"):
        reader._get_sample_name(fake_generator)


def test_load_data_fixes_genotype_truthiness_bug(reader: VcfVariantReader):
    """Test the critical bug fix: ensuring '0' is not converted to '.'."""
    # Arrange: Setup exact fake data to trigger the edge case
    fake_variant = FakeVariant(
        chrom="chr1",
        pos=100,
        ref="A",
        alt=["T"],
        formats={"PS": 500},
        genotypes=[[0, -1, True]],  # 0 = HOM_REF, -1 = Missing, True = Phased
    )
    fake_generator = FakeGenerator(
        formats=["PS"], samples=["MOCK_SAMPLE"], variants=[fake_variant]
    )

    # Act
    df = reader._load_data(fake_generator)

    # Assert: Should cleanly parse as 0|. (HOM_REF | MISSING)
    assert df.loc[0, "genotype"] == "0|."
    assert df.loc[0, "phase_set"] == "PS500_1"


def test_clean_data_removes_whitespace_and_brackets(reader: VcfVariantReader):
    """Test the Pandas dataframe cleaning logic independently."""
    # Arrange
    raw_data = pd.DataFrame(
        {
            "chrom": [" chr1 ", "chr2"],
            "pos": [10, 20],
            "ref": [" a ", "G"],
            "alt": ["['t']", "C"],
        }
    )

    # Act
    cleaned_df = reader._clean_data(raw_data)

    # Assert
    assert cleaned_df.loc[0, "chrom"] == "chr1"
    assert cleaned_df.loc[0, "ref"] == "A"
    assert cleaned_df.loc[0, "alt"] == "T"


# ---------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------
@pytest.fixture
def edge_case_vcf_content():
    """Provides an edge case VCF v4.2 string with WhatsHap phasing."""
    content = """
        ##fileformat=VCFv4.2
        ##FILTER=<ID=PASS,Description="All filters passed">
        ##FILTER=<ID=RefCall,Description="Genotyping model thinks this site is reference.">
        ##FILTER=<ID=LowQual,Description="Confidence in this variant being real is below calling threshold.">
        ##FILTER=<ID=NoCall,Description="Site has depth=0 resulting in no call.">
        ##INFO=<ID=END,Number=1,Type=Integer,Description="End position (for use with symbolic alleles)">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Conditional genotype quality">
        ##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
        ##FORMAT=<ID=MIN_DP,Number=1,Type=Integer,Description="Minimum DP observed within the GVCF block.">
        ##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Variant allele fractions.">
        ##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods rounded to the closest integer">
        ##FORMAT=<ID=MED_DP,Number=1,Type=Integer,Description="Median DP observed within the GVCF block rounded to the nearest integer.">
        ##DeepVariant_version=1.4.0
        ##contig=<ID=chr1,length=248956422>
        ##contig=<ID=chr2,length=242193529>
	    ##contig=<ID=chr3,length=198295559>
	    ##contig=<ID=chrX,length=156040895>
        ##commandline="(whatshap 1.7) ABCA4
        ##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
        #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG002
        chr1	0	.	GGGGGG	G	49.8	PASS	.	GT:GQ:DP:AD:VAF:PL	0/1:43:243:120,123:0.506173:49,0,43
        chr1	12	.	C	CTTTTT	40.6	PASS	.	GT:GQ:DP:AD:VAF:PL:PS	1|0:41:46:26,19:0.413043:40,0,58:93995982
        chr1	100	.	GATAGA	G	53.7	PASS	.	GT:GQ:DP:AD:VAF:PL:PS	1|0:53:229:132,97:0.423581:53,0,58:93995982
        chr2	110	.	G	ATTT	39.5	PASS	.	GT:GQ:DP:AD:VAF:PL:PS	1|0:39:78:43,35:0.448718:39,0,53:93995982
        chr3	93997432	.	ACCC	A	1.9	RefCall	.	GT:GQ:DP:AD:VAF:PL	0|1:4:142:63,34:0.239437:0,2,26
        chr3	93997549	.	G	A,GCAT	55.1	PASS	.	GT:GQ:DP:AD:VAF:PL:PS	1|0:47:421:213,207:0.491686:55,0,48:93995982
        chr3	93997600	.	T	TTT,TA	55.1	PASS	.	GT:GQ:DP:AD:VAF:PL:PS	0|1:47:421:213,207:0.491686:55,0,48:93995982
        chrX	93999515	.	TTT	C	41.4	PASS	.	GT:GQ:DP:AD:VAF:PL:PS	0|1:41:473:234,238:0.503171:41,0,58:93998211
        """

    return content
