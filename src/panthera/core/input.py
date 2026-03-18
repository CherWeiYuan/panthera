"""
Input files.

This module contain the function, ingest_variants, to read input VCF or TSV files.

The architecture of this module is as follows:
    1. read_variants() is executed.
    2. VariantReaderFactory recognizes TSV or VCF and load variants accordingly
       using either TsvVariantReader or VcfVariantReader into a Pandas dataframe.
       - Both reader class will inherit from VariantReader which
         enforces the same abstract method, read().
    3. VariantSchema enforces the structure of the output Pandas dataframe.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from panthera.utils.exceptions import (
    MultipleAltError,
    MultipleVcfSampleError,
    NoPhaseSetError,
)

# Configure enterprise logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Output Schema (Pandera)
# ---------------------------------------------------------
class VariantSchema(pa.DataFrameModel):
    """Enforces the structure of our resulting DataFrame."""

    # Coerce integer chrom names (e.g. 22) into string (e.g. "22")
    chrom: Series[str] = pa.Field(coerce=True)

    # Position must be greater or equal (ge) than 1
    pos: Series[int] = pa.Field(ge=1)

    # Reference and alternate alleles
    ref: Series[str]
    alt: Series[str]

    # Additional columns
    genotype: Series[str]
    genetic_background: Series[str]

    # Allow additional columns in the file but ignore them
    class Config:
        strict = False


# ---------------------------------------------------------
# Reader Interface
# ---------------------------------------------------------
class VariantReader(ABC):
    """Abstract base class for all variant file readers."""

    @abstractmethod
    def read(self, filepath: Path) -> pd.DataFrame:
        """Reads a file and returns a normalized DataFrame."""
        pass

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs data sanitization using vectorized operations"""
        if df.empty:
            return df

        # Strip whitespace and handle casing
        # "\s": matches the white space character;
        # "+": matches match the preceding element one or more times
        df["chrom"] = df["chrom"].astype(str).str.replace(r"\s+", "", regex=True)
        df["ref"] = df["ref"].str.replace(r"\s+", "", regex=True).str.upper()

        # Complex string cleaning for 'alt' using regex
        # Strip formatting brackets and quotes injected by upstream caller.
        # i.e. Removes: [ ] ' and whitespace
        # Outer square bracket "[]": indicates set
        #   (i.e., look for any of the characters within the set)
        # "\[": matches [
        # "\]": matches ]
        # ': matches a literal single quote
        # \s: matches white space character
        df["alt"] = df["alt"].str.replace(r"[\[\]'\s]", "", regex=True).str.upper()

        # Drop duplicates and sort
        df = df.drop_duplicates().sort_values(by=["chrom", "pos"], ignore_index=True)
        return df


# ---------------------------------------------------------
# Concrete Implementations
# ---------------------------------------------------------


class TsvVariantReader(VariantReader):
    """
    Handles extraction and normalization of variants from TSV files.
    """

    # Define constants
    REQUIRED_COLUMNS = ["chrom", "pos", "ref", "alt"]
    DEFAULT_GENOTYPE = "1|1"
    DEFAULT_BACKGROUND = "BG0"
    DEFAULT_PHASESET = "PST0"
    DEFAULT_SAMPLE = "S0"

    def read(self, filepath: Path) -> pd.DataFrame:
        """Main entry point to load, clean, and format the TSV."""
        df = self._load_data(filepath)
        self._validate_alleles(df)
        df = self._clean_data(df)
        df = self._apply_formatting(df)

        return df

    def _load_data(self, filepath: Path) -> pd.DataFrame:
        """Loads the raw TSV data"""
        try:
            return pd.read_csv(filepath, sep="\t")
        except Exception as e:
            logger.error(f"Failed to load TSV file {filepath}: {e}")
            raise

    def _validate_alleles(self, df: pd.DataFrame) -> None:
        """Vectorized check for multiple alleles"""
        if df["alt"].str.contains(",").any():
            logger.error("Multiple alternate alleles detected in input.")
            raise MultipleAltError("Rows with multiple ALT alleles are not supported.")

    def _apply_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds downstream-required metadata columns"""
        df["genotype"] = self.DEFAULT_GENOTYPE
        df["phase_set"] = self.DEFAULT_PHASESET
        df["sample"] = self.DEFAULT_SAMPLE

        return df


class VcfVariantReader(VariantReader):
    """
    Handles extraction and normalization of variants from VCF files.
    """

    def read(self, filepath: Path) -> pd.DataFrame:
        """Main entry point to load, clean, and format the VCF."""
        logger.info(f"Reading VCF file: {filepath}")

        try:
            generator = self._get_vcf_generator(filepath)
            self._check_phaseset_tag(generator)
            df = self._load_data(generator)
            df = self._clean_data(df)

            return df

        except Exception as e:
            logger.error(f"Failed to parse VCF {filepath}: {e}")
            raise

    def _get_vcf_generator(self, filepath: Path) -> Any:
        """Loads VCF path into cyvcf2."""
        # Using Any or a generic type because importing cyvcf2 just for typing
        # can sometimes cause circular import issues in large codebases.
        from cyvcf2 import VCF

        return VCF(str(filepath))

    def _check_phaseset_tag(self, generator: Any) -> None:
        """Validates that the VCF header contains the Phase Set (PS) definition."""
        # get_header_type returns a dict if the tag exists, or None if it's missing
        if generator.get_header_type("PS") is None:
            logger.error(
                "No PS tag in VCF format header. VCF was not phased by WhatsHap."
            )
            raise NoPhaseSetError(
                "No PS tag found in VCF's format header. Phasing information is missing."
            )

    def _get_sample_name(self, generator: Any) -> str:
        """Get sample name from generator"""
        # Ensure only one sample exist in VCF
        sample_names = generator.samples
        if len(sample_names) != 1:
            logger.error("Multiple sample names detected in VCF")
            raise MultipleVcfSampleError(
                f"Expected one sample. Got: {len(sample_names)}"
            )
        return sample_names[0]

    def _load_data(self, generator: Any) -> pd.DataFrame:
        # Get sample name
        sample_name = self._get_sample_name(generator)

        # Iterate through generator to collect variants
        ps_counter = 0
        df_seed = []
        for variant in generator:
            ps_counter += 1
            alt = str(variant.ALT)

            # Provide warning if more than one alt allele exist per VCF entry
            if len(alt) > 1:
                logger.warning(
                    f"Expect one alternate allele. Got: {alt}. "
                    + f"Only {alt[0]} is used for analysis"
                )

            # Safely extract formatting, handling numpy array returns from cyvcf2
            ps_raw = variant.format("PS")
            ps_val = str(ps_raw[0][0]) if ps_raw is not None else "UNKNOWN"

            # Get genotype (generator output in list format)
            # Use index 0 to retrieve since there is only sample
            # (number of samples is guaranteed by _get_sample_name())
            # Genotype format is [allele1, allele2, is_phased], where allele1 or
            # allele 2 is an integer:
            # 0: HOM_REF
            # 1: HET
            # 2: UNKNOWN
            # 3: HOM_ALT
            gt = variant.genotypes[0]
            # If allele is -1, set to "."
            # Ensure that if allele is 0, the allele variant is assigned 0,
            # as 0 evalutes to False (i.e., gt[0] if gt[0] else "." will fail)
            allele1 = gt[0] if gt[0] != -1 else "."
            allele2 = gt[1] if gt[1] != -1 else "."
            separator = "|" if gt[2] else "/"
            genotype = f"{allele1}{separator}{allele2}"

            # Append as dictionary for strict mapping
            df_seed.append(
                {
                    "chrom": str(variant.CHROM),
                    "pos": int(variant.POS),
                    "ref": str(variant.REF),
                    "alt": alt[0],
                    "genotype": genotype,
                    "phase_set": f"PS{ps_val}_{ps_counter}",
                    "sample_name": sample_name,
                }
            )

        return pd.DataFrame(df_seed)


# ---------------------------------------------------------
# The Factory (Router)
# ---------------------------------------------------------
class VariantReaderFactory:
    """Returns the appropriate reader based on file extension."""

    @staticmethod
    def get_reader(filepath: Path) -> VariantReader:
        # Resolve the actual file extension, handling .vcf.gz
        suffixes = filepath.suffixes
        ext = "".join(suffixes).lower()

        if ext in [".vcf", ".vcf.gz"]:
            return VcfVariantReader()
        elif ext in [".tsv", ".tsv.gz", ".txt"]:
            return TsvVariantReader()
        else:
            raise ValueError(f"Unsupported file format: {ext}")


# ---------------------------------------------------------
# Main Loading Service
# ---------------------------------------------------------
def read_variants(filepath: str | Path) -> DataFrame[VariantSchema]:
    """
    Main entrypoint for data ingestion.
    Reads the file and validates the schema.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Get the right tool for the job
    reader = VariantReaderFactory.get_reader(path)

    # Extract the data
    df = reader.read(path)

    # Validate the data structure before returning
    validated_df = VariantSchema.validate(df)

    logger.info(f"Successfully loaded and validated {len(validated_df)} variants.")
    return validated_df
