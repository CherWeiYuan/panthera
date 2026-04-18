"""Input variant reading and processing.

This module provides classes and functions to read variant data from VCF and
TSV files, normalizing them into a consistent DataFrame format.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandera.typing import DataFrame

from panthera.core.bio.blocks import VariantSchema
from panthera.utils.exceptions import (
    MultipleAltError,
    MultipleVcfSampleError,
    NoPhaseSetError,
    NoVariantsError,
)

# Set up module-level logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Reader Interface
# ---------------------------------------------------------
class VariantReader(ABC):
    """Abstract base class for all variant file readers."""

    @abstractmethod
    def read(self, filepath: Path) -> pd.DataFrame:
        """Reads a file and returns a normalized DataFrame.

        Args:
            filepath: Path to the variant file.

        Returns:
            pd.DataFrame: DataFrame containing the variant data.

        Raises:
            FileNotFoundError: If the variant file is not found.
            ValueError: If the variant file is empty.
            MultipleAltError: If the variant file contains multiple alternate alleles.
            NoPhaseSetError: If the variant file does not contain phase set information.
            NoVariantsError: If the variant file does not contain any variants.
        """
        pass

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes variant data using vectorized operations.

        Args:
            df: DataFrame containing the variant data.

        Returns:
            pd.DataFrame: Cleaned variant data.
        """
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
    """Handles extraction and normalization of variants from TSV files.

    Attributes:
        REQUIRED_COLUMNS: List of mandatory columns.
        DEFAULT_GENOTYPE: Fallback genotype for TSV variants ("1|1").
        DEFAULT_BACKGROUND: Default background identifier.
        DEFAULT_PHASESET: Default phase set identifier.
        DEFAULT_SAMPLE: Default sample identifier.
    """

    # Define constants
    REQUIRED_COLUMNS = ["chrom", "pos", "ref", "alt"]
    DEFAULT_GENOTYPE = "1|1"  # Always "1|1" for TSV
    DEFAULT_BACKGROUND = "BG0"
    DEFAULT_PHASESET = "PST0"
    DEFAULT_SAMPLE = "S0"

    def read(self, filepath: Path) -> pd.DataFrame:
        """Loads, cleans, and formats a TSV file.

        Args:
            filepath: Path to the TSV file.

        Returns:
            pd.DataFrame: Cleaned variant data.

        Raises:
            NoVariantsError: If the TSV file is empty or contains no variants.
        """
        df = self._load_data(filepath)
        if df.empty:
            error_msg = f"TSV file {filepath} contains no variants."
            logger.error(error_msg)
            raise NoVariantsError(error_msg)

        self._validate_alleles(df)
        df = self._clean_data(df)
        df = self._apply_formatting(df)

        return df

    def _load_data(self, filepath: Path) -> pd.DataFrame:
        """Loads the raw TSV data.

        Args:
            filepath: Path to the TSV file.

        Returns:
            pd.DataFrame: DataFrame containing the raw variant data.

        Raises:
            FileNotFoundError: If the TSV file is not found.
            ValueError: If the TSV file is empty.
            MultipleAltError: If the TSV file contains multiple alternate alleles.
            NoVariantsError: If the TSV file does not contain any variants.
        """
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
        df["sample_name"] = self.DEFAULT_SAMPLE

        return df


class VcfVariantReader(VariantReader):
    """Handles extraction and normalization of variants from VCF files.

    Attributes:
        REQUIRED_COLUMNS: List of mandatory columns.
    """

    # Define constants
    REQUIRED_COLUMNS = ["chrom", "pos", "ref", "alt"]

    def read(self, filepath: Path) -> pd.DataFrame:
        """Loads, cleans, and formats a VCF file.

        Args:
            filepath: Path to the VCF file.

        Returns:
            pd.DataFrame: Cleaned variant data.

        Raises:
            NoVariantsError: If the VCF file contains no variants.
        """
        logger.info(f"Reading VCF file: {filepath}")

        try:
            generator = self._get_vcf_generator(filepath)
            self._check_phaseset_tag(generator)
            df = self._load_data(generator)
            if df.empty:
                error_msg = f"VCF file {filepath} contains no variants."
                logger.error(error_msg)
                raise NoVariantsError(error_msg)
            df = self._clean_data(df)

            return df

        except Exception as e:
            logger.error(f"Failed to parse VCF {filepath}: {e}")
            raise

    def _get_vcf_generator(self, filepath: Path) -> Any:
        """Loads VCF path into cyvcf2.

        Args:
            filepath: Path to the VCF file.

        Returns:
            Any: cyvcf2 VCF object.
        """
        # Using Any or a generic type because importing cyvcf2 just for typing
        # can sometimes cause circular import issues in large codebases.
        from cyvcf2 import VCF

        # cyvcf2 loads raw and .gz VCF files using the same method
        return VCF(str(filepath))

    def _check_phaseset_tag(self, generator: Any) -> None:
        """Validates that the VCF header contains the Phase Set (PS) definition.

        Args:
            generator: cyvcf2 VCF object.

        Raises:
            NoPhaseSetError: If the VCF file does not contain phase set information.
        """
        # get_header_type returns a dict if the tag exists, or None if it's missing
        if generator.get_header_type("PS") is None:
            logger.error(
                "No PS tag in VCF format header. VCF was not phased by WhatsHap."
            )
            raise NoPhaseSetError(
                "No PS tag found in VCF's format header. "
                "Phasing information is missing."
            )

    def _get_sample_name(self, generator: Any) -> str:
        """Get sample name from generator.

        Args:
            generator: cyvcf2 VCF object.

        Returns:
            str: Sample name.

        Raises:
            MultipleVcfSampleError: If the VCF file does not contain exactly one sample.
        """
        # Ensure only one sample exist in VCF
        sample_names = generator.samples
        if len(sample_names) != 1:
            logger.error("Multiple sample names detected in VCF")
            raise MultipleVcfSampleError(
                f"Expected one sample. Got: {len(sample_names)}"
            )
        return sample_names[0]

    def _load_data(self, generator: Any) -> pd.DataFrame:
        """Loads the raw VCF data into a Pandas dataframe.

        Args:
            generator: cyvcf2 VCF object.

        Returns:
            pd.DataFrame: DataFrame containing the raw variant data.

        Raises:
            MultipleVcfSampleError: If the VCF file does not contain exactly
                one sample.
        """
        # Get sample name
        sample_name = self._get_sample_name(generator)

        # Iterate through generator to collect variants
        ps_counter = 0
        df_seed = []
        for variant in generator:
            ps_counter += 1

            # Process alternate allele
            alt = variant.ALT  # List of alleles, e.g. "['A', 'G']"
            # Provide warning if more than one alt allele exist per VCF entry
            if len(alt) > 1:
                logger.warning(
                    f"Expect one alternate allele. Got: {alt}. "
                    + f"Only {alt[0]} is used for analysis"
                )
            alt = str(alt[0])

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
        """Returns the appropriate reader based on file extension.

        Args:
            filepath: Path to the variant file.

        Returns:
            VariantReader: The appropriate reader based on file extension.

        Raises:
            ValueError: If the file extension is not supported.
        """
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
    """Reads a variant file (VCF/TSV) and validates it against the schema.

    Args:
        filepath: Path to the variant file.

    Returns:
        DataFrame[VariantSchema]: Validated variant DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
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
