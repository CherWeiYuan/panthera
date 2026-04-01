"""Region-filtered VCF input.

This module provides utilities to read VCF files with optional genomic region
filtering, normalizing them into DataFrames. It also manages background VCF
resources.
"""

from importlib import resources
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
import warnings

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from panthera.utils.exceptions import DataResolutionError, MultipleVcfSampleError

# Set up module-level logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Genomic Region (optional filter)
# ---------------------------------------------------------
@dataclass(frozen=True)
class GenomicRegion:
    """Represents a genomic interval [start, end] on a chromosome.

    Attributes:
        chrom: Chromosome name.
        start: 1-based start coordinate.
        end: 1-based end coordinate.
    """

    chrom: str
    start: int
    end: int

    def __post_init__(self) -> None:
        # Swap so that start <= end regardless of caller order
        if self.start > self.end:
            # Capture the values before overwriting
            low, high = self.end, self.start
            object.__setattr__(self, "start", low)
            object.__setattr__(self, "end", high)

    def to_region_string(self) -> str:
        """Returns a cyvcf2-compatible region string (e.g. "chr1:1000-2000")."""
        return f"{self.chrom}:{self.start}-{self.end}"


# ---------------------------------------------------------
# Output Schema (Pandera)
# ---------------------------------------------------------
class RegionVcfSchema(pa.DataFrameModel):
    """Enforces the structure of the DataFrame returned by read_vcf_region."""

    # Coerce integer chrom names (e.g. 22) into string (e.g. "22")
    chrom: Series[str] = pa.Field(coerce=True)

    # Position must be greater or equal (ge) than 1
    pos: Series[int] = pa.Field(ge=1)

    # Reference and alternate alleles
    ref: Series[str]
    alt: Series[str]

    # Diploid genotype string (e.g. "0/1", "1|0")
    genotype: Series[str]

    # Tracks the originating sample; populated from VCF header
    genetic_background: Series[str]

    # Allow additional columns in the file but ignore them
    class Config:
        strict = False


# ---------------------------------------------------------
# Reader
# ---------------------------------------------------------
_HOMOZYGOUS_REF_GENOTYPES = frozenset({"0|0", "0/0"})


class RegionVcfReader:
    """Loads, filters, and normalizes a single-sample VCF into a DataFrame.

    Attributes:
        region: Optional genomic region. When None, the entire VCF is read.
    """

    # Constant genetic_background label applied to every row
    GENETIC_BACKGROUND_LABEL: str = "SAMPLE"

    def __init__(self, region: Optional[GenomicRegion] = None) -> None:
        self._region = region

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, filepath: Path) -> pd.DataFrame:
        """Main entry point: load, filter, clean, and format the VCF.

        Args:
            filepath: Path to a plain-text or gzip-compressed VCF file.

        Returns:
            pd.DataFrame: Normalized DataFrame conforming to RegionVcfSchema.

        Raises:
            MultipleVcfSampleError: If the VCF contains more than one sample column.
        """
        logger.info(f"Reading VCF file: {filepath}")

        try:
            generator = self._get_vcf_generator(filepath)
            sample_name = self._get_sample_name(generator)

            if self._region is not None:
                records = self._fetch_region(generator)
            else:
                records = generator  # iterate the whole file

            df = self._load_data(records, sample_name)
            if df.empty:
                region_str = ""
                if self._region is not None:
                    region_str = f"in region {self._region.to_region_string()}"
                warning_msg = (
                    f"VCF file {filepath} contains no callable variants {region_str}."
                )
                logger.warning(warning_msg)
                warnings.warn(warning_msg, UserWarning)

            df = self._clean_data(df)
            df = self._apply_formatting(df)
            return df

        except Exception:
            logger.error(f"Failed to parse VCF {filepath}.")
            raise

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_vcf_generator(self, filepath: Path) -> Any:
        """Wraps the filepath in a cyvcf2 VCF generator.

        Args:
            filepath: Path to the raw or gzipped VCF file.

        Returns:
            Any: cyvcf2 VCF object.
        """
        from cyvcf2 import VCF

        return VCF(str(filepath))

    def _get_sample_name(self, generator: Any) -> str:
        """Returns the single sample name from the VCF header.

        Args:
            generator: cyvcf2 VCF object.

        Returns:
            str: The sample name.

        Raises:
            MultipleVcfSampleError: If the VCF contains more than one sample column.
        """
        sample_names = generator.samples
        if len(sample_names) != 1:
            logger.error("Multiple sample names detected in VCF.")
            raise MultipleVcfSampleError(
                f"Expected one sample. Got: {len(sample_names)}"
            )
        return sample_names[0]

    def _fetch_region(self, generator: Any) -> Any:
        """Restricts iteration to the genomic region stored on this reader.

        Args:
            generator: cyvcf2 VCF object.

        Returns:
            Any: Filtered records generator.
        """
        # _region is guaranteed non-None when this method is called
        region_str = self._region.to_region_string()  # type: ignore[union-attr]
        logger.info(f"Applying region filter: {region_str}")
        return generator(region_str)

    def _load_data(self, records: Any, sample_name: str) -> pd.DataFrame:
        """Iterates through VCF records and builds a seed DataFrame.

        Args:
            records: Iterable of VCF records.
            sample_name: Name of the sample.

        Returns:
            pd.DataFrame: Seed DataFrame containing variant data.
        """
        df_seed = []
        for variant in records:
            # --- ALT allele --------------------------------------------------
            alt_alleles = variant.ALT  # List, e.g. ["A", "G"]
            if not alt_alleles:
                logger.warning(
                    f"No ALT alleles found at {variant.CHROM}:{variant.POS}. Skipping."
                )
                continue
            if len(alt_alleles) > 1:
                logger.warning(
                    f"Expected one alternate allele. Got: {alt_alleles}. "
                    f"Only {alt_alleles[0]} will be used."
                )
            alt = str(alt_alleles[0]).upper()

            # --- Genotype ----------------------------------------------------
            # variant.genotypes returns [[allele1, allele2, is_phased], ...]
            # Use index 0 since single-sample is guaranteed by _get_sample_name.
            # Genotype integer encoding:
            #   -1 → missing (".")
            #    0 → HOM_REF
            #    1 → HET / ALT
            gt = variant.genotypes[0]
            allele1 = gt[0] if gt[0] != -1 else "."
            allele2 = gt[1] if gt[1] != -1 else "."
            separator = "|" if gt[2] else "/"
            genotype = f"{allele1}{separator}{allele2}"

            # Skip homozygous-reference calls — not informative for downstream
            if genotype in _HOMOZYGOUS_REF_GENOTYPES:
                continue

            df_seed.append(
                {
                    "chrom": str(variant.CHROM),
                    "pos": int(variant.POS),
                    "ref": str(variant.REF).upper(),
                    "alt": alt,
                    "genotype": genotype,
                    "sample_name": sample_name,
                }
            )

        if not df_seed:
            return pd.DataFrame(
                {
                    "chrom": pd.Series(dtype=str),
                    "pos": pd.Series(dtype="int64"),
                    "ref": pd.Series(dtype=str),
                    "alt": pd.Series(dtype=str),
                    "genotype": pd.Series(dtype=str),
                    "sample_name": pd.Series(dtype=str),
                }
            )

        return pd.DataFrame(df_seed)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes string columns and deduplicates rows.

        Args:
            df: The DataFrame to clean.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        if df.empty:
            return df

        # Strip stray whitespace; normalize chromosome to plain string
        # "\s+": one or more whitespace characters
        df["chrom"] = df["chrom"].astype(str).str.replace(r"\s+", "", regex=True)
        df["ref"] = df["ref"].str.replace(r"\s+", "", regex=True).str.upper()

        # Strip formatting artefacts from the alt column:
        # "\[" → literal [
        # "\]" → literal ]
        # "'"  → literal single quote
        # "\s" → whitespace
        df["alt"] = df["alt"].str.replace(r"[\[\]'\s]", "", regex=True).str.upper()

        # Remove duplicates and sort by genomic coordinate
        df = df.drop_duplicates().sort_values(by=["chrom", "pos"], ignore_index=True)
        return df

    def _apply_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds the genetic_background metadata column.

        Args:
            df: The DataFrame to format.

        Returns:
            pd.DataFrame: The formatted DataFrame.
        """
        df["genetic_background"] = self.GENETIC_BACKGROUND_LABEL
        return df


# ---------------------------------------------------------
# Loading function
# ---------------------------------------------------------
def read_vcf_region(
    filepath: str | Path,
    chrom: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> DataFrame[RegionVcfSchema]:
    """Reads a VCF file with an optional genomic region filter.

    Args:
        filepath: Path to the VCF file.
        chrom: Chromosome name for region filter.
        start: 1-based start coordinate.
        end: 1-based end coordinate.

    Returns:
        DataFrame[RegionVcfSchema]: Validated DataFrame with variant data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If only a subset of the region arguments is provided.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Validate that region arguments are either all present or all absent
    region_args = (chrom, start, end)
    if any(arg is not None for arg in region_args) and not all(
        arg is not None for arg in region_args
    ):
        raise ValueError(
            "Region filtering requires all three arguments: chrom, start, end. "
            f"Got chrom={chrom!r}, start={start!r}, end={end!r}."
        )

    region = (
        GenomicRegion(chrom=chrom, start=int(start), end=int(end))  # type: ignore[arg-type]
        if all(arg is not None for arg in region_args)
        else None
    )

    reader = RegionVcfReader(region=region)
    df = reader.read(path)

    validated_df = RegionVcfSchema.validate(df)

    logger.info(
        f"Successfully loaded and validated {len(validated_df)} variants "
        f"from {path.name}"
        + (f" [{region.to_region_string()}]." if region is not None else ".")
    )
    return validated_df


# ---------------------------------------------------------
# Loading Manager
# ---------------------------------------------------------


@dataclass
class VCFCoordinates:
    """Value object to hold genomic coordinates.

    Attributes:
        chrom: Chromosome name.
        start: Start position.
        end: End position.
    """

    chrom: str
    start: int
    end: int


class BgVcfManager:
    """Handles resolution and access of VCF resources from internal
    package storage or external user-defined directories.
    """

    PACKAGE_DATA_PATH = "panthera.data.genetic_background_vcf"

    def __init__(self, external_dir: Optional[Union[str, Path]] = None):
        self.external_dir = Path(external_dir) if external_dir else None

    def _get_resource_ref(self, filename: str) -> Any:
        """Internal logic to decide between User path or Package path.

        Args:
            filename: Name of the resource file.

        Returns:
            Any: Path or Traversable to the resource.

        Raises:
            DataResolutionError: If the external file is not found.
        """
        if self.external_dir:
            target_path = self.external_dir / filename
            if not target_path.exists():
                raise DataResolutionError(
                    f"User-provided file not found: {target_path}"
                )
            logger.info(f"Using external resource: {target_path}")
            return target_path

        # Fallback to internal resources
        logger.debug(
            "Accessing internal package resource: "
            + f"{self.PACKAGE_DATA_PATH}/{filename}"
        )
        return resources.files(self.PACKAGE_DATA_PATH).joinpath(filename)

    def fetch_region(
        self, sample_id: str, coords: VCFCoordinates
    ) -> DataFrame[RegionVcfSchema]:
        """Resolves the VCF file and reads data for a specific region.

        Args:
            sample_id: Sample identifier.
            coords: Genomic coordinates.

        Returns:
            DataFrame[RegionVcfSchema]: Region-filtered VCF data.
        """
        vcf_name = f"{sample_id}.vcf.gz"
        tbi_name = f"{vcf_name}.tbi"

        try:
            # 1. Resolve VCF Path
            vcf_ref = self._get_resource_ref(vcf_name)

            # 2. Context management for zip-safe access
            with resources.as_file(vcf_ref) as vcf_path:
                self._validate_index_exists(Path(vcf_path), tbi_name)

                logger.info(
                    f"Querying {vcf_name} for {coords.chrom}"
                    + f":{coords.start}-{coords.end}"
                )

                return read_vcf_region(
                    filepath=vcf_path,
                    chrom=coords.chrom,
                    start=coords.start,
                    end=coords.end,
                )

        except FileNotFoundError as e:
            logger.error(f"Missing required data file: {e}")
            raise DataResolutionError(f"Resource {vcf_name} is missing.") from e
        except Exception as e:
            logger.exception(f"Unexpected error during VCF processing: {str(e)}")
            raise

    def _validate_index_exists(self, vcf_path: Path, tbi_name: str) -> None:
        """Ensure the .tbi exists alongside the VCF.

        Args:
            vcf_path: Path to the VCF file.
            tbi_name: Name of the tbi index file.

        Raises:
            DataResolutionError: If the index file is missing.
        """
        tbi_path = vcf_path.parent / tbi_name
        if not tbi_path.exists():
            # In bioinformatics, a missing index is a critical failure.
            raise DataResolutionError(
                f"VCF index missing at {tbi_path}. Tabix lookup will fail."
            )
