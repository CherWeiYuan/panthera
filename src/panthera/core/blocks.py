import logging
from typing import Final, Optional

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from panthera.utils.exceptions import (
    BackgroundConflictError,
    NonUniqueChromError,
    NonUniquePhaseSetTagError,
)

# Set up module-level logging
logger = logging.getLogger(__name__)

# Constants to avoid magic numbers
# Final ensures that value will not be changed or overridden
TARGET_VARIANTS: Final = 0
BACKGROUND_VARIANTS: Final = 1


class VariantSchema(pa.DataFrameModel):
    """
    Pandera schema for validating the input variants and background DataFrames.
    Ensures that downstream vectorized operations (like string lengths and
    genomic interval math) do not fail due to bad data types.
    """

    # Coerce=True forces Pandas to convert the column to the correct type
    # if possible
    chrom: Series[str] = pa.Field(coerce=True)

    # Genomic positions are typically 1-based and strictly positive due to ge=1
    pos: Series[int] = pa.Field(coerce=True, ge=1)

    # Must be strings so `ref_len = self.vdf['ref'].str.len()` doesn't crash
    ref: Series[str] = pa.Field(coerce=True)
    alt: Series[str] = pa.Field(coerce=True)

    # Might not be strictly required so we make them Optional/nullable
    genotype: Optional[Series[str]] = pa.Field(nullable=True)
    phase_set: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)

    class Config:
        """
        Configuration for the schema.
        strict = False allows the DataFrame to contain extra columns
        (like read depth, quality scores, etc.) without throwing an error.
        """

        strict = False
        coerce = True


class HaplotypeBlock:
    """
    Class for a haplotype block

    A haplotype block is a contiguous block of variants on the same
    cis-chromosome.

    Critically, genotype column in dataframe does not matter here and is ignored
    as all variants in the dataframe are considered contiguous.
    """

    def __init__(self, variants_df: DataFrame[VariantSchema]):
        """
        Args:
            variants_df: Pandas dataframe containing the variants, genotype,
                         background and phase set (PS) tags
        """
        # Initialize self variables
        self.vdf = variants_df.assign(background=TARGET_VARIANTS)

        # Extract chromosome
        chroms = variants_df.chrom.unique()
        if len(chroms) == 1:
            self.chrom = chroms[0]
        elif len(chroms) == 0:
            self.chrom = None  # Allow empty blocks
        else:
            raise NonUniqueChromError(f"Expected one chrom. Got: {chroms}")

        # Extract phase set (PS) tag
        ps_tags = variants_df.phase_set.unique()
        if len(ps_tags) == 1:
            self.phaseset_tag = ps_tags[0]
        elif len(ps_tags) == 0:
            self.phaseset_tag = None  # Allow empty blocks
        else:
            raise NonUniquePhaseSetTagError(f"Expected one PS tag. Got: {ps_tags}")

    @property
    def name(self) -> str:
        """
        Computes the name dynamically (@property).
        Ensures 100% sync even after pandas operations.

        Generates a unique identifier for the variant combination using
        vectorized operations.

        The format is a dot-separated string of hyphenated variants:
        'chr-pos-ref-alt.chr-pos-ref-alt'.

        Note:
        Sorts the internal DataFrame by genomic coordinates to ensure
        deterministic naming regardless of input row order.
        """
        if self.vdf.empty:
            return ""

        # Sort the DataFrame
        # Using a copy or being explicit about sorting is safer
        sorted_df = self.vdf.sort_values(
            by=["chrom", "pos", "ref", "alt"], ascending=True
        )

        # Vectorized string construction
        # We use .astype(str) to ensure compatibility and 'str.cat' for
        # high-performance joining
        variant_ids = (
            sorted_df["chrom"].astype(str)
            + "-"
            + sorted_df["pos"].astype(str)
            + "-"
            + sorted_df["ref"].astype(str)
            + "-"
            + sorted_df["alt"].astype(str)
        )

        # Join all variant strings with a dot
        return ".".join(variant_ids)

    def add_background_variants(
        self,
        background_df: DataFrame[VariantSchema],
        population: str,  # e.g. "EAS"
        background_id: str,  # e.g. "NA19238"
        haplotype_id: str,  # 'A'/ 'B'
        mutation_status: str,  # "WT"/ "MT"
        resolve_conflicts: bool = False,  # True/ False
    ) -> None:
        """
        Args:
            population: Population name (e.g. "EAS" for East Asian).
            background_id: Background identity (e.g. "HG00512").
            haplotype_id: Haplotype of background (either 'A' or 'B').
            mutation_status: Either wild-type (WT) or mutant (MT).
            resolve_conflicts: Conflicts between variants in variants dataframe
                (self.vdf) and background dataframe (self.bdf) may occur due
                to sharing of the same genomic coordinates.
                If True, conflict will be resolved by removing background
                variant that share the same location as the target variant.
                If False, conflicts will raise BackgroundConflictError.
        """
        self.bdf = background_df.assign(background=BACKGROUND_VARIANTS)
        self.population = population
        self.background_id = background_id
        self.haplotype_id = haplotype_id
        self.mutation_status = mutation_status

        # Merge variants and background variants dataframe
        self.vdf = pd.concat([self.vdf, self.bdf], axis=0)

        # Resolve conflicts in the merged dataframe
        self._check_variant_conflicts(resolve_conflicts)

    def _check_variant_conflicts(self, resolve_conflicts: bool) -> None:
        """
        Checks if background variants (from non-reference genome) has
        overlapping genomic coordinates/ positions with variants (from
        variants dataframe).

        If conflicting positions are identified, raise BackgroundConflictError
        to prevent further processing if resolve == True. Otherwise, remove
        the background variant in conflict without raising error.

        There are three types of variants: SNP, INsertion and DELetion
        For SNP and INsertion, check if there are overlapping variants at
        the same genomic coordinate/ position.
        For DELetion, check the same genomic coordiate AND the coordinates of
        len(ref) - len(alt) ahead.

        Args:
            resolve_conflicts: If True, silently drops conflicting background variants.
                               If False, raises BackgroundConflictError when conflicts exist.

        Raises:
            BackgroundConflictError: If conflicts exist and resolve_conflicts is False.
        """
        if self.vdf.empty:
            return

        # Validation & Pre-processing
        # We sort self.vdf in place to prepare for the interval logic
        self.vdf["pos"] = self.vdf["pos"].astype(int)
        self.vdf.sort_values(by=["chrom", "pos"], inplace=True, ignore_index=True)

        # Calculate Genomic "Footprints" (Intervals)
        ref_len = self.vdf["ref"].str.len()
        alt_len = self.vdf["alt"].str.len()

        # Calculate the end position of the variant to determine overlap span
        end_pos = self.vdf["pos"] + np.where(
            ref_len > alt_len,  # Check if ref is longer than alt
            ref_len - alt_len,  # If True (DEL mutation), get length difference
            0,  # If False, length difference is 0
        )

        # Temporarily assign to the dataframe for vectorized operations
        self.vdf["end_pos"] = end_pos

        # Separate User Variants from Background
        target_vars = self.vdf[self.vdf["background"] == TARGET_VARIANTS]
        bg_vars = self.vdf[self.vdf["background"] == BACKGROUND_VARIANTS]

        to_remove_indices = []

        # Vectorized Overlap Detection
        if not target_vars.empty and not bg_vars.empty:
            starts = np.asarray(target_vars["pos"])
            ends = np.asarray(target_vars["end_pos"])
            bg_positions = np.asarray(bg_vars["pos"])

            # Find overlaps in O(log N) using searchsorted (binary search)
            # searchsorted treats the genomic positions as a sorted array and
            # finds overlaps in O(log N) time
            # np.searchsorted returns 1-based index so -1 converts it to 0-based
            # side='right' tells NumPy to find the index after the last suitable
            # insertion point
            idx = np.searchsorted(starts, bg_positions, side="right") - 1

            # idx >= 0: safety check. If a background variant appears before
            # the very first user variant on the chromosome, searchsorted
            # would return an index that, after subtracting 1, becomes -1
            # We ignore these.
            # bg_positions <= ends[idx] is the actual conflict check
            # We take our candidate interval and look up its end_pos.
            # If the background position is less than
            # or equal to that end position, we have a confirmed overlap
            mask = (idx >= 0) & (bg_positions <= ends[idx])

            filtered_index = bg_vars.index[mask]

            # bg_vars.index[mask] may return a single scalar value
            # (like an int) or a filtered Index object
            # Explicitly treat it as a Index object
            if isinstance(filtered_index, pd.Index):
                to_remove_indices.extend(filtered_index.tolist())
            else:
                # If it's a single scalar, wrap it in a list
                to_remove_indices.append(filtered_index)

        # Clean up the temporary calculation column
        self.vdf.drop(columns=["end_pos"], inplace=True)
        self.vdf.reset_index(drop=True, inplace=True)

        # Handle Conflicts based on `resolve_conflicts` flag
        if to_remove_indices:
            if not resolve_conflicts:
                raise BackgroundConflictError(
                    f"Found {len(to_remove_indices)} conflicting background variant(s). "
                    "Set resolve_conflicts=True to automatically remove them."
                )
            else:
                logger.info(
                    f"Resolving conflicts: Dropping {len(to_remove_indices)} background variant(s)."
                )
                self.vdf.drop(index=to_remove_indices, inplace=True)
                self.vdf.reset_index(drop=True, inplace=True)
        else:
            logger.debug("No conflicts found. Haplotype block is clean.")
