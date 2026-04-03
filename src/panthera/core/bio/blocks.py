"""Haplotype Blocks.

This module contains the HaplotypeBlock class for representing a haplotype
block and associated methods.
"""

import logging
from typing import Any, cast, Final, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from panthera.core.bio.mutation import (
    insertion_mutation,
    deletion_mutation,
    snp_mutation,
    substitute_mutation,
)

from panthera.core.bio.gene import GeneObject

from panthera.utils.exceptions import (
    AmbiguousDeletionError,
    BackgroundConflictError,
    NonUniqueChromError,
    NonUniquePhaseSetTagError,
    UnexpectedMutationError,
)

# Set up module-level logging
logger = logging.getLogger(__name__)

# Constants to avoid magic numbers
# Final ensures that value will not be changed or overridden
TARGET_VARIANTS: Final = 0
BACKGROUND_VARIANTS: Final = 1


class VariantSchema(pa.DataFrameModel):
    """Pandera schema for validating the input variants and background DataFrames.

    Attributes:
        chrom: Chromosome name.
        pos: Variant position (1-based).
        ref: Reference allele.
        alt: Alternative allele.
        genotype: Genotype string (e.g. "0/1").
        phase_set: Phase set identifier.
        sample_name: Name of the sample.
        genetic_background: Genetic background label.
        background: Integer flag distinguishing target and background variants.
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
    genotype: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
    phase_set: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
    sample_name: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
    genetic_background: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
    background: Optional[Series[int]] = pa.Field(coerce=True, nullable=True)

    class Config:
        """Configuration for the schema.

        Attributes:
            strict: If False, allows extra columns in the DataFrame.
            coerce: If True, automatically attempts to convert data types.
        """

        strict = False
        coerce = True


class HaplotypeBlock:
    """Represents a contiguous block of variants on the same cis-chromosome.

    Note:
        The genotype column is ignored as all variants in the dataframe are
        considered to be in cis.

    Attributes:
        vdf: DataFrame containing the variants.
        wt_seq: Wild-type sequence including background variants.
        mt_seq: Mutant sequence including target and background variants.
        block_id: Unique identifier for the block.
        block_type: Classification of the block (e.g., "HAPLOTYPE").
        wt_acc: Wild-type acceptor site splice probabilities.
        wt_dnr: Wild-type donor site splice probabilities.
        mt_acc: Mutant acceptor site splice probabilities.
        mt_dnr: Mutant donor site splice probabilities.
        bdf: DataFrame containing the background variants.
        population: Population identifier (e.g., "EAS").
        background_id: Background sample identifier.
        haplotype_id: Haplotype designation ('A' or 'B').
        gene_obj: Reference gene object.
        chrom: Chromosome name.
        phaseset_tag: Phase set tag.
        max_start: Start boundary for sequence extraction.
        min_end: End boundary for sequence extraction.
    """

    vdf: DataFrame[VariantSchema]
    wt_seq: str
    mt_seq: str
    block_id: int | str
    block_type: Literal["HAPLOTYPE", "SINGLE_VARIANT", "UNK"]
    wt_acc: npt.NDArray[np.float32]
    wt_dnr: npt.NDArray[np.float32]
    mt_acc: npt.NDArray[np.float32]
    mt_dnr: npt.NDArray[np.float32]

    bdf: DataFrame[VariantSchema]
    population: str
    background_id: str
    haplotype_id: str
    gene_obj: GeneObject
    chrom: Optional[str]
    phaseset_tag: Optional[str]
    max_start: int
    min_end: int

    def __init__(
        self,
        variants_df: DataFrame[VariantSchema],
        gene_obj: GeneObject,
        context_dist: int,
    ):
        """Initializes a HaplotypeBlock.

        Args:
            variants_df: DataFrame containing target variants.
            gene_obj: Gene object defining the genomic context.
        """
        # Initialize self variables
        self.vdf = cast(
            DataFrame[VariantSchema], variants_df.assign(background=TARGET_VARIANTS)
        )
        self.wt_seq = ""
        self.mt_seq = ""
        self.block_id = 0
        self.block_type = "UNK"
        self.wt_acc = cast(npt.NDArray[np.float32], np.array([], dtype=np.float32))
        self.wt_dnr = cast(npt.NDArray[np.float32], np.array([], dtype=np.float32))
        self.mt_acc = cast(npt.NDArray[np.float32], np.array([], dtype=np.float32))
        self.mt_dnr = cast(npt.NDArray[np.float32], np.array([], dtype=np.float32))

        self.bdf = cast(DataFrame[VariantSchema], None)
        self.population = ""
        self.background_id = ""
        self.haplotype_id = ""

        # Extract chromosome
        chroms = variants_df["chrom"].unique()
        if len(chroms) == 1:
            self.chrom = str(chroms[0])
        elif len(chroms) == 0:
            self.chrom = None  # Allow empty blocks
        else:
            raise NonUniqueChromError(f"Expected one chrom. Got: {chroms}")

        # Extract phase set (PS) tag
        ps_tags = variants_df["phase_set"].unique()
        if len(ps_tags) == 1:
            self.phaseset_tag = str(ps_tags[0])
        elif len(ps_tags) == 0:
            self.phaseset_tag = None  # Allow empty blocks
        else:
            raise NonUniquePhaseSetTagError(f"Expected one PS tag. Got: {ps_tags}")

        # Define acceptable genomic range using gene object
        gene_start = gene_obj.start
        gene_end = gene_obj.end

        # Use cast(Any, ...) to bypass Pyright's confusion with Pandera/Pandas min/max
        v_min = cast(Any, self.vdf["pos"].min())
        v_max = cast(Any, self.vdf["pos"].max())

        if context_dist:
            self.max_start = int(
                max(
                    int(gene_start),
                    int(v_min) - context_dist // 2 if pd.notna(v_min) else np.nan,
                )
            )
            self.min_end = int(
                min(
                    int(gene_end),
                    int(v_max) + context_dist // 2 if pd.notna(v_max) else np.nan,
                )
            )
        else:
            self.max_start = int(
                max(int(gene_start), int(v_min) if pd.notna(v_min) else np.nan)
            )
            self.min_end = int(
                min(int(gene_end), int(v_max) if pd.notna(v_max) else np.nan)
            )

        self.vdf = cast(
            DataFrame[VariantSchema],
            self.vdf[
                (self.vdf["pos"] >= self.max_start) & (self.vdf["pos"] <= self.min_end)
            ],
        )

        # Update gene information
        self.gene_obj = gene_obj

    @property
    def name(self) -> str:
        """Generates a unique identifier for the variant combination.

        The identifier is a dot-separated string of hyphenated variant strings
        (chrom-pos-ref-alt), sorted by genomic position.

        Returns:
            str: The unique name of the haplotype block.
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
        population: str,
        background_id: str,
        haplotype_id: Literal["A", "B"],
        resolve_conflicts: bool,
    ) -> None:
        """Adds background variants to the haplotype block.

        Args:
            background_df: DataFrame containing background variants.
            population: Population name (e.g. "EAS").
            background_id: Background sample identifier (e.g. "HG00512").
            haplotype_id: Haplotype designation ('A' or 'B').
            resolve_conflicts: If True, resolves coordinate conflicts by dropping
                background variants. If False, raises BackgroundConflictError.

        Raises:
            BackgroundConflictError: If resolve_conflicts is False and
                conflicts exist between target and background variants.
        """
        self.bdf = cast(
            DataFrame[VariantSchema],
            background_df.assign(background=BACKGROUND_VARIANTS),
        )
        self.population = population
        self.background_id = background_id
        self.haplotype_id = haplotype_id

        # Merge variants and background variants dataframe
        self.vdf = cast(
            DataFrame[VariantSchema], pd.concat([self.vdf, self.bdf], axis=0)
        )

        # Resolve conflicts in the merged dataframe
        self._check_variant_conflicts(resolve_conflicts)

        # Check for ambiguous deletions
        self._check_deletion_validity()

    def _check_variant_conflicts(self, resolve_conflicts: bool) -> None:
        """Identifies and handles overlapping target and background variants.

        Checks if background variants share genomic coordinates with target
        variants. Deletions are checked across their entire span.

        Args:
            resolve_conflicts: If True, silently drops conflicting background
                variants. If False, raises BackgroundConflictError.

        Raises:
            BackgroundConflictError: If conflicts exist and resolve_conflicts
                is False.
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

        # Handle Conflicts based on `resolve_conflicts` flag
        if to_remove_indices:
            if not resolve_conflicts:
                raise BackgroundConflictError(
                    f"Found {len(to_remove_indices)} conflicting background "
                    "variant(s). Set resolve_conflicts=True to automatically "
                    "remove them."
                )
            else:
                logger.info(
                    f"Resolving conflicts: Dropping {len(to_remove_indices)} "
                    "background variant(s)."
                )
                # Critical: Removing rows via index must occur before resetting index
                self.vdf.drop(index=to_remove_indices, inplace=True)
        else:
            logger.debug("No conflicts found. Haplotype block is clean.")

        # Clean up the temporary calculation column
        # Critical: vdf.reset_index must occur after vdf.drop by index
        self.vdf.drop(columns=["end_pos"], inplace=True)
        self.vdf.reset_index(drop=True, inplace=True)

    def extract_seqs(
        self,
        chrom_seq: str,
        extension_len: int,
    ) -> tuple[str, str]:
        """Generates the wild-type and mutant sequences for the block.

        The wild-type sequence includes background variants. The mutant sequence
        includes both target and background variants.

        Args:
            chrom_seq: The full chromosome sequence.
            extension_len: Length of flanking sequence to include on each side.

        Returns:
            tuple[str, str]: A tuple containing (wt_seq, mt_seq).
        """
        if self.vdf.empty:
            return "", ""

        # Determine the exact genomic interval needed
        min_pos = int(cast(Any, self.vdf["pos"].min()))
        max_pos = int(cast(Any, self.vdf["pos"].max()))

        start_bound = max(1, min_pos - extension_len)
        end_bound = max_pos + extension_len

        # Slice the chromosome once
        base_seq = chrom_seq[start_bound - 1 : end_bound]

        # Create a local copy of vdf with relative coordinates
        # This prevents the double-subtraction coordinate bug in the mutation
        # passes
        local_vdf = cast(DataFrame[VariantSchema], self.vdf.copy())
        local_vdf["pos"] = local_vdf["pos"] - start_bound + 1

        # 4. Get wild-type sequence (WT)
        # Use '}' and '{' for insertion and deletion character placeholder
        wt_seq, mt_vdf = self._modify_seq(
            vdf=local_vdf, seq=base_seq, in_char="}", del_char="{", mutation_class="WT"
        )

        # 5. Get mutant sequence (MT) from WT sequence
        # Use '>' and '<' for insertion and deletion character placeholder
        mt_seq, _ = self._modify_seq(
            vdf=mt_vdf, seq=wt_seq, in_char=">", del_char="<", mutation_class="MT"
        )

        # Update self variables
        self.wt_seq = wt_seq
        self.mt_seq = mt_seq

        return wt_seq, mt_seq

    def _check_deletion_validity(self) -> None:
        """Ensures that deletions do not overlap with other mutations.

        Raises:
            AmbiguousDeletionError: If a deletion covers coordinates where
                another mutation is specified.
        """
        # Calculate deletion length (ref - alt)
        # In VCFs, a 1-base deletion (e.g., AG -> A) has a deletion_len of 1
        deletion_len = self.vdf["ref"].str.len() - self.vdf["alt"].str.len()

        # Get the next position
        next_pos = self.vdf["pos"].shift(-1)

        # Check for overlap
        # A deletion at 'pos' of length 'L' affects coordinates from pos + 1
        # up to pos + L. We raise an error if the next mutation starts inside
        # that deleted range.
        is_ambiguous = (
            (deletion_len >= 1)
            & (next_pos > self.vdf["pos"])
            & (next_pos <= self.vdf["pos"] + deletion_len)
        )

        if is_ambiguous.any():
            raise AmbiguousDeletionError()

    def _modify_seq(
        self,
        vdf: DataFrame[VariantSchema],
        seq: str,
        in_char: str,
        del_char: str,
        mutation_class: Literal["WT", "MT"],
    ) -> tuple[str, DataFrame[VariantSchema]]:
        """Applies mutations from a DataFrame to a sequence.

        Args:
            vdf: DataFrame containing the variants to apply.
            seq: Reference DNA sequence.
            in_char: Placeholder character for insertions.
            del_char: Placeholder character for deletions.
            mutation_class: Type of mutation pass ("WT" or "MT").

        Returns:
            tuple[str, DataFrame[VariantSchema]]: A tuple containing the
                mutated sequence and a DataFrame with updated coordinates
                for subsequent passes.
        """
        # Check input validity
        if vdf.empty:
            empty_df = pd.DataFrame(columns=vdf.columns)
            return seq, cast(DataFrame[VariantSchema], empty_df)

        # Check deletion validity
        # Raise error if deletion removes position
        # where other mutations are found
        self._check_deletion_validity()

        # Initialize shift variable to track genomic coordinate shifting
        # created by insertion mutation
        shift: int = 0

        # Loop through the sequence and modify using mutation functions
        mt_vdf_records = []

        for row in vdf.to_dict(orient="records"):
            # row: dict[str, Any]
            pos: int = int(cast(Any, row["pos"]))
            ref: str = str(row["ref"])
            alt: str = str(row["alt"])
            bg: int = int(cast(Any, row["background"]))

            # Adjust current position by amount of
            # shift done by the previous iteration
            pos += shift

            # If mutation_class is "WT", create vdf entry with new coordinates
            # for MT seq for the next modify_seq(mut_type == "MT") call
            if mutation_class == "WT":
                # If current variant is target (non-background), do not
                # mutate seq
                # Add it to the mt_vdf seed with its newly shifted coordinate
                if bg == TARGET_VARIANTS:
                    # Convert the row to a dict safely to
                    # preserve ALL columns
                    row_dict = row.copy()
                    row_dict["pos"] = pos
                    mt_vdf_records.append(row_dict)
                    continue
                else:
                    pass  # Proceed to mutate background variants

            elif mutation_class == "MT":
                # Proceed to mutate all variants (which are target variants)
                pass
            else:
                raise ValueError(
                    "Expected mutation class 'WT' or 'MT'. " + f"Got {mutation_class}."
                )

            # --- Mutation Functions ---

            # Substitution
            # Substitute ref for alt when ref and alt alleles are more than 1 bp
            # No shift for equal-length substitution or deletions
            # Shift if alt is longer than ref.
            if len(ref) > 1 and len(alt) > 1:
                seq = substitute_mutation(
                    seq=seq,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    in_symbol=in_char,
                    del_symbol=del_char,
                )
                if len(ref) < len(alt):
                    # Shift for nucleotides and placeholder characters added
                    shift += 2 * (len(alt) - len(ref))
                else:
                    # No shift for substitutions and deletions
                    pass

            # SNP
            # No change in shift
            elif len(ref) == len(alt):
                seq = snp_mutation(seq=seq, pos=pos, ref=ref, alt=alt)

            # Insertion
            elif len(ref) < len(alt):
                seq = insertion_mutation(
                    seq=seq, pos=pos, ref=ref, alt=alt, in_symbol=in_char
                )
                # Shift for nucleotides and placeholder characters added
                shift += 2 * (len(alt) - 1)

            # Deletion
            elif len(ref) > len(alt):
                # No change in shift as each deleted nucleotide is replaced by
                # a placeholder character
                seq = deletion_mutation(
                    seq=seq, pos=pos, ref=ref, alt=alt, del_symbol=del_char
                )

            else:
                raise UnexpectedMutationError("Unexpected mutation type.")

        # Convert the records cleanly back to a dataframe,
        # maintaining proper columns
        # If the list is empty, it returns an empty dataframe
        # with the correct columns
        mt_vdf = pd.DataFrame(mt_vdf_records, columns=vdf.columns)
        mt_vdf = cast(DataFrame[VariantSchema], mt_vdf)

        return seq, mt_vdf
