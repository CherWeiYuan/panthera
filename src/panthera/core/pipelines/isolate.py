"""Haplotype isolate pipeline.

Generates every possible haplotype block that contains a mandatory target
variant combined with one or more non-target variants found in the same
phase set, then wraps each combination in a :class:`HaplotypeBlock`.
"""

from __future__ import annotations

import logging
from itertools import chain, combinations
from typing import Any, Generator

import pandas as pd
from pandera.typing import DataFrame
from typing import cast

from panthera.core.bio.blocks import HaplotypeBlock, VariantSchema
from panthera.core.bio.gene import GeneObject, find_genes_at_pos

__all__ = ["phase1_create_haplotype_combinations"]

logger = logging.getLogger(__name__)

# Column order expected by VariantSchema / HaplotypeBlock.
_VARIANT_COLUMNS: tuple[str, str, str, str, str] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "phase_set",
)

# dtype map applied when materialising combination DataFrames.
_VARIANT_DTYPES: dict[str, type] = {
    "chrom": str,
    "pos": int,
    "ref": str,
    "alt": str,
    "phase_set": str,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_variant_target(variant_target: str) -> tuple[str, int, str, str]:
    """Parses a variant target string into its constituent fields.

    Args:
        variant_target: Variant identifier in "chrom-pos-ref-alt" format.

    Returns:
        tuple[str, int, str, str]: A tuple of (chrom, pos, ref, alt).

    Raises:
        ValueError: If the string is malformed or the position is not an integer.
    """
    if not variant_target:
        raise ValueError("variant_target must be a non-empty string.")

    parts = variant_target.split("-")
    if len(parts) < 4:  # noqa: PLR2004
        raise ValueError(
            f"variant_target must have at least four dash-separated fields "
            f"(chrom-pos-ref-alt), got: {variant_target!r}"
        )

    chrom = parts[0]
    ref = parts[2]
    alt = parts[3]

    try:
        pos = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Position field in variant_target is not a valid integer: "
            f"{parts[1]!r} (full string: {variant_target!r})"
        ) from exc

    logger.debug("Parsed variant target: %s:%d:%s>%s", chrom, pos, ref, alt)
    return chrom, pos, ref, alt


def _find_target_gene(
    vdf: DataFrame[VariantSchema],
    chrom: str,
    gtf_dict: dict[str, list[Any]],
    gene_target: str,
) -> GeneObject:
    """Locates the GeneObject for a target gene within a variant DataFrame.

    Args:
        vdf: Variant DataFrame.
        chrom: Chromosome on which to search.
        gtf_dict: Mapping of chromosomes to lists of gene objects.
        gene_target: Gene symbol to find.

    Returns:
        GeneObject: The matching gene object.

    Raises:
        ValueError: If the gene is not found or input data is invalid.
    """
    if not gene_target:
        raise ValueError("gene_target must be a non-empty string.")
    if not chrom:
        raise ValueError("chrom must be a non-empty string.")
    if vdf.empty:
        raise ValueError("Variant dataframe is empty; cannot search for gene.")
    if len(vdf["chrom"].unique()) > 1:
        raise ValueError(
            "Expect only one chromosome in variant dataframes. "
            f"Got: {len(vdf['chrom'].unique())}"
        )

    # Restrict position search to the relevant chromosome for efficiency.
    chrom_positions: list[int] = vdf.loc[vdf["chrom"] == chrom, "pos"].unique().tolist()
    if not chrom_positions:
        raise ValueError(
            f"No positions found on chromosome {chrom!r} in the variant dataframe."
        )

    # Cache already-resolved gene objects to avoid redundant GTF lookups.
    seen_genes: list[GeneObject] = []
    found_gene: GeneObject | None = None

    for pos in chrom_positions:
        gene_objs: list[GeneObject] = find_genes_at_pos(
            chrom=chrom,
            pos=pos,
            gtf_dict=gtf_dict,
            existing_genes=seen_genes,
        )

        for g_obj in gene_objs:
            # Accumulate so subsequent calls skip already-resolved genes.
            if g_obj not in seen_genes:
                seen_genes.append(g_obj)

            if g_obj.gene_name == gene_target:
                found_gene = g_obj
                break

        if found_gene:
            break

    if found_gene is None:
        raise ValueError(
            f"Target gene {gene_target!r} was not found at any position on "
            f"chromosome {chrom!r} in the phase set."
        )

    logger.info(
        "Resolved target gene: %s (%s)", found_gene.gene_name, found_gene.gene_id
    )
    return found_gene


def _iter_haplotype_combinations(
    target_tuples: list[tuple],
    nontarget_tuples: list[tuple],
    gene_obj: GeneObject,
    context_dist: int,
) -> Generator[HaplotypeBlock, None, None]:
    """Yields HaplotypeBlock objects for every subset of non-target variants.

    Args:
        target_tuples: Rows for the mandatory target variants.
        nontarget_tuples: Rows for other variants in the phase set.
        gene_obj: Gene metadata for the blocks.

    Yields:
        HaplotypeBlock: A HaplotypeBlock for each combination.
    """
    n = len(nontarget_tuples)
    total = 0

    for r in range(1, n + 1):
        for combi in combinations(nontarget_tuples, r):
            # chain avoids allocating a merged list; tuple() forces evaluation
            # once for DataFrame construction.
            records = tuple(chain(target_tuples, combi))
            cdf = cast(
                DataFrame[VariantSchema],
                pd.DataFrame(
                    records,
                    columns=pd.Index(list(_VARIANT_COLUMNS)),
                ).astype(_VARIANT_DTYPES),
            )
            yield HaplotypeBlock(cdf, gene_obj, context_dist)
            total += 1

    logger.info("Generated %d haplotype blocks.", total)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def phase1_create_haplotype_combinations(
    vdf: DataFrame[VariantSchema],
    gtf_dict: dict[str, list[Any]],
    gene_target: str,
    variant_target: str,
    context_dist: int,
) -> list[HaplotypeBlock]:
    """Generates haplotype blocks from all target + non-target variant subsets.

    Args:
        vdf: Variant DataFrame for the phase set.
        gtf_dict: Parsed GTF metadata.
        gene_target: HGNC gene symbol.
        variant_target: Variant identifier in "chrom-pos-ref-alt" format.

    Returns:
        list[HaplotypeBlock]: A list of all generated HaplotypeBlocks.

    Raises:
        ValueError: If the target variant or gene cannot be located.
    """
    # Add phase set to vdf for HaplotypeBlock initialization
    vdf = cast(DataFrame[VariantSchema], vdf.assign(phase_set="PS"))

    # --- 1. Parse target variant to obtain chrom early (needed for gene search) ---
    chrom, pos, ref, alt = _parse_variant_target(variant_target)

    # --- 2. Resolve gene object (chrom now available) ---
    gene_obj = _find_target_gene(
        vdf=vdf,
        chrom=chrom,
        gtf_dict=gtf_dict,
        gene_target=gene_target,
    )

    # --- 3. Partition vdf into target / non-target row sets ---
    is_target: pd.Series = (
        (vdf["chrom"] == chrom)
        & (vdf["pos"] == pos)
        & (vdf["ref"] == ref)
        & (vdf["alt"] == alt)
        & (vdf["phase_set"] == "PS")
    )

    # Filter vdf to only _VARIANT_COLUMNS before creating tuples to avoid column
    # count mismatches if the input vdf has extra columns (e.g. genotype, phase_set).
    vdf_vars = vdf[list(_VARIANT_COLUMNS)]
    target_tuples: list[tuple] = list(
        vdf_vars.loc[is_target].itertuples(index=False, name=None)
    )
    nontarget_tuples: list[tuple] = list(
        vdf_vars.loc[~is_target].itertuples(index=False, name=None)
    )

    if not target_tuples:
        raise ValueError(
            f"Target variant {variant_target!r} was not found in the phase set."
        )
    if not nontarget_tuples:
        raise ValueError(
            "No non-target variants found in the phase set; "
            "at least one is required to form a combination."
        )

    logger.debug(
        "Partitioned phase set: %d target row(s), %d non-target row(s).",
        len(target_tuples),
        len(nontarget_tuples),
    )

    # --- 4. Build and return all haplotype blocks ---
    return list(
        _iter_haplotype_combinations(
            target_tuples=target_tuples,
            nontarget_tuples=nontarget_tuples,
            gene_obj=gene_obj,
            context_dist=context_dist,
        )
    )
