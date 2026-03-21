"""
Split variant dataframes

This module contain the functions to split a variant dataframe
by its phased haplotype.
"""

import logging
import pandas as pd
from pandera.typing import DataFrame
from typing import cast, Tuple

from panthera.core.input import VariantSchema

# Set up module-level logging
logger = logging.getLogger(__name__)


def split_by_haplotype(
    vdf: DataFrame[VariantSchema],
) -> Tuple[DataFrame[VariantSchema], DataFrame[VariantSchema]]:
    """
    Splits a VCF DataFrame into two DataFrames based on phased haplotypes.

    Haplotype A includes phased genotypes '1|0' and '1|1'.
    Haplotype B includes phased genotypes '0|1' and '1|1'.
    Unphased variants (e.g., '0/1'), homozygous reference ('0|0'),
    and other unexpected formats are ignored.

    Args:
        vdf (pd.DataFrame): The input VCF data. Must contain a 'genotype' column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing (haplotype_a_df, haplotype_b_df).
    """
    # Input Validation
    if not isinstance(vdf, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(vdf).__name__}")

    if "genotype" not in vdf.columns:
        logger.error("Failed to split haplotypes: 'genotype' column missing.")
        raise KeyError("The input DataFrame must contain a 'genotype' column.")

    # Vectorized Filtering
    # By strictly matching the strings, we intrinsically filter out unphased ('/'),
    # hom-ref ('0|0'), and missing data without needing regex or slow string splitting.
    mask_a = vdf["genotype"].isin(["1|0", "1|1"])
    mask_b = vdf["genotype"].isin(["0|1", "1|1"])

    # DataFrame Creation
    # .copy() is used to return independent DataFrames and prevent
    # 'SettingWithCopyWarning' if the user modifies the output later.
    df_haplotype_a = vdf[mask_a].copy()
    df_haplotype_b = vdf[mask_b].copy()

    # Return type casted dataframe to ensure vcf[mask] did not return a series
    # instead of a dataframe
    return (
        cast(DataFrame[VariantSchema], df_haplotype_a), 
        cast(DataFrame[VariantSchema], df_haplotype_b)
    )
