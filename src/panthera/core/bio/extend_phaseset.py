from typing import cast

import pandas as pd
from pandera.typing import DataFrame

from panthera.core.bio.io import VariantSchema


def extend_phaseset(
    vdf: DataFrame[VariantSchema], chrom: str, ps_id: str, ext_len: int
) -> DataFrame[VariantSchema]:
    """Extends a given phase set by incorporating immediately adjacent
    homozygous variants.

    This function searches for variants surrounding a specified phase set
    within a given extension length. It iteratively adds contiguous homozygous
    variants ('1|1' or '1/1') to the phase set boundary.

    The extension stops in a given direction at the first
    non-homozygous variant or when the extension length limit is reached.

    Args:
        vdf: Variant dataframe containing all variants.
        chrom: Target chromosome (e.g., "chr22").
        ps_id: The phase set ID tag.
        ext_len: The extension length (in base pairs) to consider for including
            homozygous variants into the phase set.

    Returns:
        DataFrame[VariantSchema]: The extended phase set variants dataframe.
            If an extension occurred, the `phase_set` column is updated to
            `{ps_id}EXT`.
    """
    # 1. Isolate the core phase set using boolean masks
    core_mask = (vdf["chrom"] == chrom) & (vdf["phase_set"] == ps_id)
    # Cast to pd.DataFrame because pandas-stubs thinks vdf[mask] might be a Series
    ps_df = cast(pd.DataFrame, vdf[core_mask]).copy()

    # Guard clause: If the phase set doesn't exist, return empty DataFrame early
    if ps_df.empty:
        return cast(DataFrame[VariantSchema], ps_df)

    start_pos = ps_df["pos"].min()
    end_pos = ps_df["pos"].max()

    homozygous_gnts = ["1|1", "1/1"]

    def _get_contiguous_homozygous(
        candidates: pd.DataFrame, sort_ascending: bool
    ) -> pd.DataFrame:
        """Filters a DataFrame to include only contiguous homozygous variants.

        Args:
            candidates: DataFrame of candidate variants to check.
            sort_ascending: Direction to sort variants before checking contiguity.

        Returns:
            pd.DataFrame: Filtering results containing only contiguous
                homozygous variants starting from the reference point.
        """
        if candidates.empty:
            return candidates

        # Sort outward from the core phase set
        sorted_cands = candidates.sort_values(by="pos", ascending=sort_ascending)

        # Create boolean mask for homozygous variants
        is_hom = sorted_cands["genotype"].isin(homozygous_gnts)

        # Use cumulative minimum (cummin)
        # cummin() ensures we only keep True values until the first False is
        # encountered
        contiguous_mask = cast(pd.Series, is_hom.cummin())

        return cast(pd.DataFrame, sorted_cands[contiguous_mask])

    # 2. Process front (upstream) extension
    front_mask = (
        (vdf["chrom"] == chrom)
        & (vdf["pos"] >= (start_pos - ext_len))
        & (vdf["pos"] < start_pos)
    )
    fps_df = _get_contiguous_homozygous(
        cast(pd.DataFrame, vdf[front_mask]), sort_ascending=False
    )

    # 3. Process back (downstream) extension
    back_mask = (
        (vdf["chrom"] == chrom)
        & (vdf["pos"] > end_pos)
        & (vdf["pos"] <= (end_pos + ext_len))
    )
    bps_df = _get_contiguous_homozygous(
        cast(pd.DataFrame, vdf[back_mask]), sort_ascending=True
    )

    # 4. Concatenate and finalize
    if fps_df.empty and bps_df.empty:
        res = ps_df.sort_values(by="pos", ignore_index=True)
        return cast(DataFrame[VariantSchema], res)

    # Extensions were found; combine, sort, and re-label
    extended_ps_df = pd.concat([fps_df, ps_df, bps_df], ignore_index=True)
    extended_ps_df = extended_ps_df.sort_values(by="pos", ignore_index=True)
    extended_ps_df["phase_set"] = f"{ps_id}EXT"

    # Final cast to satisfy the function's return type hint
    return cast(DataFrame[VariantSchema], extended_ps_df)
