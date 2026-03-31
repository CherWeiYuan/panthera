"""Generate WIG

This module provides utilities to generate Wiggle (WIG)
files for genomic track visualization.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

# Configure module-level logger
logger = logging.getLogger(__name__)

# Track Configuration Constants
TRACK_COLOR = "204,85,0"
ALT_COLOR = "0,127,255"


class WIGSchema(pa.DataFrameModel):
    pos: Series[int] = pa.Field(ge=1)
    prob: Series[float] = pa.Field(ge=-1.0, le=1.0)

    class Config:
        strict = True
        coerce = True


def prepare_wig_dataframe(
    start: int,
    acceptor_prob: npt.NDArray[np.float32],
    donor_prob: npt.NDArray[np.float32],
) -> DataFrame[WIGSchema]:
    """Vectorized preparation of the positional probability dataframe.

    Args:
        start: 1-based start position on the chromosome.
        acceptor_prob: Array of acceptor probabilities.
        donor_prob: Array of donor probabilities.

    Returns:
        pd.DataFrame: Sorted, filtered dataframe containing non-zero
                      probabilities.

    Raises:
        ValueError: If there is a collision between acceptor and donor
                    probability at the same position (pos).
    """
    # 1. Collision Check (Numpy is faster than Pandas here)
    overlap = (acceptor_prob != 0) & (donor_prob != 0)
    if np.any(overlap):
        collided_indices = np.where(overlap)[0] + start
        error_msg = f"Collision at positions: {collided_indices.tolist()}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 2. Extract non-zero indices/values efficiently
    acc_mask = acceptor_prob != 0
    dnr_mask = donor_prob != 0

    # Calculate positions only for non-zero entries
    acc_pos = np.where(acc_mask)[0] + start
    dnr_pos = np.where(dnr_mask)[0] + start

    # Combine
    pos = np.concatenate([acc_pos, dnr_pos])

    # Donors are negative values in this specific module logic
    probs = np.concatenate([acceptor_prob[acc_mask], donor_prob[dnr_mask] * -1])

    # Round probabilities to 5 decimal places
    probs = np.round(probs, 5)

    # 3. Create and Sort
    df = pd.DataFrame({"pos": pos, "prob": probs}).sort_values("pos")

    # 4. Validate
    return WIGSchema.validate(df)


def write_wig(df: DataFrame[WIGSchema], header: str, prefix: str, outdir: str) -> None:
    """Write the WIG file.

    Args:
        df: Dataframe containing the WIG data.
        header: Header for the WIG file.
        outdir: Output directory.

    Returns:
        WIG file written to outdir.
    """
    # Create output directory
    outdir = str(Path(outdir))
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Create file path
    file_path = f"{outdir}/{prefix}.wig"
    with open(file_path, "w") as f:
        f.write(header)
        # pandas can write directly to an open file handle, avoiding
        # reopening the file
        df.to_csv(f, sep="\t", header=False, index=False)

    logger.debug(f"Successfully wrote WIG track to {file_path}")


def generate_wig(
    gene_name: str,
    background_id: str,
    haplotype_id: str,
    chrom: str,
    start: int,
    outdir: Union[str, Path],
    wt_acc: npt.NDArray[np.float32],
    wt_dnr: npt.NDArray[np.float32],
    mt_acc: npt.NDArray[np.float32],
    mt_dnr: npt.NDArray[np.float32],
    block_id: str = "",
    block_type: str = "HAPLOTYPE",
) -> None:
    """Generates variableStep WIG files for Wild Type (WT) and Mutant (MT)
    splice site probabilities.

    Args:
        gene_name: Name of the target gene.
        background_id: Background strain or individual identifier.
        haplotype_id: Haplotype identifier.
        chrom: Chromosome name (e.g., 'chr1').
        start: 1-based start position for the track.
        outdir: Base output directory.
        wt_acc: Wild Type acceptor probabilities.
        wt_dnr: Wild Type donor probabilities.
        mt_acc: Mutant acceptor probabilities.
        mt_dnr: Mutant donor probabilities.
        block_id: Unique block identifier.
        block_type: Type of block (HAPLOTYPE or SINGLE_VARIANT).

    Raises:
        OSError: If there are permission/creation issues with the output
                 directory.
    """
    # 1. Robust Path Management
    base_out_path = str(Path(outdir) / "wig" / gene_name / background_id)

    try:
        Path(base_out_path).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {base_out_path}: {e}")
        raise

    # 2. Map mutation types to their arrays
    mutations = {"WT": (wt_acc, wt_dnr), "MT": (mt_acc, mt_dnr)}

    # 3. Process each mutation type
    for mut_type, (acc_prob, dnr_prob) in mutations.items():
        try:
            # Generate the cleaned, sorted dataframe
            chrom_df = prepare_wig_dataframe(start, acc_prob, dnr_prob)
            chrom_df["prob"] = chrom_df["prob"].round(5)

            if chrom_df.empty:
                logger.info(f"No non-zero probabilities for {gene_name} ({mut_type}).")

            # Construct final file path (include block_id for uniqueness)
            # .wig suffix will be added by write_wig
            filename = (
                f"{gene_name}.{background_id}.{haplotype_id}."
                + f"{block_type}.{block_id}.{mut_type}"
            )

            # Pre-format the headers
            header = (
                f'track type=wiggle_0 name="{gene_name} {background_id} {mut_type} {haplotype_id}" '
                f'description="Probability" color={TRACK_COLOR} altColor={ALT_COLOR}\n'
                f"variableStep chrom={chrom} span=1\n"
            )

            # 4. Single-pass File I/O
            write_wig(df=chrom_df, header=header, prefix=filename, outdir=base_out_path)

        except Exception as e:
            logger.error(
                f"Failed processing WIG track for {gene_name} ({mut_type}): {e}"
            )
            raise
