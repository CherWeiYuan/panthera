"""
Generate WIG

This module provides enterprise-grade utilities to generate Wiggle (WIG)
files for genomic track visualization.
"""

import logging
from pathlib import Path
from typing import cast, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

# Configure module-level logger
logger = logging.getLogger(__name__)

# Track Configuration Constants
TRACK_COLOR = "204,85,0"
ALT_COLOR = "0,127,255"


def prepare_wig_dataframe(
    start: int,
    acceptor_prob: npt.NDArray[np.float32],
    donor_prob: npt.NDArray[np.float32],
) -> pd.DataFrame:
    """
    Vectorized preparation of the positional probability dataframe.

    Args:
        start: 1-based start position on the chromosome.
        acceptor_prob: Array of acceptor probabilities.
        donor_prob: Array of donor probabilities.

    Returns:
        pd.DataFrame: Sorted, filtered dataframe containing non-zero probabilities.
    """
    # Use fast numpy arrays for positions instead of Python zip/list/range
    df_acc = pd.DataFrame(
        {"pos": np.arange(start, start + len(acceptor_prob)), "prob": acceptor_prob}
    )

    df_dnr = pd.DataFrame(
        {"pos": np.arange(start, start + len(donor_prob)), "prob": donor_prob}
    )

    # Combine dataframes
    combined_df = pd.concat([df_acc, df_dnr], ignore_index=True)

    # Filter out absolute zero probabilities to save disk space
    combined_df = combined_df[combined_df["prob"] != 0.0]

    # WIG variableStep requires strictly increasing, unique positions.
    # Group duplicates (if an acceptor and donor share the exact same base) and sort.
    # combined_df = combined_df.groupby("pos", as_index=False)["prob"].sum()
    # pyright needs a cast with pandas sometimes because the return types can be complex
    combined_df = cast(
        pd.DataFrame,
        combined_df.groupby("pos", as_index=False)["prob"].sum(),
    )
    combined_df = combined_df.sort_values(by="pos")

    return combined_df


def write_wig(df: pd.DataFrame, header: str, prefix: str, outdir: str) -> None:
    """
    Write the WIG file.

    Args:
        df: Dataframe containing the WIG data.
        header: Header for the WIG file.
        outdir: Output directory.

    Returns:
        WIG file written to outdir.

    Raises:
        OSError: If there are permission/creation issues with the output directory.
    """
    # Create output directory
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create file path
    file_path = outdir / prefix
    with open(file_path, "w") as f:
        f.write(header)
        # pandas can write directly to an open file handle, avoiding reopening the file
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
    """
    Generates variableStep WIG files for Wild Type (WT) and Mutant (MT) splice site probabilities.

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
        OSError: If there are permission/creation issues with the output directory.
    """
    # 1. Robust Path Management
    base_out_path = Path(outdir) / "wig" / gene_name / background_id

    try:
        base_out_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {base_out_path}: {e}")
        raise

    # 2. Map mutation types to their arrays
    mutations = {"WT": (wt_acc, wt_dnr * -1), "MT": (mt_acc, mt_dnr * -1)}

    # 3. Process each mutation type
    for mut_type, (acc_prob, dnr_prob) in mutations.items():
        try:
            # Generate the cleaned, sorted dataframe
            chrom_df = prepare_wig_dataframe(start, acc_prob, dnr_prob)

            if chrom_df.empty:
                logger.info(f"No non-zero probabilities for {gene_name} ({mut_type}).")

            # Construct final file path (include block_id for uniqueness)
            filename = (
                f"{gene_name}.{background_id}.{haplotype_id}."
                + f"{block_type}.{block_id}.{mut_type}.wig"
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
