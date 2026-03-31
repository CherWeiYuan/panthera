"""Query Fasta

This module contains the code to send a user-supplied FASTA file for splice site
probability prediction and return a WIG file for IGV visualization.
"""

import logging
from pathlib import Path
from typing import Literal

from panthera.core.bio.wig import prepare_wig_dataframe, write_wig
from panthera.core.bio.parse_genome import GenomeParser
from panthera.core.ssp.ssp_manager import SSPManager

logger = logging.getLogger(__name__)

# Prediction Constants
MAX_CACHE_SIZE = 1000
BATCH_SIZE = 1  # Expect low-throughput prediction

# Track Configuration Constants
TRACK_COLOR = "204,85,0"
ALT_COLOR = "0,127,255"


def run_query_fasta(
    fasta_file: str,
    model_name: Literal["modelp", "spliceai"],
    outdir: str,
    prefix: str,
) -> None:
    """Run the query fasta pipeline and writes a WIG file for IGV visualization.

    Args:
        fasta_file: Path to the fasta file.
        model_name: Name of the model to use.
        outdir: Directory to save the output files.
        prefix: Prefix for the output files.

    Returns:
        None

    Raises:
        ValueError: If any of the input arguments are invalid.
    """
    # Input validation
    if not fasta_file:
        raise ValueError("Fasta file is required.")
    if not model_name:
        raise ValueError("Model name is required.")
    if not outdir:
        raise ValueError("Output directory is required.")
    if not prefix:
        raise ValueError("Prefix is required.")

    # Load fasta
    genome_parser = GenomeParser()
    fasta_dict = genome_parser.parse_genome(fasta_file)

    # Load splice site probability prediction manager
    ssp_manager = SSPManager(
        model_name=model_name,
        batch_size=BATCH_SIZE,
        max_cache_size=MAX_CACHE_SIZE,
    )

    for name, seq in fasta_dict.items():
        predict_result = ssp_manager.predict_ssp(
            seqs=[seq],
            reverse_output=False,  # run_query_fasta assumes input seq is on plus strand
        )
        acc, dnr = predict_result[0][0], predict_result[1][0]

        try:
            wig_df = prepare_wig_dataframe(start=0, acceptor_prob=acc, donor_prob=dnr)
            # Pre-format the headers
            header = (
                f'track type=wiggle_0 name="{name}" '
                f'description="Probability" color={TRACK_COLOR} altColor={ALT_COLOR}\n'
                f"variableStep chrom={name} span=1\n"
            )

            # Write wig
            write_wig(df=wig_df, header=header, prefix=prefix, outdir=outdir)
            logger.info(
                f"Successfully wrote WIG track to {str(Path(outdir) / f'{prefix}.wig')}"
            )
        except ValueError as e:
            logger.error(f"Error generating WIG file for {name}: {e}")
