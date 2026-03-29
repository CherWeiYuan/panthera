"""
Query Genome

This module contains the code to allow a user-supplied genomic range
(e.g., "chr1-1000-2000-strand" for splice site probability prediction and
return a WIG file for IGV visualization.)
"""

import logging
from pathlib import Path
from typing import Literal

from panthera.core.bio.wig import prepare_wig_dataframe, write_wig
from panthera.core.bio.genome import GenomeParser
from panthera.core.ssp.ssp_manager import SSPManager

logger = logging.getLogger(__name__)

# Prediction Constants
MAX_CACHE_SIZE=1000
BATCH_SIZE=1 # Expect low-throughput prediction

# Track Configuration Constants
TRACK_COLOR = "204,85,0"
ALT_COLOR = "0,127,255"

def run_query_fasta(
    fasta_file: str,
    genomic_range: str,
    model_name: Literal["modelp", "spliceai"],
    outdir: str,
    prefix: str,
) -> None:
    """
    Run the query fasta pipeline.
    """
    # Input validation
    if not fasta_file:
        raise ValueError("Fasta file is required.")
    if not genomic_range:
        raise ValueError("Genomic range is required.")
    if not model_name:
        raise ValueError("Model name is required.")
    if not outdir:
        raise ValueError("Output directory is required.")
    if not prefix:
        raise ValueError("Prefix is required.")

    # Extract genomic range values
    primary_parts = genomic_range.split(":")
    if len(primary_parts) < 2:  # noqa: PLR2004
        raise ValueError(
            f"Genomic range must have at least 2 colon-separated fields "
            f"(chrom:start-end-strand), got: {genomic_range!r}"
        )
    
    secondary_parts = primary_parts[1].split("-")
    if len(secondary_parts) < 3:  # noqa: PLR2004
        raise ValueError(
            f"Genomic range must have at least 3 dash-separated fields "
            f"(start-end-strand), got: {genomic_range!r}"
        )

    chrom = primary_parts[0]
    try:
        start = int(secondary_parts[0])
        end = int(secondary_parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Position field in variant_target is not a valid integer: "
            f"start: {secondary_parts[0]!r}, end: {secondary_parts[1]!r} "
            f"(full string: {genomic_range!r})"
        ) from exc
    strand = secondary_parts[2]

    # Load fasta
    genome_parser = GenomeParser()
    fasta_dict = genome_parser.load_genome(
        fasta_file,
        chrom=chrom,
    )
    
    # Get sequence
    seq = fasta_dict[chrom]

    if strand == "+":
        seq = seq[start:end]
    elif strand == "-":
        seq = seq[start:end].reverse_complement()
    else:
        raise ValueError(f"Invalid strand: {strand}")

    # Predict
    ssp_manager = SSPManager(
        model_name=model_name,
        batch_size=BATCH_SIZE,
        max_cache_size=MAX_CACHE_SIZE,
    )

    acc, dnr = ssp_manager.predict_ssp(seq)[0]
    wig_df = prepare_wig_dataframe(start=0, acceptor_prob=acc, donor_prob=dnr)
    # Pre-format the headers
    header = (
        f'track type=wiggle_0 name="{name}" '
            f'description="Probability" color={TRACK_COLOR} altColor={ALT_COLOR}\n'
            f"variableStep chrom={name} span=1\n"
        )

    # Write wig
    write_wig(df=wig_df, header=header, prefix=prefix, outdir=outdir)
    logger.debug(f"Successfully wrote WIG track to {str(Path(outdir) / f'{prefix}.wig')}")

