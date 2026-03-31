"""Query Genome

This module contains the code to allow a user-supplied genomic range
(e.g., "chr1-1000-2000-strand" for splice site probability prediction and
return a WIG file for IGV visualization.)
"""

import logging
from pathlib import Path
from typing import Literal

from Bio.Seq import Seq

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


def run_query_genomic_range(
    fasta_file: str,
    genomic_range: str,
    model_name: Literal["modelp", "spliceai"],
    outdir: str,
    prefix: str,
) -> None:
    """Run the query genomic range pipeline and writes a WIG file for IGV visualization.

    Args:
        fasta_file: Path to the fasta file.
        genomic_range: Genomic range to query
                       (e.g., "chrX:500-1000-plus", "chr1:1000-2000-minus").
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
        start = int(secondary_parts[0].replace(",", ""))
        end = int(secondary_parts[1].replace(",", ""))
    except ValueError as exc:
        raise ValueError(
            f"Position field in variant_target is not a valid integer: "
            f"start: {secondary_parts[0]!r}, end: {secondary_parts[1]!r} "
            f"(full string: {genomic_range!r})"
        ) from exc
    strand = secondary_parts[2]

    # Load fasta
    genome_parser = GenomeParser()
    fasta_dict = genome_parser.parse_genome(
        fasta_file,
        chrom=chrom,
    )

    # Get sequence
    seq = fasta_dict[chrom]

    if strand == "+" or strand.lower() == "plus":
        seq = seq[start - 1 : end]
        reverse_ssp = False
    elif strand == "-" or strand.lower() == "minus":
        seq = str(Seq(seq[start - 1 : end]).reverse_complement())
        reverse_ssp = True
    else:
        raise ValueError(f"Invalid strand: {strand}")

    # Predict
    ssp_manager = SSPManager(
        model_name=model_name,
        batch_size=BATCH_SIZE,
        max_cache_size=MAX_CACHE_SIZE,
    )
    predict_result = ssp_manager.predict_ssp(
        seqs=[seq],
        reverse_output=reverse_ssp,
    )
    acc, dnr = predict_result[0][0], predict_result[1][0]

    try:
        wig_df = prepare_wig_dataframe(start=start, acceptor_prob=acc, donor_prob=dnr)
        # Pre-format the headers
        header = (
            f'track type=wiggle_0 name="{prefix}" '
            f'description="Probability" color={TRACK_COLOR} altColor={ALT_COLOR}\n'
            f"variableStep chrom={chrom} span=1\n"
        )

        # Write wig
        write_wig(df=wig_df, header=header, prefix=prefix, outdir=outdir)
        logger.info(
            f"Successfully wrote WIG track to {str(Path(outdir) / f'{prefix}.wig')}"
        )
    except ValueError as e:
        logger.error(f"Error generating WIG file for {prefix}: {e}")
        raise
