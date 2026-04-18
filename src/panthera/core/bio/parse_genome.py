"""Genome parser.

This module contains the functions to parse a genome fasta into
Python dictionary.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from pysam import FastxFile

from panthera.utils.exceptions import NonUniqueFastaHeader, SeqNotFoundError

# Set up module-level logging
logger = logging.getLogger(__name__)


class GenomeParser:
    """Handles genomic sequence ingestion and file management.

    This class provides static methods to parse FASTA files into dictionaries
    and manage large genome files by splitting them into chromosome-specific
    files.
    """

    @staticmethod
    def parse_genome(genome_path: str, chrom: Optional[str] = None) -> Dict[str, str]:
        """Loads genomic sequences from a FASTA file.

        If a specific chromosome is requested but missing, this method will
        attempt to split the parent genome file into individual chromosome fastas.

        Args:
            genome_path: Path to the genome FASTA file.
            chrom: Optional chromosome name to load. If None, loads all sequences.

        Returns:
            Dict[str, str]: A dictionary mapping sequence headers to sequences.

        Raises:
            SeqNotFoundError: If the requested chromosome cannot be found or
                created.
        """
        path = Path(genome_path)
        genome_dict: Dict[str, str] = {}

        # Case 1: Load the entire file
        if chrom is None:
            return GenomeParser._read_fasta_to_dict(path)

        # Case 2: Load specific chromosome
        else:
            chrom_fasta_path = path.parent / f"{path.stem}.{chrom}.fasta"

            # Check if fasta for specific chromosome exists
            if not chrom_fasta_path.exists():
                logger.warning(f"Chromosome file {chrom_fasta_path.name} not found.")

                # Create specific chromosome fasta if it does not exist
                GenomeParser._split_genome_by_chromosome(path)

                # Check if the specific chromosome fasta is created
                if not chrom_fasta_path.exists():
                    raise SeqNotFoundError(f"Could not locate sequence for {chrom}")

            genome_dict = GenomeParser._read_fasta_to_dict(chrom_fasta_path)

            if not genome_dict:
                raise SeqNotFoundError(f"Could not locate sequence for {chrom}")

            logger.info(f"Fasta for {chrom} loaded successfully.")
            return genome_dict

    @staticmethod
    def _read_fasta_to_dict(path: Path) -> Dict[str, str]:
        """Parses a FASTA file into a dictionary mapping headers to sequences.

        Args:
            path: Path to the FASTA file.

        Returns:
            Dict[str, str]: Dictionary containing the parsed sequences.

        Raises:
            NonUniqueFastaHeader: If multiple sequences share the same header.
        """
        data = {}
        # Assuming FastxFile is available in your environment
        try:
            with FastxFile(str(path)) as fasta_handler:
                for contig in fasta_handler:
                    header = str(contig.name)
                    if header in data:
                        raise NonUniqueFastaHeader(f"Duplicate header: {header}")
                    data[header] = str(contig.sequence).upper()
            return data
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            raise

    @staticmethod
    def _split_genome_by_chromosome(genome_path: Path) -> None:
        """Splits a multi-sequence FASTA file into individual chromosome files.

        Args:
            genome_path: Path to the parent genome FASTA file.
        """
        logger.info(f"Splitting {genome_path.name} into chromosome-specific files...")

        prefix = genome_path.stem  # Gets filename without extension safely

        with FastxFile(str(genome_path)) as fasta_handler:
            for contig in fasta_handler:
                chrom_name = str(contig.name)
                output_path = genome_path.parent / f"{prefix}.{chrom_name}.fasta"

                logger.debug(f"Creating {output_path.name}")
                with open(output_path, "w") as f:
                    # Scientific data integrity: Ensure sequences are normalized (upper case)
                    f.write(f">{chrom_name}\n{str(contig.sequence).upper()}\n")

        logger.info("Genome split successfully.")
