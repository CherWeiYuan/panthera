"""
Panthera API.

This module contains the function calls for Panthera API
"""

import os
from pathlib import Path
from typing import Tuple, Literal

import numpy as np
import numpy.typing as npt

from panthera.core.ssp.ssp_manager import SSPManager
from panthera.core.bio.wig import (
    TRACK_COLOR,
    ALT_COLOR,
    prepare_wig_dataframe,
    write_wig,
)


def load_model(model_name: Literal["modelp", "spliceai"]) -> SSPManager:
    """
    Load the splice site probability model.

    Args:
        model_name: Name of the model ("modelp" or "spliceai").

    Returns:
        SSPManager: SSPManager object.
    """
    manager = SSPManager(model_name, batch_size=1)
    return manager


def predict(
    seq: str, model: SSPManager
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Predict splice site probabilities for a given sequence.

    Args:
        seq: DNA/RNA sequence.
        model: SSPManager object.

    Returns:
        Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            A tuple containing (acceptor_probs, donor_probs).

    Raises:
        TypeError: If seq is not a string or model is not a SSPManager object.

    Example:
        >>> from panthera.api import load_model, predict
        >>> model = load_model("modelp")
        >>> acceptor, donor = predict("GUAG", model)
        >>> print(acceptor)
        [0.0001 0.0001 0.0001 0.0001]
        >>> print(donor)
        [0.0001 0.0001 0.0001 0.0001]
    """
    # Input validation
    if not isinstance(seq, str):
        raise TypeError("Sequence must be a string.")
    if not isinstance(model, SSPManager):
        raise TypeError("Model must be a SSPManager object.")

    # Predict
    acceptor, donor = model.predict_ssp([seq], reverse_output=False)
    return acceptor[0], donor[0]


def wig(
    acceptor_probs: npt.NDArray[np.float32] | list | tuple,
    donor_probs: npt.NDArray[np.float32] | list | tuple,
    chrom: str,
    start: int,
    strand: Literal["+", "-", "plus", "minus"],
    outdir: str,
    prefix: str,
) -> None:
    """
    Generate WIG files for acceptor and donor probabilities.

    Args:
        acceptor_probs: Array of acceptor probabilities.
        donor_probs: Array of donor probabilities.
        chrom: Chromosome name.
        start: Start position of the sequence on chromosome (1-based).
        strand: Strand of the sequence ("+"/"plus" or "-"/"minus").
        outdir: Output directory.
        prefix: Prefix for the output files.

    Example:
        >>> from panthera.api import load_model, predict, wig
        >>> model = load_model("modelp")
        >>> acceptor, donor = predict("GUAG", model)
        >>> wig(acceptor, donor, "chr1", 1, "./", "test")
    """
    # ---- Input validation ----

    # Array Coercion and Validation
    try:
        acceptor_probs = np.asarray(acceptor_probs, dtype=np.float32)
        donor_probs = np.asarray(donor_probs, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise TypeError(
            "Probabilities must be convertible to numeric numpy arrays."
        ) from e

    if acceptor_probs.ndim != 1 or donor_probs.ndim != 1:
        raise ValueError("Acceptor and donor probabilities must be 1-dimensional.")

    if acceptor_probs.size == 0 or donor_probs.size == 0:
        raise ValueError("Probability arrays cannot be empty.")

    if len(acceptor_probs) != len(donor_probs):
        raise ValueError(
            f"Length mismatch: acceptor_probs ({len(acceptor_probs)}) != "
            f"donor_probs ({len(donor_probs)})."
        )

    # Check for valid probability bounds (0 to 1)
    if np.any((acceptor_probs < 0) | (acceptor_probs > 1)) or np.any(
        (donor_probs < 0) | (donor_probs > 1)
    ):
        raise ValueError("Probabilities must be between 0.0 and 1.0.")

    # Genomic Coordinate Validation
    chrom = str(chrom).strip()
    if not chrom:
        raise ValueError("Chromosome name cannot be empty.")

    try:
        start = int(start)
    except (ValueError, TypeError):
        raise TypeError(
            f"Start position must be an integer, got {type(start).__name__}"
        )

    if start < 1:
        raise ValueError(f"Start position must be 1-based. Received: {start}")

    # Strand Normalization
    strand_norm = strand.lower().strip()
    if strand_norm not in ("+", "-", "plus", "minus"):
        raise ValueError(
            f"Strand must be '+', '-', 'plus', or 'minus'. Received: '{strand}'"
        )

    # File System and Path Validation
    if not prefix or any(sep in prefix for sep in (os.sep, os.altsep) if sep):
        raise ValueError("Prefix must be a valid file name without path separators.")

    out_path = Path(outdir)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Could not create output directory '{outdir}': {e}") from e

    # ---- Function logic ----

    # Reverse order for negative strand
    if strand.lower() in ("-", "minus"):
        acceptor_probs, donor_probs = acceptor_probs[::-1], donor_probs[::-1]

    # Prepare dataframe for WIG
    df = prepare_wig_dataframe(start, acceptor_probs, donor_probs)

    # Get header of WIG
    header = (
        f'track type=wiggle_0 name="API" '
        f'description="Probability" color={TRACK_COLOR} altColor={ALT_COLOR}\n'
        f"variableStep chrom={chrom} span=1\n"
    )

    # Export WIG file
    write_wig(df, header, prefix, outdir)
