"""
Mutation.

This module contains the functions to mutate a reference genome
to a mutated genome sequence.
"""

import logging

from panthera.utils.exceptions import (
    AlleleLengthError,
    UnexpectedRefError,
    ZeroIndexError,
)

# Set up module-level logging
logger = logging.getLogger(__name__)


def _validate_bounds(seq: str, pos: int):
    """
    Private helper function to check validity of mutation function inputs.

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.
    """
    if pos <= 0:
        raise ZeroIndexError(f"Position must be 1-based. Got: {pos}")
    if pos > len(seq):
        raise IndexError(f"Position {pos} out of bounds.")


def _convert_uppercase(ref: str, alt: str):
    """
    Private helper function to convert allele to uppercase letters.

    Args:
        ref: The expected reference allele.
        alt: The alternative allele to insert.
    """
    return ref.upper(), alt.upper()


def snp_mutation(seq: str, pos: int, ref: str, alt: str) -> str:
    """
    Apply a Single Nucleotide Polymorphism (SNP) to a sequence
    (replaces ref with alt allele).

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.
        ref: The expected reference allele.
        alt: The alternative allele to insert.

    Returns:
        The mutated sequence string.
    """
    # Input Validation
    ref, alt = _convert_uppercase(ref, alt)
    _validate_bounds(seq, pos)

    # Coordinate Conversion (1-based to 0-based)
    idx = pos - 1
    actual_ref = seq[idx]

    # Logic & Logging
    if ref != actual_ref.upper():
        # Use the local logger, not global logging
        logger.warning(
            "Ref allele mismatch. Expected %s, found %s.",
            ref,
            actual_ref,
        )
        raise UnexpectedRefError(f"Expected {ref} at pos {pos}, found {actual_ref}")

    # Return mutation
    return f"{seq[:idx]}{alt}{seq[idx + 1 :]}"


def insertion_mutation(
    seq: str,
    pos: int,
    ref: str,
    alt: str,
    in_symbol: str,
) -> str:
    """
    Apply a Insertion Mutation to a sequence when len(alt) > 1
    and len(ref) == 1.

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.
        ref: The expected reference allele.
        alt: The alternative allele to insert.
        in_symbol: Placeholder character to indicate insertion.
                   '>' for mutant insertion, '}' for background insertion.

    Returns:
        The mutated sequence string.
    """
    # Input Validation
    ref, alt = _convert_uppercase(ref, alt)
    _validate_bounds(seq, pos)

    if len(ref) != 1:
        raise AlleleLengthError(
            f"Length of reference allele must be 1. Got: {len(ref)}"
        )

    # Coordinate Conversion (1-based to 0-based)
    idx = pos - 1
    actual_ref = seq[idx]

    # Logic & Logging
    if ref == actual_ref:
        marker = in_symbol * (len(alt) - 1)
    else:
        logger.warning(
            "Ref allele mismatch. Expected %s, found %s.",
            ref,
            actual_ref,
        )
        raise UnexpectedRefError(f"Expected {ref} at pos {pos}, found {actual_ref}")
    return f"{seq[0:pos]}{marker}{alt[1:]}{seq[pos:]}"


def deletion_mutation(
    seq: str,
    pos: int,
    ref: str,
    alt: str,
    del_symbol: str,
) -> str:
    """
    Apply Deletion Mutation.
    Delete all nucleotides from ref allele except for the first base.

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.
        ref: The expected reference allele.
        alt: The alternative allele to insert.
        del_symbol: Placeholder character to indicate deleted positions.
                   '<' for mutant deletion, '{' for background deletion.

    Returns:
        The mutated sequence string.
    """
    # Input Validation
    ref, alt = _convert_uppercase(ref, alt)
    _validate_bounds(seq, pos)

    if len(ref) < len(alt):
        raise AlleleLengthError(
            "Length of reference allele must be longer than alternate allele "
            + f"Got: ref length = {len(ref)} and alt length = {len(alt)}"
        )

    # Logic & Logging
    num_del = len(ref) - len(alt)
    actual_ref = seq[pos - 1 : pos - 1 + len(ref)]
    marker = del_symbol * num_del

    if ref != actual_ref:
        logger.warning(
            "Ref allele mismatch. Expected %s, found %s.",
            ref,
            actual_ref,
        )
        raise UnexpectedRefError(f"Expected {ref} at pos {pos}, found {actual_ref}")

    if alt == "":
        seq = f"{seq[0 : pos - 1]}{marker}{seq[pos - 1 + len(ref) :]}"
    else:
        seq = f"{seq[0 : pos - 1 + len(alt)]}{marker}{seq[pos - 1 + len(ref) :]}"

    return seq


def substitute_mutation(
    seq: str,
    pos: int,
    ref: str,
    alt: str,
    in_symbol: str,
    del_symbol: str,
) -> str:
    """
    Removes reference allele and inserts alternate
    allele when len(ref) > 1 and len(alt) > 1

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.
        ref: The expected reference allele.
        alt: The alternative allele to insert.
        in_symbol: Placeholder character to indicate insertion positions.
                   '>' for mutant insertion, '}' for background insertion.
        del_symbol: Placeholder character to indicate deleted positions.
                    '<' for mutant deletion, '{' for background deletion.

    Returns:
        The mutated sequence string.
    """
    # Input Validation
    ref, alt = _convert_uppercase(ref, alt)
    _validate_bounds(seq, pos)

    if len(ref) <= 1:
        raise AlleleLengthError(
            f"Length of reference allele must more than 1. Got: {len(ref)}"
        )

    if len(alt) <= 1:
        raise AlleleLengthError(
            f"Length of alternate allele must more than 1. Got: {len(alt)}"
        )

    # Logic & Logging
    actual_ref = seq[pos - 1 : pos - 1 + len(ref)]
    if ref == actual_ref:
        # Ref is longer than alt (substitute + insertion)
        if len(ref) < len(alt):
            num_ins = len(alt) - len(ref)
            marker = in_symbol * (num_ins)
            seq = f"{seq[0 : pos - 1]}{marker}{alt}{seq[pos - 1 + len(ref) :]}"

        # Alt is longer than ref (substitute + deletion)
        elif len(ref) > len(alt):
            num_del = len(ref) - len(alt)
            marker = del_symbol * (num_del)
            seq = f"{seq[0 : pos - 1]}{alt}{marker}{seq[pos - 1 + len(ref) :]}"

        elif len(ref) == len(alt):
            seq = f"{seq[0 : pos - 1]}{alt}{seq[pos - 1 + len(ref) :]}"

    else:
        logger.warning(
            "Ref allele mismatch. Expected %s, found %s.",
            ref,
            actual_ref,
        )
        raise UnexpectedRefError(f"Expected {ref} at pos {pos}, found {actual_ref}")

    return seq
