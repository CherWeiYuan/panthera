"""Mutation.

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
    """Private helper function to check validity of mutation function inputs.

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.

    Raises:
        ZeroIndexError: If the position is less than or equal to 0.
        IndexError: If the position is greater than the length of the sequence.
    """
    if pos <= 0:
        raise ZeroIndexError(f"Position must be 1-based. Got: {pos}")
    if pos > len(seq):
        raise IndexError(f"Position {pos} out of bounds.")


def _convert_uppercase(ref: str, alt: str):
    """Private helper function to convert allele to uppercase letters.

    Args:
        ref: The expected reference allele.
        alt: The alternative allele to insert.

    Returns:
        tuple[str, str]: Tuple containing the uppercase reference and alternate
            alleles.
    """
    return ref.upper(), alt.upper()


def snp_mutation(seq: str, pos: int, ref: str, alt: str) -> str:
    """Applies a Single Nucleotide Polymorphism (SNP) to a sequence.

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.
        ref: The expected reference allele.
        alt: The alternative allele to insert.

    Returns:
        str: The mutated sequence string.

    Raises:
        ZeroIndexError: If the position is 0 or negative.
        IndexError: If the position is out of bounds.
        UnexpectedRefError: If the actual reference base does not match `ref`.
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
    """Applies an insertion mutation to a sequence.

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.
        ref: The expected reference allele.
        alt: The alternative allele to insert.
        in_symbol: Placeholder character for insertion markers.

    Returns:
        str: The mutated sequence string.

    Raises:
        ZeroIndexError: If the position is 0 or negative.
        IndexError: If the position is out of bounds.
        AlleleLengthError: If the reference allele length is not 1.
        UnexpectedRefError: If the reference allele does not match `ref`.
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
    """Applies a deletion mutation to a sequence.

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.
        ref: The expected reference allele.
        alt: The alternative allele to insert.
        del_symbol: Placeholder character for deletion markers.

    Returns:
        str: The mutated sequence string.

    Raises:
        ZeroIndexError: If the position is 0 or negative.
        IndexError: If the position is out of bounds.
        AlleleLengthError: If the reference allele is shorter than the alt allele.
        UnexpectedRefError: If the reference allele does not match `ref`.
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
    """Applies a substitution mutation (complex INDEL) to a sequence.

    Handles cases where both ref and alt alleles have length > 1.

    Args:
        seq: The reference sequence.
        pos: 1-based coordinate of the mutation.
        ref: The expected reference allele.
        alt: The alternative allele to insert.
        in_symbol: Placeholder character for insertion markers.
        del_symbol: Placeholder character for deletion markers.

    Returns:
        str: The mutated sequence string.

    Raises:
        ZeroIndexError: If the position is 0 or negative.
        IndexError: If the position is out of bounds.
        AlleleLengthError: If ref or alt allele length is not > 1.
        UnexpectedRefError: If the reference allele does not match `ref`.
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
