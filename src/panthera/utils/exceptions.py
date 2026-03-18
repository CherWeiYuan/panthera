"""
Exception codes.

This module contains application-wide custom exceptions.
"""

# ---------------------------------------------------------
# Parent classes
# ---------------------------------------------------------


class FastaException(Exception):
    """Base exception for parsing fasta"""


class VariantParsingError(Exception):
    """Base exception for variant processing"""


class MutationException(Exception):
    """Base exception for mutagenesis of reference genome sequence"""


# ---------------------------------------------------------
# FastaException Child classes
# ---------------------------------------------------------


class AlleleLengthError(FastaException):
    """Error class for unexpected allele lengths"""


class NonUniqueFastaHeader(FastaException):
    """
    Error class for non-unique Fasta sequence headers
    """


class SeqNotFoundError(FastaException):
    """
    Error class for no sequence in fasta
    """


# ---------------------------------------------------------
# MutationException Child classes
# ---------------------------------------------------------


class UnexpectedRefError(MutationException):
    """
    Error class for finding that the genomic coordinates supplied does
    not lead to the expected reference allele
    """


class ZeroIndexError(MutationException):
    """
    Error class for finding 0-indexed positions where 1-indexed positions
    are expected
    """


# ---------------------------------------------------------
# VariantParsingError child classes
# ---------------------------------------------------------
class NoVariantsError(VariantParsingError):
    """
    Error class for empty TSV or VCF
    """


class MultipleAltError(VariantParsingError):
    """
    Error class for VCF/ TSV containing multiple alternate alleles in a single row.
    """


class MultipleVcfSampleError(VariantParsingError):
    """
    Error class for having more than one sample in the VCF
    """


class NoPhaseSetError(VariantParsingError):
    """
    Error class for having no phase set format, indicating the absence of
    WhatsHap phasing performed for the input VCF file
    """
