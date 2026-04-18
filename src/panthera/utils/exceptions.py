"""Exception codes.

This module contains application-wide custom exceptions.
"""

# ---------------------------------------------------------
# Parent classes
# ---------------------------------------------------------


class FastaException(Exception):
    """Base exception for FASTA parsing errors."""


class HaplotypeBlockError(Exception):
    """Base exception for haplotype block processing errors."""


class MutationException(Exception):
    """Base exception for reference genome mutagenesis errors."""


class VariantParsingError(Exception):
    """Base exception for variant file parsing errors."""


class DataResolutionError(Exception):
    """Raised when application data or resources cannot be resolved."""

    pass


# ---------------------------------------------------------
# FastaException Child classes
# ---------------------------------------------------------


class NonUniqueFastaHeader(FastaException):
    """Raised when multiple sequences share the same header in a FASTA file."""


class SeqNotFoundError(FastaException):
    """Raised when a requested sequence is not found in the FASTA file."""


# ---------------------------------------------------------
# MutationException Child classes
# ---------------------------------------------------------


class AlleleLengthError(MutationException):
    """Raised when ref/alt allele lengths do not match the mutation type."""


class AmbiguousDeletionError(MutationException):
    """Raised when a deletion overlaps with other variants, creating ambiguity."""


class UnequalSequenceLengthError(MutationException):
    """Raised when output WT and MT sequences have unexpected length differences."""


class UnexpectedMutationError(MutationException):
    """Raised when an unhandled mutation type is encountered."""


class UnexpectedRefError(MutationException):
    """Raised when the reference allele at a coordinate doesn't match the VCF."""


class ZeroIndexError(MutationException):
    """Raised when a 0-indexed position is provided where 1-indexing is expected."""


# ---------------------------------------------------------
# VariantParsingError child classes
# ---------------------------------------------------------
class NoVariantsError(VariantParsingError):
    """Raised when a variant file contains no records."""


class MultipleAltError(VariantParsingError):
    """Raised when a record contains multiple alternate alleles."""


class MultipleVcfSampleError(VariantParsingError):
    """Raised when a single-sample VCF parser encounters multiple samples."""


class NoPhaseSetError(VariantParsingError):
    """Raised when a VCF lacks the required PS (phase set) format records."""


# ---------------------------------------------------------
# VariantParsingError child classes
# ---------------------------------------------------------
class BackgroundConflictError(HaplotypeBlockError):
    """Raised when target and background variants occupy the same coordinates."""


class NonUniqueChromError(HaplotypeBlockError):
    """Raised when a single haplotype block contains variants from multiple chroms."""


class NonUniquePhaseSetTagError(HaplotypeBlockError):
    """Raised when a single block contains variants with different phase set tags."""
