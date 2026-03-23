"""
Exception codes.

This module contains application-wide custom exceptions.
"""

# ---------------------------------------------------------
# Parent classes
# ---------------------------------------------------------


class FastaException(Exception):
    """Base exception for parsing fasta."""


class HaplotypeBlockError(Exception):
    """Base exception for haplotype blocks."""


class MutationException(Exception):
    """Base exception for mutagenesis of reference genome sequence."""


class VariantParsingError(Exception):
    """Base exception for variant processing."""


# ---------------------------------------------------------
# FastaException Child classes
# ---------------------------------------------------------


class AlleleLengthError(FastaException):
    """Error class for unexpected allele lengths."""


class NonUniqueFastaHeader(FastaException):
    """
    Error class for non-unique Fasta sequence headers.
    """


class SeqNotFoundError(FastaException):
    """
    Error class for no sequence in fasta.
    """


# ---------------------------------------------------------
# MutationException Child classes
# ---------------------------------------------------------


class AmbiguousDeletionError(MutationException):
    """
    Error class for deletion mutations deleting the positions where there are
    supposed to be other mutations.
    """


class UnequalSequenceLengthError(MutationException):
    """
    Error class for unequal WT and MT sequence length output.
    """


class UnexpectedMutationError(MutationException):
    """
    Error class for unexpected mutation types.
    """


class UnexpectedRefError(MutationException):
    """
    Error class for finding that the genomic coordinates supplied does
    not lead to the expected reference allele.
    """


class ZeroIndexError(MutationException):
    """
    Error class for finding 0-indexed positions where 1-indexed positions
    are expected.
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
    Error class for having more than one sample in the VCF.
    """


class NoPhaseSetError(VariantParsingError):
    """
    Error class for having no phase set format, indicating the absence of
    WhatsHap phasing performed for the input VCF file.
    """


# ---------------------------------------------------------
# VariantParsingError child classes
# ---------------------------------------------------------
class BackgroundConflictError(HaplotypeBlockError):
    """
    Error class for background variants from non-reference genomes sharing
    the same genomic coordinates as the input variants.
    """


class NonUniqueChromError(HaplotypeBlockError):
    """
    Error class for multiple chrom values found in the same
    variants dataframe.
    """


class NonUniquePhaseSetTagError(HaplotypeBlockError):
    """
    Error class for multiple phase set tag found in the same
    variants dataframe.
    """
