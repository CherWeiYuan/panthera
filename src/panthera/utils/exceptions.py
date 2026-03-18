"""
Exception codes.

This module contains application-wide custom exceptions.
"""


class AlleleLengthError(Exception):
    """Error class for unexpected allele lengths."""


class PantheraException(Exception):
    """Exceptions from Panthera"""


class NonUniqueFastaHeader(PantheraException):
    """
    Error class for non-unique Fasta sequence headers
    """


class SeqNotFoundError(PantheraException):
    """
    Error class for no sequence in fasta
    """


class UnexpectedRefError(PantheraException):
    """
    Error class for finding that the genomic coordinates supplied does
    not lead to the expected reference allele
    """


class ZeroIndexError(PantheraException):
    """
    Error class for finding 0-indexed positions where 1-indexed positions
    are expected
    """
