"""
Exception codes.

This module contains application-wide custom exceptions.
"""


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
