"""
Onehot-encoding.

This module contains the function for onehot-encoding of DNA or RNA sequence.
"""

from enum import Enum
from typing import Dict, List, Final
import numpy as np


class EncodingSchema(Enum):
    """
    Use EncodingSchema (Enum) to ensure type safety.
    An IDE will suggest the available schemas when SeqEncoder is specified.
    """

    SPLICEAI = "spliceai"
    MODELP = "modelp"


class SeqEncoder:
    """
    Enterprise-grade encoder for DNA/RNA sequences.
    Supports multiple one-hot encoding schemas and handles validation.
    """

    # Define mappings as constants to save memory and ensure immutability
    _SCHEMAS: Final[Dict[EncodingSchema, Dict[str, List[float]]]] = {
        EncodingSchema.SPLICEAI: {
            "A": [1.0, 0.0, 0.0, 0.0],
            "C": [0.0, 1.0, 0.0, 0.0],
            "G": [0.0, 0.0, 1.0, 0.0],
            "T": [0.0, 0.0, 0.0, 1.0],
            "U": [0.0, 0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 0.0, 0.0],
            "X": [1.0, 1.0, 1.0, 1.0],
        },
        EncodingSchema.MODELP: {
            "A": [1.0, 0.0, 0.0, 0.0],
            "T": [0.0, 1.0, 0.0, 0.0],
            "U": [0.0, 1.0, 0.0, 0.0],
            "C": [0.0, 0.0, 1.0, 0.0],
            "G": [0.0, 0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 0.0, 0.0],
            "X": [1.0, 1.0, 1.0, 1.0],
        },
    }

    @classmethod
    def one_hot_encode(cls, sequence: str, schema: EncodingSchema) -> np.ndarray:
        """
        Encodes a DNA/RNA sequence into a one-hot representation.

        Args:
            sequence: The input nucleotide string (case-insensitive).
            schema: The EncodingSchema to use for the mapping.

        Returns:
            A NumPy array of shape (len(sequence), 4). Numpy arrays uses
            contiguous memory blocks and save more space than python lists.

        Raises:
            ValueError: If the sequence contains characters
            not defined in the schema.

        Note:
            @classmethod means this function belongs to the SeqEncoder concept,
            but you don't need to build an object to use it
        """
        # Input validation
        sequence = sequence.upper().strip()

        # Handle empty sequence explicitly to maintain 2D shape consistency
        if not sequence:
            return np.empty((0, 4), dtype=np.float32)

        mapping = cls._SCHEMAS[schema]

        try:
            # Using list comprehension + numpy is usually 
            # the fastest for custom mappings
            encoded = np.array([mapping[base] for base in sequence], dtype=np.float32)
            return encoded
        except KeyError as e:
            invalid_char = e.args[0]
            raise ValueError(
                f"Invalid character '{invalid_char}' found in sequence."
            ) from None
