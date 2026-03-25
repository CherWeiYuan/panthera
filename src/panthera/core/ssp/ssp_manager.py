"""
Splice site probability prediction manager.

This module contains the class to manage splice site probability prediction.
"""

import logging
from typing import List, Literal, Tuple, Callable

from Bio.Seq import Seq
import numpy as np
import numpy.typing as npt

from panthera.core.ssp.load_model import load_frozen_graph
from panthera.core.ssp.model_paths import SPLICEAI_MODEL_PATH, MODELP_MODEL_PATH
from panthera.core.ssp.predict import spliceai_predict, modelp_predict

# Configure module-level logger
logger = logging.getLogger(__name__)


class SSPManager:
    """
    Manager for splice site probability (SSP) prediction.
    """

    # Compile the translation table once at the class level for high performance
    _INDEL_TRANS_TABLE = str.maketrans('', '', "><}{}")

    def __init__(
        self, model_name: Literal["modelp", "spliceai"], batch_size: int
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_fn = self._load_model()

    def _load_model(self) -> Callable:
        """
        Loads the frozen model graph function based on the selected model.
        """
        logger.info(f"Loading {self.model_name} model...")

        if self.model_name == "modelp":
            return load_frozen_graph(MODELP_MODEL_PATH)
        elif self.model_name == "spliceai":
            return load_frozen_graph(SPLICEAI_MODEL_PATH)
        else:
            # While Literal catches this in static analysis, this protects against
            # runtime overrides.
            raise ValueError(f"Unexpected model name: {self.model_name}")

    def predict_ssp(
        self,
        seqs: List[str],
        reverse_output: bool = False,
    ) -> Tuple[List[npt.NDArray[np.float32]], List[npt.NDArray[np.float32]]]:
        """
        Runs splice site probability prediction functions on a list of sequences.

        PANTHERA accepts DNA/RNA sequence as input so:
            - minus strand input needs to be reverse complemented
            - plus strand input can be input as it is

        Args:
            seqs: A list of DNA or RNA sequences. If the input sequence is on
                  the minus strand, it must be reverse complemented before
                  using as input to this function.
            reverse_output: If True, reverses the order of the splice site
                            probabilities before output.

        Returns:
            A tuple of two elements:
            - acceptor_probs: List of 1D arrays containing acceptor probabilities.
            - donor_probs: List of 1D arrays containing donor probabilities.
        """
        if not seqs:
            logger.warning("Empty sequence list provided to predict_ssp.")
            return [], []

        # Predict SSP
        if self.model_name == "modelp":
            acceptor_arrays, donor_arrays = modelp_predict(
                seqs=seqs, batch_size=self.batch_size, modelp_fn=self.model_fn
            )
        elif self.model_name == "spliceai":
            acceptor_arrays, donor_arrays = spliceai_predict(
                seqs=seqs, batch_size=self.batch_size, spliceai_fn=self.model_fn
            )
        else:
            raise ValueError(f"Unexpected model name: {self.model_name}")

        # Reverse the order of output probabilities 
        # (using views for memory efficiency)
        if reverse_output:
            acceptor_arrays = [arr[::-1] for arr in acceptor_arrays]
            donor_arrays = [arr[::-1] for arr in donor_arrays]

        return acceptor_arrays, donor_arrays

    def remove_indel_markers(self, seqs: List[str]) -> List[str]:
        """
        Remove INDEL placeholder markers (>/</{/}) from a list of sequences.
        """
        return [seq.translate(self._INDEL_TRANS_TABLE) for seq in seqs]

    def reverse_complement(self, seqs: List[str]) -> List[str]:
        """Reverse complement all sequences in the list."""
        return [str(Seq(seq).reverse_complement()) for seq in seqs]
