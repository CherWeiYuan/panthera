"""
Splice site probability prediction manager.

This module contains the class to manage splice site probability prediction.
"""

from collections import OrderedDict
import logging
from typing import List, Literal, Tuple, Callable, cast

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
    _INDEL_TRANS_TABLE = str.maketrans("", "", "><}{}")

    def __init__(
        self,
        model_name: Literal["modelp", "spliceai"],
        batch_size: int,
        max_cache_size: int = 500,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_fn = self._load_model()

        # Initialize the LRU Cache
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[
            str, Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        ] = OrderedDict()

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

        final_acceptors: List[npt.NDArray[np.float32] | None] = [None] * len(seqs)
        final_donors: List[npt.NDArray[np.float32] | None] = [None] * len(seqs)

        uncached_seqs = []
        uncached_indices = []

        # 1. Check cache for each sequence
        for i, seq in enumerate(seqs):
            if seq in self._cache:
                # Move to the end to mark as recently used
                self._cache.move_to_end(seq)
                acc, dnr = self._cache[seq]
                final_acceptors[i] = acc
                final_donors[i] = dnr
            else:
                uncached_seqs.append(seq)
                uncached_indices.append(i)

        # 2. Predict for uncached sequences
        if uncached_seqs:
            if self.model_name == "modelp":
                new_acc, new_dnr = modelp_predict(
                    seqs=uncached_seqs,
                    batch_size=self.batch_size,
                    modelp_fn=self.model_fn,
                )
            elif self.model_name == "spliceai":
                new_acc, new_dnr = spliceai_predict(
                    seqs=uncached_seqs,
                    batch_size=self.batch_size,
                    spliceai_fn=self.model_fn,
                )
            else:
                raise ValueError(f"Unexpected model name: {self.model_name}")

            # 3. Store new predictions in cache and place in final output
            for seq, idx, acc, dnr in zip(
                uncached_seqs, uncached_indices, new_acc, new_dnr
            ):
                final_acceptors[idx] = acc
                final_donors[idx] = dnr

                # Add to cache
                self._cache[seq] = (acc, dnr)

                # Enforce LRU size limit
                if len(self._cache) > self.max_cache_size:
                    self._cache.popitem(last=False)  # Removes the oldest entry

        # The final lists shouldn't have any None values left, but type hinting
        # requires us to cast them back to the expected return type.
        acceptor_arrays = cast(List[npt.NDArray[np.float32]], list(final_acceptors))
        donor_arrays = cast(List[npt.NDArray[np.float32]], list(final_donors))

        # Step 4: Reverse the order of output probabilities if requested
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
