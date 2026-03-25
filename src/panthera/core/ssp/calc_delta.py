"""
Calculate delta scores

This module contain functions to calculate the per-position delta 
scores between wild-type and mutant splice site probabilities.
"""

from collections import deque
import logging
from typing import Literal

import numpy as np
import numpy.typing as npt

# Configure module-level logger
logger = logging.getLogger(__name__)

class SSPScorer:
    """
    Scorer for splice site probability (SSP)

    This class handles the following:
        - initializes the variables required for all calculations
        - alignment of splice site probabilities using align_prob()
        - calculation of raw delta scores using calc_raw_delta()
        - calculation of masked delta scores using calc_masked_delta()
        - retrieve position of max masked delta score as string output using
          _find_max_mds_locations()
    
    Difference between raw and masked delta scores
        - raw: absolute difference between wild-type (WT) and mutant (MT) SSP
        - masked: raw delta scores but certain values will be masked (i.e.,
                  converted to 0.0) when any of the two conditions below is met:
        
                  1. At genomic coordinates of known splice sites, an increase 
                  in SSP is masked to 0.0 (i.e., mutation leading to increased 
                  canonical splice site probability is meaningless). 
                  
                  2. At genomic coordinates without known splice sites, a 
                  decrease in SSP is masked to 0.0 (i.e., mutation leading to 
                  decreased cryptic splice site probability is meaningless).
    
    This class is designed to be a stand-alone (i.e., contains all the 
    variables and functions for all calculations) to facilitate multiprocessing.
    """

    # Compile the translation table once at the class level for high performance
    _INDEL_TRANS_TABLE = str.maketrans('', '', "{}")

    # Define __slots__ to drastically reduce memory footprint during 
    # multiprocessing
    # Instead of creating a dictionary per class instance, Python just reserve 
    # space for these specific items
    __slots__ = [
        'chrom_start', 'splice_sites', 'wt_seq', 'mt_seq',
        'wt_acc', 'wt_dnr', 'mt_acc', 'mt_dnr',
        'reference_pos', 'aligned_prob', 'max_raw_delta', 
        'max_masked_delta', 'max_mds_loc'
    ]

    def __init__(
            self, chrom_start: int, 
            splice_sites: dict[str, list[int]],
            wt_seq: str, mt_seq:str,
            wt_acc: npt.NDArray[np.float32], wt_dnr: npt.NDArray[np.float32],
            mt_acc: npt.NDArray[np.float32], mt_dnr: npt.NDArray[np.float32],
            ) -> None:
        """
        Args:
            chrom_start: Genomic coordinate of the first nucleotide in wt_seq.
            splice_sites: Acceptor and donor positions in the gene 
                          (using genomic coordinates):
                          {"acc": list[int], "dnr": list[int]}
            wt_seq: Wild-type DNA/ RNA sequence. Contains insertion character
                    placeholder '}' or deletion placeholder '{' if background
                    variants are incorporated into the sequence.
            mt_seq: Mutant DNA/ RNA sequence with target variants. Contains both
                    INDEL characters of background variants ('}' or '{') and of
                    target variants ('>' or '<').
            wt_acc: List of wild-type acceptor probability per nucleotide.
            wt_dnr: List of wild-type donor probability per nucleotide.
            mt_acc: List of mutant acceptor probability per nucleotide.
            mt_dnr: List of mutant donor probability per nucleotide.
        """
        self.chrom_start = chrom_start
        self.splice_sites = splice_sites

        # Immutable sequences
        self.wt_seq = wt_seq
        self.mt_seq = mt_seq

        # Mutable probability lists
        self.wt_acc = wt_acc
        self.wt_dnr = wt_dnr
        self.mt_acc = mt_acc
        self.mt_dnr = mt_dnr

        # Initialize uncomputed variables with static type hints
        self.reference_pos: list[str] | None = None
        self.aligned_prob: tuple[list[float], list[float], 
                                 list[float], list[float]] | None = None
        self.max_raw_delta: float | None = None
        self.max_masked_delta: float | None = None
        self.max_mds_loc: str | None = None

    def align_prob(self) -> None:
        """
        Align splice site probabilities.

        This function uses the wild-type sequence (wt_seq, where '{' or '}' 
        placeholder markers for INDELs are removed), and mutant sequence (
        mt_seq, where '>' and '<' placeholders are kept), to align the 
        following:
            - wild-type (wt_acc) and mutant (mt_acc) acceptor probabilities
            - wild-type (wt_dnr) and mutant (mt_dnr) donor probabilities
        
        Alignment means that both wt and mt probabilities list have element-wise
        pairing and thus belongs to the same genomic coordinate on the reference
        genome (e.g., GRCh38).

        The output, aligned probabilities, can be used to for pairwise
        calculation of delta scores using the functions calc_raw_delta
        and calc_masked_delta.

        Side Effects:
            Update of self.aligned_prob to tuple containing list of splice site 
            probability (floats): 
            (new_wt_acc, new_wt_dnr, new_mt_acc, new_mt_dnr)

            Update of self.reference_pos to list of genomic coordinates 
            (integer value stored as string).
             
            The reference_pos, when matched element-wise to the
            aligned splice site probabilities, tells us the genomic coordinate
            corresponding to the element. If positions are created due to
            insertion mutations, the position will be assigned a new unique
            string: {previous genomic coordinate}p{number of insertion so far}.
        """

        # --- Prepare MT ---

        # Remove INDEL placeholders ('{' or '}') 
        # from background conversions, if any
        mt_seq = self.mt_seq.translate(self._INDEL_TRANS_TABLE)

        # Create double-ended queues for quick pops
        mt_acc_deque = deque(self.mt_acc)
        mt_dnr_deque = deque(self.mt_dnr)

        # --- Prepare WT ---

        # Create double-ended queues for quick pops
        wt_acc_deque = deque(self.wt_acc)
        wt_dnr_deque = deque(self.wt_dnr)

        # --- Alignment ---
        # Align WT and MT acceptor and donor prob
        new_wt_acc = []
        new_wt_dnr = []
        new_mt_acc = []
        new_mt_dnr = []
        reference_pos = []

        current_pos = self.chrom_start
        ignore_counter = 0
        for n in mt_seq:
            if n in {"A", "T", "C", "G", "N"}:  # Match or SNP
                if ignore_counter == 0:
                    new_wt_acc.append(wt_acc_deque.popleft())
                    new_wt_dnr.append(wt_dnr_deque.popleft())
                    new_mt_acc.append(mt_acc_deque.popleft())
                    new_mt_dnr.append(mt_dnr_deque.popleft())
                    reference_pos.append(current_pos)
                    current_pos += 1
                elif ignore_counter < 0:
                    logger.error(
                        f"ignore_counter reached negative: {ignore_counter}")
                    raise RuntimeError(
                        "Expect ignore counter to be 0 or positive. " +
                        f"Got {ignore_counter}.")
                else:
                    ignore_counter -= 1
                    continue
            elif n == ">":  # Insertion mutation, add 0 to WT score
                new_wt_acc.append(0)
                new_wt_dnr.append(0)
                new_mt_acc.append(mt_acc_deque.popleft())
                new_mt_dnr.append(mt_dnr_deque.popleft())
                ignore_counter += 1
                reference_pos.append(f"{current_pos}p{ignore_counter}")
            elif n == "<":  # Deletion mutation, add 0 to MT score
                new_wt_acc.append(wt_acc_deque.popleft())
                new_wt_dnr.append(wt_dnr_deque.popleft())
                new_mt_acc.append(0)
                new_mt_dnr.append(0)
                reference_pos.append(current_pos)
                current_pos += 1
            else:
                logger.error(f"Error: n in mt_seq is {n}")
                raise ValueError(
                    f"Expected characters are A/T/C/G/N/>/<. Got {n}"
                )
        
        # Ensure output lengths are equal
        if len(mt_seq.replace(">", "")) == len(new_wt_acc) == len(new_wt_dnr) \
            == len(new_mt_acc) == len(new_mt_dnr) == len(reference_pos):
            pass
        else:
            logger.error("Error: Sequence and splice site " +
                          "probability list are not equal length")
            raise ValueError(
                f"""
                Non-equal output lengths:
                mt_seq ('>' removed): {len(mt_seq.replace(">", ""))}
                new_wt_acc: {len(new_wt_acc)}
                new_wt_dnr: {len(new_wt_dnr)}
                new_mt_acc: {len(new_mt_acc)}
                new_mt_dnr: {len(new_mt_dnr)}
                reference_pos: {len(reference_pos)}
                """
            )
        
        # Change internal state
        self.reference_pos = reference_pos # list[str]
        self.aligned_prob = (new_wt_acc, new_wt_dnr,
                             new_mt_acc, new_mt_dnr) # tuple[list[float]]

    def calc_raw_delta(self) -> float:
        """Calculate raw delta scores"""
        # Retrieve aligned probabilities
        wt_acc, wt_dnr, mt_acc, mt_dnr, _ = self.aligned_prob

        # Calculate RAW delta score per base position
        raw_acc_deltas = [abs(wt_acc[i] - mt_acc[i]) 
                          for i in range(len(wt_acc))]
        raw_dnr_deltas = [abs(wt_dnr[i] - mt_dnr[i])
                          for i in range(len(wt_dnr))]
        
        # Get max raw delta score position
        max_raw_delta = np.max(raw_acc_deltas, raw_dnr_deltas)

        # Update internal state and return output
        self.max_raw_delta = max_raw_delta

        return max_raw_delta

    def _masked_delta_helper(
            wt_ssp: list[float], mt_ssp: list[float],
            reference_pos: list[str], splice_sites: dict[str, list[int]],
            ss_type: Literal["acc", "dnr"]
            ) -> list[float]:
        """
        Helper function to calculate masked delta scores.

        Args:
            wt_ssp: Wild-type (WT) splice site probability.
            mt_ssp: Mutant (MT) splice site probability.
            reference_pos: Genomic coordinate on the wild-type reference genome.
            splice_sites: Acceptor and donor positions in the gene 
                          (using genomic coordinates):
                          {"acc": list[int], "dnr": list[int]}
            ss_type: Acceptor ("acc") or donor ("dnr")

        Returns:
            masked_deltas: Delta masked scores
        """
        # Input validation
        if not (len(wt_ssp) == len(mt_ssp) == len(reference_pos)):
            logger.error(
                f"Expect equal length. Got wt_ssp (length {len(wt_ssp)}) " + 
                f"and mt_ssp (length {len(mt_ssp)})."
                )
            raise ValueError(
                "Expect equal length wt_ssp and mt_ssp." +
                f"Got {len(wt_ssp)} and {len(mt_ssp)}."
                )

        masked_deltas = []
        for i in range(len(wt_ssp)):
            # Get splice site probability and delta at index i
            wt    = round(float(wt_ssp[i]), 3)
            mt    = round(float(mt_ssp[i]), 3)
            pos   = reference_pos[i]
            delta = round(abs(wt - mt), 3)

            # Check if position is an annotated splice site
            if pos in splice_sites[ss_type]:
                # If position is a known splice site,
                # then WT should be higher than MT, else mask
                if wt - mt > 0:
                    masked_deltas.append(delta)
                else:
                    masked_deltas.append(0.0)
            else:
                # If position is not a known splice site,
                # then MT should be higher than WT, else mask
                if mt - wt > 0:
                    masked_deltas.append(delta)
                else:
                    masked_deltas.append(0.0)

        return masked_deltas
    
    def _find_max_mds_locations(
            self, 
            masked_acc_deltas: list[float], 
            masked_dnr_deltas: list[float],
            reference_pos: list[str]) -> str:
        """
        Find genomic positions with max Masked Delta Score (MDS) and return
        results as a string.
        """

        # Find genomic positions with max masked delta score
        max_masked_delta_indices = list(
            set([i for i, value in enumerate(masked_acc_deltas) 
                 if value == self.max_masked_delta] +\
                [i for i, value in enumerate(masked_dnr_deltas) 
                 if value == self.max_masked_delta])
                 )

        # Get string of the genomic positions of max masked delta scores
        if self.max_masked_delta > 0:
            max_mds_pos_str = ""
            for i in max_masked_delta_indices:
                max_mds_pos_str += f"{reference_pos[i]};"
        else:
            max_mds_pos_str = ""

        return max_mds_pos_str 

    def calc_masked_delta(self) -> float:
        """
        Calculate masked delta scores
        
        Returns:
            max_masked_delta: Max masked delta score in both acceptor and donor
                              delta scores
        
        
        """
        # Retrieve aligned probabilities
        wt_acc, wt_dnr, mt_acc, mt_dnr = self.aligned_prob

        # Run helper function
        masked_acc_deltas = self._masked_delta_helper(
            wt_ssp = wt_acc, 
            mt_ssp = mt_acc, 
            reference_pos = self.reference_pos, 
            splice_sites = self.splice_sites, 
            ss_type = "acc"
        )
        masked_dnr_deltas = self._masked_delta_helper(
            wt_ssp = wt_dnr, 
            mt_ssp = mt_dnr, 
            reference_pos = self.reference_pos, 
            splice_sites = self.splice_sites, 
            ss_type = "dnr"
        )

        # Get max masked delta score
        max_masked_delta = max(*masked_acc_deltas, *masked_dnr_deltas)

        # Find genomic locations of max masked delta score
        # Output is returned as a string
        max_mds_loc = self._find_max_mds_locations(
            masked_acc_deltas, masked_dnr_deltas, self.reference_pos)

        # Update internal state and return output
        self.max_mds_loc = max_mds_loc
        self.max_masked_delta = max_masked_delta

        return max_masked_delta