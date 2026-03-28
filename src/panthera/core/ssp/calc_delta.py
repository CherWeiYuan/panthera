"""
Calculate delta scores

This module contain functions to calculate the per-position delta
scores between wild-type and mutant splice site probabilities.
"""

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
    _INDEL_TRANS_TABLE = str.maketrans("", "", "{}")

    # Define __slots__ to drastically reduce memory footprint during
    # multiprocessing
    # Instead of creating a dictionary per class instance, Python just reserve
    # space for these specific items
    __slots__ = [
        "chrom_start",
        "splice_sites",
        "wt_seq",
        "mt_seq",
        "wt_acc",
        "wt_dnr",
        "mt_acc",
        "mt_dnr",
        "reference_pos",
        "aligned_prob",
        "max_raw_delta",
        "max_masked_delta",
        "max_mds_loc",
    ]

    def __init__(
        self,
        chrom_start: int,
        splice_sites: dict[str, list[int]],
        wt_seq: str,
        mt_seq: str,
        wt_acc: npt.NDArray[np.float32],
        wt_dnr: npt.NDArray[np.float32],
        mt_acc: npt.NDArray[np.float32],
        mt_dnr: npt.NDArray[np.float32],
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

        # Validate probabilities to be within 0.0 to 1.0
        # Also ensures all delta scores fall within 0.0 to 1.0
        for name, arr in [
            ("wt_acc", self.wt_acc),
            ("wt_dnr", self.wt_dnr),
            ("mt_acc", self.mt_acc),
            ("mt_dnr", self.mt_dnr),
        ]:
            if np.any((arr < 0.0) | (arr > 1.0)):
                raise ValueError(f"Input {name} contains values outside [0.0, 1.0].")

        # Initialize uncomputed variables with static type hints
        self.reference_pos: list[str] | None = None
        self.aligned_prob: (
            tuple[
                npt.NDArray[np.float32],
                npt.NDArray[np.float32],
                npt.NDArray[np.float32],
                npt.NDArray[np.float32],
            ]
            | None
        ) = None

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

        # --- Prepare Sequence ---
        mt_seq_clean = self.mt_seq.translate(self._INDEL_TRANS_TABLE)

        # --- High-Performance Alignment via Exact Pre-allocation ---
        # The exact final length is the clean sequence minus the skipped
        # literal bases. Since every '>' results in one skipped base, the final
        # length is exactly the length of the string with '>' removed.
        expected_len = len(mt_seq_clean.replace(">", ""))

        new_wt_acc = np.zeros(expected_len, dtype=np.float32)
        new_wt_dnr = np.zeros(expected_len, dtype=np.float32)
        new_mt_acc = np.zeros(expected_len, dtype=np.float32)
        new_mt_dnr = np.zeros(expected_len, dtype=np.float32)
        reference_pos = []

        # Pointers
        wt_idx = 0
        mt_idx = 0
        out_idx = 0
        current_pos = self.chrom_start
        ignore_counter = 0
        valid_bases = {"A", "T", "C", "G", "N"}

        try:
            for n in mt_seq_clean:
                if n in valid_bases:
                    if ignore_counter == 0:
                        new_wt_acc[out_idx] = self.wt_acc[wt_idx]
                        new_wt_dnr[out_idx] = self.wt_dnr[wt_idx]
                        new_mt_acc[out_idx] = self.mt_acc[mt_idx]
                        new_mt_dnr[out_idx] = self.mt_dnr[mt_idx]
                        reference_pos.append(str(current_pos))

                        current_pos += 1
                        wt_idx += 1
                        mt_idx += 1
                        out_idx += 1
                    elif ignore_counter < 0:
                        raise RuntimeError(
                            f"Expect ignore counter >= 0. Got {ignore_counter}."
                        )
                    else:
                        ignore_counter -= 1
                        continue

                elif n == ">":
                    # Insertion mutation
                    new_mt_acc[out_idx] = self.mt_acc[mt_idx]
                    new_mt_dnr[out_idx] = self.mt_dnr[mt_idx]

                    ignore_counter += 1
                    reference_pos.append(f"{current_pos - 1}p{ignore_counter}")

                    # No need to progress wt_idx as it is not used in the
                    # current update

                    # Progress mt_idx as the current one has updated
                    # new_mt_acc and new_mt_dnr
                    mt_idx += 1

                    # Progressing out_idx without updating new_wt_acc and
                    # new_wt_dnr means they are currently assigned 0
                    # (since they are fixed arrays initialized with all zeroes)
                    out_idx += 1

                elif n == "<":
                    # Deletion mutation
                    new_wt_acc[out_idx] = self.wt_acc[wt_idx]
                    new_wt_dnr[out_idx] = self.wt_dnr[wt_idx]

                    reference_pos.append(str(current_pos))

                    # No need to progress mt_idx as it is not used in the
                    # current update

                    current_pos += 1
                    wt_idx += 1

                    # Progressing out_idx without updating new_mt_acc and
                    # new_mt_dnr means they are currently assigned 0
                    # (since they are fixed arrays initialized with all zeroes)
                    out_idx += 1

                else:
                    raise ValueError(
                        f"Expected characters are A/T/C/G/N/>/<. Got '{n}'"
                    )

        except IndexError:
            # Fail-fast mechanism for array length mismatches
            raise IndexError(
                "Probability array length mismatch during alignment. "
                f"Stopped at wt_idx={wt_idx} (max size {self.wt_acc.size}), "
                + f"mt_idx={mt_idx} (max size {self.mt_acc.size})."
            )

        # Final Validation

        # 1. Ensure output length perfectly matches the pre-allocation
        if out_idx != expected_len or len(reference_pos) != expected_len:
            raise ValueError(
                f"Alignment resulted in unexpected length. "
                f"Expected: {expected_len}, Got: out_idx={out_idx}, "
                f"ref_pos={len(reference_pos)}"
            )

        # 2. Ensure ALL input probabilities were completely consumed
        if (
            wt_idx != self.wt_acc.size
            or mt_idx != self.mt_acc.size
            or wt_idx != self.wt_dnr.size
            or mt_idx != self.mt_dnr.size
        ):
            raise ValueError(
                "Not all input probabilities were consumed.\n"
                "---ACCEPTOR---\n"
                f"WT pointer at {wt_idx}/{self.wt_acc.size}.\n"
                f"MT pointer at {mt_idx}/{self.mt_acc.size}.\n"
                "---DONOR---\n"
                f"WT pointer at {wt_idx}/{self.wt_dnr.size}.\n"
                f"MT pointer at {mt_idx}/{self.mt_dnr.size}.\n"
            )

        # 3. Ensure no dangling insertions at the end of the sequence
        if ignore_counter != 0:
            raise ValueError(
                f"Sequence ended with unresolved insertions. "
                f"ignore_counter is {ignore_counter}, expected 0."
            )

        # Update internal state
        self.reference_pos = reference_pos
        self.aligned_prob = (new_wt_acc, new_wt_dnr, new_mt_acc, new_mt_dnr)

    def calc_raw_deltas(self) -> npt.NDArray[np.float32]:
        """
        Calculate raw delta scores.

        Calculates the absolute difference between wild-type and mutant
        probabilities for both acceptor and donor sites, and finds the
        maximum raw delta score across the sequence.

        Returns:
            The maximum raw delta scores (numpy array of float).

        Raises:
            RuntimeError: If align_prob() has not been called prior to this
                          method.
        """
        # Runtime safeguard for users
        if self.aligned_prob is None:
            raise RuntimeError("Must call align_prob() before calc_raw_delta().")

        # Pyright safeguard: hard-narrows the instance attribute
        assert self.aligned_prob is not None
        wt_acc, wt_dnr, mt_acc, mt_dnr = self.aligned_prob

        # --- High-Performance Vectorized Calculation ---
        # Vectorized subtraction and absolute value calculation is
        # orders of magnitude faster than a Python list comprehension.
        raw_acc_deltas = np.abs(wt_acc - mt_acc)
        raw_dnr_deltas = np.abs(wt_dnr - mt_dnr)

        # Get element-wise raw delta score
        raw_deltas = np.maximum(raw_acc_deltas, raw_dnr_deltas)

        return raw_deltas

    def _masked_delta_helper(
        self,
        wt_ssp: npt.NDArray[np.float32],
        mt_ssp: npt.NDArray[np.float32],
        ss_type: Literal["acc", "dnr"],
    ) -> npt.NDArray[np.float32]:
        """
        Calculates masked delta scores using high-performance Numpy vectorization.

        Args:
            wt_ssp: numpy array of wild-type splice site probabilities
            mt_ssp: numpy array of mutant splice site probabilities
            ss_type: type of splice site (acceptor "acc" or donor "dnr")

        Returns:
            numpy array of masked delta scores
        """
        # Ensure reference_pos is available and capture it locally
        if self.reference_pos is None:
            raise RuntimeError(
                "Reference positions unavailable. Call align_prob() first."
            )

        # Vectorized difference: MT - WT
        # Positive values mean MT > WT (increase in probability, e.g., cryptic site)
        # Negative values mean MT < WT (decrease in probability, e.g., disrupted site)
        diff = mt_ssp - wt_ssp

        # Calculate absolute delta for all positions
        delta = np.abs(diff)

        # Create a boolean mask of positions that are known splice sites.
        # We must convert the integer splice sites to strings to match reference_pos
        known_sites = set(str(pos) for pos in self.splice_sites[ss_type])
        is_known_site = np.array([pos in known_sites for pos in self.reference_pos])

        # Apply Vectorized Masking Rules:
        # Rule 1: At known sites, we only care if probability decreases (diff < 0)
        # Rule 2: At unknown sites, we only care if probability increases (diff > 0)

        # Create a boolean array where True means the delta should be KEPT
        keep_mask = np.where(is_known_site, diff < 0, diff > 0)

        # Apply the mask: keep the delta if True, otherwise set to 0.0
        masked_deltas = np.where(keep_mask, delta, 0.0).astype(np.float32)

        return masked_deltas

    def _find_max_delta_locations(
        self, max_deltas: npt.NDArray[np.float32], max_val: float
    ) -> str:
        """
        Finds genomic positions matching the max delta score.

        Args:
            max_deltas: numpy array of max delta scores
            max_val: float of max delta score

        Returns:
            String of max delta locations
        """
        if max_val <= 0.0:
            return ""

        if self.reference_pos is None:
            raise RuntimeError(
                "Reference positions are unavailable. Call align_prob() first."
            )

        # np.where returns sorted, unique indices for 1D arrays automatically
        indices = np.where(np.isclose(max_deltas, max_val, atol=1e-6))[0]

        if indices.size == 0:
            return ""

        # Efficient selection:
        # If reference_pos is a list, we use a list comprehension.
        # If it's a numpy array, self.reference_pos[indices] is faster.
        if isinstance(self.reference_pos, np.ndarray):
            relevant_pos = self.reference_pos[indices]
        else:
            relevant_pos = [self.reference_pos[i] for i in indices]

        return ";".join(relevant_pos)

    def calc_masked_deltas(self) -> npt.NDArray[np.float32]:
        """
        Calculate masked delta scores and update internal state.

        Returns:
            max_masked_delta: Max masked delta score across both acceptor
                              and donor sites.
        """
        # Runtime safeguard for users
        if self.aligned_prob is None:
            raise RuntimeError("Must call align_prob() before calc_masked_delta().")

        # Pyright safeguard: hard-narrows both instance attributes
        assert self.aligned_prob is not None

        wt_acc, wt_dnr, mt_acc, mt_dnr = self.aligned_prob

        # Calculate masked arrays
        masked_acc_deltas = self._masked_delta_helper(wt_acc, mt_acc, "acc")
        masked_dnr_deltas = self._masked_delta_helper(wt_dnr, mt_dnr, "dnr")

        # Get element-wise raw delta score
        masked_deltas = np.maximum(masked_acc_deltas, masked_dnr_deltas)

        return masked_deltas
