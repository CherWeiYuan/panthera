"""
Model prediction.

This module contain functions for model predictions.
"""

from itertools import islice
import logging
import math
from typing import Any, Callable, Iterable, List, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from panthera.core.ssp.onehotencoder import EncodingSchema, SeqEncoder

# Configure module-level logger
logger = logging.getLogger(__name__)

# Function to round floats within the splice site probability lists
def round_array(
    data: list[npt.NDArray[np.float32]],
    decimals: int = 3) -> list[npt.NDArray[np.float32]]:
    """Rounds all numpy arrays within a nested tuple-list structure.

    Args:
        data: A tuple containing two lists of float32 numpy arrays.
        decimals: The number of decimal places to round to. Defaults to 3.

    Returns:
        A new tuple with the same structure containing rounded arrays.
    """

    return [np.round(arr, decimals) for arr in data]

# --- SpliceAI Prediction --- #
def spliceai_predict(
    seqs: List[str],
    batch_size: int,
    spliceai_fn: Callable[[tf.Tensor], Any],
) -> Tuple[List[npt.NDArray[np.float32]], List[npt.NDArray[np.float32]]]:
    """
    Predicts acceptor and donor site probabilities for a list of
    DNA or RNA sequences using SpliceAI.

    PANTHERA accepts DNA/ RNA sequence as input so:
        - strand input needs to be reverse complemented
        + strand input can be input as it is

    Args:
        seqs: A list of DNA or RNA sequences. Must be all in plus strand.
              If the mRNA is found in the reverse complement of the sequence,
              reverse complement the sequence and then use as input to this
              function.
        batch_size: The number of sequences to process in a single model
                    forward pass.
        spliceai_fn: A loaded TensorFlow ConcreteFunction for SpliceAI.

    Returns:
        A tuple of two elements:
            - acceptor_prob_ndarray: Array of arrays containing acceptor
                                  probabilities per base.
            - donor_prob_ndarray: Array of arrays containing donor
                               probabilities per base.

    Raises:
        ValueError: If input lengths mismatch or an invalid strand is provided.
        RuntimeError: If model prediction fails or sequence loss is detected.
    """
    # Input validation
    if not seqs:
        logger.warning("Empty sequences provided to spliceai_predict.")
        return [], []

    if batch_size < 1:
        batch_size = 1

    # 1. Pre-calculate lengths and setup padding
    seq_lens = np.array([len(s) for s in seqs])
    max_len = seq_lens.max()
    context_pad = 5000
    pad_front = "N" * context_pad

    # Encode and pad sequences
    # pad_front: adds 5000 'N's to the beginning.
    # s: the actual DNA sequence
    # 'N' * (max_len - len(s)): Alignment padding.
    #   If you have two sequences, one 100bp long and one 200bp long,
    #   the 100bp one gets 100 extra 'N's here so that both strings end up
    #   the same length for GPU processing.
    # 'N' * context_pad: adds 5000 'N's to the end.
    #
    # The total length of the string becomes: 5000 + max_len + 5000
    encoded_seqs = [
        SeqEncoder().one_hot_encode(
            f"{pad_front}{s}{'N' * (max_len - len(s))}{'N' * context_pad}",
            EncodingSchema("spliceai"),
        )
        for s in seqs
    ]

    # 2. Batch Prediction
    all_preds = []
    try:
        # Standard list chunking
        for i in range(0, len(encoded_seqs), batch_size):
            batch = encoded_seqs[i : i + batch_size]
            tensor_batch = tf.convert_to_tensor(batch)

            # SpliceAI model returns a tuple/list
            # Index 0 contains the probabilities
            preds = spliceai_fn(tensor_batch)[0]

            # Immediately convert to numpy to free up TF graph memory
            all_preds.append(preds.numpy())

    except Exception as e:
        logger.error(f"Failed during model prediction: {e}")
        raise RuntimeError(f"Model prediction failed: {e}") from e

    # Concatenate all batches into a single contiguous NumPy array
    # Expected shape: (num_seqs, max_len, 3)
    y = np.concatenate(all_preds, axis=0)

    # Check to ensure no loss of the number of sequences
    if len(y) != len(seqs):
        raise RuntimeError(
            f"Sequence loss detected: input {len(seqs)}, output {len(y)}"
        )

    # 3. Vectorized Post-processing
    acceptor_prob_list = []
    donor_prob_list = []

    for i, seq_len in enumerate(seq_lens):
        # Use NumPy slicing
        # Index 1 = Acceptor, Index 2 = Donor
        acc = y[i, :seq_len, 1]
        dnr = y[i, :seq_len, 2]

        # Convert back to standard python lists only at the very end
        acceptor_prob_list.append(acc)
        donor_prob_list.append(dnr)

        # Ensure equal length of sequence and its splice site probabilities
        if not (seq_len == len(acc) == len(dnr)):
            logger.error(
                f"Shape mismatch: seq({seq_len}), "
                + f"acc({len(acc)}), dnr({len(dnr)})"
            )
            raise RuntimeError(
                "Prediction output length mismatch for sequence of length "
                + f"{seq_len}. Got Acceptor: {len(acc)}, Donor: {len(dnr)}."
            )

    # Round to 3 decimals
    acceptor_prob_list = round_array(acceptor_prob_list, 3)
    donor_prob_list = round_array(donor_prob_list, 3)

    return acceptor_prob_list, donor_prob_list


# --- ModelP Prediction --- #
def modelp_predict(
    seqs: List[str],
    batch_size: int,
    modelp_fn: Callable,
    crop_len: int = 1000,
    model_input_len: int = 3000,
    model_output_len: int = 1000,
) -> Tuple[List[npt.NDArray[np.float32]], List[npt.NDArray[np.float32]]]:
    """
    Highly optimized splice site prediction using dynamic batch padding.

    PANTHERA accepts DNA/ RNA sequence as input so:
        - strand input needs to be reverse complemented
        + strand input can be input as it is


    Args
        seqs: A list of DNA or RNA sequences. Must be all in plus strand.
              If the mRNA is found in the reverse complement of the sequence,
              reverse complement the sequence and then use as input to this
              function.
        batch_size: The number of sequences to process in a single model
                    forward pass.
        modelp_fn: Loaded frozen model graphs function for ModelP.
        crop_len: Number of positions removed from each end of the sequence by
                  the prediction model. Default: 1000.
        model_input_len: Length of input sequence. Default: 3000.
        model_output_len: Length of output sequence. Default: 1000.

    Returns
        A tuple of two elements:
            - acceptor_prob_ndarray: Array of arrays containing acceptor
                                  probabilities per base.
            - donor_prob_ndarray: Array of arrays containing donor
                               probabilities per base.
    """
    # Input validation
    if not seqs:
        logger.warning("Empty sequences provided to modelp_predict.")
        return [], []

    # Internal function for batching
    def _batched(iterable: Iterable, size: int) -> Iterable[Tuple]:
        """Yield successive n-sized chunks from iterable."""
        # Make iterating generator
        itrb = iter(iterable)

        # islice grabs the next size items from the generator
        # Walrus operator ":=" simultaneously does two things:
        #   1. Assigns tuple of islice output to batch
        #   2. Checks if batch is empty. If batch is empty, the loop terminates
        while batch := tuple(islice(itrb, size)):
            yield batch

    # Iterate through batches, padding ONLY to the
    # max length of the current batch
    final_acceptors = []
    final_donors = []
    for batch_idx, seq_batch in enumerate(_batched(seqs, batch_size)):
        logger.debug(f"Processing batch {batch_idx + 1}...")

        # 1. Dynamic Batch Padding

        # Calculate max length of all sequence in batch
        batch_max_seq_len = max(len(s) for s in seq_batch)

        # "rounding trick" to figure out exactly how long the sequence
        # needs to be so that it can be chopped into perfect, equal-sized blocks
        # (windows) for the model to process.

        # "batch_max_seq_len + crop_len": "Minimum Required Space."
        #   We take the longest DNA string in your current batch and add the
        #   crop_len (the extra 'N' padding at the start that the model needs
        #   for context).
        # "/ model_output_len": We divide that total length by the size of the
        #   model's prediction window. This tells us how many "windows"
        #   (including fractional ones) we need.
        # "math.ceil(...)"": This is the crucial part. If we need 2.1 windows,
        #   we can't just ignore that 0.1—the model would miss the end of your
        #   sequence. ceil (ceiling) rounds up to the next whole number
        #   (e.g., 3).
        # "* model_output_len": we multiply that whole number back by the
        #   window size to get the final padded length in bases.
        batch_max_padded = (
            math.ceil((batch_max_seq_len + crop_len) / model_output_len)
            * model_output_len
        )

        # Calculate max length of padded sequence
        batch_max_len = max(model_input_len, batch_max_padded)

        # Pad all sequence in batch
        # The length padded on the left is crop_len
        # The length padded on the right depends on batch_max_len
        padded_seq_batch = [
            (("N" * crop_len) + s).ljust(batch_max_len, "N") for s in seq_batch
        ]

        # 2. Sliding window over the batch
        batch_predictions = []
        for i in range(0, batch_max_len, model_output_len):
            window_seqs = [s[i : i + model_input_len] for s in padded_seq_batch]

            # One-hot encoding
            encoded_subseqs = [
                SeqEncoder().one_hot_encode(x, EncodingSchema("modelp"))
                for x in window_seqs
            ]

            # Convert to tensor
            subseq_tensor = tf.convert_to_tensor(encoded_subseqs, dtype=tf.float32)

            # Add entry to batch_predictions list
            if subseq_tensor.shape[1] == model_input_len:
                batch_predictions.append(modelp_fn(subseq_tensor)[0].numpy())
            else:
                break

        # Shape: (batch_size, sequence_length, 3)
        concatenated_preds = np.concatenate(batch_predictions, axis=1)

        # 3. Handle the exact trailing window by predicting it again with
        # full sequence context
        last_window_encoded = []
        for s in seq_batch:
            # Extract the last window of sequence
            last_window = s[-model_output_len:] if len(s) >= model_output_len else s

            # If window is small, pad both ends
            if len(last_window) < (model_input_len - crop_len):
                last_window = (
                    last_window.rjust(crop_len + len(last_window), "N")
                ).ljust(model_input_len, "N")

            # If window is large, pad N to the right side with ljust
            else:
                last_window = last_window.ljust(model_input_len, "N")
            last_window_encoded.append(
                SeqEncoder().one_hot_encode(last_window, EncodingSchema("modelp"))
            )

        last_window_tensor = tf.convert_to_tensor(last_window_encoded, dtype=tf.float32)
        last_preds = modelp_fn(last_window_tensor)[0].numpy()

        # 4. Vectorized Parsing Reversal
        for (
            i,
            seq,
        ) in enumerate(seq_batch):
            seq_len = len(seq)

            # Stitch main predictions and last window
            main_pred = concatenated_preds[i, : (seq_len - model_output_len)]
            tail_pred = last_preds[i]

            # Combine and slice to exact original sequence length
            full_pred = np.concatenate([main_pred, tail_pred], axis=0)[:seq_len]

            # Vectorized extraction of acceptor (index 0) and donor (index 1)
            acc = full_pred[:, 0]
            dnr = full_pred[:, 1]

            final_acceptors.append(acc)
            final_donors.append(dnr)

    # Round to 3 decimals
    final_acceptors = round_array(final_acceptors, 3)
    final_donors = round_array(final_donors, 3)

    return final_acceptors, final_donors
