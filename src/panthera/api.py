from typing import Tuple, Literal
import numpy as np
import numpy.typing as npt
from panthera.core.ssp.ssp_manager import SSPManager


def load_model(model_name: Literal["modelp", "spliceai"]) -> SSPManager:
    """
    Load the splice site probability model.

    Args:
        model_name: Name of the model ("modelp" or "spliceai").

    Returns:
        SSPManager: SSPManager object.
    """
    manager = SSPManager(model_name, batch_size=1)
    return manager


def predict(
    seq: str, model: SSPManager
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Predict splice site probabilities for a given sequence.

    Args:
        seq: DNA/RNA sequence.
        model: SSPManager object.

    Returns:
        Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            A tuple containing (acceptor_probs, donor_probs).

    Raises:
        TypeError: If seq is not a string or model is not a SSPManager object.

    Example:
        >>> from panthera.api import load_model, predict
        >>> model = load_model("modelp")
        >>> acceptor, donor = predict("GUAG", model)
        >>> print(acceptor)
        [0.0001 0.0001 0.0001 0.0001]
        >>> print(donor)
        [0.0001 0.0001 0.0001 0.0001]
    """
    # Input validation
    if not isinstance(seq, str):
        raise TypeError("Sequence must be a string.")
    if not isinstance(model, SSPManager):
        raise TypeError("Model must be a SSPManager object.")

    # Predict
    acceptor, donor = model.predict_ssp([seq], reverse_output=False)
    return acceptor[0], donor[0]
