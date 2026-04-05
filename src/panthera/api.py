from panthera.core.ssp.ssp_manager import SSPManager

def load_model(model_name: str) -> SSPManager:
    """
    Load the splice site probability model.

    Args:
        model_name: Name of the model ("modelp" or "spliceai").

    Returns:
        SSPManager: SSPManager object.
    """
    manager = SSPManager(model_name, batch_size = 1)
    return manager

def predict(
    seq: str, 
    model: SSPManager
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Predict splice site probabilities for a given sequence.

    Args:
        seq: DNA/RNA sequence.
        model: SSPManager object.

    Returns:
        Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            A tuple containing (acceptor_probs, donor_probs).
    """
    acceptor, donor = model.predict_ssp([seq], ['+'])
    return acceptor[0], donor[0]

