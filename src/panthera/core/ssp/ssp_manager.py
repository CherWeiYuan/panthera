"""
Splice site probability prediction manager.

This module contains the class to manage splice site probability prediction.
"""

from panthera.core.ssp.load_model import load_frozen_graph
from panthera.core.ssp.predict import spliceai_predict, modelp_predict


class SSPManager:
    """
    Manager of splice site probability (SSP) prediction.
    """
    def __init__(self, model_name : Literal["modelp", "spliceai"]):
        self.model_name = model_name
    
    def predict_ssp(self, seq):
        """
        Splice site probability prediction (SSP)

        Args
            seq: DNA or RNA sequence (will be uppercased and striped by
                 onehotencoder in prediction functions)
        
        Returns
            A tuple of two elements:
            - acceptor_prob_list: List of lists containing acceptor probabilities per base.
            - donor_prob_list: List of lists containing donor probabilities per base.
        """
        if self.model_name == "modelp":
