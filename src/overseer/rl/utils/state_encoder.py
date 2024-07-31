import numpy as np
from overseer.config import OverseerConfig

class StateEncoder:
    """
    Encodes ELMFIRE simulator states into RL agent observations.
    
    This class uses a dimensionality reduction model to encode high-dimensional
    ELMFIRE states into lower-dimensional representations suitable for RL agents.
    
    Attributes:
        dim_reduction_model (ConvolutionalAutoencoder): The dimensionality reduction model.
    """
    
    def __init__(self, dim_reduction_model):
        """
        Initialize the StateEncoder.
        
        Args:
            dim_reduction_model: An instance of the ConvolutionalAutoencoder class.
        """
        self.dim_reduction_model = dim_reduction_model
    
    def encode(self, state):
        """
        Encode an ELMFIRE state into an RL observation.
        
        This method applies dimensionality reduction to the ELMFIRE state and
        performs any necessary preprocessing.
        
        Args:
            state: The raw state from the ELMFIRE simulator.
        
        Returns:
            numpy.ndarray: The encoded state (observation) for the RL agent.
        """
        # Preprocess the state if necessary
        preprocessed_state = self._preprocess_state(state)
        
        # Apply dimensionality reduction
        encoded_state = self.dim_reduction_model.encode(preprocessed_state)
        
        return encoded_state
    
    def _preprocess_state(self, state):
        """
        Preprocess the raw ELMFIRE state before encoding.
        
        This method can be used to normalize, scale, or transform the raw state
        before applying dimensionality reduction.
        
        Args:
            state: The raw state from the ELMFIRE simulator.
        
        Returns:
            The preprocessed state.
        """
        # Implement preprocessing logic here
        # For example, normalizing or scaling certain features
        return state  # Placeholder, replace with actual preprocessing
    
    