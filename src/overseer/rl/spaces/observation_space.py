import gymnasium as gym
import numpy as np

class ObservationSpace:
    """
    Defines the observation space for the RL environment.
    
    This class creates and manages the observation space, which represents
    the possible observations that the RL agent can receive from the environment.
    """
    
    @staticmethod
    def create(dim_reduction_model):
        """
        Create the observation space based on the dimensionality reduction model.
        
        Args:
            dim_reduction_model: The dimensionality reduction model used in the environment.
        
        Returns:
            gym.Space: The defined observation space.
        """
        # Define the shape and bounds of the observation space
        # This should match the output of the dimensionality reduction model
        latent_dim = dim_reduction_model.latent_dim
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)