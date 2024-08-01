import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any
from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger

class ObservationSpace:
    """
    Defines the observation space for the RL environment.
    
    This class creates and manages the observation space, which represents
    the possible observations that the RL agent can receive from the environment.
    It uses an autoencoder to reduce the dimensionality of the raw state data.
    """
    
    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.latent_dim = self.config.get('latent_dim', 64)
        self.space = self._create_space()

    def _create_space(self) -> gym.Space:
        """
        Create the observation space based on the latent dimension.
        
        Returns:
            gym.Space: The defined observation space.
        """
        self.logger.info(f"Creating observation space with latent dimension {self.latent_dim}")
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32)

    def sample(self) -> np.ndarray:
        """
        Sample a random observation from the space.
        
        Returns:
            np.ndarray: A random observation.
        """
        return self.space.sample()

    def contains(self, x: np.ndarray) -> bool:
        """
        Check if the given observation is within the observation space.
        
        Args:
            x (np.ndarray): The observation to check.
        
        Returns:
            bool: True if the observation is within the space, False otherwise.
        """
        return self.space.contains(x)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the lower and upper bounds of the observation space.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The lower and upper bounds.
        """
        return self.space.low, self.space.high

    def get_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the observation space.
        
        Returns:
            Tuple[int, ...]: The shape of the observation space.
        """
        return self.space.shape

    def encode_raw_state(self, raw_state: Dict[str, Any], encoder) -> np.ndarray:
        """
        Encode the raw state into the latent space using the provided encoder.
        
        Args:
            raw_state (Dict[str, Any]): The raw state data.
            encoder: The encoder part of the autoencoder.
        
        Returns:
            np.ndarray: The encoded state in the latent space.
        """
        # This is a placeholder implementation. You'll need to implement the actual encoding logic.
        # The exact implementation will depend on how your raw state is structured and how your encoder works.
        self.logger.debug("Encoding raw state into latent space")
        # Example: Assume raw_state is a flattened numpy array
        raw_array = np.array(list(raw_state.values())).flatten()
        return encoder.predict(raw_array.reshape(1, -1))[0]

    def decode_latent_state(self, latent_state: np.ndarray, decoder) -> Dict[str, Any]:
        """
        Decode the latent state back into the raw state space using the provided decoder.
        
        Args:
            latent_state (np.ndarray): The state in the latent space.
            decoder: The decoder part of the autoencoder.
        
        Returns:
            Dict[str, Any]: The decoded raw state.
        """
        # This is a placeholder implementation. You'll need to implement the actual decoding logic.
        self.logger.debug("Decoding latent state into raw state space")
        # Example: Assume the decoder output is a flattened array that needs to be reshaped
        decoded_array = decoder.predict(latent_state.reshape(1, -1))[0]
        # You'll need to define how to convert this back into your raw state structure
        return {"decoded_state": decoded_array}

def main():
    # Setup
    config = OverseerConfig()
    obs_space = ObservationSpace(config)
    logger = OverseerLogger().get_logger("ObservationSpaceTest")

    # Test creation
    logger.info(f"Observation space created: {obs_space.space}")

    # Test sampling
    sample = obs_space.sample()
    logger.info(f"Sampled observation: {sample}")

    # Test contains
    logger.info(f"Sample in space: {obs_space.contains(sample)}")

    # Test bounds
    low, high = obs_space.get_bounds()
    logger.info(f"Observation space bounds: Low: {low}, High: {high}")

    # Test shape
    logger.info(f"Observation space shape: {obs_space.get_shape()}")

    # Test encoding and decoding (you'll need to implement a mock encoder/decoder for this)
    class MockEncoder:
        def predict(self, x):
            return np.zeros((1, obs_space.latent_dim))

    class MockDecoder:
        def predict(self, x):
            return np.zeros((1, 100))  # Assuming raw state is 100-dimensional

    mock_encoder = MockEncoder()
    mock_decoder = MockDecoder()

    raw_state = {"feature1": np.random.rand(10, 10), "feature2": np.random.rand(5, 5)}
    encoded_state = obs_space.encode_raw_state(raw_state, mock_encoder)
    logger.info(f"Encoded state: {encoded_state}")

    decoded_state = obs_space.decode_latent_state(encoded_state, mock_decoder)
    logger.info(f"Decoded state: {decoded_state}")

if __name__ == "__main__":
    main()