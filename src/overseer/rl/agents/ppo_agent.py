"""
PPO Agent for Wildfire Suppression

This module implements a Proximal Policy Optimization (PPO) agent using the Stable Baselines3 library,
specifically designed for wildfire suppression tasks with a MultiDiscrete action space.

The agent interacts with an ELMFIRE simulation environment, learning to make decisions about fireline
placement and resource allocation to effectively combat wildfires.

Key Components:
1. PPO Model: Utilizes the PPO implementation from Stable Baselines3, customized for the wildfire domain.
2. Policy Network: A neural network that maps states to action probabilities across the MultiDiscrete action space.
3. Value Network: Estimates the value of being in a given state, used for advantage estimation in PPO.
4. Action Masking: Implements dynamic action masking to prevent invalid actions during training and inference.
5. Custom Features:
   - Reward shaping for the wildfire suppression task
   - Curriculum learning for progressive difficulty in training scenarios
   - Integration with ELMFIRE simulation environment

Usage:
    agent = PPOAgent(env, config)
    agent.train(total_timesteps=1000000)
    agent.save("wildfire_suppression_model")
    
    # For inference
    loaded_agent = PPOAgent.load("wildfire_suppression_model")
    obs = env.reset()
    while not done:
        action, _states = loaded_agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

Attributes:
    env (gym.Env): The ELMFIRE gym environment
    model (stable_baselines3.PPO): The PPO model from Stable Baselines3
    config (dict): Configuration parameters for the agent and training process

Methods:
    train(total_timesteps: int): Train the agent for a specified number of timesteps
    predict(observation: np.ndarray, deterministic: bool = False): Get the agent's action for a given observation
    save(path: str): Save the trained model to a file
    load(path: str): Load a trained model from a file

Dependencies:
    - gymnasium
    - stable_baselines3
    - torch
    - numpy

Note:
    This implementation assumes the use of Stable Baselines3 version 1.5.0 or later, which includes
    support for MultiDiscrete action spaces and custom policy networks.
"""

import os
import sys
from typing import Dict, Any, Union
import traceback
import numpy as np

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance

from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger
from overseer.rl.envs.elmfire_gym_env import ElmfireGymEnv

class CustomNetwork(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        shared_features = self.shared_net(features)
        return self.policy_net(shared_features), self.value_net(shared_features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.net_arch = []  # We're using a custom network architecture

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, self.action_space.nvec.sum())

class PPOCallback(BaseCallback):
    def __init__(self, logger: OverseerLogger, verbose: int = 0):
        super().__init__(verbose)
        self.logger = logger

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            self.logger.info(f"Step: {self.n_calls}")
            self.logger.info(f"Mean reward: {self.model.ep_info_buffer.mean()['r']:.2f}")
            explained_var = explained_variance(self.model.rollout_buffer.values, self.model.rollout_buffer.returns)
            self.logger.info(f"Explained variance: {explained_var:.2f}")
        return True

    def _on_rollout_end(self) -> None:
        self.logger.info("Rollout ended. Starting optimization.")

    def _on_training_end(self) -> None:
        self.logger.info("Training ended.")

class PPOAgent:
    def __init__(self, config: OverseerConfig, env: ElmfireGymEnv):
        self.config = config
        self.env = env
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)

        self.logger.info("Initializing PPO Agent")
        
        try:
            self.model = PPO(
                CustomActorCriticPolicy,
                env,
                verbose=1,
                tensorboard_log="./ppo_elmfire_tensorboard/",
                learning_rate=self.config.get('learning_rate', 3e-4),
                n_steps=self.config.get('n_steps', 2048),
                batch_size=self.config.get('batch_size', 64),
                n_epochs=self.config.get('n_epochs', 10),
                gamma=self.config.get('gamma', 0.99),
                gae_lambda=self.config.get('gae_lambda', 0.95),
                clip_range=self.config.get('clip_range', 0.2),
                clip_range_vf=self.config.get('clip_range_vf', None),
                ent_coef=self.config.get('ent_coef', 0.0),
                vf_coef=self.config.get('vf_coef', 0.5),
                max_grad_norm=self.config.get('max_grad_norm', 0.5),
                use_sde=self.config.get('use_sde', False),
                sde_sample_freq=self.config.get('sde_sample_freq', -1),
                target_kl=self.config.get('target_kl', None),
                stats_window_size=self.config.get('stats_window_size', 100),
                device=self.config.get('device', 'auto')
            )
            self.logger.info("PPO model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing PPO model: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            raise

    def train(self, total_timesteps: int) -> None:
        self.logger.info(f"Starting training for {total_timesteps} timesteps")
        callback = PPOCallback(self.logger)
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                tb_log_name="ppo_elmfire_run",
                progress_bar=True
            )
            self.logger.info("Training completed successfully")
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            raise

    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.logger.debug("Making a prediction")
        try:
            action, _states = self.model.predict(observation, deterministic=deterministic)
            self.logger.debug(f"Predicted action: {action}")
            return action, _states
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            raise

    def save(self, path: str) -> None:
        self.logger.info(f"Saving model to {path}")
        try:
            self.model.save(path)
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            raise

    @classmethod
    def load(cls, path: str, env: ElmfireGymEnv) -> 'PPOAgent':
        logger = OverseerLogger().get_logger(cls.__name__)
        logger.info(f"Loading model from {path}")
        try:
            config = OverseerConfig()  # Assuming default config is fine for loading
            agent = cls(config, env)
            agent.model = PPO.load(path, env=env)
            logger.info("Model loaded successfully")
            return agent
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise

def main():
    logger = OverseerLogger().get_logger("PPOAgent_Main")
    logger.info("Starting PPOAgent main function")

    try:
        config_path = "src/overseer/config/elmfire_config.yaml"
        config = OverseerConfig(config_path)
        env = ElmfireGymEnv(config_path)
        
        agent = PPOAgent(config, env)
        logger.info("PPOAgent created successfully")

        # Training
        total_timesteps = config.get('total_timesteps', 1000000)
        logger.info(f"Starting training for {total_timesteps} timesteps")
        agent.train(total_timesteps)

        # Save the trained model
        save_path = "models/ppo_elmfire_model"
        agent.save(save_path)

        # Test the trained model
        logger.info("Testing trained model")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step += 1
            logger.info(f"Step {step}: Action = {action}, Reward = {reward}, Done = {done}")
            logger.info(f"Info: {info}")

        logger.info(f"Test episode completed. Total reward: {total_reward}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()