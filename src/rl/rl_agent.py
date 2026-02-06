"""
Reinforcement Learning Agent
Implements FR-5: Reinforcement Learning Engine and FR-10: Policy Update Mechanism
"""

import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Optional, Type
import os

from .reward_function import RewardFunction
from simulator.metrics import PerformanceMetrics


class RLAgent:
    """
    Adaptive Reinforcement Learning Agent for Quantum Communication
    
    Implements:
    - FR-5: RL Engine that learns optimal policies
    - FR-10: Policy update mechanism based on rewards
    - FR-4: State representation for the RL agent
    """
    
    def __init__(self, env, config: Dict):
        """
        Initialize RL agent
        
        Args:
            env: Quantum communication environment
            config: RL configuration parameters
        """
        self.config = config
        self.env = env
        
        # Wrap environment for monitoring
        self.monitor_env = Monitor(env)
        self.vec_env = DummyVecEnv([lambda: self.monitor_env])
        
        # Select algorithm
        algorithm_name = config.get('algorithm', 'PPO')
        self.algorithm_class = self._get_algorithm_class(algorithm_name)
        
        # Initialize reward function
        reward_config = config.get('reward', {})
        self.reward_function = RewardFunction(reward_config)
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        
        # Create model
        self.model = self._create_model()
        
        # Training state
        self.total_timesteps = 0
        self.best_mean_reward = -np.inf
        
    def _get_algorithm_class(self, algorithm_name: str) -> Type:
        """Get the RL algorithm class"""
        algorithms = {
            'PPO': PPO,
            'A2C': A2C,
            'DQN': DQN,
            'SAC': SAC
        }
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                           f"Available: {list(algorithms.keys())}")
        
        return algorithms[algorithm_name]
    
    def _create_model(self):
        """Create the RL model with configured parameters"""
        training_config = self.config.get('training', {})
        network_config = self.config.get('network', {})
        
        # Common parameters
        common_params = {
            'policy': network_config.get('policy_type', 'MlpPolicy'),
            'env': self.vec_env,
            'learning_rate': training_config.get('learning_rate', 3e-4),
            'verbose': self.config.get('advanced', {}).get('verbose', 1),
            'device': self.config.get('advanced', {}).get('device', 'auto'),
            'seed': self.config.get('advanced', {}).get('seed', 42)
        }
        
        # Algorithm-specific parameters
        if self.algorithm_class == PPO:
            ppo_config = self.config.get('ppo', {})
            params = {
                **common_params,
                'n_steps': ppo_config.get('n_steps', 2048),
                'batch_size': training_config.get('batch_size', 64),
                'n_epochs': training_config.get('n_epochs', 10),
                'gamma': ppo_config.get('gamma', 0.99),
                'gae_lambda': ppo_config.get('gae_lambda', 0.95),
                'clip_range': ppo_config.get('clip_range', 0.2),
                'ent_coef': ppo_config.get('ent_coef', 0.01),
                'vf_coef': ppo_config.get('vf_coef', 0.5),
                'max_grad_norm': ppo_config.get('max_grad_norm', 0.5)
            }
        else:
            params = common_params
        
        # Set policy network architecture
        policy_kwargs = {
            'net_arch': network_config.get('net_arch', [256, 256, 128]),
            'activation_fn': self._get_activation_function(
                network_config.get('activation', 'tanh')
            )
        }
        params['policy_kwargs'] = policy_kwargs
        
        return self.algorithm_class(**params)
    
    def _get_activation_function(self, activation_name: str):
        """Get activation function"""
        activations = {
            'tanh': torch.nn.Tanh,
            'relu': torch.nn.ReLU,
            'elu': torch.nn.ELU
        }
        return activations.get(activation_name, torch.nn.Tanh)
    
    def train(self, timesteps: Optional[int] = None, 
              eval_freq: Optional[int] = None,
              save_path: str = 'results/model'):
        """
        Train the RL agent (FR-10: Policy Update Mechanism)
        
        Args:
            timesteps: Number of timesteps to train
            eval_freq: Frequency of evaluation
            save_path: Path to save model checkpoints
        """
        training_config = self.config.get('training', {})
        
        if timesteps is None:
            timesteps = training_config.get('total_timesteps', 100000)
        
        if eval_freq is None:
            eval_freq = training_config.get('eval_freq', 5000)
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f"{save_path}/best_model", exist_ok=True)
        
        # Setup callbacks
        eval_callback = EvalCallback(
            self.vec_env,
            best_model_save_path=f"{save_path}/best_model",
            log_path=f"{save_path}/logs",
            eval_freq=eval_freq,
            n_eval_episodes=training_config.get('n_eval_episodes', 10),
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=training_config.get('save_freq', 10000),
            save_path=f"{save_path}/checkpoints",
            name_prefix='rl_model'
        )
        
        callbacks = [eval_callback, checkpoint_callback]
        
        # Train
        print(f"Starting training for {timesteps} timesteps...")
        self.model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            tb_log_name=self.config.get('algorithm', 'PPO')
        )
        
        self.total_timesteps += timesteps
        print(f"Training complete! Total timesteps: {self.total_timesteps}")
    
    def predict(self, observation, deterministic: bool = True):
        """
        Predict action given observation
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Selected action and additional info
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate the agent's performance
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = []
        episode_fidelities = []
        episode_success_rates = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            episode_fidelities.append(info.get('avg_fidelity', 0.0))
            
            success_rate = info.get('successful_transmissions', 0) / max(1, info.get('step', 1))
            episode_success_rates.append(success_rate)
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_fidelity': float(np.mean(episode_fidelities)),
            'mean_success_rate': float(np.mean(episode_success_rates)),
            'num_episodes': num_episodes
        }
    
    def save(self, path: str):
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load a trained model
        
        Args:
            path: Path to load the model from
        """
        self.model = self.algorithm_class.load(path, env=self.vec_env)
        print(f"Model loaded from {path}")
    
    def get_policy_parameters(self) -> Dict:
        """Get current policy parameters"""
        return {
            'total_timesteps': self.total_timesteps,
            'algorithm': self.config.get('algorithm'),
            'learning_rate': self.model.learning_rate,
        }
    
    def update_learning_rate(self, new_lr: float):
        """
        Update the learning rate dynamically
        
        Args:
            new_lr: New learning rate value
        """
        self.model.learning_rate = new_lr
        print(f"Learning rate updated to {new_lr}")
    
    def __repr__(self):
        return (f"RLAgent(algorithm={self.config.get('algorithm')}, "
                f"timesteps={self.total_timesteps})")
