"""
Policy Network for RL Agent
Custom policy networks for quantum communication optimization
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np
from typing import Optional


class PolicyNetwork(nn.Module):
    """
    Custom policy network for quantum communication
    
    Processes state representation (FR-4: State Representation)
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: list = [256, 256, 128]):
        """
        Initialize policy network
        
        Args:
            input_dim: Dimension of input state
            output_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class ValueNetwork(nn.Module):
    """
    Value network for actor-critic algorithms
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 256]):
        """
        Initialize value network
        
        Args:
            input_dim: Dimension of input state
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class QuantumStateExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for quantum state representation
    
    Implements FR-4: State Representation using current and historical data
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        """
        Initialize feature extractor
        
        Args:
            observation_space: Environment observation space
            features_dim: Dimension of extracted features
        """
        super().__init__(observation_space, features_dim)
        
        input_dim = int(np.prod(observation_space.shape))
        
        # Feature extraction network
        self.extractor = nn.Sequential(
            # First layer: expand representation
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            
            # Second layer: process features
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            
            # Third layer: compress to feature dim
            nn.Linear(256, features_dim),
            nn.Tanh()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations
        
        Args:
            observations: Raw observations from environment
            
        Returns:
            Extracted feature tensor
        """
        return self.extractor(observations)


class AdaptivePolicy:
    """
    Adaptive policy that can switch strategies based on channel conditions
    
    Implements FR-6: Adaptive Protocol Control
    """
    
    def __init__(self, num_strategies: int = 3):
        """
        Initialize adaptive policy
        
        Args:
            num_strategies: Number of different strategies to maintain
        """
        self.num_strategies = num_strategies
        self.strategy_weights = np.ones(num_strategies) / num_strategies
        self.strategy_performance = np.zeros(num_strategies)
        self.strategy_counts = np.zeros(num_strategies)
        
    def select_strategy(self, channel_state: np.ndarray) -> int:
        """
        Select strategy based on channel state
        
        Args:
            channel_state: Current channel state information
            
        Returns:
            Strategy index
        """
        # Use softmax selection based on weights
        exp_weights = np.exp(self.strategy_weights - np.max(self.strategy_weights))
        probabilities = exp_weights / np.sum(exp_weights)
        
        strategy_idx = np.random.choice(self.num_strategies, p=probabilities)
        return int(strategy_idx)
    
    def update_strategy_performance(self, strategy_idx: int, performance: float):
        """
        Update strategy performance based on outcomes
        
        Args:
            strategy_idx: Strategy that was used
            performance: Performance metric (reward)
        """
        self.strategy_counts[strategy_idx] += 1
        
        # Update running average
        alpha = 0.1  # Learning rate
        self.strategy_performance[strategy_idx] = (
            (1 - alpha) * self.strategy_performance[strategy_idx] + 
            alpha * performance
        )
        
        # Update weights based on performance
        self.strategy_weights = self.strategy_performance / (
            np.sum(self.strategy_performance) + 1e-8
        )
    
    def get_best_strategy(self) -> int:
        """Get the currently best performing strategy"""
        return int(np.argmax(self.strategy_performance))
    
    def reset(self):
        """Reset strategy tracking"""
        self.strategy_weights = np.ones(self.num_strategies) / self.num_strategies
        self.strategy_performance = np.zeros(self.num_strategies)
        self.strategy_counts = np.zeros(self.num_strategies)


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy exploration policy
    """
    
    def __init__(self, epsilon_start: float = 1.0, 
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995):
        """
        Initialize epsilon-greedy policy
        
        Args:
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate per step
        """
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy strategy
        
        Args:
            q_values: Q-values for each action
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(len(q_values))
        else:
            # Exploit: best action
            action = np.argmax(q_values)
        
        return int(action)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset(self, epsilon: Optional[float] = None):
        """Reset epsilon value"""
        if epsilon is not None:
            self.epsilon = epsilon


class BoltzmannPolicy:
    """
    Boltzmann (softmax) exploration policy
    """
    
    def __init__(self, temperature: float = 1.0, 
                 temperature_decay: float = 0.99,
                 min_temperature: float = 0.1):
        """
        Initialize Boltzmann policy
        
        Args:
            temperature: Initial temperature parameter
            temperature_decay: Temperature decay rate
            min_temperature: Minimum temperature
        """
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        
    def select_action(self, q_values: np.ndarray) -> int:
        """
        Select action using Boltzmann distribution
        
        Args:
            q_values: Q-values for each action
            
        Returns:
            Selected action index
        """
        # Apply softmax with temperature
        exp_values = np.exp((q_values - np.max(q_values)) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        action = np.random.choice(len(q_values), p=probabilities)
        return int(action)
    
    def decay_temperature(self):
        """Decay temperature"""
        self.temperature = max(self.min_temperature, 
                              self.temperature * self.temperature_decay)
