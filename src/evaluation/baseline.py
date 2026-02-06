"""
Baseline Strategies for Comparison
Implements non-adaptive baseline strategies for performance comparison
"""

import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod


class BaselineStrategy(ABC):
    """
    Abstract base class for baseline strategies
    """
    
    @abstractmethod
    def select_action(self, observation: np.ndarray, 
                     channel_state: Optional[Dict] = None) -> int:
        """Select action based on observation"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset strategy state"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass


class StaticProtocol(BaselineStrategy):
    """
    Static protocol that uses fixed parameters
    No adaptation to channel conditions
    """
    
    def __init__(self, config: Dict):
        """
        Initialize static protocol
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.action_sequence = [1, 3, 2]  # Generate, Purify, Swap
        self.step = 0
        
    def select_action(self, observation: np.ndarray,
                     channel_state: Optional[Dict] = None) -> int:
        """
        Select action from predefined sequence
        
        Args:
            observation: Environment observation
            channel_state: Channel state information (ignored in static)
            
        Returns:
            Action index
        """
        action = self.action_sequence[self.step % len(self.action_sequence)]
        self.step += 1
        return action
    
    def reset(self):
        """Reset step counter"""
        self.step = 0
    
    def get_name(self) -> str:
        return "Static Protocol"


class RandomProtocol(BaselineStrategy):
    """
    Random action selection baseline
    """
    
    def __init__(self, num_actions: int = 10):
        """
        Initialize random protocol
        
        Args:
            num_actions: Number of possible actions
        """
        self.num_actions = num_actions
        
    def select_action(self, observation: np.ndarray,
                     channel_state: Optional[Dict] = None) -> int:
        """
        Select random action
        
        Args:
            observation: Environment observation (ignored)
            channel_state: Channel state (ignored)
            
        Returns:
            Random action index
        """
        return np.random.randint(0, self.num_actions)
    
    def reset(self):
        """Nothing to reset"""
        pass
    
    def get_name(self) -> str:
        return "Random Protocol"


class GreedyProtocol(BaselineStrategy):
    """
    Greedy protocol that always tries to maximize immediate fidelity
    """
    
    def __init__(self):
        """Initialize greedy protocol"""
        self.last_fidelity = 0.0
        
    def select_action(self, observation: np.ndarray,
                     channel_state: Optional[Dict] = None) -> int:
        """
        Select action greedily based on estimated fidelity
        
        Args:
            observation: Environment observation
            channel_state: Channel state information
            
        Returns:
            Greedy action
        """
        # Extract fidelity from observation (simplified)
        if len(observation) > 0:
            avg_fidelity = np.mean(observation)
        else:
            avg_fidelity = 0.5
        
        # Decision logic
        if avg_fidelity < 0.7:
            # Low fidelity: generate new entanglement
            action = 1
        elif avg_fidelity < 0.9:
            # Medium fidelity: purify
            action = 3
        else:
            # High fidelity: swap
            action = 2
        
        self.last_fidelity = avg_fidelity
        return action
    
    def reset(self):
        """Reset fidelity tracking"""
        self.last_fidelity = 0.0
    
    def get_name(self) -> str:
        return "Greedy Protocol"


class ThresholdBasedProtocol(BaselineStrategy):
    """
    Threshold-based protocol with fixed decision rules
    """
    
    def __init__(self, 
                 purify_threshold: float = 0.8,
                 swap_threshold: float = 0.95):
        """
        Initialize threshold-based protocol
        
        Args:
            purify_threshold: Fidelity threshold to trigger purification
            swap_threshold: Fidelity threshold to trigger swapping
        """
        self.purify_threshold = purify_threshold
        self.swap_threshold = swap_threshold
        self.step = 0
        
    def select_action(self, observation: np.ndarray,
                     channel_state: Optional[Dict] = None) -> int:
        """
        Select action based on thresholds
        
        Args:
            observation: Environment observation
            channel_state: Channel state information
            
        Returns:
            Action based on threshold rules
        """
        self.step += 1
        
        # Extract average fidelity from observation
        avg_fidelity = np.mean(observation) if len(observation) > 0 else 0.5
        
        # Apply threshold rules
        if avg_fidelity >= self.swap_threshold:
            return 2  # Swap
        elif avg_fidelity >= self.purify_threshold:
            return 3  # Purify
        elif self.step % 5 == 0:
            return 1  # Generate periodically
        else:
            return 0  # Wait
    
    def reset(self):
        """Reset step counter"""
        self.step = 0
    
    def get_name(self) -> str:
        return f"Threshold Protocol ({self.purify_threshold}/{self.swap_threshold})"


class RoundRobinProtocol(BaselineStrategy):
    """
    Round-robin protocol that cycles through actions
    """
    
    def __init__(self, action_sequence: list = None):
        """
        Initialize round-robin protocol
        
        Args:
            action_sequence: List of actions to cycle through
        """
        if action_sequence is None:
            action_sequence = [1, 1, 3, 2, 0]  # Generate x2, Purify, Swap, Wait
        
        self.action_sequence = action_sequence
        self.current_index = 0
        
    def select_action(self, observation: np.ndarray,
                     channel_state: Optional[Dict] = None) -> int:
        """
        Select next action in sequence
        
        Args:
            observation: Environment observation (ignored)
            channel_state: Channel state (ignored)
            
        Returns:
            Next action in sequence
        """
        action = self.action_sequence[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.action_sequence)
        return action
    
    def reset(self):
        """Reset to beginning of sequence"""
        self.current_index = 0
    
    def get_name(self) -> str:
        return "Round-Robin Protocol"


class OptimisticProtocol(BaselineStrategy):
    """
    Optimistic protocol that assumes best-case scenario
    Minimal purification and error correction
    """
    
    def __init__(self):
        """Initialize optimistic protocol"""
        self.step = 0
        
    def select_action(self, observation: np.ndarray,
                     channel_state: Optional[Dict] = None) -> int:
        """
        Select optimistic action (minimal overhead)
        
        Args:
            observation: Environment observation
            channel_state: Channel state
            
        Returns:
            Optimistic action
        """
        self.step += 1
        
        # Mostly generate and swap, rarely purify
        if self.step % 10 == 0:
            return 3  # Occasional purification
        elif self.step % 3 == 0:
            return 2  # Frequent swapping
        else:
            return 1  # Mostly generation
    
    def reset(self):
        """Reset step counter"""
        self.step = 0
    
    def get_name(self) -> str:
        return "Optimistic Protocol"


class ConservativeProtocol(BaselineStrategy):
    """
    Conservative protocol that over-purifies and uses more resources
    """
    
    def __init__(self):
        """Initialize conservative protocol"""
        self.step = 0
        
    def select_action(self, observation: np.ndarray,
                     channel_state: Optional[Dict] = None) -> int:
        """
        Select conservative action (heavy purification)
        
        Args:
            observation: Environment observation
            channel_state: Channel state
            
        Returns:
            Conservative action
        """
        self.step += 1
        
        # Heavy purification
        if self.step % 2 == 0:
            return 3  # Purify frequently
        elif self.step % 5 == 0:
            return 2  # Occasional swapping
        else:
            return 1  # Generate
    
    def reset(self):
        """Reset step counter"""
        self.step = 0
    
    def get_name(self) -> str:
        return "Conservative Protocol"
