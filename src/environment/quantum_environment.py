"""
Quantum Communication Environment
Implements the Gymnasium environment for quantum network simulation
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional

from simulator.quantum_channel import QuantumChannel
from .noise_models import NoiseModel, DepolarizingNoise


class QuantumEnvironment(gym.Env):
    """
    Custom Gymnasium Environment for Quantum Communication
    
    The environment simulates a quantum network with multiple nodes,
    quantum channels, and various noise sources.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, config: Dict):
        """
        Initialize the quantum environment
        
        Args:
            config: Configuration dictionary containing network and simulation parameters
        """
        super().__init__()
        
        self.config = config
        self.num_nodes = config['network']['num_nodes']
        self.max_steps = config['simulation']['max_episode_steps']
        self.current_step = 0
        
        # Initialize quantum channel
        self.channel = QuantumChannel(config['quantum_channel'])
        
        # Initialize noise model
        if config['noise']['enable_noise']:
            noise_type = config['noise']['noise_model']
            if noise_type == 'depolarizing':
                self.noise_model = DepolarizingNoise(
                    rate=config['noise'].get('gate_error_rate', 0.001)
                )
            else:
                self.noise_model = None
        else:
            self.noise_model = None
        
        # Define action space
        # Actions: [0: wait, 1: generate_entanglement, 2: swap, 3: purify, ...]
        action_space_config = config.get('action_space', {})
        num_actions = action_space_config.get('num_actions', 10)
        self.action_space = spaces.Discrete(num_actions)
        
        # Define observation space
        # State includes: node states, channel states, entanglement pairs, buffers
        obs_dim = self._calculate_observation_dimension()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # State variables
        self.node_states = None
        self.entanglement_pairs = None
        self.channel_states = None
        
        # Metrics
        self.total_fidelity = 0.0
        self.successful_transmissions = 0
        self.total_transmissions = 0
        
    def _calculate_observation_dimension(self) -> int:
        """Calculate the dimension of observation space"""
        # Node states + channel states + entanglement states + buffer states
        dim = self.num_nodes * 4  # Each node: position, buffer_size, active, energy
        dim += (self.num_nodes - 1) * 3  # Each channel: distance, loss, noise_level
        dim += self.num_nodes * 2  # Entanglement: fidelity, age
        return dim
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state
        
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize node states
        self.node_states = np.zeros((self.num_nodes, 4), dtype=np.float32)
        self.node_states[:, 0] = np.linspace(0, 1, self.num_nodes)  # positions
        
        # Initialize channel states
        self.channel_states = self.channel.reset(self.num_nodes)
        
        # Initialize entanglement pairs
        self.entanglement_pairs = np.zeros((self.num_nodes, 2), dtype=np.float32)
        
        # Reset metrics
        self.total_fidelity = 0.0
        self.successful_transmissions = 0
        self.total_transmissions = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether episode has terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.current_step += 1
        
        # Execute action
        self._execute_action(action)
        
        # Update environment state
        self._update_state()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int):
        """Execute the given action"""
        if action == 0:
            # Wait action
            pass
        elif action == 1:
            # Generate entanglement
            self._generate_entanglement()
        elif action == 2:
            # Perform entanglement swap
            self._perform_swap()
        elif action == 3:
            # Purify entanglement
            self._purify_entanglement()
        # Add more actions as needed
    
    def _generate_entanglement(self):
        """Generate entanglement between adjacent nodes"""
        for i in range(self.num_nodes - 1):
            if self.entanglement_pairs[i, 0] < 0.5:  # If not already entangled
                fidelity = self.channel.generate_entanglement(i, i+1, self.noise_model)
                self.entanglement_pairs[i, 0] = fidelity
                self.entanglement_pairs[i, 1] = 0  # Reset age
    
    def _perform_swap(self):
        """Perform entanglement swapping"""
        # Simplified swapping logic
        for i in range(1, self.num_nodes - 1):
            if self.entanglement_pairs[i-1, 0] > 0 and self.entanglement_pairs[i, 0] > 0:
                # Swap reduces fidelity
                new_fidelity = self.entanglement_pairs[i-1, 0] * self.entanglement_pairs[i, 0]
                self.entanglement_pairs[i-1, 0] = new_fidelity
                self.entanglement_pairs[i, 0] = 0
    
    def _purify_entanglement(self):
        """Purify entanglement pairs"""
        for i in range(self.num_nodes - 1):
            if self.entanglement_pairs[i, 0] > 0:
                # Simplified purification
                self.entanglement_pairs[i, 0] = min(0.99, self.entanglement_pairs[i, 0] + 0.05)
    
    def _update_state(self):
        """Update environment state"""
        # Age entanglement pairs
        self.entanglement_pairs[:, 1] += 1
        
        # Degrade fidelity over time
        degradation = 0.01 * self.entanglement_pairs[:, 1]
        self.entanglement_pairs[:, 0] = np.maximum(0, self.entanglement_pairs[:, 0] - degradation)
        
        # Update channel states
        self.channel.update()
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on current state and action"""
        reward = 0.0
        
        # Reward for high fidelity
        avg_fidelity = np.mean(self.entanglement_pairs[:, 0])
        reward += avg_fidelity * self.config.get('reward', {}).get('fidelity_weight', 1.0)
        
        # Penalty for old entanglement
        avg_age = np.mean(self.entanglement_pairs[:, 1])
        reward -= avg_age * 0.01
        
        # Reward for successful end-to-end entanglement
        if self.entanglement_pairs[0, 0] > self.config['entanglement']['target_fidelity']:
            reward += 10.0
            self.successful_transmissions += 1
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if all entanglement is lost
        if np.all(self.entanglement_pairs[:, 0] < 0.01):
            return True
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = np.concatenate([
            self.node_states.flatten(),
            self.channel_states.flatten(),
            self.entanglement_pairs.flatten()
        ])
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'step': self.current_step,
            'avg_fidelity': np.mean(self.entanglement_pairs[:, 0]),
            'successful_transmissions': self.successful_transmissions,
            'total_transmissions': self.total_transmissions
        }
    
    def render(self):
        """Render the environment (optional)"""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Avg Fidelity: {np.mean(self.entanglement_pairs[:, 0]):.3f}")
    
    def close(self):
        """Clean up environment resources"""
        pass
