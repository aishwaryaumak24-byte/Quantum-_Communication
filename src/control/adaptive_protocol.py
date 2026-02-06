"""
Adaptive Protocol Control
Implements FR-6: Adaptive Protocol Control - dynamically adjust protocol parameters
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ProtocolParameters:
    """
    Data class for quantum communication protocol parameters
    """
    # Entanglement generation parameters
    generation_rate: float = 1.0  # Attempts per time step
    generation_power: float = 1.0  # Power level for generation
    
    # Purification parameters
    purification_enabled: bool = True
    purification_rounds: int = 2
    purification_threshold: float = 0.9  # Min fidelity to purify
    
    # Entanglement swapping parameters
    swap_enabled: bool = True
    swap_threshold: float = 0.95  # Min fidelity for swapping
    
    # Error correction parameters
    error_correction_enabled: bool = False
    error_correction_overhead: int = 3  # Qubits per logical qubit
    
    # Waiting/buffering parameters
    max_wait_time: int = 10  # Max time to wait for pair
    buffer_size: int = 5  # Max entangled pairs to buffer
    
    # Adaptive parameters
    noise_adaptation_rate: float = 0.1
    distance_scaling: float = 1.0


class AdaptiveProtocol:
    """
    Adaptive Protocol Controller for Quantum Communication
    
    Implements FR-6: Dynamically adjusts protocol parameters to improve
    communication reliability under noisy conditions
    """
    
    def __init__(self, config: Dict):
        """
        Initialize adaptive protocol
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize default parameters
        self.params = ProtocolParameters()
        
        # Load from config
        self._load_config()
        
        # Adaptation state
        self.channel_history = []
        self.performance_history = []
        self.adaptation_step = 0
        
        # Parameter bounds
        self.param_bounds = {
            'generation_rate': (0.1, 5.0),
            'purification_rounds': (0, 5),
            'swap_threshold': (0.8, 0.99)
        }
        
    def _load_config(self):
        """Load parameters from configuration"""
        ent_config = self.config.get('entanglement', {})
        
        self.params.purification_enabled = ent_config.get('purification_enabled', True)
        self.params.purification_rounds = ent_config.get('distillation_rounds', 2)
        
        repeater_config = self.config.get('repeater', {})
        self.params.swap_enabled = repeater_config.get('enable_repeaters', True)
        
    def adapt_to_channel(self, channel_state: Dict, 
                        performance_metrics: Dict) -> ProtocolParameters:
        """
        Adapt protocol parameters based on channel conditions
        
        Args:
            channel_state: Current channel state information
            performance_metrics: Recent performance metrics
            
        Returns:
            Adapted protocol parameters
        """
        self.adaptation_step += 1
        
        # Store history
        self.channel_history.append(channel_state)
        self.performance_history.append(performance_metrics)
        
        # Keep only recent history
        if len(self.channel_history) > 100:
            self.channel_history.pop(0)
            self.performance_history.pop(0)
        
        # Extract key metrics
        noise_level = channel_state.get('noise_level', 0.01)
        distance = channel_state.get('distance', 50)
        avg_fidelity = performance_metrics.get('avg_fidelity', 0.9)
        success_rate = performance_metrics.get('success_rate', 0.8)
        
        # Adaptation logic
        
        # 1. Adjust purification based on noise
        if noise_level > 0.05:
            # High noise: increase purification
            self.params.purification_rounds = min(5, self.params.purification_rounds + 1)
            self.params.purification_threshold = 0.85
        elif noise_level < 0.02 and avg_fidelity > 0.95:
            # Low noise and good performance: reduce purification to save resources
            self.params.purification_rounds = max(0, self.params.purification_rounds - 1)
            self.params.purification_threshold = 0.90
        
        # 2. Adjust generation rate based on success rate
        if success_rate < 0.5:
            # Poor success rate: increase generation attempts
            self.params.generation_rate = min(5.0, self.params.generation_rate * 1.2)
        elif success_rate > 0.9:
            # Good success rate: can reduce generation rate
            self.params.generation_rate = max(0.5, self.params.generation_rate * 0.9)
        
        # 3. Adjust swap threshold based on fidelity performance
        if avg_fidelity < 0.9:
            # Low fidelity: be more strict about swapping
            self.params.swap_threshold = min(0.99, self.params.swap_threshold + 0.01)
        elif avg_fidelity > 0.95:
            # High fidelity: can relax swap threshold
            self.params.swap_threshold = max(0.85, self.params.swap_threshold - 0.01)
        
        # 4. Distance-based adaptation
        if distance > 100:
            # Long distance: enable error correction
            self.params.error_correction_enabled = True
            self.params.buffer_size = 10
        else:
            # Short distance: disable error correction to save resources
            self.params.error_correction_enabled = False
            self.params.buffer_size = 5
        
        # 5. Adaptive waiting time
        if success_rate < 0.6:
            # Low success: wait longer for good pairs
            self.params.max_wait_time = min(20, self.params.max_wait_time + 2)
        else:
            # Good success: shorter wait time
            self.params.max_wait_time = max(5, self.params.max_wait_time - 1)
        
        return self.params
    
    def get_action_from_params(self) -> int:
        """
        Convert current parameters to an action
        
        Returns:
            Action index based on current protocol configuration
        """
        # Map protocol configuration to discrete action
        # This is a simplified mapping
        
        action = 0  # Default: wait
        
        if self.params.generation_rate > 2.0:
            action = 1  # Generate entanglement
        
        if self.params.purification_enabled and self.params.purification_rounds > 0:
            action = 3  # Purify
        
        if self.params.swap_enabled:
            action = 2  # Swap
        
        return action
    
    def update_from_rl_action(self, action: int):
        """
        Update protocol parameters based on RL agent's action
        
        Args:
            action: Action from RL agent
        """
        # Map actions to parameter updates
        action_map = {
            0: self._action_wait,
            1: self._action_generate,
            2: self._action_swap,
            3: self._action_purify,
            4: self._action_increase_power,
            5: self._action_decrease_power,
            6: self._action_enable_error_correction,
            7: self._action_disable_error_correction
        }
        
        if action in action_map:
            action_map[action]()
    
    def _action_wait(self):
        """Wait action - conservative parameters"""
        self.params.generation_rate = max(0.5, self.params.generation_rate * 0.9)
    
    def _action_generate(self):
        """Generate action - aggressive generation"""
        self.params.generation_rate = min(5.0, self.params.generation_rate * 1.5)
    
    def _action_swap(self):
        """Enable swapping with current threshold"""
        self.params.swap_enabled = True
    
    def _action_purify(self):
        """Increase purification"""
        self.params.purification_enabled = True
        self.params.purification_rounds = min(5, self.params.purification_rounds + 1)
    
    def _action_increase_power(self):
        """Increase generation power"""
        self.params.generation_power = min(2.0, self.params.generation_power + 0.1)
    
    def _action_decrease_power(self):
        """Decrease generation power to save energy"""
        self.params.generation_power = max(0.5, self.params.generation_power - 0.1)
    
    def _action_enable_error_correction(self):
        """Enable quantum error correction"""
        self.params.error_correction_enabled = True
    
    def _action_disable_error_correction(self):
        """Disable quantum error correction"""
        self.params.error_correction_enabled = False
    
    def get_parameter_vector(self) -> np.ndarray:
        """
        Get current parameters as a vector for RL state
        
        Returns:
            Normalized parameter vector
        """
        return np.array([
            self.params.generation_rate / 5.0,
            self.params.generation_power / 2.0,
            float(self.params.purification_enabled),
            self.params.purification_rounds / 5.0,
            self.params.swap_threshold,
            float(self.params.error_correction_enabled),
            self.params.max_wait_time / 20.0,
            self.params.buffer_size / 10.0
        ], dtype=np.float32)
    
    def reset(self):
        """Reset to default parameters"""
        self.params = ProtocolParameters()
        self.channel_history.clear()
        self.performance_history.clear()
        self.adaptation_step = 0
    
    def get_statistics(self) -> Dict:
        """Get adaptation statistics"""
        return {
            'adaptation_steps': self.adaptation_step,
            'current_generation_rate': self.params.generation_rate,
            'current_purification_rounds': self.params.purification_rounds,
            'current_swap_threshold': self.params.swap_threshold,
            'error_correction_enabled': self.params.error_correction_enabled
        }
    
    def __repr__(self):
        return (f"AdaptiveProtocol(gen_rate={self.params.generation_rate:.2f}, "
                f"purif_rounds={self.params.purification_rounds}, "
                f"swap_thresh={self.params.swap_threshold:.2f})")
