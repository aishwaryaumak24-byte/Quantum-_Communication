"""
Reward Function for Quantum Communication RL
Implements FR-9: Reward Generation
"""

import numpy as np
from typing import Dict, Optional


class RewardFunction:
    """
    Reward function for quantum communication optimization
    
    Implements FR-9: Generate reward signal reflecting both performance 
    improvement and cost efficiency
    
    Reward components:
    - Fidelity reward (higher is better)
    - Throughput reward (higher is better)
    - Latency penalty (lower is better)
    - Energy/cost penalty (lower is better)
    - Success rate reward (higher is better)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize reward function
        
        Args:
            config: Reward configuration with weights for different components
        """
        # Reward weights from FR-9 requirements
        self.fidelity_weight = config.get('fidelity_weight', 1.0)
        self.throughput_weight = config.get('throughput_weight', 0.5)
        self.latency_weight = config.get('latency_weight', -0.3)
        self.energy_weight = config.get('energy_weight', -0.1)
        self.success_rate_weight = config.get('success_rate_weight', 0.8)
        
        # Thresholds and scaling factors
        self.target_fidelity = config.get('target_fidelity', 0.95)
        self.max_latency = config.get('max_latency', 100)
        self.max_energy = config.get('max_energy', 10)
        
        # Bonus rewards
        self.success_bonus = config.get('success_bonus', 10.0)
        self.failure_penalty = config.get('failure_penalty', -5.0)
        
    def calculate_reward(self, state_info: Dict) -> float:
        """
        Calculate reward based on current state information
        
        Args:
            state_info: Dictionary containing:
                - fidelity: Current communication fidelity (0-1)
                - throughput: Current throughput (qubits/timestep)
                - latency: Current latency (timesteps)
                - energy_cost: Energy consumption
                - success: Boolean indicating successful transmission
                - error_rate: Communication error rate
                
        Returns:
            Computed reward value
        """
        reward = 0.0
        
        # 1. Fidelity component (FR-8: Performance Evaluation)
        fidelity = state_info.get('fidelity', 0.0)
        if fidelity >= self.target_fidelity:
            # Bonus for exceeding target fidelity
            fidelity_reward = self.fidelity_weight * (1.0 + (fidelity - self.target_fidelity))
        else:
            # Linear reward below target
            fidelity_reward = self.fidelity_weight * fidelity
        reward += fidelity_reward
        
        # 2. Throughput component
        throughput = state_info.get('throughput', 0.0)
        throughput_reward = self.throughput_weight * throughput
        reward += throughput_reward
        
        # 3. Latency penalty (lower is better)
        latency = state_info.get('latency', 0.0)
        normalized_latency = min(latency / self.max_latency, 1.0)
        latency_penalty = self.latency_weight * normalized_latency
        reward += latency_penalty
        
        # 4. Energy/Cost penalty (FR-8: Cost Evaluation)
        energy_cost = state_info.get('energy_cost', 0.0)
        normalized_energy = min(energy_cost / self.max_energy, 1.0)
        energy_penalty = self.energy_weight * normalized_energy
        reward += energy_penalty
        
        # 5. Success rate component
        success = state_info.get('success', False)
        if success:
            reward += self.success_bonus
        else:
            reward += self.failure_penalty
        
        # 6. Error rate penalty
        error_rate = state_info.get('error_rate', 0.0)
        error_penalty = -self.fidelity_weight * error_rate
        reward += error_penalty
        
        # 7. Additional bonuses/penalties
        
        # Stability bonus (low fidelity variance)
        fidelity_variance = state_info.get('fidelity_variance', 0.0)
        if fidelity_variance < 0.01:
            reward += 0.5  # Stability bonus
        
        # Long-distance bonus
        distance = state_info.get('distance', 0.0)
        if distance > 50 and fidelity > 0.9:
            reward += 2.0  # Long-distance high-fidelity bonus
        
        return float(reward)
    
    def calculate_shaped_reward(self, 
                                current_state: Dict,
                                previous_state: Optional[Dict] = None) -> float:
        """
        Calculate reward with reward shaping for better learning
        
        Args:
            current_state: Current state information
            previous_state: Previous state information (for delta rewards)
            
        Returns:
            Shaped reward value
        """
        # Base reward
        reward = self.calculate_reward(current_state)
        
        # Add potential-based shaping if previous state available
        if previous_state is not None:
            # Improvement in fidelity
            fidelity_improvement = (current_state.get('fidelity', 0) - 
                                   previous_state.get('fidelity', 0))
            reward += self.fidelity_weight * fidelity_improvement * 5.0
            
            # Improvement in success rate
            success_improvement = (current_state.get('success_rate', 0) - 
                                  previous_state.get('success_rate', 0))
            reward += self.success_rate_weight * success_improvement * 3.0
        
        return float(reward)
    
    def get_reward_breakdown(self, state_info: Dict) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components
        
        Args:
            state_info: State information dictionary
            
        Returns:
            Dictionary with individual reward components
        """
        breakdown = {}
        
        # Fidelity
        fidelity = state_info.get('fidelity', 0.0)
        breakdown['fidelity_reward'] = self.fidelity_weight * fidelity
        
        # Throughput
        throughput = state_info.get('throughput', 0.0)
        breakdown['throughput_reward'] = self.throughput_weight * throughput
        
        # Latency
        latency = state_info.get('latency', 0.0)
        normalized_latency = min(latency / self.max_latency, 1.0)
        breakdown['latency_penalty'] = self.latency_weight * normalized_latency
        
        # Energy
        energy_cost = state_info.get('energy_cost', 0.0)
        normalized_energy = min(energy_cost / self.max_energy, 1.0)
        breakdown['energy_penalty'] = self.energy_weight * normalized_energy
        
        # Success
        success = state_info.get('success', False)
        breakdown['success_reward'] = self.success_bonus if success else self.failure_penalty
        
        # Total
        breakdown['total_reward'] = sum(breakdown.values())
        
        return breakdown
    
    def set_weights(self, weights: Dict):
        """
        Update reward weights dynamically
        
        Args:
            weights: Dictionary with new weight values
        """
        if 'fidelity_weight' in weights:
            self.fidelity_weight = weights['fidelity_weight']
        if 'throughput_weight' in weights:
            self.throughput_weight = weights['throughput_weight']
        if 'latency_weight' in weights:
            self.latency_weight = weights['latency_weight']
        if 'energy_weight' in weights:
            self.energy_weight = weights['energy_weight']
        if 'success_rate_weight' in weights:
            self.success_rate_weight = weights['success_rate_weight']
    
    def __repr__(self):
        return (f"RewardFunction(fidelity={self.fidelity_weight}, "
                f"throughput={self.throughput_weight}, "
                f"latency={self.latency_weight}, "
                f"energy={self.energy_weight})")
