"""
Performance Metrics Module
Implements FR-3: Performance Metric Computation (fidelity, error rate)
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque


class PerformanceMetrics:
    """
    Tracks and computes performance metrics for quantum communication
    
    Metrics:
    - Communication fidelity
    - Error rate
    - Throughput
    - Latency
    - Success rate
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance metrics tracker
        
        Args:
            window_size: Number of recent measurements to track
        """
        self.window_size = window_size
        
        # Metric histories
        self.fidelity_history = deque(maxlen=window_size)
        self.error_rate_history = deque(maxlen=window_size)
        self.success_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        
        # Cumulative counters
        self.total_attempts = 0
        self.successful_transmissions = 0
        self.total_fidelity = 0.0
        
    def record_transmission(self, fidelity: float, success: bool, 
                          latency: float = 0.0):
        """
        Record a transmission attempt
        
        Args:
            fidelity: Achieved fidelity (0 to 1)
            success: Whether transmission was successful
            latency: Transmission latency in time steps
        """
        self.fidelity_history.append(fidelity)
        self.error_rate_history.append(1 - fidelity)
        self.success_history.append(1.0 if success else 0.0)
        self.latency_history.append(latency)
        
        self.total_attempts += 1
        if success:
            self.successful_transmissions += 1
            self.total_fidelity += fidelity
    
    def record_throughput(self, bits_transmitted: int, time_elapsed: float):
        """
        Record throughput measurement
        
        Args:
            bits_transmitted: Number of qubits/bits transmitted
            time_elapsed: Time elapsed in time steps
        """
        if time_elapsed > 0:
            throughput = bits_transmitted / time_elapsed
            self.throughput_history.append(throughput)
    
    def get_average_fidelity(self) -> float:
        """Get average fidelity over window"""
        if not self.fidelity_history:
            return 0.0
        return float(np.mean(self.fidelity_history))
    
    def get_average_error_rate(self) -> float:
        """Get average error rate over window"""
        if not self.error_rate_history:
            return 1.0
        return float(np.mean(self.error_rate_history))
    
    def get_success_rate(self) -> float:
        """Get success rate over window"""
        if not self.success_history:
            return 0.0
        return float(np.mean(self.success_history))
    
    def get_cumulative_success_rate(self) -> float:
        """Get cumulative success rate since start"""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_transmissions / self.total_attempts
    
    def get_average_latency(self) -> float:
        """Get average latency over window"""
        if not self.latency_history:
            return 0.0
        return float(np.mean(self.latency_history))
    
    def get_average_throughput(self) -> float:
        """Get average throughput over window"""
        if not self.throughput_history:
            return 0.0
        return float(np.mean(self.throughput_history))
    
    def get_fidelity_variance(self) -> float:
        """Get variance in fidelity measurements"""
        if len(self.fidelity_history) < 2:
            return 0.0
        return float(np.var(self.fidelity_history))
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Get all current metrics
        
        Returns:
            Dictionary of all performance metrics
        """
        return {
            'avg_fidelity': self.get_average_fidelity(),
            'avg_error_rate': self.get_average_error_rate(),
            'success_rate': self.get_success_rate(),
            'cumulative_success_rate': self.get_cumulative_success_rate(),
            'avg_latency': self.get_average_latency(),
            'avg_throughput': self.get_average_throughput(),
            'fidelity_variance': self.get_fidelity_variance(),
            'total_attempts': self.total_attempts,
            'successful_transmissions': self.successful_transmissions
        }
    
    def reset(self):
        """Reset all metrics"""
        self.fidelity_history.clear()
        self.error_rate_history.clear()
        self.success_history.clear()
        self.latency_history.clear()
        self.throughput_history.clear()
        
        self.total_attempts = 0
        self.successful_transmissions = 0
        self.total_fidelity = 0.0


class FidelityCalculator:
    """
    Calculate quantum state fidelity
    """
    
    @staticmethod
    def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate fidelity between two quantum states
        
        Args:
            state1: First quantum state (vector or density matrix)
            state2: Second quantum state (vector or density matrix)
            
        Returns:
            Fidelity value between 0 and 1
        """
        # If both are state vectors
        if state1.ndim == 1 and state2.ndim == 1:
            return float(np.abs(np.dot(state1.conj(), state2)) ** 2)
        
        # Convert vectors to density matrices if needed
        if state1.ndim == 1:
            state1 = np.outer(state1, state1.conj())
        if state2.ndim == 1:
            state2 = np.outer(state2, state2.conj())
        
        # Fidelity for density matrices: F = Tr(sqrt(sqrt(ρ1)ρ2sqrt(ρ1)))^2
        sqrt_rho1 = FidelityCalculator._matrix_sqrt(state1)
        product = sqrt_rho1 @ state2 @ sqrt_rho1
        sqrt_product = FidelityCalculator._matrix_sqrt(product)
        
        fidelity = np.trace(sqrt_product) ** 2
        return float(np.real(fidelity))
    
    @staticmethod
    def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root"""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T
    
    @staticmethod
    def bell_state_fidelity(state: np.ndarray, bell_type: str = 'phi_plus') -> float:
        """
        Calculate fidelity with a Bell state
        
        Args:
            state: Quantum state to compare
            bell_type: Type of Bell state ('phi_plus', 'phi_minus', 'psi_plus', 'psi_minus')
            
        Returns:
            Fidelity with the specified Bell state
        """
        # Define Bell states
        bell_states = {
            'phi_plus': np.array([1, 0, 0, 1]) / np.sqrt(2),
            'phi_minus': np.array([1, 0, 0, -1]) / np.sqrt(2),
            'psi_plus': np.array([0, 1, 1, 0]) / np.sqrt(2),
            'psi_minus': np.array([0, 1, -1, 0]) / np.sqrt(2)
        }
        
        target_bell = bell_states.get(bell_type, bell_states['phi_plus'])
        return FidelityCalculator.state_fidelity(state, target_bell)
    
    @staticmethod
    def average_fidelity(fidelities: List[float]) -> float:
        """Calculate average fidelity from a list"""
        if not fidelities:
            return 0.0
        return float(np.mean(fidelities))
    
    @staticmethod
    def fidelity_threshold_success_rate(fidelities: List[float], 
                                       threshold: float = 0.9) -> float:
        """
        Calculate success rate based on fidelity threshold
        
        Args:
            fidelities: List of fidelity values
            threshold: Minimum acceptable fidelity
            
        Returns:
            Fraction of fidelities above threshold
        """
        if not fidelities:
            return 0.0
        successes = sum(1 for f in fidelities if f >= threshold)
        return successes / len(fidelities)


class CostMetrics:
    """
    Track resource utilization and cost metrics
    Implements part of FR-8: Performance and Cost Evaluation
    """
    
    def __init__(self):
        """Initialize cost metrics tracker"""
        self.repeater_cost = 0.0
        self.energy_cost = 0.0
        self.time_cost = 0.0
        self.purification_cost = 0.0
        
        # Cost weights
        self.repeater_unit_cost = 1.0
        self.energy_unit_cost = 0.1
        self.time_unit_cost = 0.05
        self.purification_unit_cost = 0.5
    
    def add_repeater_cost(self, num_repeaters: int):
        """Add cost of using quantum repeaters"""
        self.repeater_cost += num_repeaters * self.repeater_unit_cost
    
    def add_energy_cost(self, energy_units: float):
        """Add energy consumption cost"""
        self.energy_cost += energy_units * self.energy_unit_cost
    
    def add_time_cost(self, time_steps: int):
        """Add time/latency cost"""
        self.time_cost += time_steps * self.time_unit_cost
    
    def add_purification_cost(self, num_rounds: int):
        """Add cost of purification operations"""
        self.purification_cost += num_rounds * self.purification_unit_cost
    
    def get_total_cost(self) -> float:
        """Get total accumulated cost"""
        return (self.repeater_cost + self.energy_cost + 
                self.time_cost + self.purification_cost)
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get detailed cost breakdown"""
        return {
            'repeater_cost': self.repeater_cost,
            'energy_cost': self.energy_cost,
            'time_cost': self.time_cost,
            'purification_cost': self.purification_cost,
            'total_cost': self.get_total_cost()
        }
    
    def reset(self):
        """Reset all costs"""
        self.repeater_cost = 0.0
        self.energy_cost = 0.0
        self.time_cost = 0.0
        self.purification_cost = 0.0
