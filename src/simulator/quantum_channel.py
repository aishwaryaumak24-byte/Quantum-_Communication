"""
Quantum Channel Simulator
Implements FR-1: Quantum Channel Simulation with distance, dynamic noise, and photon loss
"""

import numpy as np
from typing import Dict, Optional, Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity, DensityMatrix


class QuantumChannel:
    """
    Simulates a long-distance quantum communication channel
    
    Models:
    - Photon loss due to fiber attenuation
    - Dynamic channel noise (time-varying)
    - Distance-dependent degradation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize quantum channel
        
        Args:
            config: Channel configuration parameters
        """
        self.config = config
        self.loss_coefficient = config.get('fiber_loss_coefficient', 0.2)  # dB/km
        self.depolarizing_rate = config.get('depolarizing_rate', 0.01)
        self.dephasing_rate = config.get('dephasing_rate', 0.005)
        
        # Dynamic noise parameters
        self.noise_variance = 0.01
        self.time_step = 0
        
        # Channel state
        self.distances = None
        self.current_noise_levels = None
        
        # Simulator
        self.simulator = AerSimulator()
        
    def reset(self, num_nodes: int) -> np.ndarray:
        """
        Reset channel state
        
        Args:
            num_nodes: Number of communication nodes
            
        Returns:
            Initial channel states
        """
        self.time_step = 0
        
        # Initialize distances between adjacent nodes
        self.distances = np.random.uniform(10, 100, num_nodes - 1)
        
        # Initialize noise levels
        self.current_noise_levels = np.ones(num_nodes - 1) * self.depolarizing_rate
        
        return self._get_channel_state()
    
    def _get_channel_state(self) -> np.ndarray:
        """Get current channel state representation"""
        state = np.zeros((len(self.distances), 3))
        state[:, 0] = self.distances / 100.0  # Normalized distance
        state[:, 1] = self._calculate_loss()
        state[:, 2] = self.current_noise_levels
        return state
    
    def _calculate_loss(self) -> np.ndarray:
        """
        Calculate photon loss based on distance
        
        Loss formula: L = 10^(-α*d/10)
        where α is loss coefficient (dB/km) and d is distance (km)
        """
        loss_db = self.loss_coefficient * self.distances
        transmission_probability = 10 ** (-loss_db / 10)
        return 1 - transmission_probability  # Loss probability
    
    def update(self):
        """
        Update channel state with dynamic noise (FR-2: Dynamic Noise Modeling)
        
        Simulates time-varying channel conditions
        """
        self.time_step += 1
        
        # Add temporal variation to noise
        noise_drift = np.random.normal(0, self.noise_variance, len(self.current_noise_levels))
        self.current_noise_levels = np.clip(
            self.current_noise_levels + noise_drift,
            0.001, 0.1
        )
        
        # Periodic fluctuations (simulating environmental changes)
        self.current_noise_levels *= (1 + 0.05 * np.sin(2 * np.pi * self.time_step / 100))
    
    def generate_entanglement(self, node_a: int, node_b: int, 
                            noise_model=None) -> float:
        """
        Generate entanglement between two nodes
        
        Args:
            node_a: First node index
            node_b: Second node index
            noise_model: Optional noise model to apply
            
        Returns:
            Entanglement fidelity
        """
        # Create Bell state circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        
        # Simulate with noise
        if noise_model is not None:
            self.simulator.set_options(noise_model=noise_model.get_qiskit_noise_model())
        
        # Add channel effects
        distance_idx = min(node_a, node_b)
        if distance_idx < len(self.distances):
            # Apply distance-dependent depolarizing noise
            noise_prob = self.current_noise_levels[distance_idx]
            for qubit in [0, 1]:
                if np.random.random() < noise_prob:
                    # Apply random Pauli
                    pauli = np.random.choice(['x', 'y', 'z'])
                    getattr(qc, pauli)(qubit)
        
        # Measure fidelity
        qc.save_statevector()
        result = self.simulator.run(qc).result()
        statevector = result.get_statevector()
        
        # Calculate fidelity with ideal Bell state
        ideal_bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        fidelity = np.abs(np.dot(ideal_bell.conj(), statevector)) ** 2
        
        # Apply photon loss
        loss_prob = self._calculate_loss()[distance_idx] if distance_idx < len(self.distances) else 0
        if np.random.random() < loss_prob:
            fidelity = 0.0  # Photon lost
        
        return float(fidelity)
    
    def transmit_quantum_state(self, state: np.ndarray, distance: float) -> Tuple[np.ndarray, float]:
        """
        Transmit a quantum state through the channel
        
        Args:
            state: Input quantum state
            distance: Transmission distance in km
            
        Returns:
            Tuple of (output_state, fidelity)
        """
        # Calculate loss
        loss_db = self.loss_coefficient * distance
        transmission_prob = 10 ** (-loss_db / 10)
        
        # Check if photon survives
        if np.random.random() > transmission_prob:
            # Photon lost
            return np.zeros_like(state), 0.0
        
        # Apply noise
        output_state = state.copy()
        
        # Depolarizing noise
        if np.random.random() < self.depolarizing_rate:
            # Random Pauli
            pauli_idx = np.random.choice([1, 2, 3])
            if pauli_idx == 1:  # X
                output_state = np.roll(output_state, 1)
            elif pauli_idx == 2:  # Y
                output_state = np.roll(output_state, 1) * 1j
            else:  # Z
                output_state[1] *= -1
        
        # Calculate fidelity
        fidelity = np.abs(np.dot(state.conj(), output_state)) ** 2
        
        return output_state, float(fidelity)
    
    def get_channel_quality(self, node_a: int, node_b: int) -> Dict[str, float]:
        """
        Get channel quality metrics between two nodes
        
        Returns:
            Dictionary with quality metrics
        """
        distance_idx = min(node_a, node_b)
        
        if distance_idx >= len(self.distances):
            return {
                'distance': 0.0,
                'loss_probability': 0.0,
                'noise_level': 0.0,
                'estimated_fidelity': 1.0
            }
        
        distance = self.distances[distance_idx]
        loss_prob = self._calculate_loss()[distance_idx]
        noise_level = self.current_noise_levels[distance_idx]
        
        # Estimate achievable fidelity
        estimated_fidelity = (1 - loss_prob) * (1 - noise_level)
        
        return {
            'distance': float(distance),
            'loss_probability': float(loss_prob),
            'noise_level': float(noise_level),
            'estimated_fidelity': float(estimated_fidelity)
        }
    
    def set_dynamic_noise(self, noise_levels: np.ndarray):
        """
        Manually set noise levels for testing
        
        Args:
            noise_levels: Array of noise levels for each channel
        """
        self.current_noise_levels = np.clip(noise_levels, 0.0, 1.0)
    
    def __repr__(self):
        return (f"QuantumChannel(loss_coef={self.loss_coefficient}, "
                f"depolarizing={self.depolarizing_rate}, "
                f"time_step={self.time_step})")
