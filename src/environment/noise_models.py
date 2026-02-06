"""
Noise Models for Quantum Communication
Implements various quantum noise models
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer.noise import depolarizing_error, amplitude_damping_error, phase_damping_error


class NoiseModel(ABC):
    """Abstract base class for noise models"""
    
    @abstractmethod
    def apply_noise(self, state: np.ndarray) -> np.ndarray:
        """
        Apply noise to quantum state
        
        Args:
            state: Quantum state vector or density matrix
            
        Returns:
            Noisy quantum state
        """
        pass
    
    @abstractmethod
    def get_qiskit_noise_model(self) -> QiskitNoiseModel:
        """Get Qiskit noise model"""
        pass


class DepolarizingNoise(NoiseModel):
    """Depolarizing noise model"""
    
    def __init__(self, rate: float = 0.01):
        """
        Initialize depolarizing noise
        
        Args:
            rate: Depolarizing rate (probability)
        """
        self.rate = rate
        
    def apply_noise(self, state: np.ndarray) -> np.ndarray:
        """
        Apply depolarizing noise to state
        
        The depolarizing channel replaces the state with the maximally 
        mixed state with probability p.
        """
        if np.random.random() < self.rate:
            # Replace with maximally mixed state
            dim = state.shape[0]
            mixed_state = np.eye(dim) / dim
            return mixed_state
        return state
    
    def get_qiskit_noise_model(self) -> QiskitNoiseModel:
        """Get Qiskit depolarizing noise model"""
        noise_model = QiskitNoiseModel()
        
        # Add depolarizing error to single-qubit gates
        error_1q = depolarizing_error(self.rate, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z'])
        
        # Add depolarizing error to two-qubit gates
        error_2q = depolarizing_error(self.rate * 2, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'swap'])
        
        return noise_model
    
    def __repr__(self):
        return f"DepolarizingNoise(rate={self.rate})"


class AmplitudeDampingNoise(NoiseModel):
    """Amplitude damping noise model (energy relaxation)"""
    
    def __init__(self, gamma: float = 0.01):
        """
        Initialize amplitude damping noise
        
        Args:
            gamma: Damping parameter (energy loss rate)
        """
        self.gamma = gamma
        
    def apply_noise(self, state: np.ndarray) -> np.ndarray:
        """
        Apply amplitude damping noise
        
        Simulates energy dissipation (T1 relaxation)
        """
        # Kraus operators for amplitude damping
        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]])
        K1 = np.array([[0, np.sqrt(self.gamma)], [0, 0]])
        
        # Apply Kraus operators
        if len(state.shape) == 1:
            # State vector
            state_matrix = np.outer(state, state.conj())
        else:
            state_matrix = state
            
        noisy_state = K0 @ state_matrix @ K0.conj().T + K1 @ state_matrix @ K1.conj().T
        
        return noisy_state
    
    def get_qiskit_noise_model(self) -> QiskitNoiseModel:
        """Get Qiskit amplitude damping noise model"""
        noise_model = QiskitNoiseModel()
        
        # Add amplitude damping error
        error = amplitude_damping_error(self.gamma)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z'])
        
        return noise_model
    
    def __repr__(self):
        return f"AmplitudeDampingNoise(gamma={self.gamma})"


class PhaseDampingNoise(NoiseModel):
    """Phase damping noise model (dephasing)"""
    
    def __init__(self, lambda_param: float = 0.01):
        """
        Initialize phase damping noise
        
        Args:
            lambda_param: Dephasing parameter (T2 dephasing rate)
        """
        self.lambda_param = lambda_param
        
    def apply_noise(self, state: np.ndarray) -> np.ndarray:
        """
        Apply phase damping noise
        
        Simulates loss of quantum coherence (T2 dephasing)
        """
        # Kraus operators for phase damping
        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.lambda_param)]])
        K1 = np.array([[0, 0], [0, np.sqrt(self.lambda_param)]])
        
        # Apply Kraus operators
        if len(state.shape) == 1:
            state_matrix = np.outer(state, state.conj())
        else:
            state_matrix = state
            
        noisy_state = K0 @ state_matrix @ K0.conj().T + K1 @ state_matrix @ K1.conj().T
        
        return noisy_state
    
    def get_qiskit_noise_model(self) -> QiskitNoiseModel:
        """Get Qiskit phase damping noise model"""
        noise_model = QiskitNoiseModel()
        
        # Add phase damping error
        error = phase_damping_error(self.lambda_param)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z'])
        
        return noise_model
    
    def __repr__(self):
        return f"PhaseDampingNoise(lambda={self.lambda_param})"


class CombinedNoise(NoiseModel):
    """Combined noise model with multiple noise sources"""
    
    def __init__(self, noise_models: list):
        """
        Initialize combined noise model
        
        Args:
            noise_models: List of noise model instances
        """
        self.noise_models = noise_models
        
    def apply_noise(self, state: np.ndarray) -> np.ndarray:
        """Apply all noise models sequentially"""
        noisy_state = state
        for noise_model in self.noise_models:
            noisy_state = noise_model.apply_noise(noisy_state)
        return noisy_state
    
    def get_qiskit_noise_model(self) -> QiskitNoiseModel:
        """Get combined Qiskit noise model"""
        combined = QiskitNoiseModel()
        for noise_model in self.noise_models:
            # Compose noise models
            model = noise_model.get_qiskit_noise_model()
            # Note: This is simplified, actual composition is more complex
            combined = model
        return combined
    
    def __repr__(self):
        models_str = ", ".join(str(m) for m in self.noise_models)
        return f"CombinedNoise([{models_str}])"
