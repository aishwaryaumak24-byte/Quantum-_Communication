"""
Tests for Simulator Components
Tests FR-1, FR-2, FR-3
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from simulator.quantum_channel import QuantumChannel
from simulator.metrics import PerformanceMetrics, FidelityCalculator, CostMetrics
from environment.noise_models import DepolarizingNoise, AmplitudeDampingNoise


@pytest.fixture
def channel_config():
    """Provide test channel configuration"""
    return {
        'fiber_loss_coefficient': 0.2,
        'depolarizing_rate': 0.01,
        'dephasing_rate': 0.005
    }


def test_channel_creation(channel_config):
    """Test quantum channel creation"""
    channel = QuantumChannel(channel_config)
    
    assert channel is not None
    assert channel.loss_coefficient == 0.2


def test_channel_reset(channel_config):
    """Test channel reset"""
    channel = QuantumChannel(channel_config)
    state = channel.reset(num_nodes=5)
    
    assert state is not None
    assert state.shape[0] == 4  # 5 nodes -> 4 channels


def test_channel_loss_calculation(channel_config):
    """Test photon loss calculation (FR-1)"""
    channel = QuantumChannel(channel_config)
    channel.reset(num_nodes=5)
    
    loss = channel._calculate_loss()
    
    assert len(loss) == 4
    assert np.all(loss >= 0)
    assert np.all(loss <= 1)


def test_dynamic_noise_update(channel_config):
    """Test dynamic noise updates (FR-2)"""
    channel = QuantumChannel(channel_config)
    channel.reset(num_nodes=5)
    
    initial_noise = channel.current_noise_levels.copy()
    
    # Update noise multiple times
    for _ in range(10):
        channel.update()
    
    final_noise = channel.current_noise_levels
    
    # Noise should have changed
    assert not np.array_equal(initial_noise, final_noise)


def test_entanglement_generation(channel_config):
    """Test entanglement generation with noise"""
    channel = QuantumChannel(channel_config)
    channel.reset(num_nodes=5)
    
    noise_model = DepolarizingNoise(rate=0.01)
    fidelity = channel.generate_entanglement(0, 1, noise_model)
    
    assert 0 <= fidelity <= 1


def test_channel_quality_metrics(channel_config):
    """Test channel quality measurement"""
    channel = QuantumChannel(channel_config)
    channel.reset(num_nodes=5)
    
    quality = channel.get_channel_quality(0, 1)
    
    assert 'distance' in quality
    assert 'loss_probability' in quality
    assert 'noise_level' in quality
    assert 'estimated_fidelity' in quality


def test_performance_metrics():
    """Test performance metrics tracking (FR-3)"""
    metrics = PerformanceMetrics()
    
    # Record some transmissions
    metrics.record_transmission(fidelity=0.95, success=True, latency=5.0)
    metrics.record_transmission(fidelity=0.90, success=True, latency=3.0)
    metrics.record_transmission(fidelity=0.85, success=False, latency=7.0)
    
    avg_fidelity = metrics.get_average_fidelity()
    success_rate = metrics.get_success_rate()
    
    assert 0.85 <= avg_fidelity <= 0.95
    assert success_rate == 2.0 / 3.0


def test_metrics_reset():
    """Test metrics reset functionality"""
    metrics = PerformanceMetrics()
    
    metrics.record_transmission(fidelity=0.95, success=True)
    metrics.reset()
    
    assert metrics.total_attempts == 0
    assert len(metrics.fidelity_history) == 0


def test_fidelity_calculator():
    """Test fidelity calculation"""
    # Test state vector fidelity
    state1 = np.array([1, 0], dtype=complex)
    state2 = np.array([1, 0], dtype=complex)
    
    fidelity = FidelityCalculator.state_fidelity(state1, state2)
    assert np.isclose(fidelity, 1.0)
    
    # Test with orthogonal states
    state3 = np.array([0, 1], dtype=complex)
    fidelity2 = FidelityCalculator.state_fidelity(state1, state3)
    assert np.isclose(fidelity2, 0.0)


def test_bell_state_fidelity():
    """Test Bell state fidelity calculation"""
    # Create perfect Bell state |Φ+⟩
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    
    fidelity = FidelityCalculator.bell_state_fidelity(bell_state, 'phi_plus')
    assert np.isclose(fidelity, 1.0)


def test_cost_metrics():
    """Test cost tracking (FR-8)"""
    costs = CostMetrics()
    
    costs.add_repeater_cost(num_repeaters=3)
    costs.add_energy_cost(energy_units=10.0)
    costs.add_time_cost(time_steps=5)
    
    total_cost = costs.get_total_cost()
    
    assert total_cost > 0
    assert costs.repeater_cost > 0


def test_cost_breakdown():
    """Test detailed cost breakdown"""
    costs = CostMetrics()
    
    costs.add_repeater_cost(2)
    costs.add_purification_cost(3)
    
    breakdown = costs.get_cost_breakdown()
    
    assert 'repeater_cost' in breakdown
    assert 'purification_cost' in breakdown
    assert 'total_cost' in breakdown


def test_noise_models():
    """Test various noise models"""
    # Depolarizing noise
    depol = DepolarizingNoise(rate=0.1)
    state = np.array([[1, 0], [0, 0]])
    noisy_state = depol.apply_noise(state)
    assert noisy_state.shape == state.shape
    
    # Amplitude damping
    amp_damp = AmplitudeDampingNoise(gamma=0.1)
    noisy_state2 = amp_damp.apply_noise(state)
    assert noisy_state2.shape == state.shape


def test_throughput_tracking():
    """Test throughput measurement"""
    metrics = PerformanceMetrics()
    
    metrics.record_throughput(bits_transmitted=100, time_elapsed=10.0)
    metrics.record_throughput(bits_transmitted=80, time_elapsed=10.0)
    
    avg_throughput = metrics.get_average_throughput()
    assert avg_throughput == 9.0  # (10 + 8) / 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
