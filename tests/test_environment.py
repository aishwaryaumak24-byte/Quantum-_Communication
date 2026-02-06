"""
Tests for Quantum Environment
Tests FR-1, FR-2, FR-3, FR-4
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from environment.quantum_environment import QuantumEnvironment
from environment.noise_models import DepolarizingNoise


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        'network': {
            'num_nodes': 5,
            'topology': 'linear',
            'distance_range': [10, 100]
        },
        'quantum_channel': {
            'fiber_loss_coefficient': 0.2,
            'depolarizing_rate': 0.01,
            'dephasing_rate': 0.005
        },
        'noise': {
            'enable_noise': True,
            'noise_model': 'depolarizing',
            'gate_error_rate': 0.001,
            'measurement_error_rate': 0.01
        },
        'repeater': {
            'enable_repeaters': True,
            'max_repeaters': 3,
            'swap_success_probability': 0.98,
            'memory_coherence_time': 1.0
        },
        'entanglement': {
            'initial_fidelity': 0.99,
            'target_fidelity': 0.95,
            'purification_enabled': True,
            'distillation_rounds': 2
        },
        'simulation': {
            'time_slots': 1000,
            'max_episode_steps': 100,
            'random_seed': 42,
            'parallel_simulations': 1
        },
        'action_space': {
            'num_actions': 10
        },
        'reward': {
            'fidelity_weight': 1.0,
            'throughput_weight': 0.5,
            'latency_weight': -0.3
        }
    }


def test_environment_creation(test_config):
    """Test that environment can be created"""
    env = QuantumEnvironment(test_config)
    assert env is not None
    assert env.num_nodes == 5


def test_environment_reset(test_config):
    """Test environment reset functionality"""
    env = QuantumEnvironment(test_config)
    observation, info = env.reset()
    
    assert observation is not None
    assert isinstance(observation, np.ndarray)
    assert len(observation) > 0
    assert isinstance(info, dict)


def test_environment_step(test_config):
    """Test environment step execution"""
    env = QuantumEnvironment(test_config)
    observation, _ = env.reset()
    
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_space(test_config):
    """Test action space is correctly configured"""
    env = QuantumEnvironment(test_config)
    
    assert env.action_space.n == 10
    action = env.action_space.sample()
    assert 0 <= action < 10


def test_observation_space(test_config):
    """Test observation space dimensions"""
    env = QuantumEnvironment(test_config)
    observation, _ = env.reset()
    
    assert observation.shape == env.observation_space.shape
    assert np.all(observation >= env.observation_space.low)
    assert np.all(observation <= env.observation_space.high)


def test_entanglement_generation(test_config):
    """Test entanglement generation action"""
    env = QuantumEnvironment(test_config)
    env.reset()
    
    # Action 1 is generate entanglement
    obs, reward, done, truncated, info = env.step(1)
    
    # Check that entanglement pairs are created
    assert np.any(env.entanglement_pairs[:, 0] > 0)


def test_dynamic_noise(test_config):
    """Test dynamic noise updates (FR-2)"""
    env = QuantumEnvironment(test_config)
    env.reset()
    
    initial_noise = env.channel.current_noise_levels.copy()
    
    # Step through several time steps
    for _ in range(10):
        env.step(0)  # Wait action
    
    final_noise = env.channel.current_noise_levels
    
    # Noise should have changed due to dynamics
    assert not np.array_equal(initial_noise, final_noise)


def test_episode_termination(test_config):
    """Test that episodes terminate correctly"""
    test_config['simulation']['max_episode_steps'] = 10
    env = QuantumEnvironment(test_config)
    env.reset()
    
    terminated = False
    truncated = False
    steps = 0
    
    while not (terminated or truncated) and steps < 20:
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
    
    assert steps <= 10  # Should truncate at max steps


def test_reward_calculation(test_config):
    """Test reward calculation (FR-9)"""
    env = QuantumEnvironment(test_config)
    env.reset()
    
    # Take an action
    _, reward, _, _, _ = env.step(1)
    
    # Reward should be a finite number
    assert isinstance(reward, (int, float))
    assert np.isfinite(reward)


def test_noise_model_integration(test_config):
    """Test that noise models are properly integrated"""
    env = QuantumEnvironment(test_config)
    
    assert env.noise_model is not None
    assert isinstance(env.noise_model, DepolarizingNoise)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
