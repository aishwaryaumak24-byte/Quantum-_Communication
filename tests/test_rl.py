"""
Tests for Reinforcement Learning Components
Tests FR-5, FR-9, FR-10
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from environment.quantum_environment import QuantumEnvironment
from rl.rl_agent import RLAgent
from rl.reward_function import RewardFunction
from rl.policy import AdaptivePolicy, EpsilonGreedyPolicy


@pytest.fixture
def test_env_config():
    """Provide test environment configuration"""
    return {
        'network': {'num_nodes': 3, 'topology': 'linear',  'distance_range': [10, 50]},
        'quantum_channel': {'fiber_loss_coefficient': 0.2, 'depolarizing_rate': 0.01, 'dephasing_rate': 0.005},
        'noise': {'enable_noise': True, 'noise_model': 'depolarizing', 'gate_error_rate': 0.01},
        'repeater': {'enable_repeaters': False, 'max_repeaters': 2},
        'entanglement': {'initial_fidelity': 0.99, 'target_fidelity': 0.95, 'purification_enabled': True},
        'simulation': {'time_slots': 100, 'max_episode_steps': 50, 'random_seed': 42},
        'action_space': {'num_actions': 5},
        'reward': {'fidelity_weight': 1.0, 'throughput_weight': 0.5}
    }


@pytest.fixture
def test_rl_config():
    """Provide test RL configuration"""
    return {
        'algorithm': 'PPO',
        'training': {
            'total_timesteps': 1000,
            'learning_rate': 0.0003,
            'batch_size': 32,
            'n_epochs': 5
        },
        'ppo': {
            'n_steps': 128,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2
        },
        'network': {
            'policy_type': 'MlpPolicy',
            'net_arch': [64, 64],
            'activation': 'tanh'
        },
        'reward': {
            'fidelity_weight': 1.0,
            'throughput_weight': 0.5,
            'latency_weight': -0.3,
            'energy_weight': -0.1,
            'success_rate_weight': 0.8
        },
        'advanced': {
            'verbose': 0,
            'device': 'cpu',
            'seed': 42
        }
    }


def test_reward_function_creation():
    """Test reward function initialization"""
    config = {
        'fidelity_weight': 1.0,
        'throughput_weight': 0.5,
        'latency_weight': -0.3
    }
    
    reward_fn = RewardFunction(config)
    assert reward_fn is not None
    assert reward_fn.fidelity_weight == 1.0


def test_reward_calculation():
    """Test reward calculation (FR-9)"""
    config = {
        'fidelity_weight': 1.0,
        'throughput_weight': 0.5,
        'latency_weight': -0.3,
        'energy_weight': -0.1,
        'success_rate_weight': 0.8
    }
    
    reward_fn = RewardFunction(config)
    
    state_info = {
        'fidelity': 0.95,
        'throughput': 1.0,
        'latency': 10.0,
        'energy_cost': 2.0,
        'success': True,
        'error_rate': 0.05
    }
    
    reward = reward_fn.calculate_reward(state_info)
    
    assert isinstance(reward, float)
    assert np.isfinite(reward)


def test_reward_breakdown():
    """Test detailed reward breakdown"""
    config = {
        'fidelity_weight': 1.0,
        'throughput_weight': 0.5,
        'latency_weight': -0.3
    }
    
    reward_fn = RewardFunction(config)
    
    state_info = {
        'fidelity': 0.9,
        'throughput': 0.8,
        'latency': 5.0,
        'energy_cost': 1.0,
        'success': True
    }
    
    breakdown = reward_fn.get_reward_breakdown(state_info)
    
    assert 'fidelity_reward' in breakdown
    assert 'throughput_reward' in breakdown
    assert 'total_reward' in breakdown


def test_rl_agent_creation(test_env_config, test_rl_config):
    """Test RL agent initialization (FR-5)"""
    env = QuantumEnvironment(test_env_config)
    agent = RLAgent(env, test_rl_config)
    
    assert agent is not None
    assert agent.model is not None


def test_rl_agent_prediction(test_env_config, test_rl_config):
    """Test RL agent can make predictions"""
    env = QuantumEnvironment(test_env_config)
    agent = RLAgent(env, test_rl_config)
    
    obs, _ = env.reset()
    action = agent.predict(obs, deterministic=True)
    
    assert action is not None
    assert 0 <= action < env.action_space.n


def test_adaptive_policy():
    """Test adaptive policy strategy selection"""
    policy = AdaptivePolicy(num_strategies=3)
    
    channel_state = np.array([0.5, 0.2, 0.1])
    strategy = policy.select_strategy(channel_state)
    
    assert 0 <= strategy < 3


def test_adaptive_policy_update():
    """Test adaptive policy performance update"""
    policy = AdaptivePolicy(num_strategies=3)
    
    # Update strategy performance
    policy.update_strategy_performance(0, 10.0)
    policy.update_strategy_performance(1, 5.0)
    policy.update_strategy_performance(2, 15.0)
    
    best = policy.get_best_strategy()
    assert best == 2  # Strategy 2 has highest performance


def test_epsilon_greedy_policy():
    """Test epsilon-greedy exploration"""
    policy = EpsilonGreedyPolicy(epsilon_start=1.0, epsilon_end=0.1)
    
    q_values = np.array([0.5, 0.8, 0.3, 0.9])
    
    # In training mode with high epsilon, should explore
    actions = [policy.select_action(q_values, training=True) for _ in range(100)]
    assert len(set(actions)) > 1  # Should have variety due to exploration
    
    # Decay epsilon
    for _ in range(100):
        policy.decay_epsilon()
    
    # With low epsilon, should mostly exploit
    action = policy.select_action(q_values, training=False)
    assert action == 3  # Should select action with highest Q-value


def test_reward_function_weights_update():
    """Test dynamic reward weight updates"""
    config = {'fidelity_weight': 1.0}
    reward_fn = RewardFunction(config)
    
    # Update weights
    new_weights = {'fidelity_weight': 2.0, 'throughput_weight': 1.0}
    reward_fn.set_weights(new_weights)
    
    assert reward_fn.fidelity_weight == 2.0
    assert reward_fn.throughput_weight == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
