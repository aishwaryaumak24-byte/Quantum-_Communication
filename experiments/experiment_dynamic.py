"""
Dynamic Scenario Experiment
Tests RL agent's adaptation to time-varying channel conditions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import yaml
import matplotlib.pyplot as plt
from environment.quantum_environment import QuantumEnvironment
from rl.rl_agent import RLAgent
from evaluation.baseline import StaticProtocol, GreedyProtocol
from evaluation.evaluator import Evaluator
from utils.logger import ExperimentLogger
from utils.helpers import save_results, print_section


def run_dynamic_experiment(config_path: str = 'config/simulation_config.yaml',
                          rl_config_path: str = 'config/rl_config.yaml'):
    """
    Run experiment with dynamic (time-varying) channel conditions
    
    Args:
        config_path: Path to simulation config
        rl_config_path: Path to RL config
    """
    print_section("DYNAMIC SCENARIO EXPERIMENT")
    
    # Setup logger
    logger = ExperimentLogger('dynamic_experiment')
    
    # Load configurations
    with open(config_path, 'r') as f:
        sim_config = yaml.safe_load(f)
    
    with open(rl_config_path, 'r') as f:
        rl_config = yaml.safe_load(f)
    
    # Modify config for dynamic scenario
    sim_config['noise']['enable_noise'] = True
    sim_config['noise']['noise_model'] = 'depolarizing'
    
    logger.start_experiment({
        'scenario': 'dynamic',
        'noise_variation': 'time-varying'
    })
    
    # Create environment
    print("Creating environment with dynamic noise...")
    env = QuantumEnvironment(sim_config)
    
    # Create and train RL agent
    print("Creating RL agent...")
    agent = RLAgent(env, rl_config)
    
    print("Training RL agent on dynamic conditions...")
    training_timesteps = rl_config.get('training', {}).get('total_timesteps', 100000)
    agent.train(timesteps=training_timesteps, save_path='results/model/dynamic')
    
    # Test adaptation capability
    print("\nTesting adaptation to varying noise levels...")
    noise_levels = [0.001, 0.01, 0.05, 0.1]
    
    rl_results = []
    static_results = []
    
    for noise_level in noise_levels:
        print(f"\nTesting at noise level: {noise_level}")
        
        # Update environment noise
        sim_config['noise']['gate_error_rate'] = noise_level
        test_env = QuantumEnvironment(sim_config)
        
        # Evaluate RL agent
        rl_eval = Evaluator(test_env, agent)
        rl_metrics = rl_eval.evaluate(num_episodes=50, verbose=False)
        rl_results.append({
            'noise_level': noise_level,
            'fidelity': rl_metrics['mean_fidelity'],
            'success_rate': rl_metrics['mean_success_rate'],
            'strategy': 'RL Agent'
        })
        
        # Evaluate static baseline
        static_protocol = StaticProtocol(sim_config)
        static_eval = Evaluator(test_env, static_protocol)
        static_metrics = static_eval.evaluate(num_episodes=50, verbose=False)
        static_results.append({
            'noise_level': noise_level,
            'fidelity': static_metrics['mean_fidelity'],
            'success_rate': static_metrics['mean_success_rate'],
            'strategy': 'Static Protocol'
        })
    
    # Plot adaptation results
    print("\nGenerating adaptation plots...")
    plot_adaptation_results(rl_results, static_results)
    
    # Combine and save results
    all_results = rl_results + static_results
    save_results(all_results, 'results/dynamic_results.json')
    
    logger.log_result(all_results)
    
    print("\nDynamic experiment complete!")
    print(f"Results saved to results/")


def plot_adaptation_results(rl_results, static_results):
    """Plot how strategies adapt to varying noise"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    noise_levels = [r['noise_level'] for r in rl_results]
    rl_fidelities = [r['fidelity'] for r in rl_results]
    static_fidelities = [r['fidelity'] for r in static_results]
    rl_success = [r['success_rate'] for r in rl_results]
    static_success = [r['success_rate'] for r in static_results]
    
    # Plot 1: Fidelity vs Noise
    axes[0].plot(noise_levels, rl_fidelities, 'o-', label='RL Agent', 
                linewidth=2, markersize=8, color='green')
    axes[0].plot(noise_levels, static_fidelities, 's--', label='Static Protocol',
                linewidth=2, markersize=8, color='red')
    axes[0].set_xlabel('Noise Level')
    axes[0].set_ylabel('Mean Fidelity')
    axes[0].set_title('Adaptation to Varying Noise Levels')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0.95, color='gray', linestyle=':', label='Target (0.95)')
    
    # Plot 2: Success Rate vs Noise
    axes[1].plot(noise_levels, rl_success, 'o-', label='RL Agent',
                linewidth=2, markersize=8, color='green')
    axes[1].plot(noise_levels, static_success, 's--', label='Static Protocol',
                linewidth=2, markersize=8, color='red')
    axes[1].set_xlabel('Noise Level')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_title('Success Rate Under Varying Noise')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/dynamic_adaptation.png', dpi=300, bbox_inches='tight')
    print("Adaptation plot saved to results/plots/dynamic_adaptation.png")
    plt.show()


if __name__ == "__main__":
    run_dynamic_experiment()
