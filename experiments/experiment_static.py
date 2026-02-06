"""
Static Scenario Experiment
Tests RL agent on static (unchanging) channel conditions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import yaml
from environment.quantum_environment import QuantumEnvironment
from rl.rl_agent import RLAgent
from evaluation.baseline import StaticProtocol, GreedyProtocol, ThresholdBasedProtocol
from evaluation.evaluator import ComparisonEvaluator
from utils.logger import ExperimentLogger
from utils.helpers import save_results, plot_training_curve, print_section


def run_static_experiment(config_path: str = 'config/simulation_config.yaml',
                         rl_config_path: str = 'config/rl_config.yaml'):
    """
    Run experiment with static channel conditions
    
    Args:
        config_path: Path to simulation config
        rl_config_path: Path to RL config
    """
    print_section("STATIC SCENARIO EXPERIMENT")
    
    # Setup logger
    logger = ExperimentLogger('static_experiment')
    
    # Load configurations
    with open(config_path, 'r') as f:
        sim_config = yaml.safe_load(f)
    
    with open(rl_config_path, 'r') as f:
        rl_config = yaml.safe_load(f)
    
    # Modify config for static scenario
    sim_config['noise']['enable_noise'] = True
    sim_config['noise']['noise_model'] = 'depolarizing'
    sim_config['noise']['gate_error_rate'] = 0.01  # Fixed error rate
    
    logger.start_experiment({
        'scenario': 'static',
        'noise_level': sim_config['noise']['gate_error_rate']
    })
    
    # Create environment
    print("Creating environment...")
    env = QuantumEnvironment(sim_config)
    
    # Create and train RL agent
    print("Creating RL agent...")
    agent = RLAgent(env, rl_config)
    
    print("Training RL agent...")
    training_timesteps = rl_config.get('training', {}).get('total_timesteps', 50000)
    agent.train(timesteps=training_timesteps, save_path='results/model/static')
    
    # Create baseline strategies
    baselines = [
        StaticProtocol(sim_config),
        GreedyProtocol(),
        ThresholdBasedProtocol(purify_threshold=0.8, swap_threshold=0.95)
    ]
    
    # Compare strategies
    print("\nComparing strategies...")
    evaluator = ComparisonEvaluator(env)
    results_df = evaluator.compare_strategies(
        agent, 
        baselines,
        num_episodes=100,
        save_path='results/tables/static_comparison.csv'
    )
    
    # Generate plots
    print("\nGenerating comparison plots...")
    evaluator.plot_comparison(
        results_df,
        save_path='results/plots/static_comparison.png'
    )
    
    # Log results
    results_dict = results_df.to_dict('records')
    logger.log_result(results_dict)
    
    # Save results
    save_results(results_dict, 'results/static_results.json')
    
    print("\nStatic experiment complete!")
    print(f"Results saved to results/")


if __name__ == "__main__":
    run_static_experiment()
