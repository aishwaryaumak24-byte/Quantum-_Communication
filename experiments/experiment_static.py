"""
Static Scenario Experiment
Tests RL agent on static (unchanging) channel conditions
"""

import sys
from pathlib import Path
import re
import argparse
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import yaml
from environment.quantum_environment import QuantumEnvironment
from rl.rl_agent import RLAgent
from evaluation.baseline import StaticProtocol, GreedyProtocol, ThresholdBasedProtocol
from evaluation.evaluator import ComparisonEvaluator
from utils.logger import ExperimentLogger
from utils.helpers import save_results, plot_training_curve, print_section


def _find_resume_model_path(save_root: Path) -> tuple[Path | None, int | None]:
    """Find latest checkpoint to resume from, else best model.

    Returns:
        (model_path, completed_timesteps)
        completed_timesteps is available for checkpoint models.
    """
    checkpoints_dir = save_root / 'checkpoints'
    best_model_path = save_root / 'best_model' / 'best_model.zip'

    latest_checkpoint = None
    max_steps = -1

    if checkpoints_dir.exists():
        for checkpoint in checkpoints_dir.glob('rl_model_*_steps.zip'):
            match = re.search(r'rl_model_(?:interrupted_)?(\d+)_steps\.zip$', checkpoint.name)
            if match:
                steps = int(match.group(1))
                if steps > max_steps:
                    max_steps = steps
                    latest_checkpoint = checkpoint

    if latest_checkpoint is not None:
        return latest_checkpoint, max_steps

    if best_model_path.exists():
        return best_model_path, None

    return None, 0


def run_static_experiment(config_path: str = 'config/simulation_config.yaml',
                         rl_config_path: str = 'config/rl_config.yaml',
                         num_eval_episodes: int = 100,
                         save_root_dir: str = 'results/model/static',
                         static_gate_error_rate: float | None = None):
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
    
    # Enforce static scenario (fixed noise model).
    # Keep configured gate error rate unless explicitly overridden.
    sim_config['noise']['enable_noise'] = True
    sim_config['noise']['noise_model'] = 'depolarizing'
    if static_gate_error_rate is not None:
        sim_config['noise']['gate_error_rate'] = static_gate_error_rate
    
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

    save_root = Path(save_root_dir)
    resume_model_path, completed_timesteps = _find_resume_model_path(save_root)
    resume_training = resume_model_path is not None

    if resume_training:
        print(f"Resuming from saved model: {resume_model_path}")
        agent.load(str(resume_model_path))
    else:
        print("No saved model found. Starting fresh training.")
    
    print("Training RL agent...")
    target_total_timesteps = rl_config.get('training', {}).get('total_timesteps', 50000)

    if resume_training and completed_timesteps is not None:
        remaining_timesteps = max(0, target_total_timesteps - completed_timesteps)
        print(
            f"Target total timesteps: {target_total_timesteps} | "
            f"Completed: {completed_timesteps} | Remaining: {remaining_timesteps}"
        )
    else:
        remaining_timesteps = target_total_timesteps

    if remaining_timesteps > 0:
        agent.train(
            timesteps=remaining_timesteps,
            save_path=str(save_root),
            reset_num_timesteps=not resume_training
        )
    else:
        print("Target total timesteps already reached. Skipping training.")
    
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
        num_episodes=num_eval_episodes,
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
    parser = argparse.ArgumentParser(description='Run static scenario experiment')
    parser.add_argument('--sim_config', type=str, default='config/simulation_config.yaml',
                        help='Path to simulation config file')
    parser.add_argument('--rl_config', type=str, default='config/rl_config.yaml',
                        help='Path to RL config file')
    parser.add_argument('--num_eval_episodes', type=int, default=100,
                        help='Number of episodes for strategy comparison')
    parser.add_argument('--save_root', type=str, default='results/model/static',
                        help='Directory for model checkpoints and best model')
    parser.add_argument('--static_gate_error_rate', type=float, default=None,
                        help='Optional fixed gate error rate override for static scenario')

    cli_args = parser.parse_args()
    run_static_experiment(
        config_path=cli_args.sim_config,
        rl_config_path=cli_args.rl_config,
        num_eval_episodes=cli_args.num_eval_episodes,
        save_root_dir=cli_args.save_root,
        static_gate_error_rate=cli_args.static_gate_error_rate
    )
