"""
Main entry point for Quantum Communication RL Project
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from environment.quantum_environment import QuantumEnvironment
from rl.rl_agent import RLAgent
from evaluation.evaluator import Evaluator
from utils.logger import setup_logger


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    """Main execution function"""
    # Setup logger
    logger = setup_logger('quantum_rl', 'results/logs/main.log')
    logger.info("Starting Quantum Communication RL Project")
    
    # Load configurations
    sim_config = load_config(args.sim_config)
    rl_config = load_config(args.rl_config)
    
    logger.info(f"Loaded simulation config: {args.sim_config}")
    logger.info(f"Loaded RL config: {args.rl_config}")
    
    # Initialize environment
    env = QuantumEnvironment(sim_config)
    logger.info("Quantum environment initialized")
    
    # Initialize RL agent
    agent = RLAgent(env, rl_config)
    logger.info("RL agent initialized")
    
    if args.mode == 'train':
        # Train the agent
        logger.info("Starting training...")
        agent.train(timesteps=rl_config.get('total_timesteps', 100000))
        agent.save(args.model_path)
        logger.info(f"Model saved to {args.model_path}")
        
    elif args.mode == 'evaluate':
        # Load and evaluate
        logger.info("Loading model for evaluation...")
        agent.load(args.model_path)
        
        evaluator = Evaluator(env, agent)
        results = evaluator.evaluate(num_episodes=args.num_episodes)
        
        logger.info("Evaluation complete")
        logger.info(f"Results: {results}")
        
    elif args.mode == 'experiment':
        # Run experiments
        logger.info("Running experiments...")
        from experiments.compare_results import run_comparison
        run_comparison(sim_config, rl_config)
    
    logger.info("Execution complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum Communication RL Project')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'experiment'],
                        help='Execution mode')
    parser.add_argument('--sim_config', type=str, 
                        default='config/simulation_config.yaml',
                        help='Path to simulation config')
    parser.add_argument('--rl_config', type=str, 
                        default='config/rl_config.yaml',
                        help='Path to RL config')
    parser.add_argument('--model_path', type=str, 
                        default='results/model/rl_agent',
                        help='Path to save/load model')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    main(args)
