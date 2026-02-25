"""
Run inference using a trained RL model.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment.quantum_environment import QuantumEnvironment
from rl.rl_agent import RLAgent
from evaluation.evaluator import Evaluator
from utils.helpers import load_config, save_results


def run_inference(args):
    """Load a trained model and run evaluation episodes."""
    sim_config = load_config(args.sim_config)
    rl_config = load_config(args.rl_config)

    env = QuantumEnvironment(sim_config)
    agent = RLAgent(env, rl_config)

    print(f"Loading model from: {args.model_path}")
    agent.load(args.model_path)

    evaluator = Evaluator(env, agent)
    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        deterministic=not args.stochastic,
        verbose=True
    )

    if args.save_path:
        save_results(results, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained RL model")
    parser.add_argument(
        "--sim_config",
        type=str,
        default="config/simulation_config.yaml",
        help="Path to simulation config"
    )
    parser.add_argument(
        "--rl_config",
        type=str,
        default="config/rl_config.yaml",
        help="Path to RL config"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/model/static/best_model/best_model.zip",
        help="Path to the trained model (.zip)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Optional path to save results JSON"
    )

    run_inference(parser.parse_args())
