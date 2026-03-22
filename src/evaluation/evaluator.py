"""
Evaluator Module
Implements comprehensive evaluation and comparison of RL agent vs baselines
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import time

from .baseline import BaselineStrategy
from simulator.metrics import PerformanceMetrics


class Evaluator:
    """
    Evaluates the performance of RL agent or baseline strategy
    """
    
    def __init__(self, env, agent_or_strategy):
        """
        Initialize evaluator
        
        Args:
            env: Quantum communication environment
            agent_or_strategy: RL agent or baseline strategy to evaluate
        """
        self.env = env
        self.agent = agent_or_strategy
        self.metrics = PerformanceMetrics()
        
    def evaluate(self, num_episodes: int = 100, 
                deterministic: bool = True,
                verbose: bool = True) -> Dict:
        """
        Evaluate performance over multiple episodes
        
        Args:
            num_episodes: Number of episodes to run
            deterministic: Whether to use deterministic policy
            verbose: Whether to print progress
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = []
        episode_fidelities = []
        episode_success_rates = []
        episode_lengths = []
        
        iterator = tqdm(range(num_episodes), desc="Evaluating") if verbose else range(num_episodes)
        
        for episode in iterator:
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Get action
                if hasattr(self.agent, 'predict'):
                    # RL agent
                    action = self.agent.predict(obs, deterministic=deterministic)
                else:
                    # Baseline strategy
                    action = self.agent.select_action(obs)
                
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            # Record metrics
            episode_rewards.append(episode_reward)
            episode_fidelities.append(info.get('avg_fidelity', 0.0))
            episode_success_rates.append(
                info.get('successful_transmissions', 0) / max(1, info.get('step', 1))
            )
            episode_lengths.append(episode_length)
            
            # Reset baseline if needed
            if hasattr(self.agent, 'reset') and not hasattr(self.agent, 'predict'):
                self.agent.reset()
        
        # Compile results
        results = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            
            'mean_fidelity': float(np.mean(episode_fidelities)),
            'std_fidelity': float(np.std(episode_fidelities)),
            'min_fidelity': float(np.min(episode_fidelities)),
            'max_fidelity': float(np.max(episode_fidelities)),
            
            'mean_success_rate': float(np.mean(episode_success_rates)),
            'std_success_rate': float(np.std(episode_success_rates)),
            
            'mean_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths)),
            
            'num_episodes': num_episodes
        }
        
        if verbose:
            print("\n=== Evaluation Results ===")
            print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
            print(f"Mean Fidelity: {results['mean_fidelity']:.3f} ± {results['std_fidelity']:.3f}")
            print(f"Mean Success Rate: {results['mean_success_rate']:.3f} ± {results['std_success_rate']:.3f}")
            print(f"Mean Episode Length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
        
        return results


class ComparisonEvaluator:
    """
    Compares RL agent against multiple baseline strategies
    """
    
    def __init__(self, env):
        """
        Initialize comparison evaluator
        
        Args:
            env: Quantum communication environment
        """
        self.env = env
        
    def compare_strategies(self, 
                          rl_agent,
                          baseline_strategies: List[BaselineStrategy],
                          num_episodes: int = 100,
                          save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare RL agent against baseline strategies
        
        Args:
            rl_agent: Trained RL agent
            baseline_strategies: List of baseline strategies
            num_episodes: Number of episodes for evaluation
            save_path: Optional path to save results
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        # Evaluate RL agent
        try:
            print("Evaluating RL Agent...")
            rl_evaluator = Evaluator(self.env, rl_agent)
            rl_results = rl_evaluator.evaluate(num_episodes, verbose=False)
            rl_results['strategy'] = 'RL Agent (Proposed)'
            results.append(rl_results)
        except Exception as e:
            print(f"ERROR: Failed to evaluate RL Agent: {e}")
            return pd.DataFrame()  # Return empty DataFrame on failure
        
        # Evaluate baselines
        for baseline in baseline_strategies:
            try:
                print(f"Evaluating {baseline.get_name()}...")
                baseline_evaluator = Evaluator(self.env, baseline)
                baseline_results = baseline_evaluator.evaluate(num_episodes, verbose=False)
                baseline_results['strategy'] = baseline.get_name()
                results.append(baseline_results)
            except Exception as e:
                print(f"ERROR: Failed to evaluate {baseline.get_name()}: {e}")
                continue  # Skip this baseline and continue
        
        if not results:
            print("ERROR: No evaluation results obtained.")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns - ensure all columns exist before selecting
        desired_cols = ['strategy', 'mean_reward', 'mean_fidelity', 'mean_success_rate', 
                        'mean_episode_length', 'std_reward', 'std_fidelity', 'std_success_rate']
        available_cols = [col for col in desired_cols if col in df.columns]
        
        if not available_cols:
            print(f"ERROR: No expected columns found in DataFrame.")
            print(f"Available columns: {list(df.columns)}")
            return df
        
        df = df[available_cols]
        
        # Save if requested
        if save_path:
            try:
                df.to_csv(save_path, index=False)
                print(f"\nResults saved to {save_path}")
            except Exception as e:
                print(f"ERROR: Failed to save results to {save_path}: {e}")
        
        # Print comparison table
        print("\n=== Strategy Comparison ===")
        print(df.to_string(index=False))
        
        return df
    
    def plot_comparison(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot comparison results
        
        Args:
            df: DataFrame with comparison results
            save_path: Optional path to save plots
        """
        # Validate DataFrame
        if df.empty:
            print("ERROR: DataFrame is empty. Cannot generate plots.")
            return
        
        required_columns = ['strategy', 'mean_reward', 'std_reward', 'mean_fidelity', 
                           'std_fidelity', 'mean_success_rate', 'std_success_rate', 
                           'mean_episode_length', 'std_episode_length']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing columns in DataFrame: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Reset index to avoid indexing issues
        df = df.reset_index(drop=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Mean Reward comparison
        ax = axes[0, 0]
        strategies = df['strategy'].values.tolist()  # Convert to list
        rewards = df['mean_reward'].values  # Use .values for numpy array
        errors = df['std_reward'].values
        
        bars = ax.bar(range(len(strategies)), rewards, yerr=errors, 
                     capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        if len(bars) > 0:
            bars[0].set_color('green')  # Highlight RL agent
            bars[0].set_alpha(0.9)
        
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Mean Reward Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Mean Fidelity comparison
        ax = axes[0, 1]
        fidelities = df['mean_fidelity'].values
        errors = df['std_fidelity'].values
        
        bars = ax.bar(range(len(strategies)), fidelities, yerr=errors,
                     capsize=5, alpha=0.7, color='lightcoral', edgecolor='darkred')
        if len(bars) > 0:
            bars[0].set_color('green')
            bars[0].set_alpha(0.9)
        
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_ylabel('Mean Fidelity')
        ax.set_title('Mean Fidelity Comparison')
        ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target (0.95)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Success Rate comparison
        ax = axes[1, 0]
        success_rates = df['mean_success_rate'].values
        errors = df['std_success_rate'].values
        
        bars = ax.bar(range(len(strategies)), success_rates, yerr=errors,
                     capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        if len(bars) > 0:
            bars[0].set_color('green')
            bars[0].set_alpha(0.9)
        
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_ylabel('Mean Success Rate')
        ax.set_title('Success Rate Comparison')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Episode Length comparison
        ax = axes[1, 1]
        lengths = df['mean_episode_length'].values
        errors = df['std_episode_length'].values
        
        bars = ax.bar(range(len(strategies)), lengths, yerr=errors,
                     capsize=5, alpha=0.7, color='plum', edgecolor='purple')
        if len(bars) > 0:
            bars[0].set_color('green')
            bars[0].set_alpha(0.9)
        
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_ylabel('Mean Episode Length')
        ax.set_title('Episode Length Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            except Exception as e:
                print(f"ERROR: Failed to save plot to {save_path}: {e}")
        
        try:
            plt.show()
        except Exception as e:
            print(f"WARNING: Could not display plot: {e}")
    
    def statistical_significance_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform statistical significance tests
        
        Args:
            df: DataFrame with comparison results
            
        Returns:
            DataFrame with significance test results
        """
        from scipy import stats
        
        # This is a placeholder - actual implementation would require
        # running episodes and collecting samples for proper statistical tests
        
        print("\n=== Statistical Significance Analysis ===")
        print("Note: Detailed statistical tests require episode-level data.")
        print("Improvement over best baseline:")
        
        rl_reward = df.iloc[0]['mean_reward']
        baseline_rewards = df.iloc[1:]['mean_reward']
        best_baseline = baseline_rewards.max()
        
        improvement = ((rl_reward - best_baseline) / best_baseline) * 100
        print(f"Reward improvement: {improvement:.2f}%")
        
        rl_fidelity = df.iloc[0]['mean_fidelity']
        baseline_fidelities = df.iloc[1:]['mean_fidelity']
        best_baseline_fid = baseline_fidelities.max()
        
        improvement_fid = ((rl_fidelity - best_baseline_fid) / best_baseline_fid) * 100
        print(f"Fidelity improvement: {improvement_fid:.2f}%")
        
        return df
