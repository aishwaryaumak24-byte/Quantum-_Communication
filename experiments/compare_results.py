"""
Results Comparison Script
Compares and visualizes results from different experiments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import print_section, ensure_dir


def load_experiment_results(results_dir: str = 'results'):
    """
    Load all experiment results
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dictionary of loaded results
    """
    results = {}
    results_path = Path(results_dir)
    
    # Load static results
    static_file = results_path / 'static_results.json'
    if static_file.exists():
        with open(static_file, 'r') as f:
            results['static'] = json.load(f)
            print(f"Loaded static results from {static_file}")
    
    # Load dynamic results
    dynamic_file = results_path / 'dynamic_results.json'
    if dynamic_file.exists():
        with open(dynamic_file, 'r') as f:
            results['dynamic'] = json.load(f)
            print(f"Loaded dynamic results from {dynamic_file}")
    
    return results


def compare_static_results(static_results):
    """
    Create comparison visualizations for static experiments
    
    Args:
        static_results: Static experiment results
    """
    print_section("STATIC EXPERIMENT COMPARISON")
    
    df = pd.DataFrame(static_results)
    
    print("\nStatic Scenario Results:")
    print(df.to_string(index=False))
    
    # Find best strategy
    best_strategy = df.loc[df['mean_fidelity'].idxmax(), 'strategy']
    best_fidelity = df['mean_fidelity'].max()
    
    print(f"\nBest Strategy: {best_strategy}")
    print(f"Best Fidelity: {best_fidelity:.4f}")
    
    # Calculate improvement
    rl_fidelity = df[df['strategy'] == 'RL Agent (Proposed)']['mean_fidelity'].values[0]
    baseline_fidelities = df[df['strategy'] != 'RL Agent (Proposed)']['mean_fidelity']
    best_baseline = baseline_fidelities.max()
    
    improvement = ((rl_fidelity - best_baseline) / best_baseline) * 100
    print(f"RL Improvement over best baseline: {improvement:.2f}%")


def compare_dynamic_results(dynamic_results):
    """
    Create comparison visualizations for dynamic experiments
    
    Args:
        dynamic_results: Dynamic experiment results
    """
    print_section("DYNAMIC EXPERIMENT COMPARISON")
    
    df = pd.DataFrame(dynamic_results)
    
    # Separate by strategy
    rl_df = df[df['strategy'] == 'RL Agent']
    static_df = df[df['strategy'] == 'Static Protocol']
    
    print("\nRL Agent Performance Across Noise Levels:")
    print(rl_df[['noise_level', 'fidelity', 'success_rate']].to_string(index=False))
    
    print("\nStatic Protocol Performance Across Noise Levels:")
    print(static_df[['noise_level', 'fidelity', 'success_rate']].to_string(index=False))
    
    # Calculate average improvement
    avg_rl_fidelity = rl_df['fidelity'].mean()
    avg_static_fidelity = static_df['fidelity'].mean()
    improvement = ((avg_rl_fidelity - avg_static_fidelity) / avg_static_fidelity) * 100
    
    print(f"\nAverage RL Fidelity: {avg_rl_fidelity:.4f}")
    print(f"Average Static Fidelity: {avg_static_fidelity:.4f}")
    print(f"Average Improvement: {improvement:.2f}%")


def create_comprehensive_comparison(results):
    """
    Create comprehensive comparison plots
    
    Args:
        results: All experiment results
    """
    print_section("COMPREHENSIVE COMPARISON")
    
    ensure_dir('results/plots')
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Static Comparison
    if 'static' in results:
        ax1 = fig.add_subplot(gs[0, 0])
        df_static = pd.DataFrame(results['static'])
        
        strategies = df_static['strategy']
        fidelities = df_static['mean_fidelity']
        
        bars = ax1.bar(range(len(strategies)), fidelities, alpha=0.7, edgecolor='black')
        bars[0].set_color('green')
        bars[0].set_alpha(0.9)
        
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.set_ylabel('Mean Fidelity')
        ax1.set_title('Static Scenario: Strategy Comparison')
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Dynamic Adaptation
    if 'dynamic' in results:
        ax2 = fig.add_subplot(gs[0, 1])
        df_dynamic = pd.DataFrame(results['dynamic'])
        
        for strategy in df_dynamic['strategy'].unique():
            data = df_dynamic[df_dynamic['strategy'] == strategy]
            marker = 'o-' if strategy == 'RL Agent' else 's--'
            color = 'green' if strategy == 'RL Agent' else 'red'
            ax2.plot(data['noise_level'], data['fidelity'], marker,
                    label=strategy, linewidth=2, color=color)
        
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Fidelity')
        ax2.set_title('Dynamic Scenario: Noise Adaptation')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # Plot 3: Success Rate Comparison
    if 'static' in results:
        ax3 = fig.add_subplot(gs[1, 0])
        df_static = pd.DataFrame(results['static'])
        
        strategies = df_static['strategy']
        success_rates = df_static['mean_success_rate']
        
        bars = ax3.bar(range(len(strategies)), success_rates, alpha=0.7, edgecolor='black')
        bars[0].set_color('green')
        bars[0].set_alpha(0.9)
        
        ax3.set_xticks(range(len(strategies)))
        ax3.set_xticklabels(strategies, rotation=45, ha='right')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate Comparison')
        ax3.set_ylim([0, 1.0])
        ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Robustness Analysis
    if 'dynamic' in results:
        ax4 = fig.add_subplot(gs[1, 1])
        df_dynamic = pd.DataFrame(results['dynamic'])
        
        for strategy in df_dynamic['strategy'].unique():
            data = df_dynamic[df_dynamic['strategy'] == strategy]
            marker = 'o-' if strategy == 'RL Agent' else 's--'
            color = 'green' if strategy == 'RL Agent' else 'red'
            ax4.plot(data['noise_level'], data['success_rate'], marker,
                    label=strategy, linewidth=2, color=color)
        
        ax4.set_xlabel('Noise Level')
        ax4.set_ylabel('Success Rate')
        ax4.set_title('Robustness: Success Rate vs Noise')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    plt.savefig('results/plots/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("Comprehensive comparison plot saved to results/plots/comprehensive_comparison.png")
    plt.show()


def generate_summary_report(results):
    """
    Generate a text summary report
    
    Args:
        results: All experiment results
    """
    print_section("SUMMARY REPORT")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("ARL-QDCC PROJECT - EXPERIMENT SUMMARY REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Static results
    if 'static' in results:
        report_lines.append("1. STATIC SCENARIO RESULTS")
        report_lines.append("-" * 70)
        df = pd.DataFrame(results['static'])
        rl_result = df[df['strategy'] == 'RL Agent (Proposed)'].iloc[0]
        report_lines.append(f"RL Agent - Mean Fidelity: {rl_result['mean_fidelity']:.4f}")
        report_lines.append(f"RL Agent - Success Rate: {rl_result['mean_success_rate']:.4f}")
        report_lines.append("")
    
    # Dynamic results
    if 'dynamic' in results:
        report_lines.append("2. DYNAMIC SCENARIO RESULTS")
        report_lines.append("-" * 70)
        df = pd.DataFrame(results['dynamic'])
        rl_df = df[df['strategy'] == 'RL Agent']
        report_lines.append(f"RL Agent - Average Fidelity: {rl_df['fidelity'].mean():.4f}")
        report_lines.append(f"RL Agent - Average Success Rate: {rl_df['success_rate'].mean():.4f}")
        report_lines.append("")
    
    report_lines.append("=" * 70)
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report
    with open('results/summary_report.txt', 'w') as f:
        f.write(report_text)
    print("\nSummary report saved to results/summary_report.txt")


def main():
    """Main comparison function"""
    print_section("RESULTS COMPARISON AND ANALYSIS")
    
    # Load all results
    results = load_experiment_results()
    
    if not results:
        print("No results found. Please run experiments first.")
        return
    
    # Compare static results
    if 'static' in results:
        compare_static_results(results['static'])
    
    # Compare dynamic results
    if 'dynamic' in results:
        compare_dynamic_results(results['dynamic'])
    
    # Create comprehensive comparison
    create_comprehensive_comparison(results)
    
    # Generate summary report
    generate_summary_report(results)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
