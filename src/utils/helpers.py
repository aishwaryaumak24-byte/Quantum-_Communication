"""
Helper Functions
Utility functions for the quantum communication system
"""

import numpy as np
import json
import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def save_config(config: Dict, filepath: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_results(results: Dict, filepath: str, format: str = 'json'):
    """
    Save results to file
    
    Args:
        results: Results dictionary
        filepath: Path to save file
        format: Format ('json' or 'pickle')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str, format: str = 'json') -> Dict:
    """
    Load results from file
    
    Args:
        filepath: Path to results file
        format: Format ('json' or 'pickle')
        
    Returns:
        Results dictionary
    """
    if format == 'json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return results


def plot_training_curve(metrics: Dict, save_path: Optional[str] = None):
    """
    Plot training curves
    
    Args:
        metrics: Dictionary with training metrics
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward curve
    if 'rewards' in metrics:
        axes[0, 0].plot(metrics['rewards'])
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(alpha=0.3)
    
    # Fidelity curve
    if 'fidelities' in metrics:
        axes[0, 1].plot(metrics['fidelities'])
        axes[0, 1].set_title('Average Fidelity')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Fidelity')
        axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='Target')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
    
    # Success rate curve
    if 'success_rates' in metrics:
        axes[1, 0].plot(metrics['success_rates'])
        axes[1, 0].set_title('Success Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].grid(alpha=0.3)
    
    # Loss curve (if available)
    if 'losses' in metrics:
        axes[1, 1].plot(metrics['losses'])
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def moving_average(data: List[float], window: int = 10) -> np.ndarray:
    """
    Calculate moving average
    
    Args:
        data: List of values
        window: Window size
        
    Returns:
        Moving average array
    """
    data_array = np.array(data)
    weights = np.ones(window) / window
    return np.convolve(data_array, weights, mode='valid')


def normalize_data(data: np.ndarray, 
                   min_val: Optional[float] = None,
                   max_val: Optional[float] = None) -> np.ndarray:
    """
    Normalize data to [0, 1] range
    
    Args:
        data: Input data
        min_val: Minimum value (if None, use data min)
        max_val: Maximum value (if None, use data max)
        
    Returns:
        Normalized data
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    
    if max_val - min_val == 0:
        return np.zeros_like(data)
    
    return (data - min_val) / (max_val - min_val)


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for data
    
    Args:
        data: List of values
        
    Returns:
        Dictionary with statistics
    """
    data_array = np.array(data)
    
    return {
        'mean': float(np.mean(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'median': float(np.median(data_array)),
        'q25': float(np.percentile(data_array, 25)),
        'q75': float(np.percentile(data_array, 75))
    }


def create_comparison_table(results: List[Dict], 
                           metrics: List[str],
                           save_path: Optional[str] = None) -> str:
    """
    Create a formatted comparison table
    
    Args:
        results: List of result dictionaries
        metrics: List of metric names to compare
        save_path: Optional path to save table
        
    Returns:
        Formatted table string
    """
    import pandas as pd
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Select metrics
    if 'strategy' in df.columns:
        cols = ['strategy'] + [m for m in metrics if m in df.columns]
    else:
        cols = [m for m in metrics if m in df.columns]
    
    df_display = df[cols]
    
    # Format table
    table_str = df_display.to_string(index=False)
    
    # Save if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"Table saved to {save_path}")
    
    return table_str


def plot_heatmap(data: np.ndarray, 
                labels_x: Optional[List[str]] = None,
                labels_y: Optional[List[str]] = None,
                title: str = "Heatmap",
                save_path: Optional[str] = None):
    """
    Plot a heatmap
    
    Args:
        data: 2D numpy array
        labels_x: Labels for x-axis
        labels_y: Labels for y-axis
        title: Plot title
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=labels_x, yticklabels=labels_y,
                cbar_kws={'label': 'Value'})
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()


def ensure_dir(directory: str):
    """
    Ensure directory exists
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """
    Get current timestamp string
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def print_section(title: str, width: int = 60):
    """
    Print a formatted section header
    
    Args:
        title: Section title
        width: Width of the header
    """
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")
