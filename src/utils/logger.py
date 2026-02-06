"""
Logger Module
Provides logging functionality for the quantum communication system
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(name: str, 
                log_file: Optional[str] = None,
                level: int = logging.INFO,
                console_output: bool = True) -> logging.Logger:
    """
    Setup a logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Logger specifically for experiment tracking
    """
    
    def __init__(self, experiment_name: str, log_dir: str = 'results/logs'):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(
            f"experiment.{experiment_name}",
            str(log_file),
            level=logging.INFO
        )
        
        # Experiment data
        self.metrics = {}
        self.start_time = None
        
    def start_experiment(self, config: dict):
        """
        Log experiment start
        
        Args:
            config: Experiment configuration
        """
        self.start_time = datetime.now()
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Configuration: {config}")
        
    def log_step(self, step: int, metrics: dict):
        """
        Log a step in the experiment
        
        Args:
            step: Step number
            metrics: Metrics dictionary
        """
        self.logger.debug(f"Step {step}: {metrics}")
        
        # Store metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def log_epoch(self, epoch: int, metrics: dict):
        """
        Log an epoch in training
        
        Args:
            epoch: Epoch number
            metrics: Metrics dictionary
        """
        self.logger.info(f"Epoch {epoch}: {metrics}")
    
    def log_result(self, results: dict):
        """
        Log final results
        
        Args:
            results: Results dictionary
        """
        self.logger.info(f"Final results: {results}")
        
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"Total time: {elapsed:.2f} seconds")
    
    def save_metrics(self, save_path: Optional[str] = None):
        """
        Save experiment metrics to file
        
        Args:
            save_path: Path to save metrics (optional)
        """
        import json
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.log_dir / f"{self.experiment_name}_metrics_{timestamp}.json"
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {save_path}")


class TensorBoardLogger:
    """
    Wrapper for TensorBoard logging
    """
    
    def __init__(self, log_dir: str = 'results/tensorboard'):
        """
        Initialize TensorBoard logger
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
            self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars"""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close the writer"""
        if self.enabled and self.writer:
            self.writer.close()
