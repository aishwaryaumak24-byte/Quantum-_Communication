"""
Quantum Repeater Placement Optimization
Implements FR-7: Cost-aware and dynamic quantum repeater placement
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
import itertools


class RepeaterPlacement:
    """
    Manages quantum repeater placement for long-distance communication
    
    Implements FR-7: Determine cost-aware and dynamic quantum repeater 
    placement decisions based on channel conditions
    """
    
    def __init__(self, config: Dict):
        """
        Initialize repeater placement module
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        repeater_config = config.get('repeater', {})
        self.max_repeaters = repeater_config.get('max_repeaters', 5)
        self.swap_success_prob = repeater_config.get('swap_success_probability', 0.98)
        self.memory_coherence_time = repeater_config.get('memory_coherence_time', 1.0)
        
        # Cost parameters
        self.repeater_cost = 10.0  # Cost per repeater
        self.placement_cost = 2.0  # Cost to place/relocate repeater
        
        # Current repeater configuration
        self.repeater_positions = []
        self.repeater_active = []
        
    def initialize_placement(self, total_distance: float, 
                           num_nodes: int) -> List[float]:
        """
        Initialize repeater placement evenly along the path
        
        Args:
            total_distance: Total communication distance
            num_nodes: Number of communication nodes
            
        Returns:
            List of repeater positions
        """
        # Start with evenly spaced repeaters
        num_repeaters = min(self.max_repeaters, max(1, int(total_distance / 50)))
        
        positions = np.linspace(0, total_distance, num_repeaters + 2)[1:-1]
        self.repeater_positions = positions.tolist()
        self.repeater_active = [True] * len(self.repeater_positions)
        
        return self.repeater_positions
    
    def optimize_placement(self, distance_segments: np.ndarray,
                          noise_levels: np.ndarray,
                          fidelity_targets: Optional[np.ndarray] = None) -> Tuple[List[float], float]:
        """
        Optimize repeater placement based on channel conditions
        
        Args:
            distance_segments: Array of distances between nodes
            noise_levels: Array of noise levels for each segment
            fidelity_targets: Optional target fidelities for each segment
            
        Returns:
            Tuple of (optimal_positions, total_cost)
        """
        total_distance = np.sum(distance_segments)
        
        if fidelity_targets is None:
            fidelity_targets = np.ones(len(distance_segments)) * 0.95
        
        # Objective: minimize cost while meeting fidelity targets
        def objective(positions):
            cost = self._calculate_placement_cost(positions, distance_segments, noise_levels)
            fidelity_penalty = self._calculate_fidelity_penalty(
                positions, distance_segments, noise_levels, fidelity_targets
            )
            return cost + 100 * fidelity_penalty  # Weight fidelity heavily
        
        # Constraints: positions must be within total distance
        bounds = [(0, total_distance) for _ in range(self.max_repeaters)]
        
        # Initial guess: evenly spaced
        x0 = np.linspace(0, total_distance, self.max_repeaters + 2)[1:-1]
        
        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        optimal_positions = result.x.tolist()
        total_cost = result.fun
        
        # Filter out redundant positions
        optimal_positions = self._filter_close_positions(optimal_positions)
        
        self.repeater_positions = optimal_positions
        self.repeater_active = [True] * len(optimal_positions)
        
        return optimal_positions, total_cost
    
    def adaptive_placement(self, channel_quality: List[Dict],
                          performance_metrics: Dict) -> List[float]:
        """
        Adaptively adjust repeater placement based on real-time channel conditions
        
        Args:
            channel_quality: List of quality metrics for each channel segment
            performance_metrics: Current performance metrics
            
        Returns:
            Updated repeater positions
        """
        # Extract metrics
        distances = [cq['distance'] for cq in channel_quality]
        noise_levels = [cq['noise_level'] for cq in channel_quality]
        loss_probs = [cq['loss_probability'] for cq in channel_quality]
        
        # Identify problem segments (high loss or noise)
        problem_segments = []
        for i, (noise, loss) in enumerate(zip(noise_levels, loss_probs)):
            if noise > 0.05 or loss > 0.3:
                problem_segments.append(i)
        
        # Add repeaters to problem segments if available
        if len(self.repeater_positions) < self.max_repeaters and problem_segments:
            for seg_idx in problem_segments:
                if len(self.repeater_positions) >= self.max_repeaters:
                    break
                
                # Add repeater in middle of problem segment
                cumulative_dist = np.cumsum([0] + distances)
                new_position = (cumulative_dist[seg_idx] + cumulative_dist[seg_idx + 1]) / 2
                
                # Check if position not too close to existing
                if not any(abs(new_position - pos) < 10 for pos in self.repeater_positions):
                    self.repeater_positions.append(new_position)
                    self.repeater_active.append(True)
        
        # Remove repeaters from segments with good performance
        if performance_metrics.get('avg_fidelity', 0) > 0.98:
            # Performance is excellent, can reduce repeaters
            if len(self.repeater_positions) > 1:
                # Remove least useful repeater
                self.repeater_positions.pop()
                self.repeater_active.pop()
        
        return self.repeater_positions
    
    def _calculate_placement_cost(self, positions: np.ndarray,
                                  distances: np.ndarray,
                                  noise_levels: np.ndarray) -> float:
        """Calculate cost of repeater placement"""
        num_repeaters = len(positions)
        
        # Base cost: number of repeaters
        base_cost = num_repeaters * self.repeater_cost
        
        # Operational cost based on usage
        operational_cost = num_repeaters * (np.mean(noise_levels) * 10)
        
        return base_cost + operational_cost
    
    def _calculate_fidelity_penalty(self, positions: np.ndarray,
                                   distances: np.ndarray,
                                   noise_levels: np.ndarray,
                                   targets: np.ndarray) -> float:
        """Calculate penalty for not meeting fidelity targets"""
        penalty = 0.0
        
        cumulative_dist = np.cumsum([0] + distances.tolist())
        
        for i in range(len(distances)):
            # Check if segment has repeater
            segment_start = cumulative_dist[i]
            segment_end = cumulative_dist[i + 1]
            
            has_repeater = any(segment_start < pos < segment_end for pos in positions)
            
            # Estimate fidelity
            segment_fidelity = 1.0 - noise_levels[i] - (distances[i] / 100) * 0.1
            
            if has_repeater:
                segment_fidelity *= self.swap_success_prob
            
            # Penalty if below target
            if segment_fidelity < targets[i]:
                penalty += (targets[i] - segment_fidelity) ** 2
        
        return penalty
    
    def _filter_close_positions(self, positions: List[float], 
                               min_distance: float = 10.0) -> List[float]:
        """Filter out repeaters that are too close to each other"""
        if not positions:
            return []
        
        sorted_positions = sorted(positions)
        filtered = [sorted_positions[0]]
        
        for pos in sorted_positions[1:]:
            if pos - filtered[-1] >= min_distance:
                filtered.append(pos)
        
        return filtered
    
    def get_num_active_repeaters(self) -> int:
        """Get number of active repeaters"""
        return sum(self.repeater_active)
    
    def toggle_repeater(self, index: int, active: bool):
        """Activate or deactivate a repeater"""
        if 0 <= index < len(self.repeater_active):
            self.repeater_active[index] = active
    
    def get_repeater_info(self) -> List[Dict]:
        """Get information about all repeaters"""
        return [
            {
                'position': pos,
                'active': active,
                'index': i
            }
            for i, (pos, active) in enumerate(zip(self.repeater_positions, self.repeater_active))
        ]
    
    def reset(self):
        """Reset repeater configuration"""
        self.repeater_positions = []
        self.repeater_active = []


class RepeaterOptimizer:
    """
    Advanced optimizer for repeater placement using dynamic programming
    and genetic algorithms
    """
    
    def __init__(self, max_repeaters: int = 5):
        """
        Initialize repeater optimizer
        
        Args:
            max_repeaters: Maximum number of repeaters allowed
        """
        self.max_repeaters = max_repeaters
        
    def dynamic_programming_placement(self, distances: np.ndarray,
                                     loss_rates: np.ndarray,
                                     num_repeaters: int) -> List[int]:
        """
        Use dynamic programming to find optimal repeater placement
        
        Args:
            distances: Distance between each node pair
            loss_rates: Loss rate for each segment
            num_repeaters: Number of repeaters to place
            
        Returns:
            List of node indices where repeaters should be placed
        """
        n = len(distances)
        
        # DP table: dp[i][j] = min cost to cover first i segments with j repeaters
        dp = np.full((n + 1, num_repeaters + 1), np.inf)
        placement = {}
        
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(num_repeaters + 1):
                # Try placing last repeater at different positions
                for k in range(max(0, i - 10), i):  # Look back up to 10 segments
                    if j > 0 and dp[k][j - 1] < np.inf:
                        # Cost of segment from k to i without repeaters
                        segment_cost = self._calculate_segment_cost(
                            distances[k:i], loss_rates[k:i]
                        )
                        
                        total_cost = dp[k][j - 1] + segment_cost + 10  # +10 for repeater
                        
                        if total_cost < dp[i][j]:
                            dp[i][j] = total_cost
                            placement[(i, j)] = k
        
        # Backtrack to find placements
        repeater_positions = []
        i, j = n, num_repeaters
        
        while j > 0:
            if (i, j) in placement:
                pos = placement[(i, j)]
                repeater_positions.append(pos)
                i, j = pos, j - 1
            else:
                break
        
        return sorted(repeater_positions)
    
    def _calculate_segment_cost(self, distances: np.ndarray, 
                               loss_rates: np.ndarray) -> float:
        """Calculate cost for a segment without repeaters"""
        total_loss = np.sum(loss_rates * distances)
        return total_loss
    
    def greedy_placement(self, distances: np.ndarray,
                        noise_levels: np.ndarray,
                        num_repeaters: int) -> List[int]:
        """
        Greedy algorithm: place repeaters at positions with highest need
        
        Args:
            distances: Distance between nodes
            noise_levels: Noise level for each segment
            num_repeaters: Number of repeaters to place
            
        Returns:
            List of positions for repeaters
        """
        # Calculate "badness" score for each segment
        badness = distances * noise_levels
        
        # Place repeaters at highest badness segments
        top_segments = np.argsort(badness)[-num_repeaters:]
        
        return sorted(top_segments.tolist())
