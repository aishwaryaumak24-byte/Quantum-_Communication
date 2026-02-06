# ARL-QDCC Architecture and Implementation Notes

## System Architecture

### Overview
The ARL-QDCC (Adaptive Reinforcement Learning Framework for Quantum Communication under Dynamic Conditions) implements a 5-layer architecture as specified in the SRS document:

1. **Transmission Environment Layer** (`src/environment/`)
2. **Quantum Channel Simulation Layer** (`src/simulator/`)
3. **Reinforcement Learning Intelligence Layer** (`src/rl/`)
4. **Adaptive Control Layer** (`src/control/`)
5. **Performance Evaluation and Feedback Layer** (`src/evaluation/`)

### Layer Details

#### 1. Transmission Environment Layer
- **Module**: `src/environment/quantum_environment.py`
- **Purpose**: Implements the Gymnasium-compatible environment for quantum communication
- **Key Components**:
  - State space representation (node states, channel states, entanglement states)
  - Action space definition (wait, generate, swap, purify, etc.)
  - Reward calculation
  - Episode management

#### 2. Quantum Channel Simulation Layer
- **Module**: `src/simulator/`
- **Purpose**: Simulates quantum channels with realistic physics
- **Key Components**:
  - `quantum_channel.py`: Channel simulation with distance-dependent loss
  - `noise_models.py`: Various quantum noise models (depolarizing, amplitude damping, phase damping)
  - `metrics.py`: Performance metric tracking and calculation

#### 3. Reinforcement Learning Intelligence Layer
- **Module**: `src/rl/`
- **Purpose**: Implements the RL agent and learning algorithms
- **Key Components**:
  - `rl_agent.py`: Main RL agent using Stable-Baselines3
  - `reward_function.py`: Multi-objective reward function
  - `policy.py`: Policy networks and exploration strategies

#### 4. Adaptive Control Layer
- **Module**: `src/control/`
- **Purpose**: Adaptive protocol control and repeater placement
- **Key Components**:
  - `adaptive_protocol.py`: Dynamic protocol parameter adjustment (FR-6)
  - `repeater_placement.py`: Cost-aware repeater optimization (FR-7)

#### 5. Performance Evaluation and Feedback Layer
- **Module**: `src/evaluation/`
- **Purpose**: Evaluation and comparison against baselines
- **Key Components**:
  - `baseline.py`: Non-adaptive baseline strategies
  - `evaluator.py`: Comprehensive evaluation and comparison tools

### Functional Requirements Mapping

| Requirement | Implementation | Module |
|------------|----------------|---------|
| FR-1: Quantum Channel Simulation | QuantumChannel class | `simulator/quantum_channel.py` |
| FR-2: Dynamic Noise Modeling | NoiseModel classes, time-varying noise | `environment/noise_models.py` |
| FR-3: Performance Metrics | PerformanceMetrics class | `simulator/metrics.py` |
| FR-4: State Representation | Observation space in QuantumEnvironment | `environment/quantum_environment.py` |
| FR-5: RL Engine | RLAgent class | `rl/rl_agent.py` |
| FR-6: Adaptive Protocol | AdaptiveProtocol class | `control/adaptive_protocol.py` |
| FR-7: Repeater Placement | RepeaterPlacement class | `control/repeater_placement.py` |
| FR-8: Performance & Cost Evaluation | CostMetrics, PerformanceMetrics | `simulator/metrics.py` |
| FR-9: Reward Generation | RewardFunction class | `rl/reward_function.py` |
| FR-10: Policy Update | RLAgent.train() method | `rl/rl_agent.py` |

## Implementation Details

### State Representation (FR-4)
The state consists of:
- **Node states** (N×4): position, buffer_size, active status, energy
- **Channel states** ((N-1)×3): distance, loss, noise_level
- **Entanglement states** (N×2): fidelity, age

Total dimension: 9N - 3 for N nodes

### Action Space
Discrete actions (configurable):
- 0: Wait
- 1: Generate entanglement
- 2: Entanglement swap
- 3: Purify entanglement
- 4: Increase power
- 5: Decrease power
- 6: Enable error correction
- 7: Disable error correction
- 8-9: Reserved for future extensions

### Reward Function (FR-9)
Multi-objective reward combining:
```
R = w_f * R_fidelity 
  + w_t * R_throughput 
  + w_l * P_latency 
  + w_e * P_energy 
  + w_s * R_success
```

Where:
- `w_f, w_t, w_l, w_e, w_s` are configurable weights
- `R_*` are reward components
- `P_*` are penalty components

### Adaptive Protocol (FR-6)
The adaptive protocol adjusts:
1. **Purification rounds**: Based on noise level
2. **Generation rate**: Based on success rate
3. **Swap threshold**: Based on achieved fidelity
4. **Buffer size**: Based on distance
5. **Error correction**: Based on distance and performance

### Repeater Placement (FR-7)
Optimization approaches:
1. **Dynamic Programming**: Optimal placement for given configuration
2. **Greedy Algorithm**: Fast heuristic placement
3. **Gradient-based**: Continuous optimization with cost constraints

## Experiments

### Static Scenario (`experiments/experiment_static.py`)
- Tests performance under fixed channel conditions
- Compares RL agent against multiple baselines
- Metrics: fidelity, success rate, episode length

### Dynamic Scenario (`experiments/experiment_dynamic.py`)
- Tests adaptation to time-varying noise
- Evaluates robustness across noise levels
- Demonstrates superiority of adaptive approach

### Results Comparison (`experiments/compare_results.py`)
- Aggregates results from all experiments
- Generates comprehensive comparison plots
- Produces summary reports

## Usage

### Training
```bash
python run.py --mode train --sim_config config/simulation_config.yaml --rl_config config/rl_config.yaml
```

### Evaluation
```bash
python run.py --mode evaluate --model_path results/model/rl_agent --num_episodes 100
```

### Running Experiments
```bash
python experiments/experiment_static.py
python experiments/experiment_dynamic.py
python experiments/compare_results.py
```

## Configuration

### Simulation Config (`config/simulation_config.yaml`)
- Network topology and parameters
- Channel physics (loss, noise)
- Repeater configuration
- Simulation settings

### RL Config (`config/rl_config.yaml`)
- Algorithm selection (PPO, A2C, DQN, SAC)
- Training hyperparameters
- Network architecture
- Reward function weights

## Testing

Run tests with:
```bash
pytest tests/
```

Coverage report:
```bash
pytest --cov=src tests/
```

## Performance Considerations

1. **Scalability**: Environment supports 5-100 nodes
2. **Training time**: ~10-30 minutes for 100k timesteps on CPU
3. **Memory**: ~500MB for typical configuration
4. **Parallelization**: Supports vectorized environments for faster training

## Future Enhancements

1. Multi-agent RL for distributed control
2. Real quantum hardware interface
3. Advanced noise models (time-correlated)
4. Online learning and continuous adaptation
5. Transfer learning across different network topologies

## References

See `docs/srs_ieee.pdf` for complete Software Requirements Specification.
