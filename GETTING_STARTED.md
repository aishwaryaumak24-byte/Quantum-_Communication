# Getting Started Guide - ARL-QDCC Project

## Overview
This guide explains how to train the RL model and simulate the quantum communication system to get desired outputs.

---

## 🎓 Model Training Locations

### 1. Main Training Entry Point: `run.py`
The primary way to train the model:

```bash
# Train from command line
python run.py --mode train --sim_config config/simulation_config.yaml --rl_config config/rl_config.yaml
```

**What happens during training:**
- Loads configurations from YAML files
- Creates quantum environment with specified parameters
- Initializes RL agent (PPO/A2C/DQN/SAC)
- Trains for specified timesteps (default: 100,000)
- Saves trained model to `results/model/rl_agent`
- Generates TensorBoard logs for monitoring

### 2. Experiment Scripts: `experiments/`

**Static Scenario Training:**
```bash
python experiments/experiment_static.py
```
- Trains agent on fixed channel conditions
- Compares against 3 baseline strategies
- Saves results to `results/static_results.json`
- Generates comparison plots

**Dynamic Scenario Training:**
```bash
python experiments/experiment_dynamic.py
```
- Trains agent on time-varying noise
- Tests adaptation across different noise levels
- Saves results to `results/dynamic_results.json`
- Generates adaptation plots

### 3. Direct RL Agent Training: `src/rl/rl_agent.py`

For custom training in your own scripts:
```python
from environment.quantum_environment import QuantumEnvironment
from rl.rl_agent import RLAgent
import yaml

# Load configs
with open('config/simulation_config.yaml') as f:
    sim_config = yaml.safe_load(f)
with open('config/rl_config.yaml') as f:
    rl_config = yaml.safe_load(f)

# Create environment
env = QuantumEnvironment(sim_config)

# Create and train agent
agent = RLAgent(env, rl_config)
agent.train(timesteps=100000, save_path='results/model/my_model')

# Evaluate
results = agent.evaluate(num_episodes=100)
print(f"Mean Reward: {results['mean_reward']}")
print(f"Mean Fidelity: {results['mean_fidelity']}")
```

---

## 📋 External Requirements & Setup

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key packages needed:**
- `qiskit>=0.45.0` - Quantum computing simulation
- `qiskit-aer>=0.13.0` - Quantum circuit simulation backend
- `stable-baselines3>=2.0.0` - RL algorithms (PPO, A2C, DQN, SAC)
- `gymnasium>=0.29.0` - RL environment interface
- `torch>=2.0.0` - Neural network backend
- `numpy`, `scipy`, `pandas` - Scientific computing
- `matplotlib`, `seaborn` - Visualization
- `pyyaml` - Configuration files
- `pytest` - Testing

**Installation issues?**

If you get errors, try:
```bash
# For Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install others
pip install qiskit qiskit-aer
pip install stable-baselines3
pip install gymnasium
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Test imports
python -c "import qiskit; import stable_baselines3; import gymnasium; print('All imports successful!')"
```

### Step 3: Run Tests (Optional but Recommended)

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_environment.py -v
pytest tests/test_rl.py -v
pytest tests/test_simulator.py -v
```

---

## 🚀 Quick Start - Getting Desired Output

### Option A: Full Automated Workflow

```bash
# 1. Train on static scenario
python experiments/experiment_static.py

# 2. Train on dynamic scenario  
python experiments/experiment_dynamic.py

# 3. Compare all results
python experiments/compare_results.py
```

**Expected outputs:**
- `results/static_results.json` - Static scenario metrics
- `results/dynamic_results.json` - Dynamic scenario metrics
- `results/plots/static_comparison.png` - Performance comparison plot
- `results/plots/dynamic_adaptation.png` - Adaptation analysis plot
- `results/plots/comprehensive_comparison.png` - Overall comparison
- `results/tables/static_comparison.csv` - Comparison table
- `results/summary_report.txt` - Text summary

### Option B: Custom Training

```bash
# Train with custom settings
python run.py --mode train \
    --sim_config config/simulation_config.yaml \
    --rl_config config/rl_config.yaml \
    --model_path results/model/custom_model
```

### Option C: Load Pre-trained Model and Evaluate

```bash
# After training, evaluate
python run.py --mode evaluate \
    --model_path results/model/rl_agent \
    --num_episodes 100
```
``` 
new command to run
python run.py --mode evaluate --model_path results/model/static/best_model/best_model.zip --num_episodes 50
```


---

## ⚙️ Configuration for Desired Output

### Adjust Training Duration

Edit `config/rl_config.yaml`:
```yaml
training:
  total_timesteps: 200000  # Increase for better performance (default: 100000)
  eval_freq: 10000         # How often to evaluate
  save_freq: 20000         # How often to save checkpoints
```

### Adjust Simulation Complexity

Edit `config/simulation_config.yaml`:
```yaml
network:
  num_nodes: 10            # More nodes = harder problem
  distance_range: [10, 100] # Distance range in km

quantum_channel:
  fiber_loss_coefficient: 0.2  # Higher = more loss

noise:
  gate_error_rate: 0.01    # Lower = easier problem
```

### Select RL Algorithm

Edit `config/rl_config.yaml`:
```yaml
algorithm: "PPO"  # Options: PPO, A2C, DQN, SAC

# PPO is recommended for:
# - Stable training
# - Good sample efficiency
# - Continuous/discrete actions
```

---

## 📊 Understanding the Output

### Training Outputs

**Console Output:**
```
Creating environment...
Creating RL agent...
Training RL agent...
Starting training for 100000 timesteps...
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 95.2     |
|    ep_rew_mean     | 23.45    |
| time/              |          |
|    fps             | 1234     |
|    total_timesteps | 10000    |
---------------------------------
Training complete! Total timesteps: 100000
```

**Saved Files:**
- `results/model/rl_agent.zip` - Trained model
- `results/model/best_model/` - Best performing checkpoint
- `results/logs/` - Training logs

### Evaluation Outputs

```
=== Evaluation Results ===
Mean Reward: 45.234 ± 5.123
Mean Fidelity: 0.945 ± 0.023
Mean Success Rate: 0.873 ± 0.045
Mean Episode Length: 98.5 ± 12.3
```

### Comparison Outputs

```
=== Strategy Comparison ===
                strategy  mean_reward  mean_fidelity  mean_success_rate
 RL Agent (Proposed)      45.23        0.945          0.873
      Static Protocol      32.15        0.892          0.734
      Greedy Protocol      35.67        0.903          0.756
   Threshold Protocol      34.89        0.898          0.745
```

---

## 🎯 Expected Results (Based on SRS Requirements)

### Performance Targets

According to FR-3 and FR-8, the system should achieve:

1. **Fidelity**: ≥ 0.95 (target from config)
2. **Success Rate**: > 0.80 
3. **RL vs Baseline Improvement**: 10-20% better performance
4. **Adaptation**: Maintain performance despite noise variations

### Typical Training Progress

| Timesteps | Avg Reward | Avg Fidelity | Success Rate |
|-----------|------------|--------------|--------------|
| 0         | ~5.0       | ~0.70        | ~0.40        |
| 25,000    | ~20.0      | ~0.85        | ~0.65        |
| 50,000    | ~35.0      | ~0.91        | ~0.78        |
| 100,000   | ~45.0      | ~0.94        | ~0.87        |

---

## 🔧 Troubleshooting

### Issue: Training is too slow

**Solution:**
```yaml
# In config/rl_config.yaml, reduce timesteps for testing
training:
  total_timesteps: 10000  # Reduced from 100000
```

Or use fewer nodes:
```yaml
# In config/simulation_config.yaml
network:
  num_nodes: 3  # Reduced from 10
```

### Issue: Memory errors

**Solution:**
```yaml
# In config/rl_config.yaml
ppo:
  n_steps: 1024  # Reduced from 2048
  batch_size: 32  # Reduced from 64
```

### Issue: Poor convergence

**Solution:**
```yaml
# In config/rl_config.yaml
training:
  learning_rate: 0.001  # Increased from 0.0003

reward:
  fidelity_weight: 2.0  # Increased importance
```

### Issue: Import errors

```bash
# Reinstall with specific versions
pip uninstall qiskit qiskit-aer
pip install qiskit==0.45.0 qiskit-aer==0.13.0

# Or use conda
conda install -c conda-forge qiskit
```

---

## 📈 Monitoring Training

### Using TensorBoard

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir results/model/rl_agent_1/

# Open browser to: http://localhost:6006
```

You'll see:
- Reward curves over time
- Episode length trends
- Policy loss
- Value function loss

### Using Custom Logging

The training automatically logs to `results/logs/main.log`:
```bash
# Watch training progress
tail -f results/logs/main.log
```

---

## 🎬 Complete Workflow Example

```bash
# 1. Setup (one-time)
pip install -r requirements.txt
pytest tests/ -v  # Verify everything works

# 2. Quick test (fast training to verify)
# Edit config/rl_config.yaml: total_timesteps: 5000
python run.py --mode train

# 3. Full training
# Edit config/rl_config.yaml: total_timesteps: 100000
python experiments/experiment_static.py
python experiments/experiment_dynamic.py

# 4. Analyze results
python experiments/compare_results.py

# 5. View outputs
# Check: results/plots/*.png
# Check: results/summary_report.txt
```

---

## 💡 What Makes This Project Adaptive?

The RL agent learns to:

1. **Adapt purification rounds** based on noise (FR-6)
2. **Adjust generation rate** based on success (FR-6)
3. **Optimize repeater placement** for cost (FR-7)
4. **Balance fidelity vs cost** through multi-objective reward (FR-9)
5. **Respond to dynamic noise** in real-time (FR-2)

This is validated by comparing against non-adaptive baselines which use fixed strategies.

---

## 📚 Key Files Reference

| Purpose | File | Description |
|---------|------|-------------|
| **Train** | `run.py` | Main entry point |
| **Train** | `experiments/experiment_*.py` | Full experiments |
| **Config** | `config/simulation_config.yaml` | Environment settings |
| **Config** | `config/rl_config.yaml` | RL algorithm settings |
| **Core** | `src/rl/rl_agent.py` | RL agent implementation |
| **Core** | `src/environment/quantum_environment.py` | Environment |
| **Output** | `results/` | All generated outputs |
| **Docs** | `docs/explanation_notes.md` | Architecture details |

---

## ✅ Success Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Tests passing (`pytest tests/ -v`)
- [ ] Can run quick training (`python run.py --mode train`)
- [ ] Static experiment completes (`python experiments/experiment_static.py`)
- [ ] Results generated in `results/` folder
- [ ] Plots viewable in `results/plots/`
- [ ] TensorBoard accessible (optional)

---

## 🎓 Next Steps After Training

1. **Analyze Results**: Review plots and metrics
2. **Tune Hyperparameters**: Adjust configs for better performance
3. **Extend Functionality**: Add new actions, observations, or rewards
4. **Real Hardware**: Integrate with actual quantum hardware (future work)
5. **Publish Results**: Document findings for academic review

---

For questions about specific components, see `docs/explanation_notes.md`
