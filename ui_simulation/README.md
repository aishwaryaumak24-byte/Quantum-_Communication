# Quantum UI Simulation (Streamlit)

This folder contains a separate UI module for interactive simulation and demo.

## Features

- Visual network graph with node-link channel quality
- Animated qubit packet flow across active links
- Entanglement overlays with per-link fidelity
- Live metrics (reward, fidelity, success rate)
- Multiple control modes: heuristic, random, trained model

## Run

From project root:

```bash
pip install -r requirements.txt
streamlit run ui_simulation/app.py
```

## Demo Flow

1. Click **Initialize** in sidebar.
2. Choose policy mode:
   - **Heuristic** for stable visual demo
   - **Random** for baseline behavior
   - **Trained Model** after loading model
3. Use **Step** or **Run 20** for manual simulation, or click **Start Auto Run** to let the UI simulate continuously.
4. Use **Stop Auto Run** to pause automatic progression.
5. Observe metrics + entanglement table update in real time.

## Automatic Mode

- **Auto Run Delay (sec)** controls how fast the simulation advances.
- **Auto Run Steps per Tick** controls how many environment steps are executed on each refresh.
- Auto mode works with **Heuristic**, **Random**, and **Trained Model** policies.

## Notes

- This UI uses your existing backend simulator and environment.
- No changes to training scripts are required.
- For trained mode, load a valid model path from `results/model`.
