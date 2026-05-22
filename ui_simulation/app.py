"""
Streamlit UI for Quantum Communication simulation.

This dashboard visualizes network state, qubit flow, entanglement links,
and key metrics while running the existing simulator environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environment.quantum_environment import QuantumEnvironment
from rl.rl_agent import RLAgent
from stable_baselines3 import PPO
from utils.helpers import load_config


st.set_page_config(
    page_title="Quantum Communication UI Simulation",
    page_icon="Q",
    layout="wide",
)


def list_config_files() -> List[str]:
    config_dir = PROJECT_ROOT / "config"
    if not config_dir.exists():
        return []
    return sorted([str(p.relative_to(PROJECT_ROOT)) for p in config_dir.glob("*.yaml")])


def split_config_files(config_files: List[str]) -> Tuple[List[str], List[str]]:
    """Split config paths into simulation and RL config groups."""
    sim_configs: List[str] = []
    rl_configs: List[str] = []

    for cfg in config_files:
        name = Path(cfg).name.lower()
        if "simulation" in name:
            sim_configs.append(cfg)
        elif "rl_" in name or name.startswith("rl"):
            rl_configs.append(cfg)

    # Fallbacks in case naming is unconventional.
    if not sim_configs:
        sim_configs = [c for c in config_files if "simulation" in c.lower()]
    if not rl_configs:
        rl_configs = [c for c in config_files if "rl" in c.lower()]

    return sorted(sim_configs), sorted(rl_configs)


def list_model_candidates() -> List[str]:
    model_dir = PROJECT_ROOT / "results" / "model"
    if not model_dir.exists():
        return []

    candidates: List[str] = []
    for path in model_dir.rglob("*.zip"):
        candidates.append(str(path.relative_to(PROJECT_ROOT)))

    # Also include directory models saved without explicit .zip in arguments.
    for name in ["rl_agent", "smoke_agent"]:
        p = model_dir / name
        if p.exists():
            candidates.append(str(p.relative_to(PROJECT_ROOT)))

    return sorted(set(candidates))


@st.cache_data(show_spinner=False)
def inspect_model_signature(model_path: str) -> Dict[str, object]:
    """Read model observation/action space without binding to current env."""
    resolved = resolve_path(model_path)
    if not resolved.exists():
        return {"exists": False, "error": f"Model file not found: {resolved}"}

    try:
        model = PPO.load(str(resolved), device="cpu")
        obs_shape = tuple(model.observation_space.shape or ())
        action_n = getattr(model.action_space, "n", None)
        return {
            "exists": True,
            "obs_shape": obs_shape,
            "obs_dim": int(obs_shape[0]) if len(obs_shape) == 1 else None,
            "action_n": int(action_n) if action_n is not None else None,
        }
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def suggested_sim_config_from_obs(obs_dim: int | None) -> str:
    mapping = {
        33: "config/simulation_config_stage1.yaml (4 nodes)",
        42: "config/simulation_config_stage2.yaml (5 nodes)",
        51: "config/simulation_config_stage3.yaml or config/simulation_config_stable.yaml (6 nodes)",
        87: "config/simulation_config.yaml (10 nodes)",
    }
    return mapping.get(obs_dim, "Unknown. Use the same simulation config used during model training.")


def init_state() -> None:
    defaults = {
        "env": None,
        "obs": None,
        "last_info": {},
        "history": {
            "step": [],
            "reward": [],
            "avg_fidelity": [],
            "success_rate": [],
        },
        "done": False,
        "agent": None,
        "policy_mode": "Heuristic",
        "episode_reward": 0.0,
        "episode_steps": 0,
        "auto_run": False,
        "auto_delay": 0.25,
        "auto_batch_size": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def resolve_path(relative_or_abs: str) -> Path:
    path = Path(relative_or_abs)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def heuristic_action(env: QuantumEnvironment) -> int:
    """
    Simple controller for demo runs:
    - If low fidelity on links, generate entanglement (1)
    - If moderate fidelity exists, purify (3)
    - Otherwise try swapping (2)
    """
    if env.entanglement_pairs is None or len(env.entanglement_pairs) == 0:
        return 1

    fidelities = env.entanglement_pairs[:, 0]
    mean_fidelity = float(np.mean(fidelities))
    max_fidelity = float(np.max(fidelities))

    if max_fidelity < 0.35:
        return 1
    if mean_fidelity < 0.6:
        return 3
    return 2


def build_node_positions(num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.05, 0.95, num_nodes)
    y = np.ones(num_nodes) * 0.5
    return x, y


def build_network_figure(env: QuantumEnvironment, step: int) -> go.Figure:
    num_nodes = env.num_nodes
    x, y = build_node_positions(num_nodes)

    fig = go.Figure()

    # Draw channel links using estimated fidelity quality.
    for i in range(num_nodes - 1):
        quality = env.channel.get_channel_quality(i, i + 1)
        estimated_fidelity = quality.get("estimated_fidelity", 0.0)

        # Color transitions from red (poor) to green (good).
        red = int((1.0 - estimated_fidelity) * 255)
        green = int(estimated_fidelity * 255)
        link_color = f"rgb({red},{green},80)"

        fig.add_trace(
            go.Scatter(
                x=[x[i], x[i + 1]],
                y=[y[i], y[i + 1]],
                mode="lines",
                line={"width": 5, "color": link_color},
                hovertemplate=(
                    f"Link {i}-{i+1}<br>"
                    f"Distance: {quality.get('distance', 0.0):.1f} km<br>"
                    f"Loss: {quality.get('loss_probability', 0.0):.3f}<br>"
                    f"Noise: {quality.get('noise_level', 0.0):.3f}<br>"
                    f"Estimated Fidelity: {estimated_fidelity:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Draw entanglement overlays for active links.
    for i in range(num_nodes - 1):
        fidelity = float(env.entanglement_pairs[i, 0]) if env.entanglement_pairs is not None else 0.0
        if fidelity <= 0.05:
            continue

        width = 2 + 8 * min(1.0, fidelity)
        fig.add_trace(
            go.Scatter(
                x=[x[i], x[i + 1]],
                y=[y[i] + 0.05, y[i + 1] + 0.05],
                mode="lines+text",
                line={"width": width, "color": "rgba(80,180,255,0.85)"},
                text=["", f"E={fidelity:.2f}"],
                textposition="top center",
                hovertemplate=f"Entangled Pair {i}-{i+1}<br>Fidelity: {fidelity:.3f}<extra></extra>",
                showlegend=False,
            )
        )

        # Qubit flow marker to show communication movement.
        t = (step % 30) / 30.0
        px = x[i] + (x[i + 1] - x[i]) * t
        py = y[i] + (y[i + 1] - y[i]) * t
        fig.add_trace(
            go.Scatter(
                x=[px],
                y=[py],
                mode="markers",
                marker={"size": 12, "color": "gold", "symbol": "diamond"},
                name="Qubit",
                hovertemplate=(
                    f"Qubit packet on {i}-{i+1}<br>"
                    f"Animation phase: {t:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Draw nodes on top.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker={"size": 26, "color": "#1f77b4", "line": {"width": 2, "color": "white"}},
            text=[f"N{i}" for i in range(num_nodes)],
            textposition="bottom center",
            hovertemplate="Node %{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Quantum Network: Channel Quality, Qubit Flow, and Entanglement",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0.2, 0.85]},
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font={"color": "#f8fafc"},
        height=440,
    )

    return fig


def reset_episode(sim_config_path: str) -> None:
    sim_config = load_config(str(resolve_path(sim_config_path)))

    required_paths = [
        ("network", "num_nodes"),
        ("simulation", "max_episode_steps"),
        ("quantum_channel",),
        ("noise",),
        ("entanglement",),
    ]

    for path in required_paths:
        cursor = sim_config
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                joined = " -> ".join(path)
                raise ValueError(
                    f"Invalid simulation config '{sim_config_path}'. Missing key: {joined}. "
                    "Please choose a simulation_config*.yaml file."
                )
            cursor = cursor[key]

    env = QuantumEnvironment(sim_config)
    obs, info = env.reset()

    st.session_state.env = env
    st.session_state.obs = obs
    st.session_state.last_info = info
    st.session_state.done = False
    st.session_state.history = {
        "step": [],
        "reward": [],
        "avg_fidelity": [],
        "success_rate": [],
    }
    st.session_state.episode_reward = 0.0
    st.session_state.episode_steps = 0


def load_trained_agent(sim_config_path: str, rl_config_path: str, model_path: str) -> None:
    env = st.session_state.env
    if env is None:
        raise ValueError("Initialize the environment first.")

    signature = inspect_model_signature(model_path)
    if signature.get("error"):
        raise ValueError(f"Cannot inspect model: {signature['error']}")

    model_obs_shape = signature.get("obs_shape")
    env_obs_shape = tuple(env.observation_space.shape or ())
    model_action_n = signature.get("action_n")
    env_action_n = getattr(env.action_space, "n", None)

    if model_obs_shape != env_obs_shape:
        suggested = suggested_sim_config_from_obs(signature.get("obs_dim"))
        raise ValueError(
            "Observation mismatch. "
            f"Model expects {model_obs_shape}, current environment is {env_obs_shape}. "
            f"Suggested simulation config: {suggested}"
        )

    if model_action_n is not None and env_action_n is not None and model_action_n != env_action_n:
        raise ValueError(
            "Action-space mismatch. "
            f"Model expects Discrete({model_action_n}), current environment is Discrete({env_action_n}). "
            "Use matching RL/simulation configs from training."
        )

    rl_config = load_config(str(resolve_path(rl_config_path)))
    agent = RLAgent(env, rl_config)
    agent.load(str(resolve_path(model_path)))
    st.session_state.agent = agent


def run_one_step(selected_action: int | None = None) -> None:
    env = st.session_state.env
    if env is None or st.session_state.done:
        return

    mode = st.session_state.policy_mode
    obs = st.session_state.obs

    if selected_action is None:
        if mode == "Random":
            action = env.action_space.sample()
        elif mode == "Trained Model":
            if st.session_state.agent is None:
                st.warning("Load a trained model first, or switch policy mode.")
                return
            action = st.session_state.agent.predict(obs, deterministic=True)
        else:
            action = heuristic_action(env)
    else:
        action = selected_action

    obs, reward, done, truncated, info = env.step(int(action))
    st.session_state.obs = obs
    st.session_state.done = bool(done or truncated)
    st.session_state.last_info = info
    st.session_state.episode_reward += float(reward)
    st.session_state.episode_steps += 1

    step = int(info.get("step", st.session_state.episode_steps))
    success = float(info.get("successful_transmissions", 0)) / max(1, step)

    st.session_state.history["step"].append(step)
    st.session_state.history["reward"].append(float(reward))
    st.session_state.history["avg_fidelity"].append(float(info.get("avg_fidelity", 0.0)))
    st.session_state.history["success_rate"].append(success)


def run_multiple_steps(n_steps: int) -> None:
    for _ in range(n_steps):
        if st.session_state.done:
            break
        run_one_step()


def start_auto_run() -> None:
    """Enable continuous automatic simulation."""
    st.session_state.auto_run = True


def stop_auto_run() -> None:
    """Disable continuous automatic simulation."""
    st.session_state.auto_run = False


def auto_run_tick() -> None:
    """Advance the simulation automatically and rerun the app."""
    env = st.session_state.env
    if env is None or st.session_state.done:
        st.session_state.auto_run = False
        return

    run_multiple_steps(int(st.session_state.auto_batch_size))

    if st.session_state.done:
        st.session_state.auto_run = False
        return

    time.sleep(float(st.session_state.auto_delay))
    st.rerun()


def main() -> None:
    init_state()

    st.title("Quantum Communication UI Simulation")
    st.caption("Interactive visualization for qubit communication, entanglement links, and RL-driven control.")

    config_options = list_config_files()
    sim_config_options, rl_config_options = split_config_files(config_options)
    model_options = list_model_candidates()

    with st.sidebar:
        st.header("Simulation Controls")

        sim_config = st.selectbox(
            "Simulation Config",
            options=sim_config_options,
            index=sim_config_options.index("config/simulation_config.yaml") if "config/simulation_config.yaml" in sim_config_options else 0,
        )

        rl_config = st.selectbox(
            "RL Config",
            options=rl_config_options,
            index=rl_config_options.index("config/rl_config.yaml") if "config/rl_config.yaml" in rl_config_options else 0,
        )

        st.session_state.policy_mode = st.selectbox(
            "Policy Mode",
            options=["Heuristic", "Random", "Trained Model"],
            index=0,
        )

        st.session_state.auto_delay = st.slider(
            "Auto Run Delay (sec)",
            min_value=0.05,
            max_value=2.0,
            value=float(st.session_state.auto_delay),
            step=0.05,
        )
        st.session_state.auto_batch_size = st.slider(
            "Auto Run Steps per Tick",
            min_value=1,
            max_value=20,
            value=int(st.session_state.auto_batch_size),
            step=1,
        )

        model_path = st.selectbox(
            "Trained Model Path",
            options=model_options if model_options else ["results/model/static/best_model/best_model.zip"],
            index=0,
        )

        model_sig = inspect_model_signature(model_path)
        if model_sig.get("error"):
            st.caption(f"Model info: unavailable ({model_sig['error']})")
        else:
            obs_dim = model_sig.get("obs_dim")
            action_n = model_sig.get("action_n")
            hint = suggested_sim_config_from_obs(obs_dim)
            st.caption(
                f"Model expects obs_dim={obs_dim}, actions={action_n}. Recommended: {hint}"
            )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Initialize", use_container_width=True):
                try:
                    reset_episode(sim_config)
                    if st.session_state.policy_mode == "Trained Model":
                        try:
                            load_trained_agent(sim_config, rl_config, model_path)
                        except Exception as exc:
                            st.warning(f"Model load failed: {exc}")
                    st.success("Environment initialized.")
                except Exception as exc:
                    st.error(f"Initialization failed: {exc}")

        with col_b:
            if st.button("Load Model", use_container_width=True):
                try:
                    load_trained_agent(sim_config, rl_config, model_path)
                    st.success("Model loaded.")
                except Exception as exc:
                    st.error(f"Failed to load model: {exc}")

        auto_col_1, auto_col_2 = st.columns(2)
        with auto_col_1:
            if st.button("Start Auto Run", use_container_width=True):
                start_auto_run()
        with auto_col_2:
            if st.button("Stop Auto Run", use_container_width=True):
                stop_auto_run()

        st.caption(f"Auto run: {'ON' if st.session_state.auto_run else 'OFF'}")

        st.divider()
        manual_action = st.slider("Manual Action", min_value=0, max_value=9, value=1, step=1)
        col_c, col_d, col_e = st.columns(3)
        with col_c:
            if st.button("Step", use_container_width=True):
                run_one_step()
        with col_d:
            if st.button("Run 20", use_container_width=True):
                run_multiple_steps(20)
        with col_e:
            if st.button("Manual Step", use_container_width=True):
                run_one_step(selected_action=manual_action)

    env = st.session_state.env
    if env is None:
        st.info("Click Initialize in the sidebar to start simulation.")
        return

    if st.session_state.auto_run and not st.session_state.done:
        auto_run_tick()
        return

    info = st.session_state.last_info

    top_metrics = st.columns(4)
    top_metrics[0].metric("Episode Step", int(info.get("step", 0)))
    top_metrics[1].metric("Avg Fidelity", f"{float(info.get('avg_fidelity', 0.0)):.3f}")
    top_metrics[2].metric("Success Transmissions", int(info.get("successful_transmissions", 0)))
    top_metrics[3].metric("Episode Reward", f"{st.session_state.episode_reward:.3f}")

    st.plotly_chart(
        build_network_figure(env, int(info.get("step", 0))),
        use_container_width=True,
    )

    col1, col2 = st.columns([1.4, 1.0])

    with col1:
        history = st.session_state.history
        if len(history["step"]) > 0:
            df = pd.DataFrame(history)
            st.subheader("Live Metrics")
            st.line_chart(df.set_index("step")[["reward", "avg_fidelity", "success_rate"]])
        else:
            st.subheader("Live Metrics")
            st.write("Run steps to generate metrics.")

    with col2:
        st.subheader("Entanglement Table")
        ent_data = []
        for i in range(env.num_nodes - 1):
            fidelity = float(env.entanglement_pairs[i, 0])
            age = float(env.entanglement_pairs[i, 1])
            quality = env.channel.get_channel_quality(i, i + 1)
            ent_data.append(
                {
                    "link": f"{i}-{i+1}",
                    "entanglement_fidelity": round(fidelity, 4),
                    "age": round(age, 1),
                    "distance_km": round(float(quality.get("distance", 0.0)), 2),
                    "loss_probability": round(float(quality.get("loss_probability", 0.0)), 4),
                }
            )

        st.dataframe(pd.DataFrame(ent_data), use_container_width=True, hide_index=True)

    if st.session_state.done:
        st.warning("Episode ended. Click Initialize to start a new episode.")


if __name__ == "__main__":
    main()
