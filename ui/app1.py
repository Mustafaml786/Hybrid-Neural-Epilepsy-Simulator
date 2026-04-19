# ui/app.py

import streamlit as st
import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from runner.cli_1 import load_config
from adapters.hippocampus_brian2 import HippocampusBrian2Adapter as HippocampusAdapter
from adapters.worm_c302_full import WormC302FullAdapter as CElegansAdapter
from analysis.features_1 import compute_summary_metrics, apply_intervention_grid

st.set_page_config(page_title="Cross-Species Neural Simulation Dashboard", layout="wide")
st.title("Cross-Species Neural Simulation Dashboard")

# ----------------------
# 1️⃣ Load Config
# ----------------------
config = load_config()
st.sidebar.header("Simulation Config")
simulation_duration = st.sidebar.number_input(
    "Simulation duration (ms)", min_value=100, max_value=5000, value=config["simulation"]["duration"]
)

# ----------------------
# 2️⃣ Model Selection
# ----------------------
model_choice = st.sidebar.selectbox("Select Model", ["Hippocampus", "Worm"])

if model_choice == "Hippocampus":
    mode_choice = st.sidebar.selectbox("Mode", ["Healthy", "Epileptic"])
    model = HippocampusAdapter(config)
else:
    mode_choice = st.sidebar.selectbox("Variant", ["Default", "Variant"])
    model = CElegansAdapter(config)

# ----------------------
# 3️⃣ Load Stimulus
# ----------------------
st.sidebar.header("Stimulus Configuration")
stim_file = st.sidebar.text_input("Stimulus JSON file path", value=config["stimulus"]["file"])

with open(stim_file, "r") as f:
    stim = json.load(f)
st.sidebar.write("Stimulus parameters loaded:", stim)

# ----------------------
# 4️⃣ Run Simulation
# ----------------------
if st.button("Run Simulation"):
    st.info("Running simulation...")
    model.initialize()
    if hasattr(model, "apply_stimulus"):
        model.apply_stimulus(stim)
    model.run(simulation_duration)
    output = model.get_output()

    # Save outputs
    with open(f"runs/{model_choice.lower()}_raw.json", "w") as f:
        json.dump(output, f, indent=4)

    # ----------------------
    # 5️⃣ Display Results
    # ----------------------
    st.success("Simulation completed!")
    metrics = compute_summary_metrics(output)
    st.subheader("Summary Metrics")
    st.json(metrics)

    # Plot traces
    st.subheader("Voltage/Spike Trace")
    if "hipp_activity" in output:
        df = pd.DataFrame({"time_ms": range(len(output["hipp_activity"])), "voltage_mV": output["hipp_activity"]})
        st.line_chart(df.set_index("time_ms"))
    elif "spikes" in output:
        for neuron, vals in output["spikes"].items():
            st.line_chart(pd.DataFrame({neuron: vals}))

# ----------------------
# 6️⃣ Intervention Grid Search
# ----------------------
st.sidebar.header("Intervention Grid Search")

grid_search = st.sidebar.checkbox("Enable Grid Search")

if grid_search:
    st.sidebar.write("Define intervention parameter sets")
    param_grid = []
    n = st.sidebar.number_input("Number of parameter sets", min_value=1, max_value=20, value=2)
    for i in range(n):
        st.sidebar.markdown(f"**Set {i+1}**")
        amp = st.sidebar.number_input(f"Stim amplitude {i+1}", value=0.5, step=0.1, key=f"amp{i}")
        start = st.sidebar.number_input(f"Stim start {i+1}", value=100, step=10, key=f"start{i}")
        param_grid.append({"stim_amplitude": amp, "stim_start": start})

    if st.sidebar.button("Run Grid Search"):
        st.info("Running intervention grid search...")
        results = apply_intervention_grid(model, param_grid)
        st.success("Grid search completed!")

        st.subheader("Grid Search Results")
        for r in results:
            st.write("Params:", r["params"])
            st.json(r["metrics"])
