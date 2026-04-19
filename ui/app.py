# ui/app.py - Cross-Species Neural Simulation Dashboard
import sys
from pathlib import Path
import time
import os
import json
import yaml
import warnings
warnings.filterwarnings("ignore")

# Add the project root to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from adapters.hippocampus_brian2 import HippocampusBrian2Adapter
from adapters.worm_c302_full import WormC302FullAdapter
from analysis.features import compute_summary_metrics

st.set_page_config(page_title="Hybrid Neural Epilepsy Simulator", layout="wide", page_icon="🧠")

# -----------------------
# Helper functions
# -----------------------
def load_config(path="configs/base.yaml"):
    if not os.path.exists(path):
        return {"simulation": {"duration": 1000, "dt": 0.025}, "stimulus": {"file": "configs/stimuli/pulse.json"}}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_stimulus(path):
    if not os.path.exists(path):
        return {"type": "pulse", "amplitude": 0.5, "start": 100, "duration": 200}
    with open(path, "r") as f:
        return json.load(f)

def save_outputs(outdir, name_prefix, output, metrics):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(outdir, f"{name_prefix}.json"), "w") as f:
        json.dump({"output": output, "metrics": metrics}, f, indent=2)

def plot_voltage_trace(time_data, voltage_data, title="Voltage Trace"):
    """Plot voltage over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if time_data and voltage_data:
        ax.plot(time_data, voltage_data, linewidth=0.5, color="#2196F3")
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Voltage (mV)", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=-50, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
    st.pyplot(fig)
    plt.close()

def plot_spikes_histogram(spike_counts, title="Spike Counts per Neuron"):
    """Plot histogram of spike counts."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if spike_counts and len(spike_counts) > 0:
        ax.bar(range(len(spike_counts)), spike_counts, color="#4CAF50", alpha=0.7)
        ax.set_xlabel("Neuron Index", fontsize=10)
        ax.set_ylabel("Spike Count", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "No spike data", ha='center', va='center', transform=ax.transAxes)
    st.pyplot(fig)
    plt.close()

def plot_worm_heatmap(spike_data, mode_name):
    """Plot activity as color-coded heatmap."""
    neurons = list(spike_data.keys())
    values = [np.mean(spike_data[n]) for n in neurons]
    
    fig, ax = plt.subplots(figsize=(14, 3))
    
    # Create color-coded bars based on activity level
    colors = []
    for v in values:
        if v >= 90:
            colors.append('#d32f2f')  # Red - high activity
        elif v >= 50:
            colors.append('#ff9800')  # Orange - medium
        else:
            colors.append('#4caf50')  # Green - low
    
    ax.barh(0, values, color=colors, height=0.6, alpha=0.8)
    ax.set_xlim(0, max(values) * 1.1)
    ax.set_yticks([])
    ax.set_xlabel("Spike Count", fontsize=10)
    ax.set_title(f"Worm {mode_name} - Activity Heatmap (Red=High, Orange=Medium, Green=Low)", fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add neuron labels at bottom
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    st.pyplot(fig)
    plt.close()

def plot_spike_raster(spike_times_dict, title="Spike Raster Plot"):
    """Plot spike times as raster - shows when each neuron spikes."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    neurons = list(spike_times_dict.keys())
    for i, neuron in enumerate(neurons):
        times = spike_times_dict[neuron]
        if times and len(times) > 0:
            # Filter valid times
            valid_times = [t for t in times if t is not None and t > 0]
            if valid_times:
                ax.scatter(valid_times, [i] * len(valid_times), marker='|', color='black', s=50, alpha=0.7)
    
    ax.set_yticks(range(len(neurons)))
    ax.set_yticklabels(neurons, fontsize=6)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Neuron", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(spike_times_dict.values()[0][-1] if spike_times_dict.values() else 1000) * 1.1)
    
    st.pyplot(fig)
    plt.close()

def plot_voltage_distribution(voltage_data, title="Voltage Distribution"):
    """Plot histogram of membrane voltages."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if voltage_data and len(voltage_data) > 0:
        # Flatten array if needed
        if hasattr(voltage_data, 'flatten'):
            v_data = voltage_data.flatten()
        else:
            v_data = np.array(voltage_data)
        
        ax.hist(v_data, bins=50, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=np.mean(v_data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(v_data):.1f} mV')
        ax.axvline(x=np.median(v_data), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(v_data):.1f} mV')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No voltage data", ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel("Voltage (mV)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    plt.close()

def plot_statistics_panel(metrics, model_name, raw_data=None):
    """Display statistics panel with quantitative metrics."""
    st.markdown("#### 📈 Statistics Panel")
    
    if model_name.lower() == "hippocampus":
        # Extract key metrics
        num_spikes = metrics.get('num_spikes', 0)
        avg_voltage = metrics.get('hipp_activity_avg', 0)
        
        v_std = metrics.get('voltage_std_mV')
        v_max = metrics.get('voltage_max_mV')
        
        if not v_std and raw_data and raw_data.get('voltage_mean_mV'):
            voltages = np.array(raw_data['voltage_mean_mV'])
            v_std = np.std(voltages)
            v_max = np.max(voltages)
        
        if not v_std:
            v_std = 0
        if not v_max:
            v_max = avg_voltage + 20
        
        v_min = avg_voltage - v_std * 2 if v_std else avg_voltage - 5
        
        # Display statistics grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Voltage", f"{avg_voltage:.2f} mV")
        with col2:
            st.metric("Std Dev", f"{v_std:.2f} mV" if v_std else "N/A")
        with col3:
            st.metric("Min", f"{v_min:.2f} mV" if v_min else f"{avg_voltage-5:.2f} mV")
        with col4:
            st.metric("Max", f"{v_max:.2f} mV" if v_max else f"{avg_voltage+5:.2f} mV")
        
        # Firing rate
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Spikes", f"{num_spikes:,}")
        with col2:
            # Estimate firing rate
            dur_ms = raw_data.get('times_ms', [1000])[-1] if raw_data else 1000
            firing_rate = (num_spikes / dur_ms) * 1000 if dur_ms > 0 else 0
            st.metric("Firing Rate", f"{firing_rate:.2f} Hz")
    
    else:
        # Worm statistics
        spike_data = metrics.get('mean_activity', [])
        if not spike_data:
            spike_data = []
        
        values = list(spike_data) if spike_data else [0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Activity", f"{np.mean(values):.2f}")
        with col2:
            st.metric("Std Dev", f"{np.std(values):.2f}")
        with col3:
            st.metric("Min", f"{np.min(values):.2f}")
        with col4:
            st.metric("Max", f"{np.max(values):.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Spikes", f"{metrics.get('num_spikes', 0):,}")
        with col2:
            st.metric("Active Neurons", f"{metrics.get('n_neurons', 0)}")

def plot_metrics_gauge(metrics, model_name):
    """Display metrics in a nice format."""
    col1, col2, col3 = st.columns(3)
    
    if model_name.lower() == "hippocampus":
        with col1:
            st.metric("Avg Voltage", f"{metrics.get('hipp_activity_avg', 0):.1f} mV")
        with col2:
            st.metric("Total Spikes", f"{metrics.get('num_spikes', 0):,}")
        with col3:
            st.metric("Bursts", metrics.get('hipp_burst_count', 0))
    else:
        # Handle worm data - extract mean activity value
        mean_val = metrics.get('mean_activity') or metrics.get('mean_act_val', 0)
        if isinstance(mean_val, list):
            mean_val = np.mean(mean_val) if mean_val else 0
        elif isinstance(mean_val, dict):
            mean_val = sum(mean_val.values()) / len(mean_val) if mean_val else 0
        with col1:
            st.metric("Mean Activity", f"{mean_val:.2f}")
        with col2:
            st.metric("Total Spikes", f"{metrics.get('num_spikes', 0):,}")
        with col3:
            st.metric("Active Neurons", metrics.get('n_neurons', 0))

# -----------------------
# Main UI Layout
# -----------------------
st.title("🧠 Hybrid Neural Epilepsy Simulator")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_select = st.selectbox("Select Model", ["Hippocampus", "Worm"])
    
    if model_select == "Hippocampus":
        mode_select = st.selectbox("Mode", ["Healthy", "Epileptic"])
    else:
        mode_select = st.selectbox("Variant", ["Default", "Variant"])
    
    st.markdown("---")
    st.subheader("Stimulus Settings")
    stim_amp = st.slider("Amplitude (nA)", 0.1, 2.0, 0.5, 0.1)
    stim_start = st.slider("Start (ms)", 10, 500, 100, 10)
    stim_dur = st.slider("Duration (ms)", 50, 500, 200, 50)
    
    st.markdown("---")
    st.subheader("Simulation Settings")
    sim_dur = st.number_input("Duration (ms)", 100, 5000, 1000, 100)
    n_neurons = st.number_input("Neurons", 10, 100, 50, 10)
    
    # Comparison mode
    st.markdown("---")
    run_compare = st.checkbox("🔄 Run Comparison (Both Modes)", value=False)
    
    st.markdown("---")
    runs_dir = "runs"

# Build stimulus config
stim_params = {"type": "pulse", "amplitude": stim_amp, "start": stim_start, "duration": stim_dur}

# Main run button
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**Model:** {model_select} | **Mode:** {mode_select}")
with col2:
    run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Comparison mode results storage
comparison_results = {}

# Run simulation
if run_btn:
    with st.spinner("Running simulation..."):
        try:
            if model_select == "Hippocampus":
                model = HippocampusBrian2Adapter({}, mode=mode_select)
                model.initialize(n_neurons=n_neurons, stim_cfg=stim_params)
                raw_output = model.run(sim_dur)
                summary = model.get_output()
            else:
                model = WormC302FullAdapter({}, mode=mode_select)
                model.initialize()
                model.apply_stimulus(stim_params)
                raw_output = model.run(sim_dur)
                summary = model.get_output()
            
            metrics = compute_summary_metrics(summary)
            
            # Save results
            timestamp = int(time.time())
            if model_select == "Worm":
                outdir = Path("worm_runs") / f"run_{timestamp}"
            else:
                outdir = Path(runs_dir) / f"run_{timestamp}"
            save_outputs(str(outdir), f"{model_select.lower()}_{mode_select.lower()}", summary, metrics)
            
            # Success message
            folder_name = f"worm_runs/run_{timestamp}" if model_select == "Worm" else f"runs/run_{timestamp}"
            st.success(f"✅ Simulation completed! Results saved to {folder_name}/")
            
            # Display metrics nicely
            st.markdown("### 📊 Results Summary")
            plot_metrics_gauge(metrics, model_select)
            
            # Visualization
            st.markdown("### 📈 Visualizations")
            
            if model_select == "Hippocampus":
                tab1, tab2, tab3 = st.tabs(["Voltage Trace", "Spike Distribution", "Statistics"])
                
                with tab1:
                    if raw_output and raw_output.get("voltage_mean_mV"):
                        volts = raw_output["voltage_mean_mV"]
                        times = raw_output.get("times_ms", list(range(len(volts))))
                        plot_voltage_trace(times, volts, f"Hippocampus {mode_select} - Voltage Trace")
                    else:
                        st.warning("No voltage data available")
                
                with tab2:
                    if raw_output and raw_output.get("spike_counts"):
                        plot_spikes_histogram(raw_output["spike_counts"], "Spikes per Neuron")
                    else:
                        st.warning("No spike data available")
                
                with tab3:
                    st.markdown("##### Additional Analysis")
                    # Voltage distribution
                    if raw_output and raw_output.get("voltage_mean_mV"):
                        with st.expander("📊 Voltage Distribution"):
                            plot_voltage_distribution(raw_output["voltage_mean_mV"], "Voltage Distribution Across Neurons")
                    
                    # Statistics panel
                    plot_statistics_panel(metrics, model_select, raw_output)
            else:
                # Worm: show per-neuron activity
                st.markdown("#### Worm Neuron Activity (Real Connectome Data)")
                if summary and summary.get("spikes"):
                    spike_data = summary["spikes"]
                    neurons = list(spike_data.keys())
                    values = [np.mean(spike_data[n]) for n in neurons]
                    
                    # Show all neurons with actual names
                    fig, ax = plt.subplots(figsize=(14, 5))
                    colors = ["#FF5722" if mode_select == "Variant" else "#2196F3"] * len(neurons)
                    ax.bar(range(len(neurons)), values, color=colors, alpha=0.8, width=0.8)
                    ax.set_xlabel("Neuron", fontsize=10)
                    ax.set_ylabel("Spike Count", fontsize=10)
                    ax.set_title(f"Worm {mode_select}: {len(neurons)} neurons from real connectome", fontsize=12)
                    ax.set_xticks(range(len(neurons)))
                    ax.set_xticklabels(neurons, rotation=90, fontsize=6)  # Show actual neuron names
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.set_xlim(-1, len(neurons))
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show connection info
                    st.info(f"📊 Using 50 most connected neurons from herm_full_edgelist.csv ({858} connections)")
                    
                    # Show activity heatmap
                    with st.expander("🌡️ View Activity Heatmap"):
                        plot_worm_heatmap(spike_data, mode_select)
                    
                    # CSV Export
                    df = pd.DataFrame({"Neuron": neurons, "Spike_Count": values})
                    csv = df.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, f"worm_{mode_select.lower()}_results.csv", "text/csv", key="csv-export")
                    
                    # Statistics panel for worm
                    plot_statistics_panel(metrics, model_select)
            
            # Raw data expander
            with st.expander("📋 View Raw Data"):
                st.json(summary)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# =======================
# COMPARISON MODE
# =======================
if run_compare and not run_btn:
    # When comparison mode is selected, run both modes
    st.markdown("---")
    st.markdown("### 🔄 Running Comparison: Both Modes")
    
    with st.spinner("Running both modes..."):
        try:
            if model_select == "Hippocampus":
                modes = ["Healthy", "Epileptic"]
            else:
                modes = ["Default", "Variant"]
            
            results = {}
            
            for m in modes:
                if model_select == "Hippocampus":
                    model = HippocampusBrian2Adapter({}, mode=m)
                    model.initialize(n_neurons=n_neurons, stim_cfg=stim_params)
                    raw = model.run(sim_dur)
                    summary = model.get_output()
                else:
                    model = WormC302FullAdapter({}, mode=m)
                    model.initialize()
                    model.apply_stimulus(stim_params)
                    raw = model.run(sim_dur)
                    summary = model.get_output()
                
                results[m] = {"raw": raw, "summary": summary, "metrics": compute_summary_metrics(summary)}
            
            # Display comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### 🟢 {modes[0]}")
                r1 = results[modes[0]]
                m1 = r1["metrics"]
                if model_select == "Hippocampus":
                    st.metric("Spikes", f"{m1.get('num_spikes', 0):,}")
                    st.metric("Avg Voltage", f"{m1.get('hipp_activity_avg', 0):.1f} mV")
                else:
                    st.metric("Spikes", f"{m1.get('num_spikes', 0):,}")
                    st.metric("Mean Activity", f"{m1.get('mean_act_val', 0):.1f}")
            
            with col2:
                st.markdown(f"#### 🔴 {modes[1]}")
                r2 = results[modes[1]]
                m2 = r2["metrics"]
                if model_select == "Hippocampus":
                    st.metric("Spikes", f"{m2.get('num_spikes', 0):,}")
                    st.metric("Avg Voltage", f"{m2.get('hipp_activity_avg', 0):.1f} mV")
                else:
                    st.metric("Spikes", f"{m2.get('num_spikes', 0):,}")
                    st.metric("Mean Activity", f"{m2.get('mean_act_val', 0):.1f}")
            
            # Comparison table
            st.markdown("---")
            st.markdown("### 📊 Comparison Table")
            
            if model_select == "Hippocampus":
                comparison_df = pd.DataFrame({
                    "Mode": modes,
                    "Total Spikes": [m1.get('num_spikes', 0), m2.get('num_spikes', 0)],
                    "Avg Voltage (mV)": [f"{m1.get('hipp_activity_avg', 0):.2f}", f"{m2.get('hipp_activity_avg', 0):.2f}"],
                    "Difference": ["-", f"+{(m2.get('num_spikes', 0) - m1.get('num_spikes', 0))/m1.get('num_spikes', 1)*100:.1f}%"]
                })
            else:
                comparison_df = pd.DataFrame({
                    "Mode": modes,
                    "Total Spikes": [m1.get('num_spikes', 0), m2.get('num_spikes', 0)],
                    "Mean Activity": [f"{m1.get('mean_act_val', 0):.2f}", f"{m2.get('mean_act_val', 0):.2f}"],
                    "Difference": ["-", f"+{(m2.get('num_spikes', 0) - m1.get('num_spikes', 0))/m1.get('num_spikes', 1)*100:.1f}%"]
                })
            
            st.table(comparison_df)
            
            # Save comparison results
            timestamp = int(time.time())
            outdir = Path("runs") / f"comparison_{timestamp}"
            save_outputs(str(outdir), "comparison", {"results": {m: v["summary"] for m, v in results.items()}}, comparison_df.to_dict())
            st.success(f"✅ Comparison saved to runs/comparison_{timestamp}/")
            
        except Exception as e:
            st.error(f"❌ Comparison Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("🧠 Hybrid Neural Epilepsy Simulator | A Computational Approach to Modeling Epileptic Brain Behavior")