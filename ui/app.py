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


def plot_phase_portrait(voltage_data, title="Phase Portrait (Voltage vs dV/dt)"):
    """
    Phase space plot showing neuron dynamics.
    
    WHAT THIS GRAPH MEANS:
    - The X-axis shows voltage (mV) - the electrical state of the neuron
    - The Y-axis shows rate of change (dV/dt) - how fast voltage is changing
    - Together they show how the neuron behaves dynamically
    
    HOW TO READ IT:
    - Each dot represents one moment in time
    - The loop size shows how "excitable" the neuron is
    - Healthy: Small, tight orbit near rest (-65mV)
    - Epileptic: Large wide orbit reaching high dV/dt (unstable, hyperactive)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if voltage_data and len(voltage_data) > 2:
        if hasattr(voltage_data, 'flatten'):
            v = voltage_data.flatten()
        else:
            v = np.array(voltage_data)
        
        dv_dt = np.gradient(v)
        
        ax.scatter(v, dv_dt, s=1, alpha=0.5, c='#9C27B0', linewidths=0)
        
        ax.set_xlabel("Voltage (mV)", fontsize=10)
        ax.set_ylabel("dV/dt (mV/ms)", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        ax.axvline(x=-65, color='green', linestyle='--', alpha=0.5, label='Rest (~-65mV)')
        ax.axvline(x=-40, color='red', linestyle='--', alpha=0.5, label='Threshold (~-40mV)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient voltage data", ha='center', va='center', transform=ax.transAxes)
    
    st.pyplot(fig)
    plt.close()


def plot_power_spectrum(time_data, voltage_data, title="Power Spectrum (FFT)"):
    """
    Frequency analysis using Fast Fourier Transform (FFT).
    
    WHAT THIS GRAPH MEANS:
    - Shows which frequencies dominate the neural activity
    - The X-axis is frequency in Hertz (Hz) - how fast oscillations repeat
    - The Y-axis is power - how strong each frequency is
    
    KEY FREQUENCY BANDS:
    | Band | Frequency | What it means |
    |------|-----------|--------------|
    | Delta | 1-4 Hz | Slow waves (normal rest) |
    | Theta | 4-8 Hz | Drowsy/relaxed state |
    | Beta | 13-30 Hz | Active thinking |
    | Gamma | 30-100 Hz | HIGH cognitive load (memory formation) |
    | Ripple | 100-200 Hz | PATHOLOGICAL - seizure signature! |
    
    HOW TO READ IT:
    - Healthy: Small peaks in low frequencies (<20 Hz)
    - Epileptic: Large peaks in gamma (30-100 Hz) or ripple (100-200 Hz) = seizure activity
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if voltage_data and len(voltage_data) > 10:
        if hasattr(voltage_data, 'flatten'):
            v = voltage_data.flatten()
        else:
            v = np.array(voltage_data)
        
        if time_data:
            if hasattr(time_data, 'flatten'):
                t = time_data.flatten()
            else:
                t = np.array(time_data)
            dt = np.mean(np.diff(t)) if len(t) > 1 else 0.1
        else:
            dt = 0.1
            t = np.arange(len(v)) * dt
        
        n = len(v)
        freqs = np.fft.fftfreq(n, dt)[:n//2]
        fft_vals = np.fft.fft(v)[:n//2]
        power = np.abs(fft_vals) ** 2
        
        ax.semilogy(freqs, power, color='#FF5722', linewidth=1, alpha=0.8)
        
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Power (log scale)", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 500)
        
        bands = [
            (4, 8, 'theta', '#2196F3'),
            (13, 30, 'beta', '#4CAF50'),
            (30, 100, 'gamma', '#FF9800'),
            (100, 200, 'ripple', '#F44336'),
        ]
        
        for low, high, name, color in bands:
            if high <= 500 / 2:
                ax.axvspan(low, high, alpha=0.15, color=color, label=name)
        
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient voltage data", ha='center', va='center', transform=ax.transAxes)
    
    st.pyplot(fig)
    plt.close()


def plot_synchrony_over_time(spike_trains, duration_ms=1000, window_ms=50, title="Network Synchrony Over Time"):
    """
    Track how synchronized the neuron population is over time.
    
    WHAT THIS GRAPH MEANS:
    - Shows how many neurons fire together at each moment
    - The X-axis is time in milliseconds
    - The Y-axis is synchrony (0 to 1 = 0% to 100% of neurons firing together)
    
    HOW TO READ IT:
    - Each point shows how many neurons fired in a time window
    - Healthy: Low synchrony (0.05-0.2) - neurons fire independently
    - Epileptic: High synchrony (0.5-1.0) - groups of neurons firing together = seizure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if spike_trains and len(spike_trains) > 0:
        all_times = []
        for neuron_key, times in spike_trains.items():
            for t in times:
                try:
                    all_times.append(float(t))
                except (ValueError, TypeError):
                    pass
        
        if len(all_times) < 2:
            ax.text(0.5, 0.5, "Insufficient spike data", ha='center', va='center', transform=ax.transAxes)
            st.pyplot(fig)
            plt.close()
            return
        
        time_arr = np.array(all_times)
        
        time_windows = np.arange(0, duration_ms, window_ms)
        synchrony = []
        
        for i in range(len(time_windows) - 1):
            start = time_windows[i]
            end = time_windows[i + 1]
            spikes_in_window = np.sum((time_arr >= start) & (time_arr < end))
            total_neurons = len(spike_trains)
            sync = min(spikes_in_window / total_neurons, 1.0)
            synchrony.append(sync)
        
        ax.fill_between(time_windows[:-1], synchrony, alpha=0.3, color='#3F51B5')
        ax.plot(time_windows[:-1], synchrony, color='#3F51B5', linewidth=1.5)
        
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Healthy (~0.1-0.2)')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Seizure threshold (~0.5)')
        
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Synchrony Index", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No spike data", ha='center', va='center', transform=ax.transAxes)
    
    st.pyplot(fig)
    plt.close()


def plot_isi_histogram(spike_trains, title="Interspike Interval (ISI) Distribution"):
    """
    Distribution of time intervals between consecutive spikes.
    
    WHAT THIS GRAPH MEANS:
    - Shows the timing pattern of neural firing
    - X-axis: time gap between spikes in milliseconds
    - Y-axis: how often each gap occurs
    
    HOW TO READ IT:
    - Healthy: Irregular intervals - random Poisson-like distribution
    - Epileptic: Bunching around specific values = burst firing pattern
    - Long tail in healthy = occasional spontaneous spikes
    - Peak in epileptic = regular burst intervals (pathological)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if spike_trains and len(spike_trains) > 0:
        all_times = []
        for neuron_key, times in spike_trains.items():
            for t in times:
                try:
                    all_times.append(float(t))
                except (ValueError, TypeError):
                    pass
        
        if len(all_times) < 2:
            ax.text(0.5, 0.5, "Insufficient spike data", ha='center', va='center', transform=ax.transAxes)
            st.pyplot(fig)
            plt.close()
            return
        
        all_times = sorted(all_times)
        isi = np.diff(all_times)
        
        if len(isi) > 0:
            ax.hist(isi, bins=50, color='#009688', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(x=np.mean(isi), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(isi):.1f} ms')
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Not enough ISI data", ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No spike data", ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel("Interspike Interval (ms)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    plt.close()


def graph_explanation(graph_name, healthy_desc, epileptic_desc):
    """Display expandable graph explanation in simple terms."""
    with st.expander(f"📖 What does this graph mean? - {graph_name}"):
        st.markdown(f"""
**{graph_name}**

**Healthy State:** {healthy_desc}

**Epileptic State:** {epileptic_desc}
        """)


def detailed_explanation(graph_title, what_it_shows, how_to_read, expected_values, clinical_significance):
    """
    Display comprehensive, examiner-friendly explanation.
    
    Parameters:
    - graph_title: Name of the graph
    - what_it_shows: What the graph measures (string)
    - how_to_read: List of step-by-step reading instructions
    - expected_values: Dict with 'healthy' and 'epileptic' values
    - clinical_significance: Why this matters (string)
    """
    with st.expander(f"📖 Understanding the {graph_title}"):
        st.markdown(f"### What This Graph Shows")
        st.markdown(what_it_shows)
        
        st.markdown("### How to Read It (Step by Step)")
        for i, step in enumerate(how_to_read, 1):
            st.markdown(f"{i}. {step}")
        
        st.markdown("### Expected Values")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🟢 Healthy State**")
            st.markdown(expected_values.get('healthy', 'N/A'))
        with col2:
            st.markdown("**🔴 Epileptic State**")
            st.markdown(expected_values.get('epileptic', 'N/A'))
        
        st.markdown("### Clinical Significance")
        st.markdown(clinical_significance)

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


def plot_seizure_detection(metrics):
    """Display seizure detection results."""
    seizure_detected = metrics.get('seizure_detected', 0)
    seizure_prob = metrics.get('seizure_probability', 0.0)
    seizure_severity = metrics.get('seizure_severity', 'none')
    biomarkers = metrics.get('seizure_biomarkers', {})
    
    st.markdown("---")
    st.markdown("### 🩺 Seizure Detection")
    
    # Status indicator with color coding
    if seizure_detected:
        status_color = "🔴"
        status_text = "SEIZURE DETECTED"
    elif seizure_prob >= 0.2:
        status_color = "🟡"
        status_text = "ELEVATED RISK"
    else:
        status_color = "🟢"
        status_text = "NORMAL"
    
    # Display probability with progress bar
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status", f"{status_color} {status_text}")
    with col2:
        st.metric("Probability", f"{seizure_prob:.1%}")
    
    # Progress bar
    seizure_bar = st.progress(seizure_prob)
    
    # Severity badge
    severity_colors = {
        "none": "🟢",
        "mild": "🟡", 
        "moderate": "🟠",
        "severe": "����"
    }
    severity_icon = severity_colors.get(seizure_severity, "⚪")
    st.markdown(f"**Severity:** {severity_icon} {seizure_severity.upper()}")
    
    # Show biomarkers in expander
    if biomarkers:
        with st.expander("📊 View Biomarkers"):
            bm_col1, bm_col2 = st.columns(2)
            with bm_col1:
                st.metric("Spike Rate", f"{biomarkers.get('spike_rate_hz', 0):.1f} Hz")
                st.metric("Synchrony", f"{biomarkers.get('synchrony_index', 0):.2f}")
            with bm_col2:
                st.metric("Variance", f"{biomarkers.get('variance_ratio', 0):.2f}")
                st.metric("Burst", f"{biomarkers.get('burst_intensity', 0):.2f}")

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
            
            # Display seizure detection (Hippocampus only - not applicable for C302 worm model)
            if model_select == "Hippocampus":
                plot_seizure_detection(summary)
            
            # Visualization
            st.markdown("### 📈 Visualizations")
            
            if model_select == "Hippocampus":
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Voltage Trace", "Phase Plot", "Frequency Analysis", "Spike Distribution", "Statistics", "Prevention Guide"])
                
                with tab1:
                    if raw_output and raw_output.get("voltage_mean_mV"):
                        volts = raw_output["voltage_mean_mV"]
                        times = raw_output.get("times_ms", list(range(len(volts))))
                        plot_voltage_trace(times, volts, f"Hippocampus {mode_select} - Voltage Trace")
                        detailed_explanation(
                            graph_title="Voltage Trace",
                            what_it_shows="""
This graph shows the **membrane potential (V)** of a neuron over **time**. 

The membrane potential is the electrical difference between inside and outside of the neuron. 
When the voltage goes UP rapidly, the neuron is firing an **action potential** (a spike). 
When it returns DOWN, the neuron is resetting to its resting state.
                            """,
                            how_to_read=[
                                "Look at the Y-axis (Voltage): Negative values mean the neuron is at 'rest'",
                                "The horizontal red dashed line at -50mV is the **threshold** - above this, a spike will fire",
                                "Spikes going UP to +30-40mV = neuron actively firing",
                                "Look at how fast the voltage returns to rest between spikes",
                                "In Healthy: quick return to ~-65mV. In Epileptic: slow return, stays depolarized"
                            ],
                            expected_values={
                                "healthy": """
- Resting potential: **-60 to -70 mV**
- Spike peak: **+30 to +40 mV**  
- Return to rest: **< 5 ms**
- Spike frequency: **1-5 Hz (occasional)**
- Overall pattern: **Sparse, irregular spiking**
                                """,
                                "epileptic": """
- Resting potential: **-45 to -55 mV** (depolarized!)
- Spike peak: **+20 to +30 mV** (smaller due to sodium channel inactivation)
- Return to rest: **10-20 ms** (slower)
- Spike frequency: **10-50 Hz (frequent)**
- Overall pattern: **Sustained depolarization, burst-like firing**
                                """
                            },
                            clinical_significance="""
**Why this matters for epilepsy diagnosis:**

In a healthy brain, neurons fire occasionally and return quickly to rest. This allows the brain to process information in between spikes.

In epilepsy, neurons have **impaired potassium channels** that can't quickly reset the voltage. This causes:
1. **Depolarization block** - Neurons stuck at high voltage can't fire properly
2. **Hyper-excitability** - Lower threshold means easier to trigger new spikes
3. **Synchronization** - Stuck neurons can trigger their neighbors

The key diagnostic indicator is: **slow return to rest** after a spike.
                            """
                        )
                    else:
                        st.warning("No voltage data available")
                
                with tab2:
                    if raw_output and raw_output.get("voltage_mean_mV"):
                        volts = raw_output["voltage_mean_mV"]
                        plot_phase_portrait(volts, f"Hippocampus {mode_select} - Phase Portrait")
                        detailed_explanation(
                            graph_title="Phase Portrait (V vs dV/dt)",
                            what_it_shows="""
This graph shows the **dynamics of the neuron** in a different way.

Instead of voltage vs time, it plots:
- **X-axis:** Voltage (mV) - the electrical state
- **Y-axis:** dV/dt (mV/ms) - how FAST the voltage is changing

This is essentially looking at the neuron's "speed" vs its "position". 
Together, they form a **phase space** that reveals the stability of the neuron.
                            """,
                            how_to_read=[
                                "Each purple dot represents one moment in time",
                                "The pattern forms a loop/orbit - larger = more excitable",
                                "Green dashed line = rest potential (-65mV)",
                                "Red dashed line = spike threshold (-40mV)",
                                "In Healthy: Small, tight loop near rest. In Epileptic: Wide orbit reaching high dV/dt"
                            ],
                            expected_values={
                                "healthy": """
- Orbit size: **Tight, < 20 mV range**
- Maximum dV/dt: **< 5 mV/ms**
- Returns to rest: **Quickly (< 10 ms)**
- Pattern: **Stable limit cycle**
- Visual: **Compact cluster near -65mV**
                                """,
                                "epileptic": """
- Orbit size: **Wide, 40-60 mV range**
- Maximum dV/dt: **> 10 mV/ms** (fast changes!)
- Returns to rest: **Slowly (> 20 ms)**
- Pattern: **Unstable, overshoots significantly**
- Visual: **Wide spread, reaches high dV/dt values**
                                """
                            },
                            clinical_significance="""
**Why this matters for epilepsy diagnosis:**

The phase portrait reveals the **dynamical stability** of the neuron:

1. **Small orbit (healthy):** Neuron is stable - any perturbation is quickly corrected
2. **Large orbit (epileptic):** Neuron is unstable - it overshoots and oscillates

This relates to the **Hodgkin-Huxley model**:
- Sodium (Na+) channels open fast → fast rise (large dV/dt)
- Potassium (K+) channels open slow → slow fall 
- In epilepsy: Na+ channels stay open longer, K+ channels are impaired

**Key diagnostic indicator:** Orbit size and maximum dV/dt value.
                            """
                        )
                    else:
                        st.warning("No voltage data available")
                
                with tab3:
                    if raw_output and raw_output.get("voltage_mean_mV") and raw_output.get("times_ms"):
                        volts = raw_output["voltage_mean_mV"]
                        times = raw_output.get("times_ms")
                        plot_power_spectrum(times, volts, f"Hippocampus {mode_select} - Power Spectrum")
                        detailed_explanation(
                            graph_title="Power Spectrum (FFT)",
                            what_it_shows="""
This graph shows the **frequency content** of the neural signal using Fast Fourier Transform (FFT).

It answers: "What frequencies make up the voltage signal?"

Different brain waves correspond to different mental states:
- Low frequencies (delta/theta): Slow, restful states
- High frequencies (gamma/ripple): Active processing, **pathological in epilepsy**

The shaded bands show different frequency ranges, and peaks indicate strong oscillations.
                            """,
                            how_to_read=[
                                "Look at X-axis: Frequency in Hertz (Hz) - how fast oscillations repeat",
                                "Look at Y-axis: Power (log scale) - how strong each frequency is",
                                "Peaks above the noise = significant oscillations",
                                "Color bands: Blue=theta, Green=beta, Orange=gamma, Red=ripple",
                                "Healthy: Small peaks in low freq. Epileptic: Large peaks in gamma/ripple"
                            ],
                            expected_values={
                                "healthy": """
- Dominant frequencies: **1-20 Hz** (delta, theta)
- Gamma power: **< 5%** of total
- Ripple power: **Minimal or absent**
- Pattern: **Smooth decay with frequency**
- Clinical: **Normal neural oscillations**
                                """,
                                "epileptic": """
- New frequencies: **30-100 Hz (gamma), 100-200 Hz (ripple)**
- Gamma power: **20-50%** of total (elevated!)
- Ripple power: **Present** (pathological marker)
- Pattern: **Peaks in high frequencies**
- Clinical: **Seizure signature - high-frequency oscillations**
                                """
                            },
                            clinical_significance="""
**Why this matters for epilepsy diagnosis:**

High-frequency oscillations (HFOs) are a **biomarker for epileptic tissue**:

1. **Gamma (30-100 Hz):** 
   - Healthy: Associated with memory/cognition
   - Epileptic: Pathological when sustained, indicates hyper-excitability

2. **Ripple (100-200 Hz):**
   - Almost exclusively pathological
   - Localizes the seizure onset zone in pre-surgical evaluation
   - Often the clearest marker of epileptic tissue

**Key diagnostic indicators:**
- Presence of ripple (100-200 Hz) = strong seizure indicator
- Elevated gamma (>30% power) = hyper-excitable network
                            """
                        )
                    else:
                        st.warning("Insufficient data for frequency analysis")
                
                with tab4:
                    if raw_output and raw_output.get("spike_counts"):
                        plot_spikes_histogram(raw_output["spike_counts"], "Spikes per Neuron")
                        if raw_output.get("spike_trains_ms"):
                            plot_synchrony_over_time(
                                raw_output["spike_trains_ms"], 
                                duration_ms=raw_output.get("times_ms", [1000])[-1] if raw_output.get("times_ms") else 1000,
                                title="Network Synchrony"
                            )
                            detailed_explanation(
                                graph_title="Network Synchrony",
                                what_it_shows="""
This graph tracks **how synchronized** the neuron population is over time.

It answers: "Are neurons firing together or independently?"

We calculate a **synchrony index** (0 to 1):
- 0 = No neurons firing together (all independent)
- 1 = All neurons fire exactly together (perfect sync)

The synchrony index is calculated by counting spikes in sliding time windows.
                                """,
                                how_to_read=[
                                    "Look at X-axis: Time (ms) - shows when synchrony changes",
                                    "Look at Y-axis: Synchrony index (0 to 1)",
                                    "Dashed green line = healthy threshold (~0.2)",
                                    "Dashed red line = seizure threshold (~0.5)",
                                    "Above red line = coordinated network burst",
                                    "Yellow/orange regions = periods of high synchrony"
                                ],
                                expected_values={
                                    "healthy": """
- Average synchrony: **0.05 - 0.20**
- Peak synchrony: **< 0.30**
- Pattern: **Independent, random firing**
- Analogy: **Popcorn popping randomly**
- Burst count: **0-1** per simulation
- Clinical: **Normal desynchronized activity**
                                    """,
                                    "epileptic": """
- Average synchrony: **0.40 - 0.70**
- Peak synchrony: **> 0.80**
- Pattern: **Groups of neurons fire together**
- Analogy: **Applause - everyone claps at once**
- Burst count: **2-5** per simulation
- Clinical: **SEIZURE - pathologically synchronized**
                                    """
                                },
                                clinical_significance="""
**Why this matters for epilepsy diagnosis:**

Synchronization is one of the **most important seizure biomarkers**:

1. **Network bursts:** When many neurons fire together, they create a 
   larger field that can trigger even MORE neurons
2. **Positive feedback loop:** Synchronized firing → stronger field → more sync
3. **Seizure onset:** Often starts with small sync, then cascades

**How seizures spread:**
- Focus (few neurons) → Synchronize → Trigger neighbors → Larger sync → Spreads → Full seizure

**Key diagnostic indicators:**
- Average synchrony > 0.3 = Elevated risk
- Average synchrony > 0.5 = Active seizure-like activity
- Multiple network bursts = Paroxysmal depolarization shift (PDS)
                                """
                            )
                    else:
                        st.warning("No spike data available")
                
                with tab5:
                    st.markdown("##### Additional Analysis")
                    if raw_output and raw_output.get("voltage_mean_mV"):
                        with st.expander("📊 Voltage Distribution"):
                            plot_voltage_distribution(raw_output["voltage_mean_mV"], "Voltage Distribution Across Neurons")
                        if raw_output.get("spike_trains_ms"):
                            with st.expander("⏱️ Interspike Intervals"):
                                plot_isi_histogram(raw_output["spike_trains_ms"], "ISI Distribution")
                    
                    plot_statistics_panel(metrics, model_select, raw_output)
                
                with tab6:
                    st.markdown("#### 🛡️ Epilepsy Prevention & Precaution Guide")
                    st.markdown("These recommendations combine general medical advice with insights from your simulation results.")
                    
                    seizure_detected = summary.get('seizure_detected', 0)
                    seizure_prob = summary.get('seizure_probability', 0.0)
                    seizure_severity = summary.get('seizure_severity', 'none')
                    
                    if seizure_detected or seizure_prob >= 0.3:
                        st.warning("⚠️ **Based on your simulation results**, taking preventive measures is strongly recommended.")
                    else:
                        st.info("💡 These are general preventive measures for brain health.")
                    
                    st.markdown("---")
                    
                    with st.expander("🍽️ **1. Lifestyle Modifications**"):
                        st.markdown("""
                        **Sleep**
                        - Maintain regular sleep schedule (7-8 hours nightly)
                        - Sleep deprivation is a common seizure trigger
                        - Consistent bedtime and wake time
                        
                        **Stress Management**
                        - Practice relaxation techniques (deep breathing, meditation)
                        - Regular exercise (moderate, not exhaustive)
                        - Avoid overloaded schedules
                        
                        **Alcohol**
                        - Limit or avoid alcohol consumption
                        - Never drink on empty stomach
                        - Wait 6+ hours after drinking to take seizure medication
                        """)
                    
                    with st.expander("🚫 **2. Trigger Avoidance**"):
                        st.markdown("""
                        **Common Triggers:**
                        - Flashing/flicker lights (games, certain displays)
                        - Sleep deprivation
                        - Missed medications
                        - Stress and anxiety
                        - Alcohol and recreational drugs
                        - Low blood sugar
                        
                        **Personal Triggers (from simulation):**
                        """)
                        if model_select == "Epileptic" or seizure_prob >= 0.3:
                            st.markdown("""
                            - **High-frequency oscillations** detected - may indicate hyperexcitable network
                            - **Elevated synchrony** - neurons firing together excessively
                            - Consider reducing stimulant intake and managing stress
                            """)
                        else:
                            st.markdown("""
                            - Your simulation shows normal activity patterns
                            - Continue maintaining healthy lifestyle
                            """)
                    
                    with st.expander("🥑 **3. Dietary Approaches**"):
                        st.markdown("""
                        **Ketogenic Diet (for some patients):**
                        - High fat, adequate protein, low carbohydrate
                        - May reduce seizure frequency
                        - Requires medical supervision
                        
                        **General Guidelines:**
                        - Don't skip meals
                        - Stay hydrated
                        - Balance blood sugar with regular meals
                        - Consider magnesium-rich foods
                        """)
                    
                    with st.expander("⚠️ **4. Warning Signs - When to Seek Emergency Care**"):
                        st.markdown("""
                        **Seek immediate help if:**
                        - First-time seizure
                        - Seizure lasting > 5 minutes
                        - Difficulty breathing during/after seizure
                        - Person doesn't wake up between seizures
                        - Person injured during seizure
                        - Seizure in water
                        - Second seizure shortly after first
                        
                        **Call emergency services if:**
                        - Generalized tonic-clonic seizure
                        - Person is pregnant (call OB immediately)
                        - Known heart condition
                        - No seizure history but symptoms occur
                        """)
                    
                    with st.expander("🔒 **5. Safety Precautions**"):
                        st.markdown("""
                        **Daily Activities:**
                        - Showers preferred over baths
                        - Use rubber mat in shower
                        - Don't lock bathroom door
                        - Wear medical alert bracelet
                        
                        **Driving:**
                        - Follow local laws (typically 6 months seizure-free)
                        - Don't drive if seizure aura occurs
                        
                        **Swimming & Water:**
                        - Only with direct supervision
                        - Wear life jacket
                        - Stay in shallow water
                        
                        **Heights:**
                        - Avoid working at heights during seizures
                        - Use safety harness
                        """)
                    
                    with st.expander("💊 **6. Medication & Follow-up**"):
                        st.markdown("""
                        **Anti-Seizure Medications (ASM):**
                        - Take exactly as prescribed
                        - Never stop abruptly
                        - Set reminders
                        - Refill before running out
                        
                        **Medical Follow-up:**
                        - Regular neurologist visits
                        - Keep seizure diary
                        - Report side effects immediately
                        - Annual EEG if recommended
                        """)
                    
                    st.markdown("---")
                    st.caption("💡 **Disclaimer:** This information is for educational purposes only. Always consult with a qualified healthcare provider for diagnosis and treatment of epilepsy or seizures.")
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
                    
                    # Explanation for Neuron Activity Chart
                    if mode_select == "Variant":
                        exp_title = "Worm Neuron Activity (Variant Mode)"
                        exp_what = """
This bar chart shows the **spike activity** of individual neurons in the C. elegans connectome.

C. elegans is the only organism with a fully mapped neural connectome (302 neurons).
This simulation uses the real synaptic connectivity data (herm_full_edgelist.csv).
Each bar represents one neuron's activity level.
                        """
                        exp_read = [
                            "Look at X-axis: Neuron names from real C. elegans brain",
                            "Look at Y-axis: Spike count - how active each neuron is",
                            "Red bars = Variant mode (altered excitability)",
                            "Higher bars = More active neurons",
                            "Look for neurons with unusually high activity"
                        ]
                        exp_values = {
                            "default": """
- Activity level: **Low to moderate** (0-20 spikes)
- Distribution: **Scattered** across neurons
- Pattern: **Independent firing** - neurons activate sporadically
- High-activity neurons: **Few** (>10 spikes)
- Network state: **Normal baseline activity**
                            """,
                            "variant": """
- Activity level: **Elevated** (10-50 spikes)
- Distribution: **Concentrated** in specific neurons
- Pattern: **Coordinated bursts** - groups activating together
- High-activity neurons: **Many** (>20 spikes)
- Network state: **Hyper-excitable** - easier to trigger
                            """
                        }
                        exp_clinical = """
**Why this matters for understanding neural networks:**

The C. elegans connectome is a model system for studying neural disorders:

1. **Network hyperexcitability:** In Variant mode, certain neurons fire excessively,
   which can cascade to trigger their connected neighbors
   
2. **Seizure propagation:** When highly active neurons connect to others,
   they can propagate abnormal activity across the network
   
3. **Therapeutic targets:** Identifying hyperactive neurons helps find
   potential intervention points (similar to seizure focus in epilepsy)

**Key differences from hippocampus:**
- Worm neurons are smaller and simpler
- Real connectome allows study of defined pathways
- Easier to trace signal propagation
                        """
                    else:
                        exp_title = "Worm Neuron Activity (Default Mode)"
                        exp_what = """
This bar chart shows the **spike activity** of individual neurons in the C. elegans connectome.

C. elegans is the only organism with a fully mapped neural connectome (302 neurons).
This simulation uses the real synaptic connectivity data (herm_full_edgelist.csv).
Each bar represents one neuron's activity level.
                        """
                        exp_read = [
                            "Look at X-axis: Neuron names from real C. elegans brain",
                            "Look at Y-axis: Spike count - how active each neuron is",
                            "Blue bars = Default mode (normal excitability)",
                            "Most neurons have low activity",
                            "Occasional spikes are normal"
                        ]
                        exp_values = {
                            "default": """
- Activity level: **Low to moderate** (0-20 spikes)
- Distribution: **Scattered** across neurons
- Pattern: **Independent firing** - neurons activate sporadically
- High-activity neurons: **Few** (>10 spikes)
- Network state: **Normal baseline activity**
                            """,
                            "variant": """
- Activity level: **Elevated** (10-50 spikes)
- Distribution: **Concentrated** in specific neurons
- Pattern: **Coordinated bursts** - groups activating together
- High-activity neurons: **Many** (>20 spikes)
- Network state: **Hyper-excitable** - easier to trigger
                            """
                        }
                        exp_clinical = """
**Why this matters for understanding neural networks:**

The C. elegans connectome is a model system for studying neural disorders:

1. **Network baseline activity:** Default mode shows normal neural activity
   patterns in a small, well-characterized nervous system
   
2. **Comparative analysis:** Comparing Default vs Variant helps identify
   which neurons and pathways are most affected by hyperexcitability
   
3. **Educational value:** Students can see how individual neurons contribute
   to overall network behavior in a biologically realistic system
                        """
                    
                    detailed_explanation(
                        graph_title=exp_title,
                        what_it_shows=exp_what,
                        how_to_read=exp_read,
                        expected_values=exp_values,
                        clinical_significance=exp_clinical
                    )
                    
                    # Show connection info
                    st.info(f"📊 Using 50 most connected neurons from herm_full_edgelist.csv (858 connections)")
                    
                    # Show activity heatmap
                    with st.expander("🌡️ View Activity Heatmap"):
                        plot_worm_heatmap(spike_data, mode_select)
                    
                    # Explanation for Activity Heatmap
                    detailed_explanation(
                        graph_title="Activity Heatmap",
                        what_it_shows="""
This heatmap provides a **visual overview** of neural activity levels across all neurons.

The activity is color-coded:
- **Green** = Low activity (0-49 spikes)
- **Orange** = Medium activity (50-89 spikes)  
- **Red** = High activity (90+ spikes)

This allows quick identification of which brain regions are most active.
                        """,
                        how_to_read=[
                            "Look at the color bar on the right",
                            "Green = Low activity (normal)",
                            "Orange = Elevated activity",
                            "Red = High activity (hyperactive)",
                            "Find the pattern: scattered vs concentrated red"
                        ],
                        expected_values={
                            "default": """
- Mostly **green** bars (low activity)
- Few **orange** bars (moderate)
- No **red** bars (no hyperactive neurons)
- Pattern: **Scattered** activity across brain
- Interpretation: **Normal functioning network**
                            """,
                            "variant": """
- More **orange** and **red** bars
- Concentrated **red** regions in specific areas
- Pattern: **Clustered** high activity
- Interpretation: **Hyperactive subnetworks**
                            """
                        },
                        clinical_significance="""
**Clinical interpretation:**

1. **Activity hotspots:** Red regions indicate neurons that are
   hyperactive and may be triggering nearby neurons
   
2. **Network propagation:** If hyperactive neurons are well-connected,
   they can spread abnormal activity throughout the network
   
3. **Intervention points:** Treating (inhibiting) red neurons may help
   reduce overall network excitability
   
4. **Biomarker localization:** Similar to EEG/fMRI in humans,
   heatmaps help localize problem areas
                        """
                    )
                    
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