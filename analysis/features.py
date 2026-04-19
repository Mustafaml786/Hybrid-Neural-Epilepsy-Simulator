# analysis/features.py

import numpy as np
from copy import deepcopy

def compute_summary_metrics(output):
    """
    Compute metrics from hippocampus or worm simulation output.
    Returns dictionary of metrics.
    """
    metrics = {}
    
    # Handle HippocampusBrian2Adapter format: {"hipp_activity_mean_mV": float, "num_spikes": int}
    if "hipp_activity_mean_mV" in output:
        avg = output.get("hipp_activity_mean_mV")
        metrics["hipp_activity_avg"] = avg
        metrics["hipp_burst_count"] = detect_bursts_from_avg(avg) if avg else 0
        metrics["num_spikes"] = output.get("num_spikes", 0)
        metrics["n_neurons"] = output.get("n_neurons", 0)
        metrics["active_neurons"] = output.get("active_neurons", 0)
        metrics["mean_activity"] = output.get("mean_activity", 0)
    # Handle legacy format: {"hipp_activity": [...]} 
    elif "hipp_activity" in output:
        voltages = output.get("hipp_activity", [])
        metrics["hipp_activity_avg"] = np.mean(voltages) if voltages else None
        metrics["hipp_burst_count"] = detect_bursts(voltages)
        metrics["hipp_spike_frequency"] = compute_spike_frequency(voltages)
    # Handle worm format: {"spikes": {...}}
    elif "spikes" in output:
        spikes = output["spikes"]
        for neuron, vals in spikes.items():
            metrics[f"{neuron}_avg_spike"] = np.mean(vals) if vals else None
            metrics[f"{neuron}_spike_count"] = len([v for v in vals if v > 0])
        metrics["mean_activity"] = output.get("mean_activity", 0)
        metrics["mean_act_val"] = output.get("mean_act_val", 0)
        metrics["num_spikes"] = output.get("num_spikes", 0)
        metrics["n_neurons"] = output.get("n_neurons", len(spikes))
    return metrics


def detect_bursts_from_avg(avg_voltage):
    """
    Detect bursts from average voltage - if avg is above -50mV, likely bursting.
    """
    if avg_voltage and avg_voltage > -50:
        return 1
    return 0

def detect_bursts(voltage_trace, threshold=-50, min_duration=3):
    """
    Simple burst detection based on threshold crossing.
    Counts consecutive points above threshold with minimum duration.
    """
    bursts = 0
    above_thr = [v > threshold for v in voltage_trace]
    count = 0
    for v in above_thr:
        if v:
            count += 1
        elif count >= min_duration:
            bursts += 1
            count = 0
        else:
            count = 0
    if count >= min_duration:
        bursts += 1
    return bursts

def compute_spike_frequency(voltage_trace, dt=0.1):
    """
    Compute number of spikes per millisecond.
    Assumes spikes are positive voltage crossings.
    """
    spikes = [v for v in voltage_trace if v > 0]
    return len(spikes) / (len(voltage_trace) * dt) if voltage_trace else 0

def apply_intervention_grid(model, param_grid):
    """
    Apply a grid search intervention on the model.

    Args:
        model: HippocampusAdapter or WormC302Adapter instance
        param_grid: list of dictionaries with intervention parameters

    Returns:
        List of dictionaries with params and resulting metrics
    """
    results = []

    for param_set in param_grid:
        # Deepcopy to avoid modifying the original model
        temp_model = deepcopy(model)
        if hasattr(temp_model, "apply_intervention"):
            temp_model.apply_intervention(param_set)
        else:
            raise AttributeError(f"{type(temp_model).__name__} has no method 'apply_intervention'")

        # Run simulation
        temp_model.run(temp_model.config["simulation"]["duration"])

        # Get output and compute metrics
        output = temp_model.get_output()
        metrics = compute_summary_metrics(output)

        # Store results
        results.append({"params": param_set, "metrics": metrics})

    return results
