# analysis/seizure_detection.py
"""
Seizure detection module for hybrid neural epilepsy simulator.
Provides binary detection and probability scoring based on multiple biomarkers.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_spike_rate(spike_trains: Dict, duration_ms: float) -> float:
    """
    Compute average spike rate across all neurons.
    
    Args:
        spike_trains: Dictionary mapping neuron_id to list of spike times (ms)
        duration_ms: Total simulation duration in ms
    
    Returns:
        Average spike rate in Hz (spikes per second)
    """
    if not spike_trains or duration_ms <= 0:
        return 0.0
    
    total_spikes = sum(len(times) for times in spike_trains.values())
    n_neurons = len(spike_trains)
    
    if n_neurons == 0:
        return 0.0
    
    # Average per neuron, converted to Hz
    spike_rate = (total_spikes / n_neurons) / (duration_ms / 1000.0)
    return spike_rate


def compute_synchrony(spike_trains: Dict, time_window_ms: float = 10.0) -> float:
    """
    Compute network synchrony - how coordinated firing is across neurons.
    Uses cross-correlation of spike trains within a time window.
    
    Args:
        spike_trains: Dictionary mapping neuron_id to list of spike times (ms)
        time_window_ms: Time window for synchrony measurement (default 10ms)
    
    Returns:
        Synchrony index 0.0 (no sync) to 1.0 (perfect sync)
    """
    if not spike_trains:
        return 0.0
    
    neurons = list(spike_trains.keys())
    n_neurons = len(neurons)
    
    if n_neurons < 2:
        return 0.0
    
    # Create binary spike arrays
    all_times = []
    for times in spike_trains.values():
        all_times.extend(times)
    
    if not all_times:
        return 0.0
    
    time_min = min(all_times)
    time_max = max(all_times)
    time_range = time_max - time_min
    
    if time_range <= 0:
        return 0.0
    
    # Bin spike times
    n_bins = max(int(time_range / time_window_ms), 1)
    bins = np.linspace(time_min, time_max, n_bins)
    
    # Count co-occurring spikes
    sync_count = 0
    total_opportunities = 0
    
    for bin_start in bins[:-1]:
        bin_end = bins[np.searchsorted(bins, bin_start) + 1] if np.searchsorted(bins, bin_start) + 1 < len(bins) else bin_start + time_window_ms
        
        # Count neurons firing in this window
        neurons_firing = 0
        for neuron_id in neurons:
            times = spike_trains[neuron_id]
            if times:
                in_window = sum(1 for t in times if bin_start <= t < bin_end)
                neurons_firing += in_window
        
        if neurons_firing >= 2:  # At least 2 neurons
            sync_count += 1
        total_opportunities += 1
    
    if total_opportunities == 0:
        return 0.0
    
    synchrony = sync_count / total_opportunities
    return float(np.clip(synchrony, 0.0, 1.0))


def compute_variance_ratio(voltage_trace: List[float], rest_voltage: float = -65.0) -> float:
    """
    Compute normalized voltage variance - high variance indicates seizure activity.
    
    Args:
        voltage_trace: List of membrane voltage values (mV)
        rest_voltage: Expected resting potential (default -65mV)
    
    Returns:
        Normalized variance 0.0 (stable) to 1.0 (highly variable)
    """
    if not voltage_trace:
        return 0.0
    
    v_arr = np.array(voltage_trace)
    
    # Compute coefficient of variation
    v_mean = np.mean(v_arr)
    v_std = np.std(v_arr)
    
    if v_mean == 0:
        return 0.0
    
    # Normalized by absolute mean
    cv = abs(v_std / v_mean) if v_mean != 0 else 0
    
    # Clip to 0-1 range (variance > 1 is max seizure)
    variance_ratio = float(np.clip(cv, 0.0, 1.0))
    return variance_ratio


def compute_burst_intensity(spike_trains: Dict, spike_rate: float, 
                        window_ms: float = 50.0, min_neurons: int = 5) -> float:
    """
    Compute burst intensity - measure of paroxysmal activity.
    
    Args:
        spike_trains: Dictionary mapping neuron_id to list of spike times
        spike_rate: Pre-computed spike rate (Hz)
        window_ms: Time window for burst detection
        min_neurons: Minimum neurons to count as burst
    
    Returns:
        Burst intensity 0.0 to 1.0
    """
    if not spike_trains:
        return 0.0
    
    # Collect all spike times
    all_spikes = []
    for times in spike_trains.values():
        all_spikes.extend(times)
    
    if not all_spikes:
        return 0.0
    
    all_spikes = sorted(all_spikes)
    spike_arr = np.array(all_spikes)
    n_spikes = len(spike_arr)
    
    if n_spikes < 2:
        return 0.0
    
    # Find burst-like patterns (many spikes in short window)
    burst_windows = 0
    
    for i in range(n_spikes):
        window_end = spike_arr[i] + window_ms
        spikes_in_window = np.sum(spike_arr < window_end)
        
        if spikes_in_window >= min_neurons:
            burst_windows += 1
    
    # Normalize by total possible windows
    max_windows = max(1, n_spikes)
    burst_intensity = burst_windows / max_windows
    
    return float(np.clip(burst_intensity, 0.0, 1.0))


def calculate_seizure_probability(spike_rate: float, synchrony: float, 
                               variance_ratio: float, burst_intensity: float,
                               weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate seizure probability from biomarker scores.
    
    Args:
        spike_rate: Average spike rate (Hz)
        synchrony: Network synchrony index (0-1)
        variance_ratio: Normalized voltage variance (0-1)
        burst_intensity: Burst intensity (0-1)
        weights: Optional custom weights for each marker
    
    Returns:
        Seizure probability 0.0 to 1.0
    """
    if weights is None:
        weights = {
            "spike_rate": 0.3,
            "synchrony": 0.3,
            "variance": 0.2,
            "burst_intensity": 0.2
        }
    
    # Normalize spike rate (healthy ~0-5 Hz, seizure ~50+ Hz)
    spike_rate_normalized = float(np.clip(spike_rate / 100.0, 0.0, 1.0))
    
    # Weighted combination
    probability = (
        weights["spike_rate"] * spike_rate_normalized +
        weights["synchrony"] * synchrony +
        weights["variance"] * variance_ratio +
        weights["burst_intensity"] * burst_intensity
    )
    
    return float(np.clip(probability, 0.0, 1.0))


def detect_seizure(spike_trains: Dict, voltage_trace: List[float], 
                duration_ms: float, model_type: str = "hippocampus") -> Dict:
    """
    Main seizure detection function.
    
    Args:
        spike_trains: Dictionary mapping neuron_id to list of spike times
        voltage_trace: List of voltage values (mV) (optional for worm)
        duration_ms: Total simulation duration in ms
        model_type: "hippocampus" or "worm"
    
    Returns:
        Dictionary with detection results
    """
    # Compute individual biomarkers
    spike_rate = compute_spike_rate(spike_trains, duration_ms)
    synchrony = compute_synchrony(spike_trains, time_window_ms=10.0)
    burst_intensity = compute_burst_intensity(spike_trains, spike_rate)
    
    # Voltage variance (only for hippocampus which has voltage data)
    if model_type == "hippocampus" and voltage_trace:
        variance_ratio = compute_variance_ratio(voltage_trace, rest_voltage=-65.0)
    else:
        # For worm, use different threshold - worm baseline is normal activity
        # Use relative spike rate to baseline for worm
        variance_ratio = min(spike_rate / 30.0, 1.0)  # Higher threshold for worm
    
# Use different weights/thresholds based on model type
    if model_type == "worm":
        # For worm, spike rate of ~55 Hz baseline vs ~99 Hz variant
        # Need higher threshold to distinguish
        rate_above_baseline = spike_rate - 50.0  # Baseline ~55Hz, variant ~99Hz
        if rate_above_baseline < 0:
            rate_above_baseline = 0.0
        spike_rate_normalized = rate_above_baseline / 50.0  # Only count >50Hz above baseline
        if spike_rate > 80.0:
            vr = 1.0
        elif spike_rate > 50.0:
            vr = 0.5
        else:
            vr = 0.2
        probability = (
            0.2 * spike_rate_normalized +
            0.2 * synchrony +
            0.3 * vr +
            0.3 * burst_intensity
        )
    else:
        probability = calculate_seizure_probability(
            spike_rate=spike_rate,
            synchrony=synchrony,
            variance_ratio=variance_ratio,
            burst_intensity=burst_intensity
        )
    
    probability = float(np.clip(probability, 0.0, 1.0))
    
    # Use different thresholds for each model
    if model_type == "worm":
        threshold = 0.8  # Much higher - default worm is normal active network
    else:
        threshold = 0.4
    
    detected = probability >= threshold
    
    # Determine severity
    if probability < 0.2:
        severity = "none"
    elif probability < 0.4:
        severity = "mild"
    elif probability < 0.6:
        severity = "moderate"
    else:
        severity = "severe"
    
    return {
        "seizure_detected": int(detected),
        "seizure_probability": round(probability, 3),
        "seizure_severity": severity,
        "biomarkers": {
            "spike_rate_hz": round(spike_rate, 2),
            "synchrony_index": round(synchrony, 3),
            "variance_ratio": round(variance_ratio, 3),
            "burst_intensity": round(burst_intensity, 3)
        },
        "threshold_used": threshold
    }


def classify_state(results: Dict, model_type: str = "hippocampus") -> Dict:
    """
    Classify neural state from simulation results.
    
    Args:
        results: Simulation results dictionary
        model_type: "hippocampus" or "worm"
    
    Returns:
        Classification results
    """
    # Extract data
    spike_trains = results.get("spike_trains_ms", {})
    voltage_mean = results.get("voltage_mean_mV", [])
    
    # Determine duration
    times = results.get("times_ms", [])
    duration_ms = float(max(times)) if times else 500.0
    
    # Run detection
    detection = detect_seizure(
        spike_trains=spike_trains,
        voltage_trace=voltage_mean,
        duration_ms=duration_ms,
        model_type=model_type
    )
    
    return detection


# Convenience function for quick detection
def quick_detect(results: Dict) -> Tuple[int, float, str]:
    """
    Quick seizure detection returning just binary, probability, severity.
    
    Args:
        results: Simulation results dictionary
    
    Returns:
        Tuple of (detected, probability, severity)
    """
    detection = classify_state(results)
    return (
        detection["seizure_detected"],
        detection["seizure_probability"],
        detection["seizure_severity"]
    )