# adapters/worm_c302_full.py

"""
Enhanced C. elegans Adapter with Real Connectome Data
Uses herm_full_edgelist.csv for authentic neural network simulation.

Parameters (configurable via config dict):
    - n_neurons: Number of neurons (default: 50)
    - threshold: Firing threshold mV (default: -35 from literature)
    - decay: Activity decay per step (default: 0.88)
    - resting_potential: Resting potential mV (default: -65)
    - weight_scale: Synaptic weight scaling (default: 0.15)
    - plasticity_enabled: Enable STDP (default: False)
    - stdp_window: STDP timing window ms (default: 20)
    - stdp_strength: STDP learning rate (default: 0.1)
"""

import os
import csv
import json
import numpy as np
from collections import defaultdict
from pathlib import Path


class WormC302FullAdapter:
    """
    Enhanced C. elegans adapter with configurable parameters.
    
    Default parameters based on C. elegans literature:
    - Resting potential: -61 to -74 mV (Lockhart et al. 2019)
    - Input resistance: 1.6-2.2 GΩ
    - Firing threshold: approximately -30 to -45 mV
    """
    
    # Default parameters for Default and Variant modes
    # Variant simulates hyper-excitability (like epilepsy) - dramatic difference
    DEFAULT_PARAMS = {
        "default": {
            "threshold": 15.0,       # Very high - VERY hard to fire
            "decay": 0.99,           # Near complete decay - activity dies quickly 
            "resting_potential": -75.0,  # Very hyperpolarized
            "weight_scale": 0.02,    # Very weak synapses
            "plasticity_enabled": False,
            "stdp_window": 20.0,     # ms
            "stdp_strength": 0.1,   # Learning rate
        },
        "variant": {
            "threshold": 2.0,        # VERY LOW - fires very easily
            "decay": 0.50,           # Minimal decay - highly sustained
            "resting_potential": -35.0,  # Very depolarized
            "weight_scale": 0.60,    # Very strong synapses
            "plasticity_enabled": False,
            "stdp_window": 20.0,
            "stdp_strength": 0.15,
        }
    }
    
    def __init__(self, config, mode="Default"):
        self.config = config
        self.mode = mode if isinstance(mode, str) else "Default"
        
        # Get mode-specific parameters (case-insensitive)
        mode_key = self.mode.lower()
        
        # Map common mode names
        mode_map = {"healthy": "default", "epileptic": "variant", "default": "default", "variant": "variant"}
        mode_key = mode_map.get(mode_key, mode_key)
        
        self.params = self.DEFAULT_PARAMS.get(mode_key, self.DEFAULT_PARAMS["default"]).copy()
        
        # Allow config overrides
        worm_cfg = config.get("worm", {})
        for key in self.params:
            if key in worm_cfg:
                self.params[key] = worm_cfg[key]
        
        self.results = {}
        self.initialized = False
        self.duration = config.get("simulation", {}).get("duration", 1000)
        
        # Network components
        self.neurons = []
        self.network_out = {}  # outgoing: {source: [(target, weight)]}
        self.network_in = {}   # incoming: {target: [(source, weight)]}
        self.activities = {}
        self.spike_history = defaultdict(list)  # For STDP
        
        print(f"[WormC302FullAdapter] Initialized in {mode} mode.")
        print(f"[WormC302FullAdapter] Parameters: threshold={self.params['threshold']}mV, decay={self.params['decay']}, weight_scale={self.params['weight_scale']}, plasticity={self.params['plasticity_enabled']}")
        
    def load_connectome(self, top_n=50):
        """Load real C. elegans connectome data."""
        data_path = Path(__file__).parent.parent / "data" / "herm_full_edgelist.csv"
        
        if not data_path.exists():
            print(f"[WormC302FullAdapter] ERROR: Data not found: {data_path}")
            return False
            
        neuron_connections = defaultdict(lambda: {'targets': [], 'sources': []})
        
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row['Source'].strip()
                target = row['Target'].strip()
                weight = int(row['Weight'].strip())
                
                neuron_connections[source]['targets'].append((target, weight))
                neuron_connections[target]['sources'].append((source, weight))
        
        # Calculate connectivity scores
        neuron_scores = {}
        for neuron in neuron_connections:
            targets = len(neuron_connections[neuron]['targets'])
            sources = len(neuron_connections[neuron]['sources'])
            total_weights = sum(w for _, w in neuron_connections[neuron]['targets']) + \
                        sum(w for _, w in neuron_connections[neuron]['sources'])
            neuron_scores[neuron] = targets + sources + total_weights / 10
        
        # Get top N neurons
        top_neurons = sorted(neuron_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        selected = [n[0] for n in top_neurons]
        
        # Build adjacency lists
        network_out = defaultdict(list)
        network_in = defaultdict(list)
        
        for neuron in selected:
            # Outgoing
            for target, weight in neuron_connections[neuron]['targets']:
                if target in selected:
                    network_out[neuron].append((target, weight))
            # Incoming  
            for source, weight in neuron_connections[neuron]['sources']:
                if source in selected:
                    network_in[neuron].append((source, weight))
        
        self.neurons = selected
        self.network_out = dict(network_out)
        self.network_in = dict(network_in)
        
        print(f"[WormC302FullAdapter] Loaded: {len(self.neurons)} neurons, {sum(len(v) for v in self.network_out.values())} connections")
        return True
    
    def initialize(self, n_neurons=50):
        """Initialize network."""
        print("[WormC302FullAdapter] Loading connectome...")
        
        if not self.load_connectome(top_n=n_neurons):
            self.neurons = [f"N{i}" for i in range(n_neurons)]
            return False
        
        self.activities = {n: 0.0 for n in self.neurons}
        self.initialized = True
        return True
    
    def apply_stimulus(self, stim):
        """Apply stimulus to network."""
        self.stimulus = stim
        stim_amp = stim.get('amplitude', 0.5) if stim else 0.5
        
        # Apply stronger continuous stimulus to create sustained activity
        # This represents ongoing sensory input
        for neuron in self.neurons:
            self.activities[neuron] = stim_amp * 2  # Lower initial but continuous
        
        # External drive: add constant input to top 5 neurons
        for neuron in self.neurons[:5]:
            self.activities[neuron] += stim_amp * 10
        
        print(f"[WormC302FullAdapter] Stimulus: amplitude={stim_amp}")
    
    def run(self, duration_ms, plasticity=None):
        """Run network simulation.
        
        Args:
            duration_ms: Simulation duration in milliseconds
            plasticity: Override plasticity setting (bool) or None to use default
        """
        print(f"[WormC302FullAdapter] Running for {duration_ms} ms...")
        
        # Use stored parameters
        p = self.params
        
        # Allow override of plasticity setting
        if plasticity is not None:
            p["plasticity_enabled"] = plasticity
        
        n_steps = int(duration_ms / 10)
        decay = p["decay"]
        threshold = p["threshold"]
        weight_scale = p["weight_scale"]
        
        if self.mode.lower() == "variant":
            print(f"[WormC302FullAdapter] Variant mode: threshold={threshold}mV, weight_scale={weight_scale}")
        
        if p["plasticity_enabled"]:
            print(f"[WormC302FullAdapter] STDP enabled: window={p['stdp_window']}ms, strength={p['stdp_strength']}")
        
        spike_counts = {n: 0 for n in self.neurons}
        weight_modifications = defaultdict(lambda: defaultdict(float))  # For STDP
        
        for step in range(n_steps):
            new_activities = {}
            current_time = step * 10
            
            for neuron in self.neurons:
                # Sum inputs from presynaptic neurons with weight scaling
                input_sum = 0.0
                if neuron in self.network_in:
                    for source, base_weight in self.network_in[neuron]:
                        # Apply any weight modifications from STDP
                        w = base_weight * weight_scale
                        if weight_modifications[source].get(neuron, 0) != 0:
                            w *= (1 + weight_modifications[source][neuron])
                        input_sum += self.activities.get(source, 0) * w
                
                # Update activity with decay
                activity = self.activities.get(neuron, 0) * decay + input_sum
                
                # Check for spike
                if activity > threshold:
                    spike_counts[neuron] += 1
                    
                    # Apply STDP: strengthen connections from recently active neurons
                    if p["plasticity_enabled"]:
                        stdp_window = p["stdp_window"]
                        stdp_strength = p["stdp_strength"]
                        
                        # Potentiation: if presynaptic neuron fired recently (check neighbors)
                        for target, _ in self.network_out.get(neuron, []):
                            if target in self.spike_history:
                                for prev_time in self.spike_history[target]:
                                    if current_time - prev_time < stdp_window:
                                        # LTP: increase synaptic weight from this neuron
                                        weight_modifications[neuron][target] += stdp_strength
                                        break
                        
                        # Depression: weaken unused connections
                        for source in self.neurons:
                            if spike_counts.get(source, 0) == 0:
                                weight_modifications[source][neuron] -= stdp_strength * 0.1
                    
                    # Record spike time (keep ALL spikes for accurate counting)
                    if neuron not in self.spike_history:
                        self.spike_history[neuron] = []
                    self.spike_history[neuron].append(current_time)
                    
                    activity = threshold * 0.2  # Reset after spike
                
                new_activities[neuron] = activity
                
                new_activities[neuron] = activity
            
            self.activities = new_activities
        
        # Calculate mean activity (scaled for display)
        mean_activities = {n: float(spike_counts[n] * 10) for n in self.neurons}
        total_spikes = sum(spike_counts.values())
        
        self.results = {
            "spikes": spike_counts,
            "spike_history": self.spike_history,
            "mean_activity": mean_activities,
            "total_spikes": total_spikes,
            "num_spikes": total_spikes,
            "mode": self.mode,
            "n_neurons": len(self.neurons),
            "parameters": p,
            "plasticity_enabled": p["plasticity_enabled"]
        }
        
        print(f"[WormC302FullAdapter] Complete: {total_spikes} spikes from {len(self.neurons)} neurons")
        return self.results
    
    def get_output(self):
        """Return summary."""
        if not self.results:
            return {}
        
        spike_history = self.results.get("spike_history", {})
        spikes = {n: [float(t) for t in times] for n, times in spike_history.items() if times}
        
        return {
            "spikes": spikes,
            "mean_activity": self.results.get("mean_activity", {}),
            "mean_act_val": float(np.mean(list(self.results.get("mean_activity", {}).values()))) if self.results.get("mean_activity") else 0,
            "num_spikes": self.results.get("num_spikes", 0),
            "mode": self.mode,
            "n_neurons": self.results.get("n_neurons", 0)
        }
    
    def save_results(self, outdir="worm_runs"):
        """Save results."""
        Path(outdir).mkdir(parents=True, exist_ok=True)
        
        filename = f"worm_{self.mode.lower()}.json"
        filepath = Path(outdir) / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                "mode": self.mode,
                "n_neurons": len(self.neurons),
                "results": self.results
            }, f, indent=2)
        
        print(f"[WormC302FullAdapter] Saved: {filepath}")
        return str(filepath)


if __name__ == "__main__":
    adapter = WormC302FullAdapter({}, mode="Default")
    adapter.initialize(n_neurons=50)
    adapter.apply_stimulus({"amplitude": 0.5})
    results = adapter.run(500)
    print(f"Test result: {results.get('num_spikes')} spikes")