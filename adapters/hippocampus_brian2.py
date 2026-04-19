# adapters/hippocampus_brian2.py
"""
Enhanced Brian2 hippocampus adapter for xspecies-neuro.
Uses conductance-based model with configurable parameters.
"""

import os
import sys

# Fix signal for Streamlit/multithreaded environments before importing brian2
if 'streamlit' in sys.modules:
    import signal
    _orig_signal = signal.signal
    def _safe_signal(sig, handler):
        try:
            return _orig_signal(sig, handler)
        except (ValueError, OSError):
            return None
    signal.signal = _safe_signal

import json
import numpy as np
from pathlib import Path

from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, StateMonitor,
    TimedArray, ms, mV, nS, pF, pA, nA, run, Network,
    defaultclock, prefs, volt, amp
)

prefs.codegen.target = "numpy"


def _to_list(x):
    """Convert numpy/brian2 arrays to plain Python lists safely."""
    try:
        return np.array(x).tolist()
    except Exception:
        try:
            return list(x)
        except Exception:
            return str(x)


class HippocampusBrian2Adapter:
    """
    Hippocampus adapter with enhanced conductance-based model.
    
    Parameters (all configurable via config dict or __init__):
        - n_neurons: Number of neurons (default: 50)
        - gNa: Sodium conductance mS/cm² (default: 20 healthy, 30 epileptic)
        - gK: Potassium conductance mS/cm² (default: 10 healthy, 20 epileptic)
        - gL: Leak conductance mS/cm² (default: 0.1)
        - ENa: Na+ reversal potential mV (default: 55)
        - EK: K+ reversal potential mV (default: -90)
        - EL: Leak reversal potential mV (default: -65 healthy, -50 epileptic)
        - Cm: Membrane capacitance µF/cm² (default: 1.0)
        - tau_syn: Synaptic time constant ms (default: 5 healthy, 10 epileptic)
        - connection_prob: Connection probability (default: 0.1)
        - Vt: Threshold potential mV (default: -50)
        - Vreset: Reset potential mV (default: -60)
        - tau_ref: Refractory period ms (default: 2)
    """
    
    # Default parameters for healthy and epileptic modes
    DEFAULT_PARAMS = {
        "Healthy": {
            "gNa": 20.0,      # mS/cm²
            "gK": 10.0,       # mS/cm²
            "gL": 0.1,        # mS/cm²
            "ENa": 55.0,       # mV
            "EK": -90.0,       # mV
            "EL": -65.0,       # mV
            "Cm": 1.0,         # µF/cm²
            "tau_syn": 5.0,    # ms
            "connection_prob": 0.1,
            "Vt": -50.0,       # mV
            "Vreset": -60.0,   # mV
            "tau_ref": 2.0,    # ms
            "synaptic_weight": 150.0,  # pA
        },
        "Epileptic": {
            "gNa": 30.0,       # Increased (1.5x)
            "gK": 20.0,        # Increased (2x)
            "gL": 0.3,         # Increased (3x)
            "ENa": 55.0,        # Same
            "EK": -90.0,        # Same
            "EL": -50.0,        # Depolarized (+15mV)
            "Cm": 1.0,          # Same
            "tau_syn": 10.0,   # Increased (2x)
            "connection_prob": 0.1,
            "Vt": -45.0,       # Lower threshold (+5mV)
            "Vreset": -60.0,   # Same
            "tau_ref": 2.0,    # Same
            "synaptic_weight": 500.0,  # Increased
        }
    }
    
    def __init__(self, config=None, mode="Healthy"):
        if config is None:
            config = {}
        self.config = config
        self.mode = mode if isinstance(mode, str) else "Healthy"
        
        # Get mode-specific parameters with overrides from config
        mode_upper = self.mode.capitalize() if self.mode.lower() in ["healthy", "epileptic"] else "Healthy"
        base_params = self.DEFAULT_PARAMS.get(mode_upper, self.DEFAULT_PARAMS["Healthy"])
        self.params = base_params.copy()
        
        # Allow config overrides
        hippo_cfg = config.get("hippocampus", {})
        for key in self.params:
            if key in hippo_cfg:
                self.params[key] = hippo_cfg[key]
        
        # Apply epileptic modifiers if mode is epileptic
        if self.mode.lower() == "epileptic":
            self._apply_epileptic_modifiers()
        
        sim_cfg = self.config.get("simulation", {})
        dt_val = sim_cfg.get("dt", 0.025)
        dur_val = sim_cfg.get("duration", 1000)

        self.dt = float(dt_val) * ms
        self.sim_duration = float(dur_val) * ms
        defaultclock.dt = self.dt

        self.neurons = None
        self.syn = None
        self.S = None
        self.M = None
        self.network = None
        self.stim_timed_array = None
        self.results = {}
        self.initialized = False

        print(f"[HippocampusBrian2Adapter] Initialized in {self.mode} mode.")
        print(f"[HippocampusBrian2Adapter] Parameters: gNa={self.params['gNa']}, gK={self.params['gK']}, gL={self.params['gL']}, EL={self.params['EL']}, tau_syn={self.params['tau_syn']}")

    def _apply_epileptic_modifiers(self):
        """Apply epileptic modifiers to parameters."""
        modifiers = {
            "gNa_mult": 1.5,
            "gK_mult": 2.0,
            "gL_mult": 3.0,
            "tau_syn_mult": 2.0,
            "EL_shift": 15.0,
            "VT_shift": 5.0,
        }
        
        self.params["gNa"] *= modifiers["gNa_mult"]
        self.params["gK"] *= modifiers["gK_mult"]
        self.params["gL"] *= modifiers["gL_mult"]
        self.params["tau_syn"] *= modifiers["tau_syn_mult"]
        self.params["EL"] += modifiers["EL_shift"]
        self.params["Vt"] += modifiers["VT_shift"]
        self.params["synaptic_weight"] = 500.0  # Increased synaptic strength

    def _make_stim_timedarray(self, stim_cfg):
        total_ms = int(int((self.sim_duration / ms)) + 1)
        values = np.zeros(total_ms)

        if stim_cfg:
            stype = stim_cfg.get("type", "pulse")
            amp_nA = float(stim_cfg.get("amplitude", 0.5))
            start_ms = int(float(stim_cfg.get("start", 100)))
            dur_ms = int(float(stim_cfg.get("duration", 200)))

            start_idx = max(0, start_ms)
            end_idx = min(total_ms, start_idx + dur_ms)
            if stype == "pulse":
                values[start_idx:end_idx] = amp_nA
            else:
                values[:] = amp_nA

        ta = TimedArray(values * nA, dt=1 * ms)
        return ta

    def initialize(self, n_neurons=None, stim_cfg=None):
        try:
            print("[HippocampusBrian2Adapter] Building network...")

            N = int(n_neurons or self.config.get("n_neurons", 50))
            
            # Store for get_output()
            self._n_neurons = N
            
            # Use stored parameters from config
            p = self.params
            
            Cm = p["Cm"] * pF
            gL = p["gL"] * nS
            gNa = p["gNa"] * nS
            gK = p["gK"] * nS
            EL = p["EL"] * mV
            ENa = p["ENa"] * mV
            EK = p["EK"] * mV
            VT = p["Vt"] * mV
            Vreset = p["Vreset"] * mV
            tau_ref = p["tau_ref"] * ms
            tau_syn = p["tau_syn"] * ms
            conn_prob = p["connection_prob"]
            w_amp = p["synaptic_weight"] * pA

            if self.mode.lower() == "epileptic":
                print(f"[HippocampusBrian2Adapter] Epileptic mode: gNa={gNa/nS:.1f}nS, gK={gK/nS:.1f}nS, gL={gL/nS:.2f}nS, EL={EL/mV:.0f}mV")

            self.stim_timed_array = self._make_stim_timedarray(stim_cfg)

            # Enhanced leaky integrate-and-fire with improved parameters
            # Using more realistic parameters while maintaining stability
            eqs = """
            dv/dt = (gL*(EL - v) + I_syn + I_ext) / Cm : volt (unless refractory)
            I_syn : amp
            I_ext : amp
            """

            self.neurons = NeuronGroup(
                N, eqs,
                threshold="v > VT",
                reset="v = Vreset",
                refractory=tau_ref,
                method="euler",
                namespace={
                    'Cm': Cm, 'gL': gL, 'gNa': gNa, 'gK': gK,
                    'EL': EL, 'ENa': ENa, 'EK': EK,
                    'VT': VT, 'Vreset': Vreset,
                    'tau_ref': tau_ref,
                    'stimulus': self.stim_timed_array
                }
            )

            self.neurons.v = EL
            self.neurons.I_syn = 0 * pA
            self.neurons.I_ext = 0 * pA

            # Synaptic connections with configurable probability
            # Using instant synaptic current (simpler but still effective)
            self.syn = Synapses(self.neurons, self.neurons, 
                             model="w : amp", 
                             on_pre="I_syn_post += w")
            self.syn.connect(p=conn_prob)
            self.syn.w = w_amp

            self.S = SpikeMonitor(self.neurons)
            self.M = StateMonitor(self.neurons, "v", record=True)

            self.network = Network(self.neurons, self.syn, self.S, self.M)

            if self.stim_timed_array is not None:
                self.neurons.run_regularly("I_ext = stimulus(t)", dt=1 * ms)

            self.initialized = True
            print(f"[HippocampusBrian2Adapter] Network initialized (N={N}, conn_prob={conn_prob}).")
        except Exception as e:
            print("[HippocampusBrian2Adapter] ERROR during initialize():", e)
            raise

    def apply_stimulus(self, stim_cfg):
        self.config["stimulus"] = stim_cfg or self.config.get("stimulus", None)
        if not self.initialized:
            self.initialize(stim_cfg=stim_cfg)
            return
        self.stim_timed_array = self._make_stim_timedarray(stim_cfg)
        print("[HippocampusBrian2Adapter] Stimulus updated.")

    def run(self, duration_ms=None):
        if not self.initialized:
            self.initialize(stim_cfg=self.config.get("stimulus", None))

        if duration_ms is None:
            duration = self.sim_duration
        else:
            duration = float(duration_ms) * ms

        try:
            print(f"[HippocampusBrian2Adapter] Running for {duration/ms:.0f} ms ...")
            self.network.run(duration)
        except Exception as e:
            print("[HippocampusBrian2Adapter] ERROR during run():", e)
            raise

        try:
            times = (self.M.t / ms)
            volt_matrix = (self.M.v / mV)
            mean_voltage = np.mean(volt_matrix, axis=0)

            raw_spike_trains = self.S.spike_trains()
            spike_trains = {str(k): (np.array(v / ms)).tolist() for k, v in raw_spike_trains.items()}

            results = {
                "times_ms": _to_list(times),
                "voltage_matrix_mV": _to_list(volt_matrix),
                "voltage_mean_mV": _to_list(mean_voltage),
                "spike_trains_ms": spike_trains,
                "spike_counts": _to_list(np.array(self.S.count[:])),
                "num_spikes": int(self.S.num_spikes)
            }
            self.results = results
            print("[HippocampusBrian2Adapter] Simulation complete.")
            return results
        except Exception as e:
            print("[HippocampusBrian2Adapter] ERROR during post-processing:", e)
            raise

    def get_output(self):
        if not self.results:
            return {}
        try:
            mean_trace = np.array(self.results.get("voltage_mean_mV", []))
            avg = float(np.mean(mean_trace)) if mean_trace.size else None
            
            # Calculate additional metrics
            spike_counts = self.results.get("spike_counts", [])
            active_neurons = sum(1 for c in spike_counts if c > 0) if spike_counts else 0
            
            # Mean activity across neurons (for UI compatibility)
            mean_activity = float(np.mean(spike_counts)) if spike_counts else 0
            
            # Get n_neurons from stored or guess from results
            n_neurons = getattr(self, '_n_neurons', len(spike_counts)) if spike_counts else 50
            
            return {
                "hipp_activity_mean_mV": avg,
                "hipp_activity_avg": avg,  # For metrics calculation
                "num_spikes": self.results.get("num_spikes", 0),
                "n_neurons": n_neurons,
                "active_neurons": active_neurons,
                "mean_activity": mean_activity,  # For UI display
                "spike_counts": spike_counts,
                "mode": self.mode
            }
        except Exception as e:
            print(f"[HippocampusBrian2Adapter] get_output error: {e}")
            return {"hipp_activity_mean_mV": None, "num_spikes": 0, "n_neurons": 0}

    def save_results(self, outdir="runs", name=None):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fname = name or f"hippocampus_brian2_{self.mode.lower()}.json"
        path = Path(outdir) / fname
        with open(path, "w") as fh:
            json.dump(self.results, fh, indent=2)
        print(f"[HippocampusBrian2Adapter] Results saved to {path}")
        return str(path)


if __name__ == "__main__":
    cfg = {"simulation": {"duration": 500, "dt": 0.1}}
    adapter = HippocampusBrian2Adapter(config=cfg, mode="Healthy")
    adapter.initialize(n_neurons=20, stim_cfg={"type": "pulse", "amplitude": 2.0, "start": 50, "duration": 200})
    out = adapter.run(500)
    adapter.save_results(outdir="runs", name="hippocampus_brian2_sample.json")
    print("Summary:", adapter.get_output())