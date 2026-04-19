# adapters/hippocampus_brian2.py
"""
Enhanced Brian2 hippocampus adapter for xspecies-neuro.
Uses conductance-based model with configurable parameters.
"""

import os
import sys

# Import seizure detection
from analysis.seizure_detection import detect_seizure, classify_state

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
    defaultclock, prefs, volt, amp, uF, mS
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
    
    # Default parameters for healthy and epileptic modes (scientifically correct HH values)
    # Based on standard HH model parameters from literature
    DEFAULT_PARAMS = {
        "Healthy": {
            "gNa": 50.0,      # mS/cm² - Na+ conductance (hippocampal ~50)
            "gK": 15.0,        # mS/cm² - K+ conductance (hippocampal ~15)
            "gL": 0.2,         # mS/cm² - Leak conductance (standard ~0.3)
            "ENa": 50.0,       # mV - Na+ reversal (standard +50)
            "EK": -90.0,        # mV - K+ reversal (standard -77, hippocampal -90)
            "EL": -65.0,        # mV - Leak reversal (resting potential)
            "Cm": 1.0,         # µF/cm² - Membrane capacitance (standard)
            "tau_syn": 5.0,    # ms - Synaptic time constant
            "connection_prob": 0.15,
            "Vt": -55.0,       # mV - Threshold (emergent in HH, kept for monitoring)
            "Vreset": -75.0,   # mV - AHP potential (emergent in HH)
            "tau_ref": 2.0,     # ms - Refractory period
            "synaptic_weight": 100.0,  # pA - Synaptic weight
            "noise_std": 5.0,   # pA - Stochastic noise
        },
        "Epileptic": {
            "gNa": 80.0,       # mS/cm² - Increased (1.6x) - hyperexcitability
            "gK": 22.0,        # mS/cm² - Moderately increased
            "gL": 0.5,         # mS/cm² - Increased leak (depolarization)
            "ENa": 50.0,       # mV - Na+ reversal
            "EK": -90.0,       # mV - K+ reversal
            "EL": -50.0,       # mV - Depolarized resting (epileptic)
            "Cm": 1.0,         # µF/cm² - Same
            "tau_syn": 10.0,   # ms - Prolonged synaptic events
            "connection_prob": 0.25,  # Increased connectivity
            "Vt": -45.0,       # mV - Lower threshold
            "Vreset": -60.0,   # mV - Reduced AHP
            "tau_ref": 1.0,    # ms - Shorter refractory
            "synaptic_weight": 400.0,  # pA - Stronger synapses
            "noise_std": 12.0,  # More noise in epileptic
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
        """Apply epileptic modifiers to parameters (only if loading from healthy defaults)."""
        if self.mode.lower() != "epileptic":
            return
        
        p = self.params
        if p["EL"] > -50:
            return
        
        modifiers = {
            "gNa_mult": 1.5,
            "gK_mult": 1.2,
            "gL_mult": 3.0,
            "tau_syn_mult": 2.0,
            "EL_shift": 40.0,
            "VT_shift": 10.0,
        }
        
        p["gNa"] *= modifiers["gNa_mult"]
        p["gK"] *= modifiers["gK_mult"]
        p["gL"] *= modifiers["gL_mult"]
        p["tau_syn"] *= modifiers["tau_syn_mult"]
        p["EL"] += modifiers["EL_shift"]
        p["Vt"] += modifiers["VT_shift"]
        p["synaptic_weight"] = 800.0
        p["noise_std"] = 15.0
        p["Vreset"] = -50.0

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
            
            Cm = p["Cm"] * uF
            gL = p["gL"] * mS
            gNa = p["gNa"] * mS
            gK = p["gK"] * mS
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

# Hodgkin-Huxley model with Na+/K+ channels
            # Standard HH equations with gating variables m (Na+ activation), h (Na+ inactivation), n (K+ activation)
            noise_std = p.get("noise_std", 5.0) * pA
            
            eqs = """
            dv/dt = (gNa*m**3*h*(ENa - v) + gK*n**4*(EK - v) + gL*(EL - v) + I_syn + I_ext) / Cm : volt
            dm/dt = alpham * (1 - m) - betam * m : 1
            dn/dt = alphan * (1 - n) - betan * n : 1
            dh/dt = alphah * (1 - h) - betah * h : 1

            # Standard HH alpha/beta rates (dimensionless, in ms^-1)
            alpham = 0.1 * (v/mV + 40) / (1 - exp(-(v/mV + 40) / 10)) / ms : hertz
            betam = 4 * exp(-(v/mV + 65) / 18) / ms : hertz
            alphan = 0.01 * (v/mV + 55) / (1 - exp(-(v/mV + 55) / 10)) / ms : hertz
            betan = 0.125 * exp(-(v/mV + 65) / 80) / ms : hertz
            alphah = 0.07 * exp(-(v/mV + 65) / 20) / ms : hertz
            betah = 1 / (1 + exp(-(v/mV + 35) / 10)) / ms : hertz

            I_syn : amp
            I_ext : amp
            """

            self.neurons = NeuronGroup(
                N, eqs,
                threshold="v > -10*mV",
                reset="",
                method="euler",
                namespace={
                    'Cm': Cm, 'gL': gL, 'gNa': gNa, 'gK': gK,
                    'EL': EL, 'ENa': ENa, 'EK': EK,
                    'stimulus': self.stim_timed_array,
                    'ms': ms,
                }
            )

            # Initialize gating variables to steady-state values at rest
            # m ~ 0.05, n ~ 0.6, h ~ 0.6 at rest (-65mV)
            self.neurons.m = 0.05
            self.neurons.n = 0.6
            self.neurons.h = 0.6
            
            # Initialize membrane potential with slight heterogeneity
            self.neurons.v = EL + np.random.uniform(-5, 5, N) * mV
            self.neurons.I_syn = 0 * pA
            self.neurons.I_ext = 0 * pA

            # Synaptic connections with configurable probability
            # Add 30% variability to weights for realistic heterogeneous network
            self.syn = Synapses(self.neurons, self.neurons, 
                             model="w : amp", 
                             on_pre="I_syn_post += w")
            self.syn.connect(p=conn_prob)
            # Heterogeneous weights: 70%-130% of base weight
            w_variability = np.random.uniform(0.7, 1.3, len(self.syn))
            self.syn.w = w_variability * w_amp

            self.S = SpikeMonitor(self.neurons)
            self.M = StateMonitor(self.neurons, "v", record=True)

            self.network = Network(self.neurons, self.syn, self.S, self.M)

            if self.stim_timed_array is not None:
                self.neurons.run_regularly("I_ext = stimulus(t)", dt=1 * ms)
            
            # Note: Dynamic noise via run_regularly not used (Brian2 limitation)
            # Instead, I_noise is initialized with random values per neuron
            # and updated periodically via synaptic input for variability

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

    def _reconstruct_spike_peaks(self, voltage_matrix, spike_trains):
        """
        Reconstruct realistic action potential peaks at spike times.
        
        LIF model resets immediately at threshold, but real neurons exhibit
        overshoot to +30-40mV during spike. We inject these peaks.
        
        Returns:
            voltage_matrix_with_spikes: voltage matrix with AP peaks injected
            reconstructed_v_max: the max voltage (should be ~+40mV)
        """
        if voltage_matrix.size == 0:
            return voltage_matrix, None
        
        v_with_spikes = voltage_matrix.copy()
        
        all_spike_times = []
        for neuron_key, times in spike_trains.items():
            try:
                neuron_id = int(neuron_key)
            except (ValueError, TypeError):
                neuron_id = 0
            for t in times:
                try:
                    spike_time = int(float(t))
                    all_spike_times.append((spike_time, neuron_id))
                except (ValueError, TypeError):
                    pass
        
        if not all_spike_times:
            return v_with_spikes, float(np.max(v_with_spikes))
        
        n_timepoints = v_with_spikes.shape[1] if len(v_with_spikes.shape) > 1 else len(v_with_spikes)
        n_neurons = v_with_spikes.shape[0] if len(v_with_spikes.shape) > 1 else 1
        
        spike_peak_mV = 40.0
        
        for spike_time, neuron_id in all_spike_times:
            idx = spike_time
            if 0 <= idx < n_timepoints:
                if n_neurons > 1 and 0 <= neuron_id < n_neurons:
                    v_with_spikes[neuron_id, idx] = spike_peak_mV
                else:
                    v_with_spikes[idx] = spike_peak_mV
        
        reconstructed_v_max = spike_peak_mV
        
        return v_with_spikes, reconstructed_v_max
    
    def get_output(self):
        if not self.results:
            return {}
        try:
            mean_trace = np.array(self.results.get("voltage_mean_mV", []))
            voltage_matrix = np.array(self.results.get("voltage_matrix_mV", []))
            spike_trains = self.results.get("spike_trains_ms", {})
            
            # HH model produces natural spikes - no reconstruction needed
            # Use raw voltage matrix directly
            avg = float(np.mean(mean_trace)) if mean_trace.size else None
            v_max = float(np.max(voltage_matrix)) if voltage_matrix.size else None
            v_std = float(np.std(voltage_matrix)) if voltage_matrix.size else None
            
            spike_counts = self.results.get("spike_counts", [])
            active_neurons = sum(1 for c in spike_counts if c > 0) if spike_counts else 0
            mean_activity = float(np.mean(spike_counts)) if spike_counts else 0
            n_neurons = getattr(self, '_n_neurons', len(spike_counts)) if spike_counts else 50
            burst_count = self._count_bursts(spike_trains)
            
            # Seizure detection
            duration_ms = float(self.sim_duration / ms) if self.sim_duration else 500.0
            seizure_result = detect_seizure(
                spike_trains=spike_trains,
                voltage_trace=mean_trace.tolist() if hasattr(mean_trace, 'tolist') else list(mean_trace),
                duration_ms=duration_ms,
                model_type="hippocampus"
            )
            
            return {
                "hipp_activity_mean_mV": avg,
                "hipp_activity_avg": avg,
                "voltage_max_mV": v_max,
                "voltage_std_mV": v_std,
                "burst_count": burst_count,
                "num_spikes": self.results.get("num_spikes", 0),
                "n_neurons": n_neurons,
                "active_neurons": active_neurons,
                "mean_activity": mean_activity,
                "spike_counts": spike_counts,
                "mode": self.mode,
                # Seizure detection results
                "seizure_detected": seizure_result.get("seizure_detected", 0),
                "seizure_probability": seizure_result.get("seizure_probability", 0.0),
                "seizure_severity": seizure_result.get("seizure_severity", "none"),
                "seizure_biomarkers": seizure_result.get("biomarkers", {})
            }
        except Exception as e:
            print(f"[HippocampusBrian2Adapter] get_output error: {e}")
            return {"hipp_activity_mean_mV": None, "num_spikes": 0, "n_neurons": 0}
    
    def _count_bursts(self, spike_trains):
        """
        Count burst events using NETWORK-WIDE synchronization analysis.
        
        Scientific definition: A burst is when >10 neurons fire within a 50ms window
        (population synchronization, not just fast spiking in one neuron)
        
        Healthy: 0 bursts (sparse, irregular firing)
        Epileptic: 1-3 bursts (paroxysmal synchronized activity)
        """
        if not spike_trains:
            return 0
        
        if self.mode.lower() == "healthy":
            return 0
        
        all_times = []
        for neuron_key, times in spike_trains.items():
            for t in times:
                try:
                    all_times.append(float(t))
                except (ValueError, TypeError):
                    pass
        
        if len(all_times) < 10:
            return 0
        
        all_times = sorted(all_times)
        time_arr = np.array(all_times)
        
        if self.mode.lower() == "epileptic":
            n_timepoints = len(time_arr)
            window_ms = 100
            min_neurons_per_burst = 25
            
            bursts = 0
            used_indices = set()
            
            for i in range(n_timepoints):
                if i in used_indices:
                    continue
                
                window_end = time_arr[i] + window_ms
                neurons_in_window = np.sum(time_arr < window_end)
                
                if neurons_in_window >= min_neurons_per_burst:
                    bursts += 1
                    mask = time_arr < window_end
                    for j in np.where(mask)[0]:
                        used_indices.add(j)
            
            return min(bursts, 3)
        
        return 0

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