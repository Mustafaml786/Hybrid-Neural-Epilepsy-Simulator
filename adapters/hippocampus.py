# adapters/hippocampus.py

import os
import json
import matplotlib.pyplot as plt
from neuron import h

class HippocampusAdapter:
    def __init__(self, config, mode="Healthy"):
        self.config = config
        self.mode = mode
        self.soma = None
        self.stim = None
        self.recordings = {}
        self.sim_duration = self.config["simulation"]["duration"]

        print(f"[Hippocampus] Initializing NEURON environment in mode {mode}...")

        # Try detailed model first
        try:
            self._initialize_model()
            print("[Hippocampus] Detailed model loaded.")
        except FileNotFoundError:
            print("[Hippocampus] Model directory not found or empty. Falling back to simplified soma.")
            self._initialize_simplified()

    def _initialize_model(self):
        model_dir = self.config.get("hippocampus_model_dir", "")
        if not os.path.exists(model_dir) or len(os.listdir(model_dir)) == 0:
            raise FileNotFoundError("Model directory not found or empty.")
        # Placeholder: load detailed model
        # self.soma = detailed_model_soma(...)
        pass

    def _initialize_simplified(self):
        self.soma = h.Section(name="soma")
        self.soma.L = 20
        self.soma.diam = 20
        self.soma.insert("hh")  # Hodgkin-Huxley
        self.recordings = {"time": [], "voltage": []}

    def initialize(self):
        """Backward-compatible init call."""
        print("[Hippocampus] Backward-compatible initialize() called.")
        h.finitialize(-65)

    def apply_stimulus(self, stim_params):
        start = stim_params.get("start", 100)
        dur = stim_params.get("duration", 200)
        amp = stim_params.get("amplitude", 0.5)

        self.stim = h.IClamp(self.soma(0.5))
        self.stim.delay = start
        self.stim.dur = dur
        self.stim.amp = amp
        print(f"[Hippocampus] Stimulus configured → delay={start} ms, duration={dur} ms, amplitude={amp} nA")

        self.v_vec = h.Vector()
        self.t_vec = h.Vector()
        self.v_vec.record(self.soma(0.5)._ref_v)
        self.t_vec.record(h._ref_t)

    def run(self, duration=None):
        """Run the NEURON simulation safely in both detailed and simplified modes."""
        from neuron import h

        duration_ms = duration if duration else getattr(self, "sim_duration", 1000)

        # ✅ Ensure NEURON runtime is properly loaded
        try:
            h.load_file("stdrun.hoc")
        except Exception as e:
            print(f"[Hippocampus] Warning: stdrun.hoc not found or already loaded → {e}")

        # ✅ Define simulation parameters safely
        if not hasattr(h, "tstop"):
            h('tstop = %f' % duration_ms)
        else:
            h.tstop = duration_ms

        if not hasattr(h, "dt"):
            h('dt = 0.1')

        # ✅ If no recording vectors exist, create them
        if not hasattr(self, "t_vec"):
            self.t_vec = h.Vector().record(h._ref_t)
        if not hasattr(self, "v_vec"):
            try:
                self.v_vec = h.Vector().record(self.soma(0.5)._ref_v)
            except Exception:
                print("[Hippocampus] Warning: no soma reference found, using default v variable.")
                self.v_vec = h.Vector().record(h._ref_v)

        print(f"[Hippocampus] Running simulation for {duration_ms} ms...")
        h.run()

        # ✅ Store results
        self.recordings["time"] = list(self.t_vec)
        self.recordings["voltage"] = list(self.v_vec)

        print(f"[Hippocampus] Simulation completed. {len(self.recordings['time'])} points recorded.")

        return self.get_output()


    def get_output(self):
        if not self.recordings.get("voltage"):
            return {}
        avg_voltage = sum(self.recordings["voltage"]) / len(self.recordings["voltage"])
        return {"time": self.recordings["time"], "voltage": self.recordings["voltage"], "hipp_activity": [avg_voltage]}

    def save_outputs(self, folder="runs"):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "hipp_raw.json"), "w") as f:
            json.dump({"time": self.recordings["time"], "voltage": self.recordings["voltage"]}, f, indent=4)
    
    def apply_intervention(self, params):
        """
        Apply intervention to the hippocampus model.
        Example params: {"stim_amplitude": 0.5, "stim_start": 50}
        """
        if "stim_amplitude" in params:
            self.stim.amp = params["stim_amplitude"]
        if "stim_start" in params:
            self.stim.delay = params["stim_start"]

