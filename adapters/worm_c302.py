# adapters/worm_c302.py

import os
import json
import random  # placeholder for actual simulation

class WormC302Adapter:
    def __init__(self, config, mode="Default"):
        self.config = config
        self.mode = mode
        self.spikes = {}
        self.voltages = {}

    def initialize(self):
        print(f"[Worm] Initializing C. elegans model in {self.mode} mode...")

    def apply_stimulus(self, stim_input):
        """Input from hippocampus output"""
        # Placeholder: distribute hippocampal activity to worm neurons
        self.spikes = {"AVAL": [s * 0.01 for s in stim_input.get("hipp_activity", [0])]}
        self.voltages = {"AVAL": [s * 10 for s in stim_input.get("hipp_activity", [0])]}
        print("[Worm] Stimulus applied based on hippocampus output.")

    def run(self, duration):
        # Simple placeholder for spiking dynamics
        for neuron in self.spikes:
            self.spikes[neuron] = [v + random.uniform(-0.1, 0.1) for v in self.spikes[neuron]]
            self.voltages[neuron] = [v + random.uniform(-0.5, 0.5) for v in self.voltages[neuron]]

    def get_output(self):
        return {"spikes": self.spikes, "voltages": self.voltages}

    def save_outputs(self, folder="runs"):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "worm_raw.json"), "w") as f:
            json.dump({"spikes": self.spikes, "voltages": self.voltages}, f, indent=4)
    
    def apply_intervention(self, params):
        """
        Apply intervention to worm model.
        Example params: {"input_scale": 1.0}
        """
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)

