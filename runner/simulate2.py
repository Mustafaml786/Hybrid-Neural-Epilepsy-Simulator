# runner/simulate.py
import os
import json
import yaml
import time
from pathlib import Path

from adapters.hippocampus_brian2 import HippocampusBrian2Adapter as HippocampusAdapter
from adapters.worm_c302_full import WormC302FullAdapter


DEFAULT_CONFIG = {
    "simulation": {"duration": 1000, "dt": 0.1},
    "stimulus": {"file": "configs/stimuli/pulse.json"},
    "model_dir": os.path.join("third_party", "hippocampus"),
    "worm_model_dir": os.path.join("third_party", "c302")
}


def load_config(path="configs/base.yaml"):
    if not os.path.exists(path):
        print(f"[Runner] Config {path} not found; using defaults.")
        return DEFAULT_CONFIG
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_run_json(outdir, filename, obj):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(outdir, filename), "w") as f:
        json.dump(obj, f, indent=2)


def run_simulation():
    config = load_config()
    print("[Runner] Configuration loaded from YAML.")

    # load stimulus
    stim_path = config.get("stimulus", {}).get("file", DEFAULT_CONFIG["stimulus"]["file"])
    if not os.path.exists(stim_path):
        print(f"[Runner] Stimulus {stim_path} not found; using default pulse.")
        stim = {"type": "pulse", "amplitude": 0.5, "start": 100, "duration": 200}
    else:
        with open(stim_path, "r") as f:
            stim = json.load(f)
    print(f"[Runner] Stimulus parameters loaded: {stim}")

    # Initialize hippocampus
    cfg = config.copy()
    cfg["model_dir"] = config.get("model_dir", DEFAULT_CONFIG["model_dir"])
    cfg["simulation"] = config.get("simulation", DEFAULT_CONFIG["simulation"])

    # Use 'Healthy' mode by default for demo
    hippocampus = HippocampusAdapter(cfg, mode="Healthy")
    hippocampus.initialize()

    hippocampus.apply_stimulus(stim)
    hippo_result = hippocampus.run(cfg["simulation"]["duration"])
    hippo_summary = hippocampus.get_output()

    # Prepare run directory
    stamp = int(time.time())
    outdir = os.path.join("runs", f"run_{stamp}")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Save raw results, summary
    save_run_json(outdir, "hipp_raw.json", hippo_result if hippo_result else {})
    save_run_json(outdir, "hipp_summary.json", hippo_summary)

    print("\n=== Simulation Complete ===")
    print("Hippocampus summary:", hippo_summary)
    print(f"Results saved in {outdir}")

    return hippo_result, hippo_summary


if __name__ == "__main__":
    run_simulation()
