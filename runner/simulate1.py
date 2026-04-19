# runner/simulate.py
import os
import json
import yaml
import time
from pathlib import Path
# from adapters.hippocampus import HippocampusAdapter
from adapters.hippocampus_brian2 import HippocampusBrian2Adapter as HippocampusAdapter
from adapters.hippocampus_brian2 import HippocampusBrian2Adapter
from adapters.worm_c302_full import WormC302FullAdapter

# Optional worm adapter import
try:
    from adapters.worm_c302_full import WormC302FullAdapter as WormC302Adapter
    worm_available = True
except Exception:
    worm_available = False

import matplotlib.pyplot as plt
import csv

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

def save_trace_csv(outdir, t, v):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(outdir, "hipp_trace.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["time_ms", "voltage_mV"])
        for ti, vi in zip(t, v):
            writer.writerow([ti, vi])
    return csv_path

def plot_trace(outdir, t, v):
    plt.figure()
    plt.plot(t, v)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("Hippocampus membrane potential")
    plt.tight_layout()
    png = os.path.join(outdir, "hipp_trace.png")
    plt.savefig(png)
    plt.close()
    return png

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

    # init hippocampus
    # merge config values
    cfg = config.copy()
    cfg["model_dir"] = config.get("model_dir", DEFAULT_CONFIG["model_dir"])
    cfg["simulation"] = config.get("simulation", DEFAULT_CONFIG["simulation"])

    hippocampus = HippocampusAdapter(cfg)
    hippocampus.initialize()

    hippocampus.apply_stimulus(stim)
    hippo_result = hippocampus.run(cfg["simulation"]["duration"])
    hippo_summary = hippocampus.get_output()

    # prepare run directory
    stamp = int(time.time())
    outdir = os.path.join("runs", f"run_{stamp}")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # save raw results, summary
    save_run_json(outdir, "hipp_raw.json", hippo_result if hippo_result else {})
    save_run_json(outdir, "hipp_summary.json", hippo_summary)

    # save csv + png if traces available
    if hippo_result and hippo_result.get("time") and hippo_result.get("voltage"):
        csv_path = save_trace_csv(outdir, hippo_result["time"], hippo_result["voltage"])
        png = plot_trace(outdir, hippo_result["time"], hippo_result["voltage"])
        print(f"[Runner] Saved hippocampus trace to {csv_path} and plot to {png}")

    # worm
    if worm_available:
        worm = WormC302Adapter(cfg)
        worm.initialize()
        worm.apply_stimulus(hippo_summary)  # pass compact summary for now
        worm_result = worm.run(cfg["simulation"]["duration"])
        save_run_json(outdir, "worm_raw.json", worm_result if worm_result else {})
        print("[Runner] Worm simulation complete.")
    else:
        worm_result = {"status": "worm adapter not available"}

    print("\n=== Simulation Complete ===")
    print("Hippocampus summary:", hippo_summary)
    print("Worm result:", ("available" if worm_available else "not available"))

    return hippo_result, worm_result

if __name__ == "__main__":
    run_simulation()
