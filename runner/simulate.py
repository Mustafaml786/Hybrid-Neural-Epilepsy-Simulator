# runner/simulate.py
import os
import json
import yaml
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from adapters.hippocampus_brian2 import HippocampusBrian2Adapter as HippocampusAdapter
from adapters.worm_c302_full import WormC302FullAdapter


# ======================= DEFAULT CONFIG ==========================
DEFAULT_CONFIG = {
    "simulation": {"duration": 1000, "dt": 0.1},
    "stimulus": {"file": "configs/stimuli/pulse.json"},
    "model_dir": os.path.join("third_party", "hippocampus"),
    "worm_model_dir": os.path.join("third_party", "c302"),
    "worm_csv_dir": os.path.join("third_party", "c302", "connectome_data")
}


# ======================= CONFIG & UTILS ==========================
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


# ======================= PLOTTING FUNCTIONS ==========================
def plot_hippocampus_voltage(raw_data, mode, outdir):
    """Plot hippocampus voltage traces for a given mode."""
    try:
        times = raw_data.get("times_ms", [])
        voltage_mean = raw_data.get("voltage_mean_mV", [])

        if not times or not voltage_mean:
            print(f"[Runner] No voltage data available for Hippocampus {mode}")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(times, voltage_mean, linewidth=2, color='steelblue')
        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Mean Membrane Potential (mV)', fontsize=12)
        plt.title(f'Hippocampus {mode} Mode - Voltage Trace', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = os.path.join(outdir, f"hippo_{mode.lower()}_voltage.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Runner] Saved graph: {filename}")
    except Exception as e:
        print(f"[Runner] Error plotting Hippocampus {mode}: {e}")


def plot_hippocampus_spikes(raw_data, mode, outdir):
    """Plot hippocampus spike raster for a given mode."""
    try:
        spike_trains = raw_data.get("spike_trains_ms", {})

        if not spike_trains:
            print(f"[Runner] No spike data available for Hippocampus {mode}")
            return

        plt.figure(figsize=(12, 6))
        for neuron_id, spikes in spike_trains.items():
            if spikes:
                plt.vlines(spikes, int(neuron_id), int(neuron_id) + 0.8, colors='red', linewidth=1)

        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Neuron ID', fontsize=12)
        plt.title(f'Hippocampus {mode} Mode - Spike Raster', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        filename = os.path.join(outdir, f"hippo_{mode.lower()}_spikes.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Runner] Saved graph: {filename}")
    except Exception as e:
        print(f"[Runner] Error plotting Hippocampus spikes {mode}: {e}")


def plot_worm_activity(raw_data, mode, outdir):
    """Plot C302 worm neural activity for a given mode."""
    try:
        spikes = raw_data.get("spikes", {})
        mean_activity = raw_data.get("mean_activity", 0)

        if not spikes:
            print(f"[Runner] No spike data available for Worm {mode}")
            return

        neurons = sorted(spikes.keys())
        neuron_indices = range(len(neurons))
        activities = [spikes[n][0] if spikes[n] else 0 for n in neurons]

        plt.figure(figsize=(12, 6))
        plt.bar(neuron_indices, activities, color='coral', alpha=0.7, edgecolor='darkred')
        plt.xlabel('Neuron ID', fontsize=12)
        plt.ylabel('Spike Activity', fontsize=12)
        plt.title(f'C302 Worm {mode} Mode - Neural Activity', fontsize=14, fontweight='bold')
        plt.axhline(y=mean_activity, color='green', linestyle='--', linewidth=2,
                    label=f'Mean Activity: {mean_activity:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        filename = os.path.join(outdir, f"worm_{mode.lower()}_activity.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Runner] Saved graph: {filename}")
    except Exception as e:
        print(f"[Runner] Error plotting Worm {mode}: {e}")


# ======================= CUSTOM RUNNERS ==========================
def run_hippocampus_modes_custom(config, stim, outdir, modes):
    print("\n" + "=" * 60)
    print("HIPPOCAMPUS SIMULATIONS (Custom Selection)")
    print("=" * 60)

    cfg_hippo = config.copy()
    cfg_hippo["model_dir"] = config.get("model_dir", DEFAULT_CONFIG["model_dir"])
    cfg_hippo["simulation"] = config.get("simulation", DEFAULT_CONFIG["simulation"])

    hippocampus_results = {}
    for mode in modes:
        try:
            print(f"\n[Runner] Initializing Hippocampus in {mode} mode...")
            hippocampus = HippocampusAdapter(cfg_hippo, mode=mode)
            hippocampus.initialize(stim_cfg=stim)
            hippocampus.apply_stimulus(stim)
            print(f"[Runner] Running Hippocampus {mode} simulation...")
            hippo_result = hippocampus.run(cfg_hippo["simulation"]["duration"])
            hippo_summary = hippocampus.get_output()

            mode_lower = mode.lower()
            save_run_json(outdir, f"hippo_{mode_lower}_raw.json", hippo_result or {})
            save_run_json(outdir, f"hippo_{mode_lower}_summary.json", hippo_summary)
            print(f"[Runner] Hippocampus {mode} complete. Summary:", hippo_summary)

            hippocampus_results[mode] = {"raw": hippo_result, "summary": hippo_summary}
            plot_hippocampus_voltage(hippo_result, mode, outdir)
            plot_hippocampus_spikes(hippo_result, mode, outdir)
        except Exception as e:
            print(f"[Runner] Error in Hippocampus {mode}: {e}")
            hippocampus_results[mode] = {"raw": None, "summary": {"error": str(e)}}
    return hippocampus_results


def run_worm_modes_custom(config, stim, outdir, modes):
    print("\n" + "=" * 60)
    print("C302 WORM SIMULATIONS (Custom Selection)")
    print("=" * 60)

    cfg_worm = config.copy()
    cfg_worm["model_dir"] = config.get("worm_model_dir", DEFAULT_CONFIG["worm_model_dir"])
    cfg_worm["csv_dir"] = config.get("worm_csv_dir", DEFAULT_CONFIG["worm_csv_dir"])
    cfg_worm["simulation"] = config.get("simulation", DEFAULT_CONFIG["simulation"])

    worm_results = {}
    for mode in modes:
        try:
            print(f"\n[Runner] Initializing C302 Worm in {mode} mode...")
            worm = WormC302FullAdapter(cfg_worm, mode=mode)
            worm.initialize()
            worm.apply_stimulus(stim)
            print(f"[Runner] Running C302 Worm {mode} simulation...")
            worm_result = worm.run(cfg_worm["simulation"]["duration"])
            worm_summary = worm.get_output()

            mode_lower = mode.lower()
            save_run_json(outdir, f"worm_{mode_lower}_raw.json", worm_result or {})
            save_run_json(outdir, f"worm_{mode_lower}_summary.json", worm_summary)
            print(f"[Runner] Worm {mode} complete. Summary:", worm_summary)

            worm_results[mode] = {"raw": worm_result, "summary": worm_summary}
            plot_worm_activity(worm_result, mode, outdir)
        except Exception as e:
            print(f"[Runner] Error in Worm {mode}: {e}")
            worm_results[mode] = {"raw": None, "summary": {"error": str(e)}}
    return worm_results


# ======================= MAIN SIMULATION RUNNER ==========================
def run_simulation():
    config = load_config()
    print("[Runner] Configuration loaded from YAML.")

    stim_path = config.get("stimulus", {}).get("file", DEFAULT_CONFIG["stimulus"]["file"])
    if not os.path.exists(stim_path):
        print(f"[Runner] Stimulus {stim_path} not found; using default pulse.")
        stim = {"type": "pulse", "amplitude": 0.5, "start": 100, "duration": 200}
    else:
        with open(stim_path, "r") as f:
            stim = json.load(f)
    print(f"[Runner] Stimulus parameters loaded: {stim}")

    # ---------------------- USER SELECTION SECTION ----------------------
    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    print("Choose model(s) to simulate:")
    print("1. Hippocampus only")
    print("2. C302 Worm only")
    print("3. Both")

    choice = input("Enter your choice (1/2/3): ").strip()
    simulate_hippo = choice in ["1", "3"]
    simulate_worm = choice in ["2", "3"]

    selected_modes = {"hippocampus": [], "worm": []}

    if simulate_hippo:
        print("\nSelect Hippocampus mode:")
        print("1. Healthy only")
        print("2. Epileptic only")
        print("3. Both")
        hippo_choice = input("Enter choice (1/2/3): ").strip()
        if hippo_choice == "1":
            selected_modes["hippocampus"] = ["Healthy"]
        elif hippo_choice == "2":
            selected_modes["hippocampus"] = ["Epileptic"]
        else:
            selected_modes["hippocampus"] = ["Healthy", "Epileptic"]

    if simulate_worm:
        print("\nSelect C302 Worm mode:")
        print("1. Default only")
        print("2. Variant only")
        print("3. Both")
        worm_choice = input("Enter choice (1/2/3): ").strip()
        if worm_choice == "1":
            selected_modes["worm"] = ["Default"]
        elif worm_choice == "2":
            selected_modes["worm"] = ["Variant"]
        else:
            selected_modes["worm"] = ["Default", "Variant"]
    # -------------------------------------------------------------------

    stamp = int(time.time())
    outdir = os.path.join("runs", f"run_{stamp}")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    hippocampus_results, worm_results = {}, {}

    if simulate_hippo:
        hippocampus_results = run_hippocampus_modes_custom(config, stim, outdir, selected_modes["hippocampus"])
    if simulate_worm:
        worm_results = run_worm_modes_custom(config, stim, outdir, selected_modes["worm"])

    # Save combined summaries
    dual_model_comparison = {
        "hippocampus": hippocampus_results,
        "worm_c302": worm_results,
        "stimulus_config": stim,
        "simulation_config": config.get("simulation", DEFAULT_CONFIG["simulation"]),
        "timestamp": stamp
    }
    save_run_json(outdir, "dual_model_comparison.json", dual_model_comparison)
    print("[Runner] Dual model comparison file saved.")
    print("\nSimulation complete. Results stored in:", outdir)
    return {"hippocampus": hippocampus_results, "worm": worm_results}


# ======================= ENTRY POINT ==========================
if __name__ == "__main__":
    run_simulation()
