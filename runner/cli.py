# runner/cli.py

import json
import yaml
import os
from adapters.hippocampus_brian2 import HippocampusBrian2Adapter as HippocampusAdapter
from adapters.worm_c302_full import WormC302FullAdapter as CElegansAdapter
from analysis.viz import plot_activity
from analysis.features import compute_summary_metrics

def load_config(path="configs/base.yaml"):
    """Load simulation configuration from YAML."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print("[Runner] Configuration loaded from YAML.")
    return config

def load_stimulus(stim_path):
    """Load stimulus parameters from JSON."""
    with open(stim_path, "r") as f:
        stim = json.load(f)
    print(f"[Runner] Stimulus parameters loaded: {stim}")
    return stim

def select_model():
    """CLI menu for model selection and mode."""
    print("\nSelect Model:")
    print("1️⃣ Hippocampus")
    print("2️⃣ Worm (C. elegans)")
    choice = input("Enter choice (1/2): ").strip()
    if choice == "1":
        mode = input("Select Hippocampus mode (Healthy/Epileptic): ").strip()
        return "hippocampus", mode
    elif choice == "2":
        mode = input("Select Worm mode (Default/Variant): ").strip()
        return "worm", mode
    else:
        print("Invalid choice. Defaulting to Hippocampus Healthy.")
        return "hippocampus", "Healthy"

def apply_intervention(model_name, output_metrics, stim_params):
    """Example closed-loop intervention."""
    print(f"\n[Intervention] Evaluating {model_name} output for intervention...")
    # Simple rule: if avg activity exceeds threshold, reduce amplitude by 20%
    threshold = -60 if model_name == "hippocampus" else 0.5
    avg_metric = output_metrics.get("hipp_activity", output_metrics.get("spikes", {}))
    
    if model_name == "hippocampus" and avg_metric[0] > threshold:
        stim_params["amplitude"] *= 0.8
        print(f"[Intervention] Hippocampus overactive → reducing amplitude to {stim_params['amplitude']}")
        return True
    elif model_name == "worm":
        # Example: spike count intervention (if needed)
        return False
    return False

def run_simulation():
    # 1️⃣ Load configuration and stimulus
    config = load_config()
    stim = load_stimulus(config["stimulus"]["file"])
    
    # 2️⃣ Model selection
    model_name, mode = select_model()
    
    # 3️⃣ Initialize appropriate model
    if model_name == "hippocampus":
        model = HippocampusAdapter(config, mode=mode)
    else:
        model = CElegansAdapter(config, mode=mode)

    model.initialize()

    # 4️⃣ Apply stimulus and run
    model.apply_stimulus(stim)
    model.run(config["simulation"]["duration"])
    
    # 5️⃣ Fetch outputs - run() returns raw data, get_output() returns compact metrics
    output = model.get_output()
    metrics = compute_summary_metrics(output)
    
    print("\n=== Simulation Complete ===")
    print(f"{model_name.capitalize()} Output Summary:", metrics)

    # 6️⃣ Save outputs
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/{model_name}_raw.json", "w") as f:
        json.dump(output, f, indent=4)
    with open(f"runs/{model_name}_summary.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # 7️⃣ Plot voltage trace if available
    if "time" in output and "voltage" in output:
        plot_activity({"voltage_mV": output["voltage"], "time_ms": output["time"]})

    # 8️⃣ Check pathology & optional intervention
    pathology_detected = apply_intervention(model_name, metrics, stim)
    while pathology_detected:
        print("[Runner] Re-running simulation with intervention...")
        model.apply_stimulus(stim)
        model.run(config["simulation"]["duration"])
        output = model.get_output()
        metrics = compute_summary_metrics(output)
        pathology_detected = apply_intervention(model_name, metrics, stim)

    print(f"[Runner] Final {model_name} output saved in runs/ folder.")
    return output, metrics

if __name__ == "__main__":
    run_simulation()
