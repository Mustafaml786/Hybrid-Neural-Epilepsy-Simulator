# A Computational Approach to Modeling Epileptic Brain Behavior Using Hybrid Neural Simulation

**Short Name:** Hybrid Neural Epilepsy Simulator

A cross-species neural simulation platform for modeling and comparing hippocampal and C. elegans neural activity, with emphasis on studying epileptic brain behavior through computational neuroscience.

---

## 📖 Overview

This project simulates neural networks from two different biological systems:

1. **Hippocampus** (Mammalian brain) - Using Brian2 neural simulator
2. **C. elegans** (Nematode worm) - Using real connectome data

The platform enables researchers to compare healthy vs pathological (epileptic) brain states through computational modeling and visualization.

---

## 🧠 Key Features

### Models
- **Hippocampus**: 50-neuron network using Brian2 with Leaky Integrate-and-Fire dynamics
  - Healthy Mode: Normal parameters
  - Epileptic Mode: Lower threshold, stronger synapses, depolarized resting potential

- **C. elegans**: 50-neuron network using real connectome data
  - Default Mode: Normal connectivity weights
  - Variant Mode: Increased synaptic gain (50% more excitable)

### Visualizations
- Voltage trace over time
- Spike distribution per neuron
- Voltage distribution histogram
- Activity heatmap (for worm)
- Statistics panel (mean, std, min, max, firing rate)
- Comparison mode (side-by-side + table)

### Data Sources
- Input: `configs/stimuli/pulse.json` (stimulus parameters)
- Input: `data/herm_full_edgelist.csv` (C. elegans connectome - 7,378 connections)

---

## 📁 Project Structure

```
xspecies-neuro/
├── adapters/              # Neural simulation adapters
│   ├── hippocampus_brian2.py
│   └── worm_c302_full.py
├── analysis/             # Data analysis
│   └── features.py
├── configs/               # Configuration files
│   └── stimuli/pulse.json
├── data/                 # Data files
│   └── herm_full_edgelist.csv
├── runs/                 # Simulation outputs (Hippocampus)
├── worm_runs/            # Simulation outputs (Worm)
├── ui/                   # Streamlit web interface
│   └── app.py
├── runner/               # CLI runner
│   └── cli.py
└── README.md            # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.12+
- Virtual environment (recommended)

### Setup

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
See `requirements.txt` for full list:
- brian2 (neural simulator)
- numpy, matplotlib, pandas (data processing)
- pyyaml (configuration)
- streamlit (web UI)

---

## 🎯 Quick Start

### Option 1: Web UI (Recommended)

```bash
# Set PYTHONPATH and run Streamlit
PYTHONPATH=. streamlit run ui/app.py
```

Then open your browser to **http://localhost:8501**

### Option 2: Command Line Interface

```bash
PYTHONPATH=. python runner/cli.py
```

### Option 3: Direct Python

```bash
PYTHONPATH=. python -c "
from adapters.hippocampus_brian2 import HippocampusBrian2Adapter

model = HippocampusBrian2Adapter({}, 'Healthy')
model.initialize(n_neurons=50, stim_cfg={'type':'pulse', 'amplitude': 0.5, 'start': 100, 'duration': 200})
model.run(1000)
print(model.get_output())
"
```

---

## 📊 Using the Web UI

### Main Controls
1. **Model Selection**: Choose Hippocampus or Worm
2. **Mode Selection**: 
   - Hippocampus: Healthy or Epileptic
   - Worm: Default or Variant
3. **Stimulus Settings**: Amplitude, Start time, Duration
4. **Simulation Settings**: Duration (ms), Number of neurons

### Running Simulations
1. Configure model and parameters
2. Click "🚀 Run Simulation"
3. View results in tabs

### Comparison Mode
Check "🔄 Run Comparison (Both Modes)" to run both healthy/epileptic (or default/variant) simultaneously and see side-by-side comparison.

### Export
- Click "📥 Download CSV" to save results
- Results also saved to `runs/` or `worm_runs/` folders

---

## 🔬 Scientific Background

### Hippocampus Model
- Uses Brian2 Leaky Integrate-and-Fire neurons
- 50 neurons with 10% synaptic connectivity
- Parameters:
  - Healthy: gL=10nS, EL=-70mV, VT=-50mV, w=150pA
  - Epileptic: gL=30nS, EL=-50mV, VT=-45mV, w=500pA

### C. elegans Model
- Uses real connectome data from White et al. (1986)
- Top 50 most connected neurons selected
- Network propagation based on synaptic weights

---

## 📋 Configuration

### Input Files

**Stimulus** (`configs/stimuli/pulse.json`):
```json
{
  "type": "pulse",
  "amplitude": 0.5,
  "start": 100,
  "duration": 200
}
```

### Output Files
- Results saved to `runs/` (Hippocampus) or `worm_runs/` (Worm)
- Format: JSON with raw data and metrics

---

## 🛠️ Troubleshooting

### PYTHONPATH Issues
If you get "ModuleNotFoundError", ensure PYTHONPATH is set:
```bash
export PYTHONPATH=.  # On Mac/Linux
set PYTHONPATH=.     # On Windows
```

### Streamlit Not Found
```bash
pip install streamlit
```

### Brian2 Errors
Ensure Brian2 is installed:
```bash
pip install brian2
```

---

## 📚 References

- **Brian2**: A free Python package for spiking neural network simulations
  - https://brian2.readthedocs.io/

- **C. elegans Connectome**: White et al. (1986)
  - herm_full_edgelist.csv contains 7,378 synaptic connections

- **Hippocampus**: Brain region involved in memory and spatial navigation

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🤝 Contact

For questions or contributions, please contact the project maintainers.

---

**Project**: A Computational Approach to Modeling Epileptic Brain Behavior Using Hybrid Neural Simulation  
**Version**: 1.0.0