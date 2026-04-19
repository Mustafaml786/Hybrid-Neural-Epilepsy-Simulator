# analysis/viz.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_activity(time, voltage, title="Neuron Activity", save_path=None):
    """Plot voltage vs time and optionally save figure."""
    plt.figure(figsize=(8, 4))
    plt.plot(time, voltage)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_voltage(time, voltage, savepath=None):
    plt.figure()
    plt.plot(time, voltage)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Membrane potential")
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
