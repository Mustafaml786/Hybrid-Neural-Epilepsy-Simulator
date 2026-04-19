"""
Parameter validation schemas for xspecies-neuro models.
Uses dataclasses for runtime validation with min/max bounds.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HippocampusParams:
    """Hippocampus model parameters with validation."""
    
    n_neurons: int = 50
    gNa: float = 20.0       # mS/cm² - Sodium conductance
    gK: float = 10.0       # mS/cm² - Potassium conductance  
    gL: float = 0.1         # mS/cm² - Leak conductance
    ENa: float = 55.0       # mV - Sodium reversal potential
    EK: float = -90.0       # mV - Potassium reversal potential
    EL: float = -65.0       # mV - Leak reversal potential
    Cm: float = 1.0         # µF/cm² - Membrane capacitance
    tau_syn: float = 5.0      # ms - Synaptic time constant
    connection_prob: float = 0.1  # Connection probability
    Vt: float = -50.0       # mV - Threshold potential
    Vreset: float = -60.0     # mV - Reset potential
    tau_ref: float = 2.0       # ms - Refractory period
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_bounds()
    
    def _validate_bounds(self):
        """Validate all parameter bounds."""
        errors = []
        
        if not 1 <= self.n_neurons <= 1000:
            errors.append(f"n_neurons must be 1-1000, got {self.n_neurons}")
        
        if not 0.01 <= self.gNa <= 500:
            errors.append(f"gNa must be 0.01-500 mS/cm², got {self.gNa}")
        
        if not 0.01 <= self.gK <= 500:
            errors.append(f"gK must be 0.01-500 mS/cm², got {self.gK}")
        
        if not 0.01 <= self.gL <= 10:
            errors.append(f"gL must be 0.01-10 mS/cm², got {self.gL}")
        
        if not 30 <= self.ENa <= 80:
            errors.append(f"ENa must be 30-80 mV, got {self.ENa}")
        
        if not -120 <= self.EK <= -50:
            errors.append(f"EK must be -120 to -50 mV, got {self.EK}")
        
        if not -100 <= self.EL <= -30:
            errors.append(f"EL must be -100 to -30 mV, got {self.EL}")
        
        if not 0.1 <= self.Cm <= 10:
            errors.append(f"Cm must be 0.1-10 µF/cm², got {self.Cm}")
        
        if not 0.1 <= self.tau_syn <= 100:
            errors.append(f"tau_syn must be 0.1-100 ms, got {self.tau_syn}")
        
        if not 0.01 <= self.connection_prob <= 1.0:
            errors.append(f"connection_prob must be 0.01-1.0, got {self.connection_prob}")
        
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(errors))


@dataclass
class HippocampusEpilepticModifier:
    """Parameters that modify hippocampus for epileptic mode."""
    
    gNa_mult: float = 1.5      # Increase sodium conductance
    gK_mult: float = 1.2      # Increase potassium conductance  
    gL_mult: float = 3.0      # Increase leak (depolarization)
    tau_syn_mult: float = 2.0  # Increase synaptic time
    EL_shift: float = 15.0     # Shift EL by this amount (depolarization)
    VT_shift: float = 5.0       # Shift threshold (easier to fire)
    
    def __post_init__(self):
        errors = []
        
        if not 0.5 <= self.gNa_mult <= 5.0:
            errors.append(f"gNa_mult must be 0.5-5.0, got {self.gNa_mult}")
        
        if not 0.5 <= self.gK_mult <= 5.0:
            errors.append(f"gK_mult must be 0.5-5.0, got {self.gK_mult}")
        
        if not 1.0 <= self.gL_mult <= 10.0:
            errors.append(f"gL_mult must be 1.0-10.0, got {self.gL_mult}")
        
        if not 0.5 <= self.tau_syn_mult <= 5.0:
            errors.append(f"tau_syn_mult must be 0.5-5.0, got {self.tau_syn_mult}")
        
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(errors))


@dataclass
class WormParams:
    """Worm C302 model parameters with validation."""
    
    n_neurons: int = 50
    threshold: float = -35.0      # mV - Firing threshold (from literature)
    decay: float = 0.88          # Activity decay per step
    resting_potential: float = -65.0  # mV - Resting potential
    weight_scale: float = 0.15       # Synaptic weight scaling
    plasticity_enabled: bool = False   # Enable STDP
    stdp_window: float = 20.0       # ms - STDP timing window
    stdp_strength: float = 0.1      # STDP learning rate
    
    def __post_init__(self):
        errors = []
        
        if not 1 <= self.n_neurons <= 302:
            errors.append(f"n_neurons must be 1-302, got {self.n_neurons}")
        
        if not -80 <= self.threshold <= -10:
            errors.append(f"threshold must be -80 to -10 mV, got {self.threshold}")
        
        if not 0.5 <= self.decay <= 1.0:
            errors.append(f"decay must be 0.5-1.0, got {self.decay}")
        
        if not -100 <= self.resting_potential <= -30:
            errors.append(f"resting_potential must be -100 to -30 mV, got {self.resting_potential}")
        
        if not 0.01 <= self.weight_scale <= 1.0:
            errors.append(f"weight_scale must be 0.01-1.0, got {self.weight_scale}")
        
        if not 1 <= self.stdp_window <= 100:
            errors.append(f"stdp_window must be 1-100 ms, got {self.stdp_window}")
        
        if not 0.001 <= self.stdp_strength <= 1.0:
            errors.append(f"stdp_strength must be 0.001-1.0, got {self.stdp_strength}")
        
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(errors))


@dataclass
class WormVariantModifier:
    """Parameters that modify worm for variant mode."""
    
    threshold_mult: float = 0.7   # Lower threshold (easier to fire)
    weight_mult: float = 1.3      # Increased synaptic weight
    decay_mult: float = 1.0       # No change to decay
    
    def __post_init__(self):
        errors = []
        
        if not 0.3 <= self.threshold_mult <= 1.0:
            errors.append(f"threshold_mult must be 0.3-1.0, got {self.threshold_mult}")
        
        if not 0.5 <= self.weight_mult <= 3.0:
            errors.append(f"weight_mult must be 0.5-3.0, got {self.weight_mult}")
        
        if not 0.8 <= self.decay_mult <= 1.0:
            errors.append(f"decay_mult must be 0.8-1.0, got {self.decay_mult}")
        
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(errors))


@dataclass
class StimulusParams:
    """Stimulus parameters with validation."""
    
    type: str = "pulse"
    amplitude: float = 0.5       # nA
    start: int = 100            # ms
    duration: int = 200          # ms
    frequency: Optional[float] = None  # Hz (for oscillatory stimuli)
    
    def __post_init__(self):
        errors = []
        
        valid_types = ["pulse", "oscillatory", "noise", "step"]
        if self.type not in valid_types:
            errors.append(f"type must be one of {valid_types}, got {self.type}")
        
        if not 0.01 <= self.amplitude <= 10.0:
            errors.append(f"amplitude must be 0.01-10.0 nA, got {self.amplitude}")
        
        if not 0 <= self.start <= 10000:
            errors.append(f"start must be 0-10000 ms, got {self.start}")
        
        if not 1 <= self.duration <= 10000:
            errors.append(f"duration must be 1-10000 ms, got {self.duration}")
        
        if self.frequency is not None and not 0.1 <= self.frequency <= 200:
            errors.append(f"frequency must be 0.1-200 Hz, got {self.frequency}")
        
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(errors))


@dataclass
class SimulationParams:
    """General simulation parameters."""
    
    duration: int = 1000       # ms
    dt: float = 0.025         # ms
    seed: Optional[int] = None     # Random seed for reproducibility
    
    def __post_init__(self):
        errors = []
        
        if not 10 <= self.duration <= 60000:
            errors.append(f"duration must be 10-60000 ms, got {self.duration}")
        
        if not 0.001 <= self.dt <= 1.0:
            errors.append(f"dt must be 0.001-1.0 ms, got {self.dt}")
        
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(errors))


def get_default_hippocampus_params(mode: str = "Healthy") -> HippocampusParams:
    """Get default hippocampus parameters for given mode."""
    params = HippocampusParams()
    
    if mode.lower() == "epileptic":
        mod = HippocampusEpilepticModifier()
        params.gNa *= mod.gNa_mult
        params.gK *= mod.gK_mult
        params.gL *= mod.gL_mult
        params.tau_syn *= mod.tau_syn_mult
        params.EL += mod.EL_shift
        params.Vt += mod.VT_shift
    
    return params


def get_default_worm_params(mode: str = "Default") -> WormParams:
    """Get default worm parameters for given mode."""
    params = WormParams()
    
    if mode.lower() == "variant":
        mod = WormVariantModifier()
        params.threshold *= mod.threshold_mult
        params.weight_scale *= mod.weight_mult
        params.decay *= mod.decay_mult
    
    return params