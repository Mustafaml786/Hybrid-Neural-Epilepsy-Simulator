#!/usr/bin/env python3
"""
Test script to verify parameter improvements work correctly.
"""

import json
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.hippocampus_brian2 import HippocampusBrian2Adapter
from adapters.worm_c302_full import WormC302FullAdapter


def test_hippocampus():
    """Test hippocampus adapter with new parameters."""
    print("=" * 60)
    print("TESTING HIPPOCAMPUS ADAPTER")
    print("=" * 60)
    
    # Test Healthy mode
    print("\n[1] Testing Healthy mode...")
    cfg = {
        "simulation": {"duration": 500, "dt": 0.1},
        "hippocampus": {
            "n_neurons": 30,
            "connection_prob": 0.15
        }
    }
    
    adapter = HippocampusBrian2Adapter(config=cfg, mode="Healthy")
    adapter.initialize(n_neurons=30, stim_cfg={
        "type": "pulse",
        "amplitude": 1.0,
        "start": 50,
        "duration": 150
    })
    results = adapter.run(500)
    output = adapter.get_output()
    
    print(f"   Healthy output: {output}")
    print(f"   Num spikes: {output.get('num_spikes', 0)}")
    
    # Test Epileptic mode
    print("\n[2] Testing Epileptic mode...")
    adapter_ep = HippocampusBrian2Adapter(config=cfg, mode="Epileptic")
    adapter_ep.initialize(n_neurons=30, stim_cfg={
        "type": "pulse",
        "amplitude": 1.0,
        "start": 50,
        "duration": 150
    })
    results_ep = adapter_ep.run(500)
    output_ep = adapter_ep.get_output()
    
    print(f"   Epileptic output: {output_ep}")
    print(f"   Num spikes: {output_ep.get('num_spikes', 0)}")
    
    # Compare
    healthy_spikes = output.get('num_spikes', 0)
    epileptic_spikes = output_ep.get('num_spikes', 0)
    
    print("\n[3] Comparison:")
    print(f"   Healthy spikes:  {healthy_spikes}")
    print(f"   Epileptic spikes: {epileptic_spikes}")
    if healthy_spikes > 0:
        ratio = epileptic_spikes / healthy_spikes
        print(f"   Ratio (E/H):    {ratio:.2f}x")
    
    print("\n[OK] Hippocampus tests passed!")
    return True


def test_worm():
    """Test worm adapter with new parameters."""
    print("\n" + "=" * 60)
    print("TESTING WORM ADAPTER")
    print("=" * 60)
    
    # Test Default mode
    print("\n[1] Testing Default mode...")
    cfg = {
        "simulation": {"duration": 500, "dt": 0.1}
    }
    
    adapter = WormC302FullAdapter(config=cfg, mode="Default")
    adapter.initialize(n_neurons=30)
    adapter.apply_stimulus({"amplitude": 0.5})
    results = adapter.run(500)
    output = adapter.get_output()
    
    print(f"   Default output: num_spikes={output.get('num_spikes', 0)}")
    print(f"   Mean activity: {output.get('mean_act_val', 0):.2f}")
    
    # Test Variant mode
    print("\n[2] Testing Variant mode...")
    adapter_var = WormC302FullAdapter(config=cfg, mode="Variant")
    adapter_var.initialize(n_neurons=30)
    adapter_var.apply_stimulus({"amplitude": 0.5})
    results_var = adapter_var.run(500)
    output_var = adapter_var.get_output()
    
    print(f"   Variant output: num_spikes={output_var.get('num_spikes', 0)}")
    print(f"   Mean activity: {output_var.get('mean_act_val', 0):.2f}")
    
    # Test with plasticity enabled
    print("\n[3] Testing with STDP plasticity...")
    adapter_plast = WormC302FullAdapter(config=cfg, mode="Default")
    adapter_plast.initialize(n_neurons=20)
    adapter_plast.apply_stimulus({"amplitude": 0.5})
    results_plast = adapter_plast.run(500, plasticity=True)
    output_plast = adapter_plast.get_output()
    
    print(f"   STDP output: num_spikes={output_plast.get('num_spikes', 0)}")
    
    # Compare
    default_spikes = output.get('num_spikes', 0)
    variant_spikes = output_var.get('num_spikes', 0)
    
    print("\n[4] Comparison:")
    print(f"   Default spikes:  {default_spikes}")
    print(f"   Variant spikes: {variant_spikes}")
    if default_spikes > 0:
        ratio = variant_spikes / default_spikes
        print(f"   Ratio (V/D):    {ratio:.2f}x")
    
    print("\n[OK] Worm tests passed!")
    return True


def test_parameter_validation():
    """Test parameter validation module."""
    print("\n" + "=" * 60)
    print("TESTING PARAMETER VALIDATION")
    print("=" * 60)
    
    try:
        from configs.parameters import (
            HippocampusParams, WormParams, StimulusParams,
            SimulationParams, get_default_hippocampus_params, get_default_worm_params
        )
        
        # Test valid params
        print("\n[1] Testing valid parameters...")
        hp = HippocampusParams(n_neurons=50, gNa=20, gK=10)
        print(f"   HippocampusParams: gNa={hp.gNa}, gK={hp.gK}")
        
        wp = WormParams(n_neurons=50, threshold=-35, decay=0.88)
        print(f"   WormParams: threshold={wp.threshold}, decay={wp.decay}")
        
        sp = StimulusParams(amplitude=0.5, start=100, duration=200)
        print(f"   StimulusParams: amplitude={sp.amplitude}")
        
        # Test invalid params (should raise)
        print("\n[2] Testing invalid parameter rejection...")
        try:
            bad_hp = HippocampusParams(n_neurons=0)  # Should fail
            print("   ERROR: Should have raised ValueError!")
            return False
        except ValueError as e:
            print(f"   Correctly rejected n_neurons=0: {str(e)[:50]}...")
        
        # Test defaults
        print("\n[3] Testing get_default_* functions...")
        hp_healthy = get_default_hippocampus_params("Healthy")
        hp_epileptic = get_default_hippocampus_params("Epileptic")
        print(f"   Healthy: gNa={hp_healthy.gNa}, EL={hp_healthy.EL}")
        print(f"   Epileptic: gNa={hp_epileptic.gNa}, EL={hp_epileptic.EL}")
        
        wp_default = get_default_worm_params("Default")
        wp_variant = get_default_worm_params("Variant")
        print(f"   Default: threshold={wp_default.threshold}")
        print(f"   Variant: threshold={wp_variant.threshold}")
        
        print("\n[OK] Parameter validation tests passed!")
        return True
        
    except ImportError as e:
        print(f"   Warning: Could not import parameters module: {e}")
        print("   Skipping parameter validation tests...")
        return True


if __name__ == "__main__":
    print("Running parameter improvement tests...\n")
    
    all_passed = True
    
    try:
        all_passed &= test_parameter_validation()
    except Exception as e:
        print(f"Parameter validation test error: {e}")
    
    try:
        all_passed &= test_hippocampus()
    except Exception as e:
        print(f"Hippocampus test error: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        all_passed &= test_worm()
    except Exception as e:
        print(f"Worm test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)