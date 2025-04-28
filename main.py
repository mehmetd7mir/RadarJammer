"""
RadarJammer - Main Entry Point
--------------------------------
Electronic warfare radar jamming simulator.

Usage:
    python main.py --mode noise --duration 0.001
    python main.py --mode drfm --visualize
    python main.py --demo

Author: Mehmet Demir
"""

import argparse
import numpy as np

from src.signals.radar_generator import RadarSignalGenerator
from src.jamming.jammer import JammingGenerator, ElectronicCountermeasures


def run_demo():
    """Run demonstration of all jamming techniques."""
    print("="*50)
    print("  RadarJammer Demo")
    print("  Electronic Warfare Simulation")
    print("="*50)
    print()
    
    sample_rate = 10e6
    
    # create radar signal
    print("[1] Generating radar signal...")
    radar = RadarSignalGenerator(frequency=1e6, sample_rate=sample_rate)
    t, chirp = radar.generate_lfm_chirp(duration=100e-6, bandwidth=2e6)
    print(f"    Generated LFM chirp: {len(chirp)} samples")
    
    # create jammer
    jammer = JammingGenerator(sample_rate=sample_rate)
    
    # noise jamming
    print("\n[2] Generating noise jamming...")
    _, noise_jam = jammer.noise_jamming(duration=100e-6, bandwidth=2e6)
    print(f"    Noise jamming: {len(noise_jam)} samples")
    
    # sweep jamming
    print("\n[3] Generating sweep jamming...")
    _, sweep_jam = jammer.sweep_jamming(
        duration=100e-6,
        start_freq=0.5e6,
        end_freq=2.5e6
    )
    print(f"    Sweep jamming: {len(sweep_jam)} samples")
    
    # deceptive jamming
    print("\n[4] Generating deceptive jamming...")
    deceptive_jam = jammer.deceptive_jamming(chirp, num_false_targets=5)
    print(f"    Created 5 false targets")
    
    # DRFM jamming
    print("\n[5] Simulating DRFM jamming...")
    drfm_jam = jammer.drfm_jamming(
        chirp,
        doppler_shift_hz=5000,  # simulate moving target
        time_delay_s=5e-6
    )
    print(f"    Added 5kHz doppler shift, 5us delay")
    
    # calculate effectiveness
    print("\n[6] Calculating JSR...")
    jsr = jammer.calculate_jsr(signal_power=1.0, jammer_power=10.0)
    print(f"    Jamming-to-Signal Ratio: {jsr:.1f} dB")
    
    if jsr > 0:
        print("    Jamming is EFFECTIVE")
    else:
        print("    Jamming is INEFFECTIVE")
    
    print("\n[+] Demo complete!")
    
    return {
        "radar": (t, chirp),
        "noise": (t, noise_jam),
        "sweep": (t, sweep_jam),
        "deceptive": (t, deceptive_jam),
        "drfm": (t, drfm_jam)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Electronic Warfare Radar Jamming Simulator"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="noise",
        choices=["noise", "sweep", "drfm", "deceptive", "spot"],
        help="Jamming mode"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=0.001,
        help="Signal duration in seconds"
    )
    parser.add_argument(
        "--frequency", "-f",
        type=float,
        default=1e6,
        help="Center frequency in Hz"
    )
    parser.add_argument(
        "--bandwidth", "-b",
        type=float,
        default=1e6,
        help="Bandwidth in Hz"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show plots"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run full demonstration"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        signals = run_demo()
        
        if args.visualize:
            try:
                from src.visualization.plots import plot_comparison
                import matplotlib.pyplot as plt
                
                sample_rate = 10e6
                signal_list = [
                    (signals["radar"][0], signals["radar"][1], "Radar Chirp"),
                    (signals["noise"][0], signals["noise"][1], "Noise Jamming"),
                    (signals["sweep"][0], signals["sweep"][1], "Sweep Jamming"),
                ]
                
                fig = plot_comparison(signal_list, sample_rate)
                if fig:
                    plt.savefig("demo_output.png", dpi=150)
                    print("\nSaved visualization to demo_output.png")
                    plt.show()
            except ImportError:
                print("Matplotlib not available for visualization")
        
        return
    
    # single mode operation
    sample_rate = 10e6
    jammer = JammingGenerator(sample_rate=sample_rate)
    
    print(f"Generating {args.mode} jamming...")
    
    if args.mode == "noise":
        t, signal = jammer.noise_jamming(
            args.duration,
            args.bandwidth,
            args.frequency
        )
    elif args.mode == "sweep":
        t, signal = jammer.sweep_jamming(
            args.duration,
            args.frequency - args.bandwidth/2,
            args.frequency + args.bandwidth/2
        )
    elif args.mode == "spot":
        t, signal = jammer.spot_jamming(args.duration, args.frequency)
    else:
        # for drfm and deceptive, we need radar signal first
        radar = RadarSignalGenerator(frequency=args.frequency, sample_rate=sample_rate)
        t, radar_sig = radar.generate_pulse(1e-6, prf=10000, num_pulses=5)
        
        if args.mode == "drfm":
            signal = jammer.drfm_jamming(radar_sig, doppler_shift_hz=1000)
        else:
            signal = jammer.deceptive_jamming(radar_sig)
    
    print(f"Generated {len(signal)} samples")
    
    if args.visualize:
        try:
            from src.visualization.plots import plot_time_domain, plot_spectrum
            import matplotlib.pyplot as plt
            
            fig1 = plot_time_domain(t, signal, f"{args.mode.title()} Jamming")
            fig2 = plot_spectrum(signal, sample_rate, f"{args.mode.title()} Spectrum")
            plt.show()
        except ImportError:
            print("Matplotlib not available")


if __name__ == "__main__":
    main()
