"""
Jamming Techniques Module
--------------------------
Various electronic warfare jamming techniques.

Types of jamming:
    - Noise jamming: overwhelm with noise
    - Deceptive jamming: create false targets
    - DRFM: digital copy and replay
    - Sweep jamming: sweep across frequency band
    - Spot jamming: focus on single frequency

Used in electronic warfare to disrupt radar systems.

Author: Mehmet Demir
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class JammerParams:
    """Jammer parameters"""
    center_frequency: float
    bandwidth: float
    power: float
    technique: str


class JammingGenerator:
    """
    Generate jamming waveforms.
    
    Example:
        jammer = JammingGenerator(sample_rate=100e6)
        noise = jammer.noise_jamming(duration=1e-3, bandwidth=10e6)
        deceptive = jammer.deceptive_jamming(target_signal)
    """
    
    def __init__(self, sample_rate: float = 100e6):
        """
        Initialize jammer.
        
        Args:
            sample_rate: sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
    
    def noise_jamming(
        self,
        duration: float,
        bandwidth: float,
        center_freq: float = 0.0,
        power_db: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate broadband noise jamming.
        
        Simplest form of jamming - just blast noise across the band.
        Effective but requires lot of power.
        
        Args:
            duration: signal duration
            bandwidth: noise bandwidth in Hz
            center_freq: center frequency
            power_db: output power in dB
        
        Returns:
            time, jamming signal
        """
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) * self.dt
        
        # generate white noise
        noise = np.random.randn(n_samples)
        
        # bandlimit the noise
        if bandwidth < self.sample_rate / 2:
            from scipy import signal as sig
            
            # design bandpass filter
            nyq = self.sample_rate / 2
            low = max(0.01, (center_freq - bandwidth/2)) / nyq
            high = min(0.99, (center_freq + bandwidth/2)) / nyq
            
            if low < high and low > 0 and high < 1:
                b, a = sig.butter(4, [low, high], btype='band')
                noise = sig.filtfilt(b, a, noise)
        
        # apply power
        power_linear = 10**(power_db / 20)
        noise = noise * power_linear
        
        return t, noise
    
    def spot_jamming(
        self,
        duration: float,
        frequency: float,
        power_db: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spot (single frequency) jamming.
        
        Focus all power on one frequency.
        Very efficient but can be avoided with frequency hopping.
        """
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) * self.dt
        
        power_linear = 10**(power_db / 20)
        signal = power_linear * np.cos(2 * np.pi * frequency * t)
        
        return t, signal
    
    def sweep_jamming(
        self,
        duration: float,
        start_freq: float,
        end_freq: float,
        sweep_time: float = 1e-3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sweep jamming.
        
        Frequency sweeps back and forth across band.
        good balance between spot and noise jamming.
        """
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) * self.dt
        
        # create sawtooth modulation of frequency
        mod_freq = 1.0 / sweep_time
        sweep_phase = 2 * np.pi * mod_freq * t
        
        # frequency varies with triangle wave
        freq_offset = (end_freq - start_freq) * (0.5 + 0.5 * np.sin(sweep_phase))
        inst_freq = start_freq + freq_offset
        
        # integrate frequency to get phase
        phase = 2 * np.pi * np.cumsum(inst_freq) * self.dt
        
        signal = np.cos(phase)
        
        return t, signal
    
    def deceptive_jamming(
        self,
        target_signal: np.ndarray,
        num_false_targets: int = 5,
        delay_range: Tuple[float, float] = (1e-6, 10e-6)
    ) -> np.ndarray:
        """
        Generate deceptive jamming with false targets.
        
        Copy the target signal and retransmit with delays
        to create multiple false echoes.
        
        Args:
            target_signal: signal to copy
            num_false_targets: number of false echoes
            delay_range: range of delays (min, max) in seconds
        
        Returns:
            jamming signal with false targets
        """
        jammed = np.zeros_like(target_signal)
        
        for i in range(num_false_targets):
            # random delay
            delay = np.random.uniform(delay_range[0], delay_range[1])
            delay_samples = int(delay * self.sample_rate)
            
            # random amplitude (simulate different ranges)
            amplitude = np.random.uniform(0.5, 2.0)
            
            # shift and add
            if delay_samples < len(target_signal):
                shifted = np.zeros_like(target_signal)
                shifted[delay_samples:] = target_signal[:-delay_samples]
                jammed += amplitude * shifted
        
        return jammed
    
    def drfm_jamming(
        self,
        target_signal: np.ndarray,
        doppler_shift_hz: float = 0.0,
        time_delay_s: float = 0.0,
        amplitude_mod: float = 1.0
    ) -> np.ndarray:
        """
        Digital RF Memory (DRFM) jamming.
        
        Record, modify, and retransmit enemy radar signal.
        Can add doppler to simulate moving target, etc.
        
        Args:
            target_signal: captured radar signal
            doppler_shift_hz: frequency shift to add
            time_delay_s: delay before retransmit
            amplitude_mod: amplitude modification
        
        Returns:
            modified signal
        """
        n = len(target_signal)
        t = np.arange(n) * self.dt
        
        # apply doppler shift
        doppler_phase = 2 * np.pi * doppler_shift_hz * t
        signal = target_signal * np.cos(doppler_phase)
        
        # apply delay
        delay_samples = int(time_delay_s * self.sample_rate)
        if delay_samples > 0:
            delayed = np.zeros(n)
            if delay_samples < n:
                delayed[delay_samples:] = signal[:-delay_samples]
            signal = delayed
        
        # apply amplitude mod
        signal = signal * amplitude_mod
        
        return signal
    
    def barrage_jamming(
        self,
        duration: float,
        freq_bands: list,
        power_per_band_db: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Barrage jamming - jam multiple frequency bands.
        
        Args:
            duration: signal duration
            freq_bands: list of (center_freq, bandwidth) tuples
            power_per_band_db: power for each band
        
        Returns:
            combined jamming signal
        """
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) * self.dt
        
        combined = np.zeros(n_samples)
        
        for center_freq, bandwidth in freq_bands:
            _, noise = self.noise_jamming(
                duration, bandwidth, center_freq, power_per_band_db
            )
            combined += noise
        
        # normalize
        combined = combined / len(freq_bands)
        
        return t, combined
    
    def calculate_jsr(
        self,
        signal_power: float,
        jammer_power: float,
        signal_gain: float = 1.0,
        jammer_gain: float = 1.0
    ) -> float:
        """
        Calculate Jamming-to-Signal Ratio (JSR).
        
        JSR > 1 (or > 0 dB) means jamming is effective.
        """
        jsr = (jammer_power * jammer_gain) / (signal_power * signal_gain)
        jsr_db = 10 * np.log10(jsr)
        return jsr_db


class ElectronicCountermeasures:
    """
    Collection of ECM techniques.
    
    Higher level wrapper around jamming generator.
    """
    
    def __init__(self, sample_rate: float = 100e6):
        self.jammer = JammingGenerator(sample_rate)
        self.sample_rate = sample_rate
    
    def respond_to_radar(
        self,
        radar_signal: np.ndarray,
        technique: str = "noise"
    ) -> np.ndarray:
        """
        Generate ECM response to detected radar.
        
        Args:
            radar_signal: detected radar pulse
            technique: 'noise', 'drfm', 'deceptive'
        
        Returns:
            jamming response
        """
        if technique == "noise":
            duration = len(radar_signal) / self.sample_rate
            _, jam = self.jammer.noise_jamming(duration, self.sample_rate/4)
            return jam
        
        elif technique == "drfm":
            return self.jammer.drfm_jamming(radar_signal, doppler_shift_hz=1000)
        
        elif technique == "deceptive":
            return self.jammer.deceptive_jamming(radar_signal, num_false_targets=3)
        
        else:
            return radar_signal


# test
if __name__ == "__main__":
    jammer = JammingGenerator(sample_rate=10e6)
    
    # test noise jamming
    t, noise = jammer.noise_jamming(duration=1e-3, bandwidth=1e6)
    print(f"Noise jamming: {len(noise)} samples")
    
    # test sweep jamming
    t, sweep = jammer.sweep_jamming(duration=1e-3, start_freq=1e6, end_freq=5e6)
    print(f"Sweep jamming: {len(sweep)} samples")
    
    # test deceptive
    dummy_signal = np.cos(2 * np.pi * 1e6 * t)
    deceptive = jammer.deceptive_jamming(dummy_signal, num_false_targets=5)
    print(f"Deceptive jamming: {len(deceptive)} samples")
    
    # calculate JSR
    jsr = jammer.calculate_jsr(1.0, 10.0)
    print(f"JSR: {jsr:.1f} dB")
