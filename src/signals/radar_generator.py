"""
Radar Signal Generator
-----------------------
Generate various radar waveforms for simulation.

Types of radar signals:
    - Pulsed radar: short bursts of energy
    - CW (Continuous Wave): constant frequency
    - FMCW (Frequency Modulated): chirp signals
    - LFM (Linear Frequency Modulation): linear chirps

This is core module for electronic warfare simulation.

Author: Mehmet Demir
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RadarParams:
    """Radar system parameters"""
    frequency_hz: float  # carrier frequency
    bandwidth_hz: float  # signal bandwidth
    pulse_width_s: float  # pulse duration
    prf_hz: float  # pulse repetition frequency
    sample_rate_hz: float  # ADC sample rate


class RadarSignalGenerator:
    """
    Generate radar waveforms for simulation.
    
    Example:
        gen = RadarSignalGenerator(frequency=10e9, sample_rate=100e6)
        pulse = gen.generate_pulse(width=1e-6)
        chirp = gen.generate_lfm_chirp(bandwidth=50e6)
    """
    
    def __init__(
        self,
        frequency: float = 10e9,  # 10 GHz (X-band)
        sample_rate: float = 100e6  # 100 MHz
    ):
        """
        Initialize generator.
        
        Args:
            frequency: center frequency in Hz
            sample_rate: sample rate in Hz
        """
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
    
    def generate_time_array(self, duration: float) -> np.ndarray:
        """Create time array for given duration."""
        n_samples = int(duration * self.sample_rate)
        return np.arange(n_samples) * self.dt
    
    def generate_cw(self, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Continuous Wave signal.
        
        Simple sinusoidal signal at carrier frequency.
        """
        t = self.generate_time_array(duration)
        signal = np.cos(2 * np.pi * self.frequency * t)
        return t, signal
    
    def generate_pulse(
        self,
        pulse_width: float,
        prf: float = 1000,
        num_pulses: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate pulsed radar signal.
        
        Args:
            pulse_width: pulse duration in seconds
            prf: pulse repetition frequency
            num_pulses: number of pulses to generate
        
        Returns:
            time array, signal array
        """
        pri = 1.0 / prf  # pulse repetition interval
        total_duration = num_pulses * pri
        
        t = self.generate_time_array(total_duration)
        signal = np.zeros_like(t)
        
        samples_per_pulse = int(pulse_width * self.sample_rate)
        samples_per_pri = int(pri * self.sample_rate)
        
        # create pulses
        for i in range(num_pulses):
            start = i * samples_per_pri
            end = start + samples_per_pulse
            if end > len(signal):
                end = len(signal)
            
            # modulate with carrier
            pulse_t = t[start:end] - t[start]
            signal[start:end] = np.cos(2 * np.pi * self.frequency * pulse_t)
        
        return t, signal
    
    def generate_lfm_chirp(
        self,
        duration: float,
        bandwidth: float,
        start_freq: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Linear Frequency Modulation (LFM) chirp.
        
        Frequency sweeps linearly from start to end.
        Very common in modern radars.
        
        Args:
            duration: chirp duration
            bandwidth: frequency sweep range
            start_freq: starting frequency (default: center - bandwidth/2)
        
        Returns:
            time, signal arrays
        """
        t = self.generate_time_array(duration)
        
        if start_freq is None:
            start_freq = self.frequency - bandwidth / 2
        
        # chirp rate
        k = bandwidth / duration
        
        # instantaneous frequency: f(t) = f0 + k*t
        # phase: phi(t) = 2*pi*integral(f(t)) = 2*pi*(f0*t + k*t^2/2)
        phase = 2 * np.pi * (start_freq * t + k * t**2 / 2)
        
        signal = np.cos(phase)
        
        return t, signal
    
    def generate_fmcw(
        self,
        chirp_duration: float,
        bandwidth: float,
        num_chirps: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate FMCW (Frequency Modulated Continuous Wave) signal.
        
        Repeated chirps, common in automotive radar.
        """
        total_duration = chirp_duration * num_chirps
        t = self.generate_time_array(total_duration)
        signal = np.zeros_like(t)
        
        samples_per_chirp = int(chirp_duration * self.sample_rate)
        
        for i in range(num_chirps):
            start = i * samples_per_chirp
            end = start + samples_per_chirp
            if end > len(signal):
                end = len(signal)
            
            chirp_t = t[start:end] - t[start]
            start_freq = self.frequency - bandwidth / 2
            k = bandwidth / chirp_duration
            
            phase = 2 * np.pi * (start_freq * chirp_t + k * chirp_t**2 / 2)
            signal[start:end] = np.cos(phase)
        
        return t, signal
    
    def add_noise(
        self,
        signal: np.ndarray,
        snr_db: float = 20
    ) -> np.ndarray:
        """
        Add white Gaussian noise to signal.
        
        Args:
            signal: input signal
            snr_db: signal to noise ratio in dB
        
        Returns:
            noisy signal
        """
        # calculate signal power
        signal_power = np.mean(signal**2)
        
        # calculate noise power from SNR
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # generate noise
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        
        return signal + noise
    
    def get_spectrum(
        self,
        signal: np.ndarray,
        window: str = "hann"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency spectrum of signal.
        
        Returns:
            frequencies, magnitude (dB)
        """
        from scipy import signal as sig
        
        n = len(signal)
        
        # apply window
        if window == "hann":
            win = np.hanning(n)
        elif window == "hamming":
            win = np.hamming(n)
        else:
            win = np.ones(n)
        
        windowed = signal * win
        
        # FFT
        spectrum = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(n, self.dt)
        
        # positive frequencies only
        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        spectrum = spectrum[positive_mask]
        
        # magnitude in dB
        magnitude = 20 * np.log10(np.abs(spectrum) + 1e-10)
        
        return freqs, magnitude


# test
if __name__ == "__main__":
    gen = RadarSignalGenerator(frequency=1e6, sample_rate=10e6)
    
    # generate chirp
    t, chirp = gen.generate_lfm_chirp(duration=100e-6, bandwidth=1e6)
    print(f"Generated LFM chirp: {len(chirp)} samples")
    
    # add noise
    noisy = gen.add_noise(chirp, snr_db=10)
    
    # get spectrum
    freqs, spectrum = gen.get_spectrum(chirp)
    print(f"Spectrum: {len(freqs)} frequency bins")
