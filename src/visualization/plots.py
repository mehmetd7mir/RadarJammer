"""
Signal Visualization Module
----------------------------
Plot radar and jamming signals.

Includes:
    - Time domain plots
    - Frequency spectrum (FFT)
    - Spectrogram (time-frequency)
    - Waterfall display

Author: Mehmet Demir
"""

import numpy as np
from typing import Optional, Tuple, List

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_time_domain(
    t: np.ndarray,
    signal: np.ndarray,
    title: str = "Signal",
    ylabel: str = "Amplitude"
) -> Optional[Figure]:
    """Plot signal in time domain."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(t * 1e6, signal, linewidth=0.5)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_spectrum(
    signal: np.ndarray,
    sample_rate: float,
    title: str = "Spectrum"
) -> Optional[Figure]:
    """Plot frequency spectrum."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    n = len(signal)
    
    # apply window
    windowed = signal * np.hanning(n)
    
    # FFT
    spectrum = np.fft.fft(windowed)
    freqs = np.fft.fftfreq(n, 1/sample_rate)
    
    # positive frequencies
    mask = freqs >= 0
    freqs = freqs[mask]
    spectrum = spectrum[mask]
    
    # magnitude in dB
    mag_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(freqs / 1e6, mag_db, linewidth=0.5)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, sample_rate/2/1e6])
    
    return fig


def plot_spectrogram(
    signal: np.ndarray,
    sample_rate: float,
    title: str = "Spectrogram",
    window_size: int = 256
) -> Optional[Figure]:
    """Plot spectrogram (time-frequency representation)."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.specgram(
        signal,
        NFFT=window_size,
        Fs=sample_rate,
        noverlap=window_size//2,
        cmap="viridis"
    )
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    
    return fig


def plot_comparison(
    signals: List[Tuple[np.ndarray, np.ndarray, str]],
    sample_rate: float
) -> Optional[Figure]:
    """
    Plot multiple signals for comparison.
    
    Args:
        signals: list of (time, signal, name) tuples
        sample_rate: sampling rate
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    n = len(signals)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3*n))
    
    if n == 1:
        axes = [axes]
    
    for i, (t, sig, name) in enumerate(signals):
        # time domain
        axes[i][0].plot(t * 1e6, sig, linewidth=0.5)
        axes[i][0].set_xlabel("Time (us)")
        axes[i][0].set_ylabel("Amplitude")
        axes[i][0].set_title(f"{name} - Time Domain")
        axes[i][0].grid(True, alpha=0.3)
        
        # frequency domain
        spectrum = np.fft.fft(sig * np.hanning(len(sig)))
        freqs = np.fft.fftfreq(len(sig), 1/sample_rate)
        mask = freqs >= 0
        mag_db = 20 * np.log10(np.abs(spectrum[mask]) + 1e-10)
        
        axes[i][1].plot(freqs[mask] / 1e6, mag_db, linewidth=0.5)
        axes[i][1].set_xlabel("Frequency (MHz)")
        axes[i][1].set_ylabel("Magnitude (dB)")
        axes[i][1].set_title(f"{name} - Spectrum")
        axes[i][1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_waterfall(
    signal: np.ndarray,
    sample_rate: float,
    segment_duration: float = 1e-3,
    title: str = "Waterfall"
) -> Optional[Figure]:
    """
    Create waterfall display (stacked spectra over time).
    
    good for seeing how spectrum changes with time.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    samples_per_segment = int(segment_duration * sample_rate)
    n_segments = len(signal) // samples_per_segment
    
    if n_segments < 2:
        print("Signal too short for waterfall")
        return None
    
    # compute spectra for each segment
    waterfall_data = []
    
    for i in range(n_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = signal[start:end]
        
        windowed = segment * np.hanning(len(segment))
        spectrum = np.fft.fft(windowed)
        mag_db = 20 * np.log10(np.abs(spectrum[:len(spectrum)//2]) + 1e-10)
        waterfall_data.append(mag_db)
    
    waterfall_data = np.array(waterfall_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(
        waterfall_data,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, sample_rate/2/1e6, 0, n_segments * segment_duration * 1e3]
    )
    
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    
    plt.colorbar(im, label="Magnitude (dB)")
    
    return fig


# test
if __name__ == "__main__":
    if MATPLOTLIB_AVAILABLE:
        # create test signal
        sample_rate = 10e6
        duration = 1e-3
        n = int(duration * sample_rate)
        t = np.arange(n) / sample_rate
        
        # chirp signal
        f0 = 1e6
        f1 = 4e6
        k = (f1 - f0) / duration
        signal = np.cos(2 * np.pi * (f0 * t + k * t**2 / 2))
        
        # add noise
        signal += 0.1 * np.random.randn(len(signal))
        
        # plot
        fig1 = plot_time_domain(t, signal, "Test Chirp Signal")
        fig2 = plot_spectrum(signal, sample_rate, "Chirp Spectrum")
        fig3 = plot_spectrogram(signal, sample_rate, "Chirp Spectrogram")
        
        plt.show()
    else:
        print("Matplotlib not available")
