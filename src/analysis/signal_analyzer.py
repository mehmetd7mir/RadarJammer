"""
Signal Analyzer
-----------------
Some extra analysis tools for radar and jammed signals.
Helps to understand how well the jamming is working.
"""

import numpy as np
from typing import Tuple, Dict


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.

    Just the ratio of signal power to noise power, in decibels.
    Higher SNR = cleaner signal, lower SNR = more noise.
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)

    if noise_power == 0:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)


def calculate_bandwidth(signal: np.ndarray, sample_rate: float,
                        threshold_db: float = -3.0) -> float:
    """
    Estimate the -3dB bandwidth of a signal.

    Finds where the spectrum drops below the threshold
    and measures the width. -3dB is standard in RF.
    """
    # get the frequency spectrum
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
    power_db -= np.max(power_db)  # normalize to 0 dB peak

    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/sample_rate))

    # find where power is above threshold
    above_threshold = power_db > threshold_db
    if not np.any(above_threshold):
        return 0.0

    freq_above = freqs[above_threshold]
    bandwidth = freq_above[-1] - freq_above[0]

    return abs(bandwidth)


def calculate_peak_frequency(signal: np.ndarray,
                              sample_rate: float) -> float:
    """
    Find the dominant frequency in the signal.

    Uses FFT and finds the peak. Simple but works well
    for single-tone signals like CW and spot jamming.
    """
    spectrum = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)

    # only look at positive frequencies
    positive = freqs > 0
    peak_idx = np.argmax(spectrum[positive])

    return freqs[positive][peak_idx]


def measure_jamming_effectiveness(
    radar_signal: np.ndarray,
    jammed_signal: np.ndarray,
    sample_rate: float
) -> Dict:
    """
    Measure how effective the jamming is.

    Compares the original radar signal with the jammed version
    and gives some metrics about the jamming quality.

    Returns dict with:
        - power_ratio_db: how much stronger the jam is
        - correlation: how similar they are (lower = better jam)
        - bandwidth_ratio: bandwidth comparison
    """
    # power comparison
    radar_power = np.mean(np.abs(radar_signal) ** 2)
    jam_power = np.mean(np.abs(jammed_signal) ** 2)

    if radar_power == 0:
        power_ratio = 0.0
    else:
        power_ratio = 10 * np.log10(jam_power / radar_power)

    # correlation (how similar the signals are)
    # lower correlation = jammming is injecting different content
    if len(radar_signal) == len(jammed_signal):
        correlation = abs(np.corrcoef(
            np.abs(radar_signal), np.abs(jammed_signal)
        )[0, 1])
    else:
        correlation = 0.0

    # bandwidth comparison
    radar_bw = calculate_bandwidth(radar_signal, sample_rate)
    jam_bw = calculate_bandwidth(jammed_signal, sample_rate)

    if radar_bw > 0:
        bw_ratio = jam_bw / radar_bw
    else:
        bw_ratio = 0.0

    return {
        "power_ratio_db": round(power_ratio, 2),
        "correlation": round(correlation, 4),
        "bandwidth_ratio": round(bw_ratio, 2),
        "radar_power_dbm": round(10 * np.log10(radar_power + 1e-12), 2),
        "jam_power_dbm": round(10 * np.log10(jam_power + 1e-12), 2),
    }


def detect_signal_type(signal: np.ndarray, sample_rate: float) -> str:
    """
    Try to guess what type of signal this is.

    Looks at the spectrum shape and time-domain properties
    to classify it. Not perfect but gives a rough idea.

    returns one of: "CW", "pulsed", "chirp", "noise", "unknown"
    """
    magnitude = np.abs(signal)

    # check if it's pulsed (has big gaps of silence)
    zero_fraction = np.sum(magnitude < 0.01 * np.max(magnitude)) / len(magnitude)
    if zero_fraction > 0.3:
        return "pulsed"

    # check bandwidth - narrow = CW, wide = chirp or noise
    bw = calculate_bandwidth(signal, sample_rate)
    relative_bw = bw / sample_rate

    if relative_bw < 0.01:
        return "CW"

    # check if spectrum is flat (noise) or shaped (chirp)
    spectrum = np.abs(np.fft.fft(signal))
    # remove DC
    spectrum[0] = 0
    spec_std = np.std(spectrum) / (np.mean(spectrum) + 1e-12)

    if spec_std < 0.5:
        return "noise"
    elif relative_bw > 0.05:
        return "chirp"

    return "unknown"


# quick test
if __name__ == "__main__":
    from src.signals.radar_generator import RadarSignalGenerator
    from src.jamming.jammer import JammingGenerator

    gen = RadarSignalGenerator(sample_rate=1e6)
    jammer = JammingGenerator(sample_rate=1e6)

    # make a chirp and jam it
    t, chirp = gen.generate_lfm_chirp(1e3, 50e3, 0.001)
    noise_jam = jammer.noise_jamming(len(chirp), 100e3)

    # analyze
    result = measure_jamming_effectiveness(chirp, noise_jam, 1e6)
    print("Jamming effectiveness:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # detect signal type
    sig_type = detect_signal_type(chirp, 1e6)
    print(f"\nSignal type: {sig_type}")
