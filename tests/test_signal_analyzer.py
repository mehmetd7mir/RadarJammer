"""
Tests for signal analyzer.
"""

import pytest
import numpy as np
from src.analysis.signal_analyzer import (
    calculate_snr,
    calculate_bandwidth,
    calculate_peak_frequency,
    detect_signal_type,
)


def test_snr_high():
    signal = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 10000))
    noise = np.random.normal(0, 0.01, 10000)
    snr = calculate_snr(signal, noise)
    assert snr > 10


def test_bandwidth():
    t = np.linspace(0, 1, 10000)
    signal = np.sin(2 * np.pi * 100 * t)
    bw = calculate_bandwidth(signal, sample_rate=10000)
    assert bw > 0


def test_peak_frequency():
    t = np.linspace(0, 1, 10000)
    signal = np.sin(2 * np.pi * 500 * t)
    peak = calculate_peak_frequency(signal, sample_rate=10000)
    assert abs(peak - 500) < 50


def test_detect_signal_type():
    t = np.linspace(0, 1, 10000)
    signal = np.sin(2 * np.pi * 1000 * t)
    sig_type = detect_signal_type(signal, sample_rate=10000)
    assert sig_type in ["CW", "pulsed", "chirp", "noise", "unknown"]
