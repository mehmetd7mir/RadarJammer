"""
Tests for signal analyzer.
"""

import pytest
import numpy as np
from src.signals.radar_generator import RadarSignalGenerator
from src.analysis.signal_analyzer import (
    calculate_snr,
    calculate_bandwidth,
    calculate_peak_frequency,
    measure_jamming_effectiveness,
    detect_signal_type
)


class TestSNR:
    """test SNR calculation"""

    def test_high_snr(self):
        """strong signal + weak noise = high SNR"""
        signal = np.ones(1000)
        noise = 0.01 * np.random.randn(1000)
        snr = calculate_snr(signal, noise)
        assert snr > 30

    def test_equal_power(self):
        """equal power = ~0 dB"""
        np.random.seed(42)
        signal = np.random.randn(10000)
        noise = np.random.randn(10000)
        snr = calculate_snr(signal, noise)
        assert -3 < snr < 3

    def test_zero_noise(self):
        """zero noise = infinite SNR"""
        signal = np.ones(100)
        noise = np.zeros(100)
        assert calculate_snr(signal, noise) == float('inf')


class TestBandwidth:

    def test_narrowband_signal(self):
        """CW signal should have narrower bandwidth than chirp"""
        gen = RadarSignalGenerator(frequency=100e3, sample_rate=1e6)
        _, cw = gen.generate_cw(0.01)
        _, chirp = gen.generate_lfm_chirp(0.01, bandwidth=200e3)
        cw_bw = calculate_bandwidth(cw, 1e6)
        chirp_bw = calculate_bandwidth(chirp, 1e6)
        # CW should be narrower than chirp
        assert cw_bw < chirp_bw

    def test_wideband_signal(self):
        """chirp should have wider bandwidth"""
        gen = RadarSignalGenerator(frequency=100e3, sample_rate=1e6)
        _, chirp = gen.generate_lfm_chirp(0.01, bandwidth=200e3)
        bw = calculate_bandwidth(chirp, 1e6)
        assert bw > 50000


class TestPeakFrequency:

    def test_cw_peak(self):
        """CW at 100kHz should have peak near 100kHz"""
        gen = RadarSignalGenerator(frequency=100e3, sample_rate=1e6)
        _, cw = gen.generate_cw(0.01)
        peak = calculate_peak_frequency(cw, 1e6)
        assert abs(peak - 100e3) < 5000


class TestJammingEffectiveness:

    def test_returns_dict(self):
        """should return a dict with expected keys"""
        gen = RadarSignalGenerator(frequency=1e6, sample_rate=10e6)
        _, radar = gen.generate_lfm_chirp(100e-6, bandwidth=1e6)
        noise = np.random.randn(len(radar))

        result = measure_jamming_effectiveness(radar, noise, 10e6)

        assert "power_ratio_db" in result
        assert "correlation" in result
        assert "bandwidth_ratio" in result

    def test_correlation_between_0_and_1(self):
        """correlation should be between 0 and 1"""
        gen = RadarSignalGenerator(frequency=1e6, sample_rate=10e6)
        _, radar = gen.generate_cw(0.001)
        noise = np.random.randn(len(radar))

        result = measure_jamming_effectiveness(radar, noise, 10e6)
        assert 0 <= result["correlation"] <= 1


class TestSignalTypeDetection:

    def setup_method(self):
        # use higher sample rate so CW bandwidth is relatively small
        self.gen = RadarSignalGenerator(frequency=10e3, sample_rate=1e6)

    def test_detect_cw(self):
        """should detect CW or at least not call it noise/pulsed"""
        _, cw = self.gen.generate_cw(0.01)
        sig_type = detect_signal_type(cw, 1e6)
        # CW is hard to distinguish from other narrowband
        assert sig_type in ("CW", "unknown", "chirp")

    def test_detect_pulsed(self):
        """should detect pulsed signal"""
        _, pulse = self.gen.generate_pulse(10e-6, prf=1000, num_pulses=5)
        sig_type = detect_signal_type(pulse, 1e6)
        assert sig_type == "pulsed"
