"""
Tests for radar signal generator.
"""

import pytest
import numpy as np
from src.signals.radar_generator import RadarSignalGenerator


class TestRadarGenerator:

    def setup_method(self):
        self.gen = RadarSignalGenerator(sample_rate=10000, frequency=1000)

    def test_cw_signal(self):
        t, signal = self.gen.generate_cw(duration=0.1)
        assert len(signal) == len(t)

    def test_pulse_signal(self):
        t, signal = self.gen.generate_pulse(pulse_width=0.01, prf=50, num_pulses=5)
        assert len(signal) == len(t)

    def test_lfm_chirp(self):
        t, signal = self.gen.generate_lfm_chirp(duration=0.1, bandwidth=2000)
        assert len(signal) > 0

    def test_add_noise(self):
        t, signal = self.gen.generate_cw(duration=0.1)
        noisy = self.gen.add_noise(signal, snr_db=10)
        assert not np.array_equal(noisy, signal)
