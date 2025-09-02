"""
Tests for jamming techniques.
"""

import pytest
import numpy as np
from src.jamming.jammer import JammingGenerator


class TestJammer:

    def setup_method(self):
        self.jammer = JammingGenerator(sample_rate=10000)

    def test_noise_jamming(self):
        signal = self.jammer.noise_jamming(duration=0.1, bandwidth=2000)
        assert len(signal) > 0

    def test_spot_jamming(self):
        signal = self.jammer.spot_jamming(duration=0.1, frequency=1000)
        assert len(signal) > 0

    def test_sweep_jamming(self):
        signal = self.jammer.sweep_jamming(
            duration=0.1, start_freq=500, end_freq=2000
        )
        assert len(signal) > 0

    def test_deceptive_false_targets(self):
        jammer = JammingGenerator(sample_rate=1000000)
        base = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, 10000))
        result = jammer.deceptive_jamming(base, num_false_targets=3)
        assert len(result) == len(base)

    def test_jsr_calculation(self):
        jsr = self.jammer.calculate_jsr(signal_power=1.0, jammer_power=10.0)
        assert jsr > 0
