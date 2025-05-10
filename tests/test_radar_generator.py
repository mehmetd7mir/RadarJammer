"""
Tests for radar signal generator.
I wrote these to make sure the signals have correct length and frequency.
"""

import pytest
import numpy as np
from src.signals.radar_generator import RadarSignalGenerator


class TestRadarGenerator:
    """basic tests for radar signal generation"""

    def setup_method(self):
        # use lower freq and sample rate so tests are fast
        self.gen = RadarSignalGenerator(frequency=1e6, sample_rate=10e6)

    def test_create_generator(self):
        """just check it creates without error"""
        gen = RadarSignalGenerator(frequency=5e9, sample_rate=50e6)
        assert gen.sample_rate == 50e6
        assert gen.frequency == 5e9

    def test_cw_signal_length(self):
        """CW signal should have correct number of samples"""
        duration = 0.001  # 1 ms
        t, signal = self.gen.generate_cw(duration=duration)

        expected_samples = int(self.gen.sample_rate * duration)
        assert len(signal) == expected_samples
        assert len(t) == expected_samples

    def test_cw_signal_is_real(self):
        """CW signal should be real valued cosine"""
        t, signal = self.gen.generate_cw(duration=0.001)
        # this generator returns real signals (cosine)
        assert not np.iscomplexobj(signal)

    def test_cw_amplitude(self):
        """CW signal should have max amplitude of 1 (cosine)"""
        t, signal = self.gen.generate_cw(duration=0.001)
        assert np.max(np.abs(signal)) <= 1.01

    def test_pulse_signal_has_gaps(self):
        """pulsed signal should have zeros between pulses"""
        t, signal = self.gen.generate_pulse(
            pulse_width=10e-6, prf=1000, num_pulses=3
        )

        # there should be some zeros (gap between pulses)
        zero_count = np.sum(np.abs(signal) < 0.01)
        assert zero_count > 0

    def test_pulse_total_length(self):
        """total signal should match num_pulses * PRI"""
        prf = 1000  # 1 kHz PRF
        num_pulses = 5
        t, signal = self.gen.generate_pulse(
            pulse_width=10e-6, prf=prf, num_pulses=num_pulses
        )

        pri = 1.0 / prf
        expected_len = int(num_pulses * pri * self.gen.sample_rate)
        assert len(signal) == expected_len

    def test_lfm_chirp_length(self):
        """LFM chirp should have right duration"""
        duration = 0.001
        t, signal = self.gen.generate_lfm_chirp(
            duration=duration, bandwidth=500e3
        )

        expected_samples = int(self.gen.sample_rate * duration)
        assert len(signal) == expected_samples

    def test_lfm_chirp_is_real(self):
        """chirp should be real signal"""
        t, signal = self.gen.generate_lfm_chirp(
            duration=0.001, bandwidth=500e3
        )
        assert not np.iscomplexobj(signal)

    def test_fmcw_signal_length(self):
        """FMCW should have correct total duration"""
        t, signal = self.gen.generate_fmcw(
            chirp_duration=100e-6, bandwidth=500e3, num_chirps=4
        )

        expected_samples = int(4 * 100e-6 * self.gen.sample_rate)
        assert len(signal) == expected_samples

    def test_add_noise_changes_signal(self):
        """adding noise should modify the signal"""
        t, clean = self.gen.generate_cw(duration=0.001)
        noisy = self.gen.add_noise(clean, snr_db=10)

        # noisy signal should be different from clean
        assert not np.array_equal(clean, noisy)

    def test_add_noise_preserves_length(self):
        """noise should not change signal length"""
        t, signal = self.gen.generate_cw(duration=0.001)
        noisy = self.gen.add_noise(signal, snr_db=20)
        assert len(noisy) == len(signal)

    def test_spectrum_output(self):
        """spectrum should return frequency and magnitude arrays"""
        t, signal = self.gen.generate_cw(duration=0.01)
        freqs, spectrum = self.gen.get_spectrum(signal)

        assert len(freqs) == len(spectrum)
        assert len(freqs) > 0

    def test_different_sample_rates(self):
        """should work with different sample rates"""
        gen_low = RadarSignalGenerator(frequency=1e6, sample_rate=5e6)
        gen_high = RadarSignalGenerator(frequency=1e6, sample_rate=20e6)

        t1, s1 = gen_low.generate_cw(0.001)
        t2, s2 = gen_high.generate_cw(0.001)

        # higher sample rate = more samples for same duration
        assert len(s2) > len(s1)

    def test_time_array(self):
        """time array should start at 0 and have correct spacing"""
        t = self.gen.generate_time_array(0.001)
        assert t[0] == 0.0
        # check spacing is 1/sample_rate
        dt = t[1] - t[0]
        assert abs(dt - 1.0 / self.gen.sample_rate) < 1e-15
