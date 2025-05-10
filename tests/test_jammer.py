"""
Tests for jamming module.
Testing different jamming techniques and JSR calculations.
"""

import pytest
import numpy as np
from src.signals.radar_generator import RadarSignalGenerator
from src.jamming.jammer import JammingGenerator, ElectronicCountermeasures


class TestJammingGenerator:
    """test each jamming technique"""

    def setup_method(self):
        self.gen = RadarSignalGenerator(frequency=1e6, sample_rate=10e6)
        self.jammer = JammingGenerator(sample_rate=10e6)

        # create a simple radar chirp to jam
        _, self.radar_signal = self.gen.generate_lfm_chirp(
            duration=100e-6, bandwidth=1e6
        )

    def test_noise_jamming_returns_signal(self):
        """noise jamming should return time and signal"""
        t, jammed = self.jammer.noise_jamming(
            duration=100e-6, bandwidth=1e6
        )
        assert len(jammed) > 0
        assert len(t) == len(jammed)

    def test_noise_jamming_not_zero(self):
        """noise should actually have some power"""
        t, jammed = self.jammer.noise_jamming(
            duration=100e-6, bandwidth=1e6
        )
        assert np.mean(np.abs(jammed)) > 0

    def test_spot_jamming_returns_signal(self):
        """spot jamming should return time and signal"""
        t, jammed = self.jammer.spot_jamming(
            duration=100e-6, frequency=1e6
        )
        assert len(jammed) > 0
        assert len(t) == len(jammed)

    def test_spot_jamming_is_narrowband(self):
        """spot jam should have fairly constant amplitude"""
        t, jammed = self.jammer.spot_jamming(
            duration=100e-6, frequency=1e6
        )
        # single tone should have consistent magnitude
        assert np.std(np.abs(jammed)) < np.mean(np.abs(jammed))

    def test_sweep_jamming_returns_signal(self):
        """sweep should return a signal"""
        t, jammed = self.jammer.sweep_jamming(
            duration=100e-6, start_freq=500e3, end_freq=2e6
        )
        assert len(jammed) > 0

    def test_deceptive_jamming_length(self):
        """deceptive jam output should match radar signal length"""
        jammed = self.jammer.deceptive_jamming(
            self.radar_signal, num_false_targets=3
        )
        assert len(jammed) == len(self.radar_signal)

    def test_deceptive_jamming_has_content(self):
        """deceptive jam should not be all zeros"""
        jammed = self.jammer.deceptive_jamming(
            self.radar_signal, num_false_targets=2
        )
        assert np.mean(np.abs(jammed)) > 0

    def test_drfm_jamming_length(self):
        """DRFM output should match input length"""
        jammed = self.jammer.drfm_jamming(
            self.radar_signal, doppler_shift_hz=1000,
            time_delay_s=1e-6
        )
        assert len(jammed) == len(self.radar_signal)

    def test_drfm_with_no_modification(self):
        """DRFM with no shift should still produce output"""
        jammed = self.jammer.drfm_jamming(
            self.radar_signal, doppler_shift_hz=0, time_delay_s=0
        )
        assert np.mean(np.abs(jammed)) > 0

    def test_barrage_jamming(self):
        """barrage jamming should cover multiple bands"""
        bands = [(1e6, 500e3), (3e6, 500e3)]
        t, jammed = self.jammer.barrage_jamming(
            duration=100e-6, freq_bands=bands
        )
        assert len(jammed) > 0

    def test_jsr_calculation(self):
        """JSR should return a float value"""
        jsr = self.jammer.calculate_jsr(
            signal_power=1.0, jammer_power=10.0
        )
        assert isinstance(jsr, float)
        assert not np.isnan(jsr)

    def test_jsr_increases_with_jammer_power(self):
        """more jammer power = higher JSR"""
        jsr_low = self.jammer.calculate_jsr(
            signal_power=1.0, jammer_power=1.0
        )
        jsr_high = self.jammer.calculate_jsr(
            signal_power=1.0, jammer_power=100.0
        )
        assert jsr_high > jsr_low


class TestECM:
    """test the electronic countermeasures wrapper"""

    def setup_method(self):
        self.ecm = ElectronicCountermeasures(sample_rate=10e6)

    def test_ecm_creation(self):
        """ECM should initialize ok"""
        assert self.ecm is not None

    def test_respond_noise(self):
        """ECM noise response should return a signal"""
        gen = RadarSignalGenerator(frequency=1e6, sample_rate=10e6)
        _, radar = gen.generate_lfm_chirp(duration=100e-6, bandwidth=1e6)

        response = self.ecm.respond_to_radar(radar, technique="noise")
        assert len(response) > 0
        assert np.mean(np.abs(response)) > 0

    def test_respond_drfm(self):
        """ECM DRFM response should return a signal"""
        gen = RadarSignalGenerator(frequency=1e6, sample_rate=10e6)
        _, radar = gen.generate_lfm_chirp(duration=100e-6, bandwidth=1e6)

        response = self.ecm.respond_to_radar(radar, technique="drfm")
        assert len(response) > 0

    def test_respond_deceptive(self):
        """ECM deceptive response should return a signal"""
        gen = RadarSignalGenerator(frequency=1e6, sample_rate=10e6)
        _, radar = gen.generate_lfm_chirp(duration=100e-6, bandwidth=1e6)

        response = self.ecm.respond_to_radar(radar, technique="deceptive")
        assert len(response) > 0
