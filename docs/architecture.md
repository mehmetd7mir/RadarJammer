# RadarJammer - Architecture

## Overview

Electronic warfare simulation that generates radar signals and applies jamming techniques.

## Data Flow

```
Radar Generator ──→ Signal (CW/Pulse/Chirp/FMCW)
                         │
                         ▼
                   Jammer Module
                    ├── Noise Jamming
                    ├── Spot Jamming
                    ├── Sweep Jamming
                    ├── Deceptive (false targets)
                    ├── DRFM (record + replay)
                    └── Barrage (multi-band)
                         │
                         ▼
                  Signal Analyzer ──→ JSR, SNR, Effectiveness
                         │
                         ▼
                   Visualization (spectrum, spectrogram)
```

## Modules

### `src/signals/`
- **radar_generator.py** - Generates radar waveforms: CW (continuous wave), pulsed radar, LFM chirp, and FMCW. Includes noise injection and spectrum calculation.

### `src/jamming/`
- **jammer.py** - Implements 6 jamming techniques. Also includes `ElectronicCountermeasures` wrapper that automatically responds to detected radar signals.

### `src/analysis/`
- **signal_analyzer.py** - Signal analysis tools: SNR calculation, bandwidth estimation, peak frequency detection, jamming effectiveness measurement, and signal type classification.

### `src/visualization/`
- **plots.py** - Matplotlib-based visualization for time-domain signals, frequency spectra, and spectrograms.

## Key Concepts

- **JSR (Jamming-to-Signal Ratio)**: Ratio of jamming power to radar signal power. JSR > 0 dB means jamming is effective
- **DRFM**: Digital RF Memory - records enemy radar signal and retransmits modified copy
- **LFM Chirp**: Linear Frequency Modulation - frequency sweeps linearly over time
