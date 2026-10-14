# Stadium sound analysis

A Python project for investigating recorded whistling at football matches, including allegations concerning Galatasaray. The first increment imports audio/video, computes STFTs, and compares spectrograms. It does not automatically classify natural or artificial sound, and no match recordings have been analyzed yet.

## Setup

Use [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer (3.12 recommended):

```sh
uv sync --extra video
```

`uv.lock` pins the resolved dependencies; `uv sync` installs the project and development tools in `.venv`. The `video` extra supplies a packaged **FFmpeg** executable through imageio-ffmpeg. An FFmpeg already on PATH takes precedence. WAV/FLAC and formats supported by SoundFile work without this extra; other formats fall back to FFmpeg. Extraction preserves the chosen stream's native sample rate and channels in float WAV, without volume normalization. It cannot undo compression already present in the source.

Put original recordings in `data/raw/`; recordings and generated outputs are excluded from Git. Preserve originals and record match, source URL, recording position, broadcast feed, acquisition date, and relevant match events separately.

## Commands

Extract a reusable audio file from a video (existing destinations are refused):

```sh
uv run --extra video stadium-audio extract data/raw/match.mp4 data/raw/match.wav
```

Plot a 20-second interval starting five minutes into the recording:

```sh
uv run --extra video stadium-audio spectrogram data/raw/match.wav \
  --start 300 --duration 20 --channel 0 \
  --n-fft 2048 --hop-length 512 \
  --min-frequency 1000 --max-frequency 8000 \
  --output outputs/match-whistles.png
```

Video paths can also be passed directly to `spectrogram`; decoding the entire audio stream into a temporary WAV can be slow for a full match, so extract once for repeated work. Segment reads from audio files load only the requested interval. The CLI defaults to 30 seconds to limit memory use. A duration extending past the end is truncated and the actual duration is saved.

Compare multiple recordings with identical analysis parameters and a common sample rate:

```sh
uv run --extra video stadium-audio spectrogram data/raw/match-a.wav data/raw/match-b.wav \
  --start 300 --duration 20 --sample-rate 24000 \
  --max-frequency 10000 --output outputs/comparison.png
```

`--start` and `--duration` apply to every input in a CLI invocation. For different offsets use the Python API below. Times label source recording seconds, not the on-screen match clock. `--channel mix` averages channels; use `--channel 0` or `1` to avoid cancellation or examine microphone differences. `--stream 1` selects the second audio stream of a video. Resampling cannot recover frequencies absent from the original; use a target rate no higher than the lowest original rate for controlled comparisons.

Plots share a frequency range and color scale. `--reference shared-peak` uses the largest spectral amplitude across all panels; `--reference full-scale` uses unit digital amplitude. The default displayed range is -80 to 0 dB; amplitudes above 0 dB are color-clipped in full-scale mode. These values are **not sound pressure levels**. Microphone gain and broadcast processing prevent direct physical loudness comparison. The plot is a coherent-gain-normalized STFT amplitude display, not a power spectral density estimate.

Each image receives a `.png.json` (or equivalent) sidecar containing source paths, actual intervals, channel/stream, sample rates, transform parameters, reference, and dependency versions. Add `--show` to open a Matplotlib window. `uv run --extra video python -m stadium_audio` also works.

## Python API

```python
from stadium_audio import load_recording, compute_spectrogram, STFTConfig
from stadium_audio.plotting import plot_spectrograms

config = STFTConfig(n_fft=2048, hop_length=512)
a = load_recording("data/raw/match-a.wav", start=300, duration=20,
                   sample_rate=24000, channel=0)
b = load_recording("data/raw/match-b.wav", start=725, duration=20,
                   sample_rate=24000, channel=0)
results = [compute_spectrogram(r, config) for r in (a, b)]
fig = plot_spectrograms(results, min_frequency=1000, max_frequency=10000)
fig.savefig("outputs/comparison.png", dpi=160)
# results[0].stft: complex coefficients for subsequent correlation work
# results[0].amplitude, .frequencies, .times: numerical arrays
```

## Demo and verification

```sh
uv run --extra video python examples/synthetic_demo.py
uv run --extra video pytest
```

The demo produces two labeled synthetic signals and `outputs/demo/comparison.png`: irregular drifting oscillators, and an exactly repeated chirp phrase under independent noise. These are teaching examples, not validated models of crowds or evidence about any club.

## Project structure and next increments

- `src/stadium_audio/audio.py`: decoding, segments, channel selection, optional resampling.
- `src/stadium_audio/spectrogram.py`: numerical STFT and amplitude representation.
- `src/stadium_audio/plotting.py`: shared-scale visualization.
- `src/stadium_audio/cli.py`: extraction, plotting, metadata.
- `docs/methodology.md`: Fourier analysis, stochastic-process reasoning, and investigation design.
- `examples/synthetic_demo.py` and `tests/`: reproducible demo and numerical checks.

Next increments will add Welch PSD, normalized autocorrelation, cross-correlation, and repeated spectral-pattern search in separate analysis modules. Their mathematics and pitfalls are outlined in the methodology; they are not implemented in this increment.
