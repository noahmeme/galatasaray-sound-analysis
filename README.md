# Stadium sound analysis

A Python project for investigating recorded whistling at football matches, including allegations concerning Galatasaray. It imports audio/video and provides STFT spectrograms, Welch PSD, normalized waveform autocorrelation and cross-correlation, and repeated spectral-sequence search. It does not automatically classify natural or artificial sound, and no match recordings have been analyzed yet.

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

Each CLI image receives `.png.json` and `.png.npz` (or equivalent) sidecars. The NPZ contains numerical arrays; JSON records the method, parameters and summary, including source paths, actual intervals, channel/stream, sample rates, transform parameters, reference, and dependency versions. Correlation and pattern arrays use NaN for undefined/excluded entries. Load arrays with `np.load("outputs/result.png.npz", allow_pickle=False)`. Add `--show` to open a Matplotlib window. `uv run --extra video python -m stadium_audio` also works.

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

## Welch PSD, correlation and spectral repetition

Compare frequency content using density estimates in digital-amplitude squared per Hz:

```sh
uv run --extra video stadium-audio psd data/raw/match-a.wav data/raw/match-b.wav \
  --start 300 --duration 20 --sample-rate 24000 \
  --nperseg 2048 --noverlap 1024 --average mean --output outputs/psd.png
```

Welch detrends each frame by its mean and averages one-sided Hann-window periodograms. `--average median` is also supported. PSD integration estimates detrended mean-square power; overlapping frame counts are saved but are not independent observation counts. PSD describes average frequency content, not the timing of repetition.

Search waveform recurrence within one interval:

```sh
uv run --extra video stadium-audio autocorrelation data/raw/match.wav \
  --start 300 --duration 20 --max-lag 8 --min-overlap 5 \
  --min-peak-lag 0.2 --output outputs/autocorrelation.png
```

Align two recordings or distinct intervals from the same recording:

```sh
uv run --extra video stadium-audio cross-correlation data/raw/match.wav data/raw/match.wav \
  --start 300 --second-start 725 --duration 20 --second-duration 15 \
  --sample-rate 24000 --max-lag 5 --min-overlap 5 \
  --output outputs/cross-correlation.png
```

Correlation is normalized using means and energies of the overlapping samples at each lag. Positive lag means matching content occurs later within the second selected segment. To obtain the source-time displacement, add the difference between segment starts. Choose a common sample rate; different rates are rejected unless explicitly resampled. Both channels use `--channel`, with an optional `--second-channel` override. Source start/duration defaults apply to both inputs unless overridden.

The JSON reports the strongest absolute coefficient (including negative polarity), its lag and overlap. An undefined result is `null`. Zero-lag autocorrelation is trivial; `--min-peak-lag` excludes it from peak reporting, without changing the plotted data. Tonal oscillations at sub-millisecond lags are not evidence of a multi-second recorded loop. Gain changes are handled by normalization, but microphones, compression and timing drift can reduce waveform agreement.

Search nonoverlapping spectral sequences in one recording:

```sh
uv run --extra video stadium-audio patterns data/raw/match.wav \
  --start 300 --duration 20 --sample-rate 24000 \
  --window-seconds 1 --min-gap 0.25 --min-frequency 1000 --max-frequency 8000 \
  --threshold 0.8 --max-matches 10 --output outputs/patterns.png
```

Patterns compare aligned, log-spectral sequences after gain and stationary-background suppression. The image shows a sequence-similarity matrix; blank areas exclude overlapping windows or undefined features. Circles mark selected candidates, and JSON contains their source start times, actual duration and similarity. These scores are **not probabilities**. The 0.8 default is a starting threshold for exploration, not a validated decision threshold. Defaults limit search to 2000 STFT frames; at 48 kHz and hop 512 a 30-second interval exceeds that limit. Shorten the interval, resample appropriately, or increase hop length; memory and work grow quadratically with frame count.

The Python API exposes the same methods:

```python
from stadium_audio import (
    estimate_psd, WelchConfig, autocorrelation, cross_correlation,
    compute_spectrogram, detect_repeated_patterns,
)

psd = estimate_psd(a, WelchConfig(nperseg=2048))
auto = autocorrelation(a, max_lag_seconds=5, min_overlap_seconds=5)
cross = cross_correlation(a, b, max_lag_seconds=5, min_overlap_seconds=5)
patterns = detect_repeated_patterns(compute_spectrogram(a),
                                    window_seconds=1, max_frequency=8000)
print(cross.strongest())
print(patterns.matches)
# Plot helpers: plot_psds, plot_correlation, plot_pattern_search
```

Read [the methodology](docs/methodology.md) for normalization formulas, stochastic-process assumptions, and interpretation limits. No method alone establishes stadium-speaker playback.

## Demo and verification

```sh
uv run --extra video python examples/analysis_demo.py
uv run --extra video pytest
```

The demo produces two labeled synthetic signals and `outputs/demo/comparison.png`: irregular drifting oscillators, and an exactly repeated chirp phrase under independent noise. The full demo also creates PSD, autocorrelation, cross-correlation and pattern plots plus `analysis.json`. A separate delayed copy has a known 0.25-second delay and a changed gain. Pattern plots compare a broad band with a focused 2300–2900 Hz band selected from the generator's known chirp range. Background noise dilutes the broad-band score, illustrating why a negative thresholded search cannot rule out replay. For real recordings, predefine bands from independent information or controls rather than choosing them to maximize a match. These are teaching examples, not validated models of crowds or evidence about any club.

## Project structure and further work

- `src/stadium_audio/audio.py`: decoding, segments, channel selection, optional resampling.
- `src/stadium_audio/spectrogram.py`: numerical STFT and amplitude representation.
- `src/stadium_audio/psd.py`: Welch density estimates and integrated power.
- `src/stadium_audio/correlation.py`: overlap-normalized waveform correlation.
- `src/stadium_audio/patterns.py`: nonoverlapping spectral sequence search.
- `src/stadium_audio/plotting.py`: spectrogram, PSD, correlation and similarity plots.
- `src/stadium_audio/cli.py`: extraction, plotting, metadata.
- `docs/methodology.md`: Fourier analysis, stochastic-process reasoning, and investigation design.
- `examples/` and `tests/`: reproducible simulations, numerical checks and CLI/video integration checks.

Further work can add band-limited envelope correlation, cross-recording spectral templates, constrained timing-drift alignment, surrogate-based threshold calibration, and annotated real-data validation. The current methods report measurements and candidate recurrences, not natural/artificial labels.
