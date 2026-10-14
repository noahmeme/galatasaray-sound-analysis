"""STFT computation is independent of plotting and future comparison methods."""
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import get_window

from .audio import AudioRecording


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 2048
    hop_length: int = 512

    def __post_init__(self):
        if not isinstance(self.n_fft, int) or self.n_fft < 2:
            raise ValueError("n_fft must be an integer of at least 2.")
        if not isinstance(self.hop_length, int) or not 1 <= self.hop_length <= self.n_fft:
            raise ValueError("hop_length must be an integer between 1 and n_fft.")


@dataclass(frozen=True)
class Spectrogram:
    recording: AudioRecording
    config: STFTConfig
    stft: np.ndarray
    amplitude: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray


def compute_spectrogram(recording: AudioRecording,
                        config: STFTConfig | None = None) -> Spectrogram:
    config = config or STFTConfig()
    if recording.samples.size < config.n_fft:
        raise ValueError("Interval is shorter than one STFT window; reduce n_fft or use a longer interval.")
    window = get_window("hann", config.n_fft, fftbins=True)
    transform = librosa.stft(recording.samples, n_fft=config.n_fft,
                            hop_length=config.hop_length, window=window, center=False)
    # Coherent-gain normalization: a bin-centered unit sinusoid has amplitude 1.
    amplitude = np.abs(transform) / window.sum()
    amplitude[1:-1 if config.n_fft % 2 == 0 else None] *= 2
    frequencies = np.fft.rfftfreq(config.n_fft, d=1 / recording.sample_rate)
    # No synthetic edge padding: label each complete frame by its center.
    times = (np.arange(transform.shape[1]) * config.hop_length + config.n_fft / 2)
    times = times / recording.sample_rate + recording.start_seconds
    return Spectrogram(recording, config, transform, amplitude, frequencies, times)


def amplitude_db(result: Spectrogram, *, reference: float = 1.0) -> np.ndarray:
    """20 log10(A/reference), with a numerical floor; no per-recording normalization."""
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Reference amplitude must be finite and positive.")
    return librosa.amplitude_to_db(result.amplitude, ref=reference, amin=1e-10, top_db=None)
