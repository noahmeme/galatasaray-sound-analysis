"""Welch estimates in digital-amplitude squared per Hz."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window, welch

from .audio import AudioRecording


@dataclass(frozen=True)
class WelchConfig:
    nperseg: int = 2048
    noverlap: int | None = None
    average: str = "mean"

    def __post_init__(self):
        if not isinstance(self.nperseg, int) or self.nperseg < 2:
            raise ValueError("nperseg must be an integer of at least 2.")
        if self.noverlap is not None and (
            not isinstance(self.noverlap, int) or not 0 <= self.noverlap < self.nperseg
        ):
            raise ValueError("noverlap must be an integer from 0 to nperseg - 1.")
        if self.average not in {"mean", "median"}:
            raise ValueError("average must be 'mean' or 'median'.")

    @property
    def overlap(self) -> int:
        return self.nperseg // 2 if self.noverlap is None else self.noverlap


@dataclass(frozen=True)
class PowerSpectralDensity:
    recording: AudioRecording
    config: WelchConfig
    frequencies: np.ndarray
    density: np.ndarray
    segment_count: int

    @property
    def integrated_power(self) -> float:
        """Sum density times bin width; includes DC/Nyquist bins once."""
        return float(self.density.sum() * self.recording.sample_rate / self.config.nperseg)


def estimate_psd(recording: AudioRecording,
                 config: WelchConfig | None = None) -> PowerSpectralDensity:
    config = config or WelchConfig()
    if recording.samples.size < config.nperseg:
        raise ValueError("Interval is shorter than nperseg; reduce nperseg or use a longer interval.")
    window = get_window("hann", config.nperseg, fftbins=True)
    frequencies, density = welch(
        recording.samples.astype(np.float64), fs=recording.sample_rate,
        window=window, nperseg=config.nperseg, noverlap=config.overlap,
        detrend="constant", return_onesided=True, scaling="density", average=config.average,
    )
    count = 1 + (recording.samples.size - config.nperseg) // (config.nperseg - config.overlap)
    return PowerSpectralDensity(recording, config, frequencies, density, count)
