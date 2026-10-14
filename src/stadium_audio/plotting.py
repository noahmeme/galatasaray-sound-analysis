"""Plots use consistent frequency and dB axes across recordings."""
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .spectrogram import Spectrogram, amplitude_db


def plot_spectrograms(results: Sequence[Spectrogram], *, min_frequency: float = 0,
                      max_frequency: float | None = None, dynamic_range: float = 80,
                      reference: str = "shared-peak"):
    if not results:
        raise ValueError("Provide at least one spectrogram.")
    nyquist = min(r.recording.sample_rate / 2 for r in results)
    upper = nyquist if max_frequency is None else max_frequency
    if not 0 <= min_frequency < upper <= nyquist:
        raise ValueError(f"Frequency limits must lie between 0 and {nyquist:g} Hz.")
    if not np.isfinite(dynamic_range) or dynamic_range <= 0:
        raise ValueError("Dynamic range must be finite and positive.")
    if reference not in {"shared-peak", "full-scale"}:
        raise ValueError("Reference must be 'shared-peak' or 'full-scale'.")
    peak = max(float(np.max(r.amplitude)) for r in results)
    ref = max(peak, 1e-10) if reference == "shared-peak" and peak > 0 else 1.0
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.6 * len(results)),
                             squeeze=False, layout="constrained")
    for ax, result in zip(axes[:, 0], results):
        step = result.config.hop_length / result.recording.sample_rate
        bin_width = result.recording.sample_rate / result.config.n_fft
        time_edges = np.r_[result.times - step / 2, result.times[-1] + step / 2]
        frequency_edges = np.r_[result.frequencies - bin_width / 2,
                                result.frequencies[-1] + bin_width / 2]
        frequency_edges = np.clip(frequency_edges, 0, result.recording.sample_rate / 2)
        image = ax.pcolormesh(time_edges, frequency_edges,
                              amplitude_db(result, reference=ref), shading="flat",
                              cmap="magma", vmin=-dynamic_range, vmax=0, rasterized=True)
        ax.set_ylim(min_frequency, upper)
        ax.set_xlabel("Time in source recording (s)")
        ax.set_ylabel("Frequency (Hz)")
        r = result.recording
        ax.set_title(f"{r.source.name} | channel {r.channel} | {r.sample_rate} Hz | "
                     f"Hann {result.config.n_fft}, hop {result.config.hop_length}")
    label = "dB relative to shared peak" if reference == "shared-peak" else "dB relative to unit amplitude"
    fig.colorbar(image, ax=axes[:, 0].tolist(), label=label)
    return fig
