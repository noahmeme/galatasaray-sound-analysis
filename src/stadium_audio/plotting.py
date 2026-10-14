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


def plot_psds(results, *, min_frequency=0, max_frequency=None):
    """Overlay density estimates without independent peak normalization."""
    if not results:
        raise ValueError("Provide at least one PSD estimate.")
    nyquist = min(r.recording.sample_rate / 2 for r in results)
    upper = nyquist if max_frequency is None else max_frequency
    if not 0 <= min_frequency < upper <= nyquist:
        raise ValueError("Frequency limits must lie within the common Nyquist range.")
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for result in results:
        recording = result.recording
        db = 10 * np.log10(np.maximum(result.density, 1e-20))
        ax.plot(result.frequencies, db,
                label=f"{recording.source.name} [{recording.start_seconds:g}s, "
                      f"{recording.duration:g}s], ch {recording.channel}")
    ax.set(xlim=(min_frequency, upper), xlabel="Frequency (Hz)",
           ylabel="PSD (dB relative to unit digital amplitude²/Hz)",
           title="Welch power spectral density")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_correlation(result):
    """Undefined coefficients leave gaps; lower panel exposes changing overlap."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             height_ratios=[3, 1], layout="constrained")
    axes[0].plot(result.lags_seconds, result.coefficients, linewidth=0.9)
    axes[0].set(ylim=(-1.05, 1.05), ylabel="Overlap Pearson coefficient",
                title=f"{result.kind.capitalize()} | {result.first.source.name} / "
                      f"{result.second.source.name}")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    if not np.isfinite(result.coefficients).any():
        axes[0].text(0.5, 0.5, "No overlap has sufficient variance for correlation",
                     ha="center", transform=axes[0].transAxes)
    axes[1].plot(result.lags_seconds, result.overlap_seconds)
    axes[1].set(ylabel="Overlap (s)", xlabel="Segment-relative lag (s)")
    if result.kind == "cross-correlation":
        axes[1].set_xlabel("Segment-relative lag (s); positive = content later in second segment")
    for ax in axes:
        ax.grid(alpha=0.25)
    return fig


def plot_pattern_search(result):
    """Show only tested pairs; excluded and undefined comparisons remain blank."""
    step = result.spectrogram.config.hop_length / result.spectrogram.recording.sample_rate
    edges = np.r_[result.starts_seconds - step / 2, result.starts_seconds[-1] + step / 2]
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    image = ax.pcolormesh(edges, edges, np.ma.masked_invalid(result.similarity),
                          cmap="coolwarm", vmin=-1, vmax=1, shading="flat", rasterized=True)
    for match in result.matches:
        ax.plot(match.second_start_seconds, match.first_start_seconds,
                "o", markerfacecolor="none", markeredgecolor="black", markersize=7)
    ax.set(xlabel="Window start in source recording (s)",
           ylabel="Window start in source recording (s)",
           title=f"{result.spectrogram.recording.source.name} | "
                 f"{result.frequency_limits[0]:g}–{result.frequency_limits[1]:g} Hz\n"
                 f"Spectral sequence similarity | {result.window_seconds:.3f}s windows\n"
                 f"{len(result.matches)} candidates ≥ {result.threshold:g}; "
                 f"{result.tested_pairs} valid pairs tested")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, label="Cosine similarity (not probability)")
    return fig
