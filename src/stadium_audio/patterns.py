"""Find candidate recurring, nonoverlapping log-spectral sequences."""
from dataclasses import dataclass

import numpy as np

from .spectrogram import Spectrogram


@dataclass(frozen=True)
class PatternMatch:
    first_start_seconds: float
    second_start_seconds: float
    duration_seconds: float
    similarity: float


@dataclass(frozen=True)
class PatternSearch:
    spectrogram: Spectrogram
    starts_seconds: np.ndarray
    similarity: np.ndarray
    matches: tuple[PatternMatch, ...]
    window_frames: int
    window_seconds: float
    min_gap_seconds: float
    frequency_limits: tuple[float, float]
    threshold: float
    dynamic_range_db: float
    tested_pairs: int


def _rolling_sum(values: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    return cumulative[width:] - cumulative[:-width]


def detect_repeated_patterns(spectrogram: Spectrogram, *, window_seconds: float = 1,
                             min_gap_seconds: float = 0.25, min_frequency: float = 1000,
                             max_frequency: float | None = None, threshold: float = 0.8,
                             max_matches: int = 10, dynamic_range_db: float = 60,
                             max_frames: int = 2000) -> PatternSearch:
    """Cosine similarity of stacked, background-centered log-spectral features.

    Remove each frame's mean log amplitude (gain invariance), then each frequency
    bin's temporal mean (stationary-background suppression). Compare aligned
    windows without time warping. Scores are candidates, not probabilities.
    max_frames bounds the quadratic matrix size; shorten the interval or increase
    the hop rather than silently dropping frames.
    """
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive.")
    if not np.isfinite(min_gap_seconds) or min_gap_seconds < 0:
        raise ValueError("min_gap_seconds must be finite and nonnegative.")
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must lie between -1 and 1.")
    if not isinstance(max_matches, int) or max_matches < 1:
        raise ValueError("max_matches must be a positive integer.")
    if not isinstance(max_frames, int) or max_frames < 2:
        raise ValueError("max_frames must be an integer of at least 2.")
    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0:
        raise ValueError("dynamic_range_db must be finite and positive.")
    r, config = spectrogram.recording, spectrogram.config
    upper = r.sample_rate / 2 if max_frequency is None else max_frequency
    if not 0 <= min_frequency < upper <= r.sample_rate / 2:
        raise ValueError("Frequency limits must lie within the recording's Nyquist range.")
    band = (spectrogram.frequencies >= min_frequency) & (spectrogram.frequencies <= upper)
    if np.count_nonzero(band) < 2:
        raise ValueError("Frequency band must contain at least two STFT bins.")
    frames = spectrogram.amplitude.shape[1]
    if frames > max_frames:
        raise ValueError(f"Pattern search has {frames} frames (limit {max_frames}); "
                         "shorten the interval or increase hop_length.")
    width = 1 + int(np.ceil(max(0, window_seconds * r.sample_rate - config.n_fft)
                           / config.hop_length))
    actual_duration = ((width - 1) * config.hop_length + config.n_fft) / r.sample_rate
    separation = int(np.ceil((actual_duration + min_gap_seconds) * r.sample_rate
                             / config.hop_length))
    size = frames - width + 1
    if size <= separation:
        raise ValueError("Interval is too short for two nonoverlapping pattern windows with this gap.")
    amplitudes = spectrogram.amplitude[band].astype(np.float64)
    peak = float(amplitudes.max())
    # Relative floor prevents low-amplitude numerical leakage dominating log features.
    floor = max(peak * 10 ** (-dynamic_range_db / 20), np.finfo(float).tiny)
    features = 20 * np.log10(np.maximum(amplitudes, floor) / max(peak, floor))
    features -= features.mean(axis=0, keepdims=True)
    features -= features.mean(axis=1, keepdims=True)
    energies = _rolling_sum(np.sum(features**2, axis=0), width)
    raw_energy = _rolling_sum(np.sum(amplitudes**2, axis=0), width)
    usable = (energies > 1e-10) & (raw_energy > max(float(raw_energy.max()) * 1e-6, 0))
    frame_products = features.T @ features
    scores = np.full((size, size), np.nan, dtype=np.float64)
    for lag in range(separation, size):
        products = _rolling_sum(np.diagonal(frame_products, offset=lag), width)
        indices = np.arange(size - lag)
        other = indices + lag
        valid = usable[indices] & usable[other]
        denominator = np.sqrt(energies[indices] * energies[other])
        values = np.full(indices.size, np.nan)
        np.divide(products, denominator, out=values, where=valid)
        values[valid] = np.clip(values[valid], -1, 1)
        scores[indices, other] = values
        scores[other, indices] = values
    starts = spectrogram.times[:size] - config.n_fft / (2 * r.sample_rate)
    rows, cols = np.where(np.triu(np.isfinite(scores) & (scores >= threshold), k=1))
    order = np.argsort(scores[rows, cols])[::-1]
    matches = []
    chosen = []
    # Suppress nearby duplicate alignments of the same pair; distinct loop cycles remain.
    radius = max(1, width // 2)
    for index in order:
        i, j = int(rows[index]), int(cols[index])
        if any(abs(i - a) < radius and abs(j - b) < radius for a, b in chosen):
            continue
        chosen.append((i, j))
        matches.append(PatternMatch(float(starts[i]), float(starts[j]), actual_duration,
                                    float(scores[i, j])))
        if len(matches) == max_matches:
            break
    return PatternSearch(spectrogram, starts, scores, tuple(matches), width,
                         actual_duration, min_gap_seconds, (min_frequency, upper), threshold,
                         dynamic_range_db, int(np.count_nonzero(np.isfinite(scores)) // 2))
