"""Overlap-normalized Pearson correlation; FFT numerator, cumulative-sum moments."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags

from .audio import AudioRecording


@dataclass(frozen=True)
class CorrelationResult:
    first: AudioRecording
    second: AudioRecording
    kind: str
    lags_seconds: np.ndarray
    coefficients: np.ndarray
    overlap_seconds: np.ndarray
    min_overlap_seconds: float

    def strongest(self, *, min_abs_lag: float = 0) -> dict | None:
        """Largest absolute coefficient (including polarity inversion), or None.

        This is a descriptive maximum, not a calibrated significance estimate.
        """
        if not np.isfinite(min_abs_lag) or min_abs_lag < 0:
            raise ValueError("min_abs_lag must be finite and nonnegative.")
        valid = np.isfinite(self.coefficients) & (np.abs(self.lags_seconds) >= min_abs_lag)
        indices = np.flatnonzero(valid)
        if not indices.size:
            return None
        index = indices[np.argmax(np.abs(self.coefficients[indices]))]
        return {"lag_seconds": float(self.lags_seconds[index]),
                "coefficient": float(self.coefficients[index]),
                "overlap_seconds": float(self.overlap_seconds[index])}


def _interval_sums(values, start, count):
    cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    return cumulative[start + count] - cumulative[start]


def cross_correlation(first: AudioRecording, second: AudioRecording, *,
                      max_lag_seconds: float = 10,
                      min_overlap_seconds: float = 1) -> CorrelationResult:
    """Positive lag means matching content occurs later in the second segment.

    Segment-relative lag excludes source start offsets. At each lag, subtract the
    means and divide by the standard deviations of the overlapping samples only.
    Undefined (constant/silent) overlaps are NaN; not false perfect matches.
    """
    if first.sample_rate != second.sample_rate:
        raise ValueError("Correlation requires equal sample rates; resample explicitly when loading.")
    if not np.isfinite(max_lag_seconds) or max_lag_seconds < 0:
        raise ValueError("max_lag_seconds must be finite and nonnegative.")
    if not np.isfinite(min_overlap_seconds) or min_overlap_seconds <= 0:
        raise ValueError("min_overlap_seconds must be finite and positive.")
    sr = first.sample_rate
    minimum = max(2, int(np.ceil(min_overlap_seconds * sr)))
    if minimum > min(first.samples.size, second.samples.size):
        raise ValueError("Minimum overlap exceeds the shorter recording.")
    # Global centering and scaling improve cumulative-sum numerical stability;
    # the overlap-specific centering below still defines the actual statistic.
    x = first.samples.astype(np.float64) - np.mean(first.samples, dtype=np.float64)
    y = second.samples.astype(np.float64) - np.mean(second.samples, dtype=np.float64)
    x /= max(float(np.max(np.abs(x))), np.finfo(float).tiny)
    y /= max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    lags = correlation_lags(y.size, x.size, mode="full")
    xs = np.maximum(0, -lags)
    ys = np.maximum(0, lags)
    count = np.minimum(x.size - xs, y.size - ys)
    valid = (count >= minimum) & (np.abs(lags) <= int(np.floor(max_lag_seconds * sr)))
    products = correlate(y, x, mode="full", method="fft")[valid]
    lags, xs, ys, count = (a[valid] for a in (lags, xs, ys, count))
    sx, sy = _interval_sums(x, xs, count), _interval_sums(y, ys, count)
    ex, ey = _interval_sums(x*x, xs, count), _interval_sums(y*y, ys, count)
    vx, vy = np.maximum(0, ex - sx*sx/count), np.maximum(0, ey - sy*sy/count)
    denominator = np.sqrt(vx * vy)
    # Very low residual variance cannot be distinguished from sum-rounding error.
    usable = (vx > 1e-12 * np.maximum(ex, 1)) & (vy > 1e-12 * np.maximum(ey, 1))
    coefficients = np.full(lags.size, np.nan)
    np.divide(products - sx*sy/count, denominator, out=coefficients, where=usable)
    coefficients[usable] = np.clip(coefficients[usable], -1, 1)
    return CorrelationResult(first, second, "cross-correlation", lags / sr,
                             coefficients, count / sr, min_overlap_seconds)


def autocorrelation(recording: AudioRecording, *, max_lag_seconds: float = 10,
                    min_overlap_seconds: float = 1) -> CorrelationResult:
    """Nonnegative-lag waveform autocorrelation; lag zero is a trivial reference."""
    result = cross_correlation(recording, recording, max_lag_seconds=max_lag_seconds,
                               min_overlap_seconds=min_overlap_seconds)
    positive = result.lags_seconds >= 0
    return CorrelationResult(recording, recording, "autocorrelation",
                             result.lags_seconds[positive], result.coefficients[positive],
                             result.overlap_seconds[positive], min_overlap_seconds)
