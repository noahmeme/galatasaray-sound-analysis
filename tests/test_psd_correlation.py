from pathlib import Path

import numpy as np
import pytest

from stadium_audio import (AudioRecording, WelchConfig, estimate_psd,
                           autocorrelation, cross_correlation)


def recording(values, sr=8000):
    return AudioRecording(np.asarray(values, dtype=float), sr, Path("synthetic.wav"))


@pytest.mark.parametrize("nperseg", [512, 511])
def test_welch_tone_frequency_and_integrated_power(nperseg):
    t = np.arange(16000) / 8000
    result = estimate_psd(recording(0.5 * np.sin(2 * np.pi * 1000 * t)),
                          WelchConfig(nperseg))
    assert result.frequencies[np.argmax(result.density)] == pytest.approx(1000, abs=8000/nperseg)
    assert result.integrated_power == pytest.approx(0.5**2 / 2, rel=0.001)
    assert result.segment_count == 1 + (16000 - nperseg) // (nperseg - nperseg // 2)


def test_welch_noise_variance_and_silence():
    rng = np.random.default_rng(4)
    values = rng.normal(0, 0.2, 80000)
    result = estimate_psd(recording(values), WelchConfig(1024))
    assert result.integrated_power == pytest.approx(np.var(values), rel=0.04)
    quiet = estimate_psd(recording(np.zeros(8000)))
    assert np.all(quiet.density == 0)
    median = estimate_psd(recording(values), WelchConfig(1024, average="median"))
    assert median.integrated_power == pytest.approx(np.var(values), rel=0.1)


def test_welch_bad_config_and_short_signal():
    with pytest.raises(ValueError):
        WelchConfig(128, 128)
    with pytest.raises(ValueError):
        WelchConfig(average="unknown")
    with pytest.raises(ValueError, match="shorter"):
        estimate_psd(recording(np.ones(16)))


def test_correlation_matches_brute_force_for_unequal_lengths():
    x = np.array([1, -3, 2, 4, -2, 1, 0, 3, 4]) + 50
    y = np.array([-2, 4, 1, -1, 2, 3, 0, -5, 2, 2, -3, 4]) - 70
    result = cross_correlation(recording(x, 20), recording(y, 20),
                               min_overlap_seconds=0.15, max_lag_seconds=1)
    for seconds, coefficient, overlap in zip(result.lags_seconds, result.coefficients,
                                             result.overlap_seconds):
        lag = round(seconds * 20)
        xs, ys = max(0, -lag), max(0, lag)
        count = min(x.size - xs, y.size - ys)
        expected = np.corrcoef(x[xs:xs+count], y[ys:ys+count])[0, 1]
        assert coefficient == pytest.approx(expected, abs=1e-12)
        assert overlap == pytest.approx(count / 20)


def test_delayed_copy_gain_offset_polarity_and_lag_sign():
    rng = np.random.default_rng(11)
    x = rng.normal(size=2000)
    y = np.r_[np.zeros(173), -3*x + 5, np.zeros(61)]
    result = cross_correlation(recording(x, 1000), recording(y, 1000),
                               max_lag_seconds=0.5, min_overlap_seconds=1)
    peak = result.strongest()
    assert peak["lag_seconds"] == pytest.approx(0.173)
    assert peak["coefficient"] == pytest.approx(-1)
    assert peak["overlap_seconds"] == 2
    reverse = cross_correlation(recording(y, 1000), recording(x, 1000),
                                max_lag_seconds=0.5, min_overlap_seconds=1)
    assert reverse.strongest()["lag_seconds"] == pytest.approx(-0.173)


def test_autocorrelation_loop_and_noise_control():
    rng = np.random.default_rng(12)
    phrase = rng.normal(size=1000)
    loop = recording(np.tile(phrase, 5), 1000)
    result = autocorrelation(loop, max_lag_seconds=2.5, min_overlap_seconds=1)
    assert result.lags_seconds.min() == 0
    assert result.coefficients[0] == pytest.approx(1)
    for lag in (1, 2):
        assert result.coefficients[round(lag*1000)] == pytest.approx(1)
    noise = autocorrelation(recording(rng.normal(size=5000), 1000),
                            max_lag_seconds=2.5, min_overlap_seconds=1)
    assert abs(noise.strongest(min_abs_lag=0.1)["coefficient"]) < 0.1


def test_correlation_silence_and_invalid_inputs():
    quiet = recording(np.zeros(2000), 1000)
    result = autocorrelation(quiet, max_lag_seconds=1)
    assert np.isnan(result.coefficients).all()
    assert result.strongest() is None
    constant = recording(np.ones(2000), 1000)
    assert cross_correlation(quiet, constant).strongest() is None
    with pytest.raises(ValueError, match="equal sample rates"):
        cross_correlation(quiet, recording(np.zeros(2000), 2000))
    with pytest.raises(ValueError, match="Minimum overlap"):
        autocorrelation(quiet, min_overlap_seconds=3)
    with pytest.raises(ValueError):
        autocorrelation(quiet, max_lag_seconds=-1)
