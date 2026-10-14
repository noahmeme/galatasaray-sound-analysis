from pathlib import Path

import numpy as np
import pytest

from stadium_audio import AudioRecording, STFTConfig, compute_spectrogram, detect_repeated_patterns


def spectrogram(samples):
    return compute_spectrogram(AudioRecording(np.asarray(samples), 8000, Path("synthetic.wav"), 20),
                               STFTConfig(256, 64))


def test_repeated_phrase_matches_are_nonoverlapping_and_in_source_time():
    rng = np.random.default_rng(42)
    t = np.arange(4096) / 8000
    phrase = np.sin(2*np.pi*(1400*t + 1000*t*t)) * np.sin(np.pi*t / t[-1])**2
    samples = np.tile(phrase, 4) + rng.normal(0, 0.0001, 4*phrase.size)
    result = detect_repeated_patterns(spectrogram(samples), window_seconds=0.3,
                                      min_gap_seconds=0.05, max_frequency=3500, threshold=0.99)
    assert result.matches
    assert result.tested_pairs > 0
    assert result.window_seconds >= 0.3
    for match in result.matches:
        assert match.first_start_seconds >= 20
        assert match.second_start_seconds >= match.first_start_seconds + match.duration_seconds + 0.05 - 1e-9
        assert 0.99 <= match.similarity <= 1
    best = result.matches[0]
    # The true phrase repeats every 4096/8000 seconds, at one or more periods.
    periods = (best.second_start_seconds - best.first_start_seconds) / (4096 / 8000)
    assert periods == pytest.approx(round(periods), abs=0.02)
    assert np.isnan(np.diag(result.similarity)).all()


def test_silence_stationary_tone_and_noise_do_not_match_at_high_threshold():
    rng = np.random.default_rng(123)
    t = np.arange(16000) / 8000
    for samples in [np.zeros(16000), np.sin(2*np.pi*1500*t), rng.normal(0, 0.1, t.size)]:
        result = detect_repeated_patterns(spectrogram(samples), window_seconds=0.3,
                                          max_frequency=3500, threshold=0.99)
        assert not result.matches


def test_pattern_search_limits_and_parameter_errors():
    spec = spectrogram(np.zeros(8000))
    with pytest.raises(ValueError, match="limit"):
        detect_repeated_patterns(spec, max_frames=10)
    with pytest.raises(ValueError, match="too short"):
        detect_repeated_patterns(spec, window_seconds=2)
    with pytest.raises(ValueError, match="two STFT bins"):
        detect_repeated_patterns(spec, min_frequency=1500, max_frequency=1501)
    with pytest.raises(ValueError):
        detect_repeated_patterns(spec, threshold=1.1)
