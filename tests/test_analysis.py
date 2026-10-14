from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf

from stadium_audio import AudioRecording, STFTConfig, amplitude_db, compute_spectrogram, load_recording
from stadium_audio.plotting import plot_spectrograms


def tone(frequency=2000, amplitude=0.5, rate=16000, seconds=1):
    t = np.arange(round(rate * seconds)) / rate
    return AudioRecording(amplitude * np.sin(2 * np.pi * frequency * t), rate, Path("tone.wav"), 12)


def test_tone_frequency_amplitude_and_frame_times():
    result = compute_spectrogram(tone(), STFTConfig(1024, 256))
    peak = np.argmax(result.amplitude[:, 0])
    assert result.frequencies[peak] == 2000
    assert result.amplitude[peak, 0] == pytest.approx(0.5, abs=1e-5)
    assert amplitude_db(result)[peak, 0] == pytest.approx(-6.0206, abs=1e-4)
    assert result.times[0] == pytest.approx(12 + 512 / 16000)
    assert result.times[-1] + 512 / 16000 <= 13


def test_silence_is_finite_and_dark():
    quiet = AudioRecording(np.zeros(4096), 16000, Path("silence.wav"))
    result = compute_spectrogram(quiet)
    assert np.isfinite(amplitude_db(result)).all()
    assert np.max(amplitude_db(result)) < -100
    fig = plot_spectrograms([result])
    assert fig.axes[0].collections[0].norm.vmax == 0
    assert np.max(fig.axes[0].collections[0].get_array()) < -100
    plt.close(fig)


def test_shared_scale_preserves_amplitude_difference():
    loud = compute_spectrogram(tone(amplitude=0.5))
    quiet = compute_spectrogram(tone(amplitude=0.05))
    fig = plot_spectrograms([loud, quiet])
    grids = [ax.collections[0] for ax in fig.axes[:2]]
    assert float(np.max(grids[0].get_array())) == pytest.approx(0)
    assert float(np.max(grids[1].get_array())) == pytest.approx(-20, abs=0.01)
    assert grids[0].get_clim() == grids[1].get_clim()
    plt.close(fig)


def test_segment_channels_and_resampling(tmp_path):
    path = tmp_path / "stereo.wav"
    samples = tone(rate=16000, seconds=2).samples
    sf.write(path, np.column_stack([samples, -samples]), 16000, subtype="FLOAT")
    mixed = load_recording(path, start=0.5, duration=0.25)
    left = load_recording(path, start=0.5, duration=0.25, channel=0, sample_rate=8000)
    assert mixed.duration == pytest.approx(0.25)
    assert np.max(np.abs(mixed.samples)) == 0
    assert left.samples.size == 2000
    assert left.start_seconds == 0.5
    assert np.max(np.abs(left.samples)) > 0.4
    with pytest.raises(ValueError, match="Channel"):
        load_recording(path, channel=2)
    with pytest.raises(ValueError, match="beyond"):
        load_recording(path, start=3)


def test_short_interval_and_bad_config():
    with pytest.raises(ValueError, match="shorter"):
        compute_spectrogram(tone(seconds=0.01))
    with pytest.raises(ValueError):
        STFTConfig(1024, 2048)
    with pytest.raises(ValueError):
        plot_spectrograms([compute_spectrogram(tone())], max_frequency=9000)
