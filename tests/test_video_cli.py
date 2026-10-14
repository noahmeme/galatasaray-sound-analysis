"""Exercise a real video container, decoder, plotting command, and metadata export."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from stadium_audio import extract_audio, load_recording


def test_video_import_and_cli(tmp_path):
    executable = shutil.which("ffmpeg")
    if executable is None:
        executable = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    sr = 16000
    t = np.arange(sr) / sr
    wave = 0.3 * np.sin(2 * np.pi * 2000 * t)
    audio = tmp_path / "original.wav"
    video = tmp_path / "test video.mp4"
    decoded = tmp_path / "decoded.wav"
    sf.write(audio, np.column_stack([wave, wave * 0.5]), sr, subtype="FLOAT")
    subprocess.run([executable, "-nostdin", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=black:s=64x64:r=10:d=1", "-i", str(audio),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                    "-c:a", "aac", "-shortest", str(video)], check=True)
    extract_audio(video, decoded)
    info = sf.info(decoded)
    assert info.samplerate == sr
    assert info.channels == 2
    with pytest.raises(FileExistsError):
        extract_audio(video, decoded)
    recording = load_recording(video, start=0.1, duration=0.5, channel=0)
    assert recording.sample_rate == sr
    assert recording.duration == pytest.approx(0.5)
    assert np.max(np.abs(recording.samples)) > 0.1
    output = tmp_path / "spectrogram.png"
    subprocess.run([sys.executable, "-m", "stadium_audio", "spectrogram", str(video),
                    "--start", "0.1", "--duration", "0.5", "--channel", "0",
                    "--output", str(output)], check=True, capture_output=True, text=True)
    assert output.stat().st_size > 10000
    metadata = json.loads(Path(str(output) + ".json").read_text())
    assert metadata["recordings"][0]["duration_seconds"] == pytest.approx(0.5)
    assert metadata["recordings"][0]["source"] == str(video)
    assert metadata["n_fft"] == 2048
    assert metadata["versions"]["librosa"]
