"""Decode audio without peak normalization; retain source time and channel choice."""
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioRecording:
    samples: np.ndarray
    sample_rate: int
    source: Path
    start_seconds: float = 0.0
    channel: str = "mix"

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        if self.samples.ndim != 1 or not self.samples.size:
            raise ValueError("Samples must be a nonempty mono array.")
        if not np.isfinite(self.samples).all():
            raise ValueError("Samples contain NaN or infinity.")
        if not np.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("Start time must be finite and nonnegative.")

    @property
    def duration(self) -> float:
        return self.samples.size / self.sample_rate


def extract_audio(source: str | Path, destination: str | Path, *, stream: int = 0) -> Path:
    """Extract the nth audio stream to float WAV at its native rate/channel count.

    FFmpeg is required (on PATH or provided by the video extra).
    Existing destinations are never overwritten.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    if stream < 0:
        raise ValueError("Audio stream index must be nonnegative.")
    executable = shutil.which("ffmpeg")
    if executable is None:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("FFmpeg is required for video import. Run 'uv sync --extra video' "
                               "or install ffmpeg on PATH.") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stadium-extract-", dir=destination.parent) as folder:
        temporary = Path(folder) / "decoded.wav"
        result = subprocess.run(
            [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
             "-i", str(source), "-map", f"0:a:{stream}", "-vn", "-c:a", "pcm_f32le",
             "-f", "wav", str(temporary)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"FFmpeg extraction failed: {result.stderr.strip()}")
        # Publish a complete file exclusively, including if another writer raced us.
        os.link(temporary, destination)
    return destination


def _read_segment(path: Path, start: float, duration: float | None):
    with sf.SoundFile(path) as handle:
        rate = handle.samplerate
        first = round(start * rate)
        if first >= len(handle):
            raise ValueError("Start time is beyond the recording.")
        handle.seek(first)
        count = -1 if duration is None else round(duration * rate)
        samples = handle.read(frames=count, dtype="float32", always_2d=True)
    return samples, rate, first / rate


def load_recording(source: str | Path, *, start: float = 0,
                   duration: float | None = None, sample_rate: int | None = None,
                   channel: str | int = "mix", stream: int = 0) -> AudioRecording:
    """Read an interval; fall back to FFmpeg for videos/unsupported audio codecs.

    channel='mix' averages channels; an integer selects a zero-based channel.
    Resampling is optional, and does not recover information lost in the source.
    """
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not np.isfinite(start) or start < 0:
        raise ValueError("Start must be finite and nonnegative.")
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("Duration must be finite and positive.")
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError("Target sample rate must be positive.")
    if stream < 0:
        raise ValueError("Audio stream index must be nonnegative.")
    try:
        if stream != 0:
            raise sf.LibsndfileError(0, "Selecting a nondefault stream requires FFmpeg")
        samples, rate, actual_start = _read_segment(source, start, duration)
    except sf.LibsndfileError:
        with tempfile.TemporaryDirectory(prefix="stadium-audio-") as folder:
            decoded = extract_audio(source, Path(folder) / "decoded.wav", stream=stream)
            samples, rate, actual_start = _read_segment(decoded, start, duration)
    if channel == "mix":
        mono = samples.mean(axis=1)
    elif isinstance(channel, int) and 0 <= channel < samples.shape[1]:
        mono = samples[:, channel]
    else:
        raise ValueError(f"Channel must be 'mix' or an index from 0 to {samples.shape[1] - 1}.")
    if not mono.size:
        raise ValueError("The requested interval contains no samples.")
    if sample_rate is not None and sample_rate != rate:
        mono = librosa.resample(mono, orig_sr=rate, target_sr=sample_rate)
        rate = sample_rate
    return AudioRecording(mono, rate, source, actual_start, str(channel))
