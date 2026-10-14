"""Generate labeled simulations for inspecting STFT behavior, not a classifier.

Run from the project root: uv run --extra video python examples/synthetic_demo.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from stadium_audio import STFTConfig, compute_spectrogram, load_recording
from stadium_audio.plotting import plot_spectrograms


def main():
    folder = Path("outputs/demo")
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    sr = 24000
    t = np.arange(6 * sr) / sr
    # Independent, drifting whistle-like oscillators with slowly varying envelopes.
    irregular = np.zeros_like(t)
    for _ in range(8):
        f = rng.uniform(1800, 4300) + rng.uniform(40, 200) * np.sin(
            2 * np.pi * rng.uniform(0.3, 1.7) * t + rng.uniform(0, 2 * np.pi))
        phase = 2 * np.pi * np.cumsum(f) / sr
        envelope = (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 0.8) * t
                                     + rng.uniform(0, 2 * np.pi))) ** 4
        irregular += 0.045 * envelope * np.sin(phase)
    irregular += rng.normal(0, 0.015, t.size)
    # Repeat the exact same one-second chirping phrase six times, under new noise.
    u = np.arange(sr) / sr
    phrase = 0.16 * np.sin(2 * np.pi * (2400 * u + 200 * u**2)) * np.sin(np.pi * u)**2
    looped = np.tile(phrase, 6) + rng.normal(0, 0.015, t.size)
    paths = [folder / "synthetic_irregular.wav", folder / "synthetic_loop.wav"]
    for path, samples in zip(paths, [irregular, looped]):
        sf.write(path, samples, sr, subtype="FLOAT")
    results = [compute_spectrogram(load_recording(path), STFTConfig()) for path in paths]
    fig = plot_spectrograms(results, min_frequency=1000, max_frequency=6000)
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
    print((folder / "comparison.png").resolve())


if __name__ == "__main__":
    main()
