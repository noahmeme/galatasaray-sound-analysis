"""Run every method on known synthetic signals; outputs are not evidence about a match.

uv run --extra video python examples/analysis_demo.py
"""
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from synthetic_demo import main as make_signals
from stadium_audio import (STFTConfig, WelchConfig, load_recording, compute_spectrogram,
                           estimate_psd, autocorrelation, cross_correlation,
                           detect_repeated_patterns)
from stadium_audio.plotting import plot_psds, plot_correlation, plot_pattern_search


def save(figure, path):
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    make_signals()
    folder = Path("outputs/demo")
    irregular = load_recording(folder / "synthetic_irregular.wav")
    loop = load_recording(folder / "synthetic_loop.wav")
    save(plot_psds([estimate_psd(r, WelchConfig()) for r in (irregular, loop)],
                    min_frequency=1000, max_frequency=6000), folder / "psd.png")
    auto = autocorrelation(loop, max_lag_seconds=3, min_overlap_seconds=2)
    save(plot_correlation(auto), folder / "autocorrelation.png")
    # This copy reuses both phrase and background noise; the two segments have known delay.
    delay = int(0.25 * loop.sample_rate)
    copied = np.r_[np.zeros(delay), 0.6 * loop.samples + 0.01]
    path = folder / "synthetic_delayed_copy.wav"
    sf.write(path, copied, loop.sample_rate, subtype="FLOAT")
    cross = cross_correlation(loop, load_recording(path), max_lag_seconds=0.5,
                              min_overlap_seconds=2)
    save(plot_correlation(cross), folder / "cross-correlation.png")
    spectrogram = compute_spectrogram(loop, STFTConfig())
    broad = detect_repeated_patterns(spectrogram, window_seconds=0.6,
                                      min_gap_seconds=0.1, max_frequency=6000, threshold=0.8)
    save(plot_pattern_search(broad), folder / "patterns_broad.png")
    # The generator's chirp is known to sweep 2400–2800 Hz. Select its band in
    # advance; unrelated noisy bins dilute the broad-band similarity score.
    patterns = detect_repeated_patterns(spectrogram, window_seconds=0.6,
                                         min_gap_seconds=0.1, min_frequency=2300,
                                         max_frequency=2900, threshold=0.8)
    save(plot_pattern_search(patterns), folder / "patterns.png")
    summary = {"synthetic_only": True,
               "autocorrelation_strongest": auto.strongest(min_abs_lag=0.2),
               "known_copy_delay_seconds": 0.25,
               "cross_correlation_strongest": cross.strongest(),
               "broad_band_hz": broad.frequency_limits,
               "broad_band_candidate_count": len(broad.matches),
               "broad_band_maximum_similarity": float(np.nanmax(broad.similarity)),
               "focused_band_hz": patterns.frequency_limits,
               "pattern_tested_pairs": patterns.tested_pairs,
               "pattern_candidates": [asdict(m) for m in patterns.matches]}
    (folder / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
