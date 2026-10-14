"""Command-line workflows with images, numerical arrays and provenance sidecars."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from .audio import extract_audio, load_recording
from .correlation import autocorrelation, cross_correlation
from .patterns import detect_repeated_patterns
from .psd import WelchConfig, estimate_psd
from .spectrogram import STFTConfig, compute_spectrogram


def _audio_options(parser, *, multiple=True):
    parser.add_argument("sources", type=Path, nargs="+" if multiple else 1)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--channel", default="mix", help="mix or a zero-based channel index")
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(f"outputs/{parser.prog.split()[-1]}.png"))
    parser.add_argument("--show", action="store_true")


def _frequency_options(parser, *, minimum=0):
    parser.add_argument("--min-frequency", type=float, default=minimum)
    parser.add_argument("--max-frequency", type=float)


def _stft_options(parser):
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)


def _load(args, index, *, second=False):
    start = args.second_start if second and args.second_start is not None else args.start
    duration = args.second_duration if second and args.second_duration is not None else args.duration
    selection = args.second_channel if second and args.second_channel is not None else args.channel
    channel = "mix" if selection == "mix" else int(selection)
    return load_recording(args.sources[index], start=start, duration=duration,
                          sample_rate=args.sample_rate, channel=channel, stream=args.stream)


def _write_results(args, fig, recordings, metadata, arrays):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    array_path = Path(str(args.output) + ".npz")
    np.savez_compressed(array_path, **arrays)
    metadata.update({
        "command": args.command,
        "settings": {k: v for k, v in vars(args).items() if k != "show"},
        "numerical_arrays": str(array_path.resolve()),
        "versions": {name: version(name) for name in
                     ["numpy", "scipy", "matplotlib", "librosa", "soundfile"]},
        "recordings": [{"source": str(r.source), "start_seconds": r.start_seconds,
                        "duration_seconds": r.duration, "sample_rate": r.sample_rate,
                        "channel": r.channel, "audio_stream": args.stream} for r in recordings],
    })
    Path(str(args.output) + ".json").write_text(
        json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n")
    print(args.output.resolve())


def main():
    parser = argparse.ArgumentParser(description="Exploratory stadium audio analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Extract native audio from video with FFmpeg")
    extract.add_argument("source", type=Path)
    extract.add_argument("destination", type=Path)
    extract.add_argument("--stream", type=int, default=0)
    spec = sub.add_parser("spectrogram", help="Plot recordings on a shared scale")
    _audio_options(spec)
    _stft_options(spec)
    _frequency_options(spec)
    spec.add_argument("--dynamic-range", type=float, default=80)
    spec.add_argument("--reference", choices=["shared-peak", "full-scale"], default="shared-peak")
    psd = sub.add_parser("psd", help="Compare Welch power spectral density estimates")
    _audio_options(psd)
    _frequency_options(psd)
    psd.add_argument("--nperseg", type=int, default=2048)
    psd.add_argument("--noverlap", type=int)
    psd.add_argument("--average", choices=["mean", "median"], default="mean")
    auto = sub.add_parser("autocorrelation", help="Search positive-lag waveform repetition")
    _audio_options(auto, multiple=False)
    cross = sub.add_parser("cross-correlation", help="Align exactly two recordings/segments")
    _audio_options(cross)
    cross.add_argument("--second-start", type=float)
    cross.add_argument("--second-duration", type=float)
    cross.add_argument("--second-channel")
    for command in (auto, cross):
        command.add_argument("--max-lag", type=float, default=10)
        command.add_argument("--min-overlap", type=float, default=1)
        command.add_argument("--min-peak-lag", type=float, default=0.05 if command is auto else 0,
                             help="Minimum absolute lag for the reported maximum; does not filter plot")
    patterns = sub.add_parser("patterns", help="Find repeated nonoverlapping spectral sequences")
    _audio_options(patterns, multiple=False)
    _stft_options(patterns)
    _frequency_options(patterns, minimum=1000)
    patterns.add_argument("--window-seconds", type=float, default=1)
    patterns.add_argument("--min-gap", type=float, default=0.25)
    patterns.add_argument("--threshold", type=float, default=0.8)
    patterns.add_argument("--max-matches", type=int, default=10)
    patterns.add_argument("--dynamic-range", type=float, default=60)
    patterns.add_argument("--max-frames", type=int, default=2000)
    args = parser.parse_args()
    try:
        if args.command == "extract":
            print(extract_audio(args.source, args.destination, stream=args.stream))
            return
        if args.command == "cross-correlation" and len(args.sources) != 2:
            raise ValueError("Cross-correlation requires exactly two sources (the same path may be used twice).")
        outputs = {Path(str(args.output) + suffix).resolve() for suffix in ("", ".json", ".npz")}
        if outputs.intersection(source.resolve() for source in args.sources):
            raise ValueError("Output paths must not overwrite an input recording.")
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plotting import (plot_spectrograms, plot_psds, plot_correlation, plot_pattern_search)
        recordings = [_load(args, i, second=args.command == "cross-correlation" and i == 1)
                      for i in range(len(args.sources))]
        metadata, arrays = {}, {}
        if args.command == "spectrogram":
            config = STFTConfig(args.n_fft, args.hop_length)
            results = [compute_spectrogram(r, config) for r in recordings]
            fig = plot_spectrograms(results, min_frequency=args.min_frequency,
                                   max_frequency=args.max_frequency,
                                   dynamic_range=args.dynamic_range, reference=args.reference)
            metadata.update(method="librosa STFT, periodic Hann, center=False, coherent-gain amplitude",
                            n_fft=config.n_fft, hop_length=config.hop_length,
                            reference=args.reference, dynamic_range_db=args.dynamic_range,
                            frequency_limits_hz=[args.min_frequency, args.max_frequency],
                            shared_peak_amplitude=max(float(r.amplitude.max()) for r in results))
            for i, result in enumerate(results):
                arrays.update({f"stft_{i}": result.stft, f"amplitude_{i}": result.amplitude,
                               f"frequencies_{i}": result.frequencies, f"times_{i}": result.times})
        elif args.command == "psd":
            config = WelchConfig(args.nperseg, args.noverlap, args.average)
            results = [estimate_psd(r, config) for r in recordings]
            fig = plot_psds(results, min_frequency=args.min_frequency, max_frequency=args.max_frequency)
            metadata.update(method="Welch, periodic Hann, constant detrend, one-sided density",
                            nperseg=config.nperseg, noverlap=config.overlap, average=config.average,
                            units="digital amplitude squared per Hz",
                            integrated_power=[r.integrated_power for r in results],
                            segment_counts=[r.segment_count for r in results])
            for i, result in enumerate(results):
                arrays.update({f"frequencies_{i}": result.frequencies, f"density_{i}": result.density})
        elif args.command in {"autocorrelation", "cross-correlation"}:
            options = dict(max_lag_seconds=args.max_lag, min_overlap_seconds=args.min_overlap)
            result = (autocorrelation(recordings[0], **options) if args.command == "autocorrelation"
                      else cross_correlation(recordings[0], recordings[1], **options))
            peak = result.strongest(min_abs_lag=args.min_peak_lag)
            fig = plot_correlation(result)
            metadata.update(method="FFT waveform correlation, overlap-specific Pearson normalization",
                            lag_convention="Positive lag means content later in the second segment; "
                                           "source time offsets excluded",
                            strongest_absolute_match=peak,
                            undefined_lags=int(np.count_nonzero(~np.isfinite(result.coefficients))))
            arrays.update(lags_seconds=result.lags_seconds, coefficients=result.coefficients,
                          overlap_seconds=result.overlap_seconds)
        else:
            spectrogram = compute_spectrogram(recordings[0], STFTConfig(args.n_fft, args.hop_length))
            result = detect_repeated_patterns(
                spectrogram, window_seconds=args.window_seconds, min_gap_seconds=args.min_gap,
                min_frequency=args.min_frequency, max_frequency=args.max_frequency,
                threshold=args.threshold, max_matches=args.max_matches,
                dynamic_range_db=args.dynamic_range, max_frames=args.max_frames)
            fig = plot_pattern_search(result)
            metadata.update(method="Aligned cosine similarity of doubly centered log-spectral sequences",
                            actual_window_seconds=result.window_seconds, window_frames=result.window_frames,
                            frequency_limits_hz=result.frequency_limits, tested_pairs=result.tested_pairs,
                            candidates=[asdict(match) for match in result.matches],
                            score_interpretation="Similarity, not a probability or significance estimate")
            arrays.update(starts_seconds=result.starts_seconds, similarity=result.similarity)
        _write_results(args, fig, recordings, metadata, arrays)
        if args.show:
            plt.show()
        plt.close(fig)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
