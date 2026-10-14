import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .audio import extract_audio, load_recording
from .spectrogram import STFTConfig, compute_spectrogram


def main():
    parser = argparse.ArgumentParser(description="Exploratory stadium audio analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Extract native audio from video with FFmpeg")
    extract.add_argument("source", type=Path)
    extract.add_argument("destination", type=Path)
    extract.add_argument("--stream", type=int, default=0)
    spec = sub.add_parser("spectrogram", help="Plot one or more recordings on a shared scale")
    spec.add_argument("sources", type=Path, nargs="+")
    spec.add_argument("--start", type=float, default=0)
    spec.add_argument("--duration", type=float, default=30)
    spec.add_argument("--sample-rate", type=int)
    spec.add_argument("--channel", default="mix", help="mix or a zero-based channel index")
    spec.add_argument("--stream", type=int, default=0)
    spec.add_argument("--n-fft", type=int, default=2048)
    spec.add_argument("--hop-length", type=int, default=512)
    spec.add_argument("--min-frequency", type=float, default=0)
    spec.add_argument("--max-frequency", type=float)
    spec.add_argument("--dynamic-range", type=float, default=80)
    spec.add_argument("--reference", choices=["shared-peak", "full-scale"], default="shared-peak")
    spec.add_argument("--output", type=Path, default=Path("outputs/spectrogram.png"))
    spec.add_argument("--show", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "extract":
            print(extract_audio(args.source, args.destination, stream=args.stream))
            return
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plotting import plot_spectrograms
        channel = "mix" if args.channel == "mix" else int(args.channel)
        output_paths = {args.output.resolve(),
                        args.output.with_suffix(args.output.suffix + ".json").resolve()}
        if output_paths.intersection(source.resolve() for source in args.sources):
            raise ValueError("Output paths must not overwrite an input recording.")
        config = STFTConfig(args.n_fft, args.hop_length)
        results = [compute_spectrogram(load_recording(
            source, start=args.start, duration=args.duration, sample_rate=args.sample_rate,
            channel=channel, stream=args.stream), config) for source in args.sources]
        fig = plot_spectrograms(results, min_frequency=args.min_frequency,
                               max_frequency=args.max_frequency,
                               dynamic_range=args.dynamic_range, reference=args.reference)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        metadata = {
            "method": "librosa STFT, periodic Hann, center=False, coherent-gain amplitude",
            "n_fft": config.n_fft, "hop_length": config.hop_length,
            "reference": args.reference, "dynamic_range_db": args.dynamic_range,
            "frequency_limits_hz": [args.min_frequency, args.max_frequency],
            "shared_peak_amplitude": max(float(np.max(r.amplitude)) for r in results),
            "versions": {name: version(name) for name in
                         ["numpy", "scipy", "matplotlib", "librosa", "soundfile"]},
            "recordings": [{"source": str(r.recording.source),
                            "start_seconds": r.recording.start_seconds,
                            "duration_seconds": r.recording.duration,
                            "sample_rate": r.recording.sample_rate,
                            "channel": r.recording.channel, "audio_stream": args.stream}
                           for r in results],
        }
        args.output.with_suffix(args.output.suffix + ".json").write_text(
            json.dumps(metadata, indent=2) + "\n")
        print(args.output.resolve())
        if args.show:
            plt.show()
        plt.close(fig)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
