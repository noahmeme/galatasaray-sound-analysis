"""Exploratory tools; measurements do not classify the origin of a sound."""
from .audio import AudioRecording, extract_audio, load_recording
from .spectrogram import STFTConfig, Spectrogram, amplitude_db, compute_spectrogram

__all__ = ["AudioRecording", "extract_audio", "load_recording", "STFTConfig",
           "Spectrogram", "amplitude_db", "compute_spectrogram"]

from .psd import WelchConfig, PowerSpectralDensity, estimate_psd
from .correlation import CorrelationResult, autocorrelation, cross_correlation
from .patterns import PatternMatch, PatternSearch, detect_repeated_patterns

__all__ += ["WelchConfig", "PowerSpectralDensity", "estimate_psd", "CorrelationResult",
            "autocorrelation", "cross_correlation", "PatternMatch", "PatternSearch",
            "detect_repeated_patterns"]
