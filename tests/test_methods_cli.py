import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf


@pytest.mark.parametrize("command, array_name", [
    ("psd", "density_0"), ("autocorrelation", "coefficients"),
    ("cross-correlation", "coefficients"), ("patterns", "similarity"),
])
def test_analysis_command_exports_arrays_and_metadata(tmp_path, command, array_name):
    sr = 8000
    t = np.arange(sr) / sr
    phrase = 0.3 * np.sin(2*np.pi*(1400*t + 600*t*t)) * np.sin(np.pi*t)**2
    source = tmp_path / "synthetic.wav"
    sf.write(source, np.tile(phrase, 5), sr, subtype="FLOAT")
    output = tmp_path / f"{command}.png"
    sources = [str(source), str(source)] if command == "cross-correlation" else [str(source)]
    arguments = [sys.executable, "-m", "stadium_audio", command, *sources,
                 "--duration", "3", "--output", str(output)]
    if command == "cross-correlation":
        arguments += ["--second-start", "1", "--second-duration", "2"]
    subprocess.run(arguments, check=True, capture_output=True, text=True)
    metadata = json.loads(Path(str(output) + ".json").read_text())
    assert metadata["command"] == command
    assert output.stat().st_size > 10000
    with np.load(str(output) + ".npz", allow_pickle=False) as arrays:
        assert arrays[array_name].size
    if command == "cross-correlation":
        assert metadata["recordings"][1]["start_seconds"] == 1
        assert metadata["recordings"][1]["duration_seconds"] == 2
        assert metadata["strongest_absolute_match"]["coefficient"] > 0.99
