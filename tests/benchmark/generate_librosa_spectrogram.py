# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "librosa",
#     "numpy",
# ]
# ///
"""
Script to generate spectrograms using librosa for comparison with spectrs.
Reads test parameters from command line and outputs spectrogram as JSON.
"""

import sys
import json
import numpy as np
import librosa

def compute_librosa_stft(
    audio_file: str,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    center: bool = False,
) -> np.ndarray:
    """Compute STFT power spectrogram using librosa."""
    # Load audio (librosa loads as mono by default)
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    
    # Compute STFT
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=center
    )
    
    # Convert to power spectrogram
    power_spectrogram = np.abs(stft) ** 2
    
    return power_spectrogram, sr

def compute_librosa_mel(
    audio_file: str,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 40,
    f_min: float = 0.0,
    f_max: float = None,
    htk: bool = True,
    center: bool = False,
) -> np.ndarray:
    """Compute mel spectrogram using librosa."""
    # First compute power spectrogram
    power_spec, sr = compute_librosa_stft(audio_file, n_fft, hop_length, win_length, center)
    
    if f_max is None:
        f_max = sr / 2.0
    
    # Apply mel filterbank
    mel_spec = librosa.feature.melspectrogram(
        S=power_spec,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        htk=htk
    )
    
    return mel_spec, sr

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_librosa_spectrogram.py <audio_file> <output_json> [params_json]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Default parameters
    params = {
        "type": "stft",  # or "mel"
        "n_fft": 512,
        "hop_length": 160,
        "win_length": 400,
        "center": False,
        "n_mels": 40,
        "f_min": 0.0,
        "f_max": None,
        "htk": True,
    }
    
    # Load custom parameters if provided
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'r') as f:
            custom_params = json.load(f)
            params.update(custom_params)
    
    # Compute spectrogram
    if params["type"] == "mel":
        spec, sr = compute_librosa_mel(
            audio_file,
            n_fft=params["n_fft"],
            hop_length=params["hop_length"],
            win_length=params["win_length"],
            n_mels=params["n_mels"],
            f_min=params["f_min"],
            f_max=params["f_max"],
            htk=params["htk"],
            center=params["center"],
        )
    else:  # stft
        spec, sr = compute_librosa_stft(
            audio_file,
            n_fft=params["n_fft"],
            hop_length=params["hop_length"],
            win_length=params["win_length"],
            center=params["center"],
        )
    
    # Convert to list for JSON serialization
    output = {
        "data": spec.tolist(),
        "shape": spec.shape,
        "sample_rate": int(sr),
        "params": params
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output, f)
    
    print(f"Librosa spectrogram saved to {output_file}")
    print(f"Shape: {spec.shape}")
    print(f"Sample rate: {sr}")

if __name__ == "__main__":
    main()

