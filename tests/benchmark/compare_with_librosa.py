# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "librosa",
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
Script to compare Rust mel spectrogram output with librosa implementation.
"""

import numpy as np
import librosa
import json
import matplotlib.pyplot as plt

def load_rust_output(json_file):
    """Load the Rust spectrogram output from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert to numpy array
    spectrogram = np.array(data['data'])
    print(f"Rust output shape: {spectrogram.shape}")
    return spectrogram

def compute_librosa_spectrogram(audio_file, n_fft=512, hop_length=160, win_length=400):
    """Compute mel spectrogram using librosa for comparison."""
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)
    print(f"Audio shape: {y.shape}, Sample rate: {sr}")
    
    # Compute STFT (Short-Time Fourier Transform) - similar to what Rust code does
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=False)
    
    # Convert to power spectrogram (magnitude squared)
    power_spectrogram = np.abs(stft) ** 2
    
    print(f"Librosa STFT shape: {power_spectrogram.shape}")
    return power_spectrogram

def compare_spectrograms(rust_spec, librosa_spec):
    """Compare the two spectrograms."""
    print("\n=== Comparison Results ===")
    print(f"Rust spectrogram shape: {rust_spec.shape}")
    print(f"Librosa spectrogram shape: {librosa_spec.shape}")
    
    # Check if shapes match
    if rust_spec.shape != librosa_spec.shape:
        print("WARNING: Shapes don't match!")
        min_freq = min(rust_spec.shape[0], librosa_spec.shape[0])
        min_time = min(rust_spec.shape[1], librosa_spec.shape[1])
        rust_spec = rust_spec[:min_freq, :min_time]
        librosa_spec = librosa_spec[:min_freq, :min_time]
        print(f"Trimmed to common shape: {rust_spec.shape}")
    
    # Compute statistics
    rust_mean = np.mean(rust_spec)
    librosa_mean = np.mean(librosa_spec)
    
    rust_std = np.std(rust_spec)
    librosa_std = np.std(librosa_spec)
    
    # Compute correlation
    correlation = np.corrcoef(rust_spec.flatten(), librosa_spec.flatten())[0, 1]
    
    # Compute relative error
    relative_error = np.mean(np.abs(rust_spec - librosa_spec) / (librosa_spec + 1e-10))
    
    print(f"\nStatistics:")
    print(f"Rust - Mean: {rust_mean:.6f}, Std: {rust_std:.6f}")
    print(f"Librosa - Mean: {librosa_mean:.6f}, Std: {librosa_std:.6f}")
    print(f"Correlation: {correlation:.6f}")
    print(f"Mean Relative Error: {relative_error:.6f}")
    
    return rust_spec, librosa_spec

def plot_comparison(rust_spec, librosa_spec, output_file="spectrogram_comparison.png"):
    """Plot both spectrograms for visual comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Rust spectrogram
    im1 = axes[0, 0].imshow(10 * np.log10(rust_spec + 1e-10), aspect='auto', origin='lower')
    axes[0, 0].set_title('Rust Spectrogram (dB)')
    axes[0, 0].set_xlabel('Time Frame')
    axes[0, 0].set_ylabel('Frequency Bin')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot Librosa spectrogram
    im2 = axes[0, 1].imshow(10 * np.log10(librosa_spec + 1e-10), aspect='auto', origin='lower')
    axes[0, 1].set_title('Librosa Spectrogram (dB)')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Frequency Bin')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot difference
    diff = np.abs(rust_spec - librosa_spec)
    #im3 = axes[1, 0].imshow(10 * np.log10(diff + 1e-10), aspect='auto', origin='lower')
    im3 = axes[1, 0].imshow(10 * np.log10(rust_spec + 1e-10) - 10 * np.log10(librosa_spec + 1e-10), aspect='auto', origin='lower')
    axes[1, 0].set_title('Absolute Difference (dB)')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Frequency Bin')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot scatter comparison
    axes[1, 1].scatter(librosa_spec.flatten()[::100], rust_spec.flatten()[::100], alpha=0.5, s=1)
    axes[1, 1].plot([librosa_spec.min(), librosa_spec.max()], [librosa_spec.min(), librosa_spec.max()], 'r--')
    axes[1, 1].set_xlabel('Librosa Values')
    axes[1, 1].set_ylabel('Rust Values')
    axes[1, 1].set_title('Scatter Plot Comparison')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved as: {output_file}")

def main():
    # Parameters (should match your Rust code)
    n_fft = 512
    hop_length = 160
    win_length = 400
    
    # File paths
    audio_file = "harvard.wav"
    rust_json = "output_spectrogram.json"
    
    print("Loading Rust spectrogram output...")
    rust_spec = load_rust_output(rust_json)
    
    print("\nComputing librosa spectrogram...")
    librosa_spec = compute_librosa_spectrogram(audio_file, n_fft, hop_length, win_length)
    
    print("\nComparing spectrograms...")
    rust_spec, librosa_spec = compare_spectrograms(rust_spec, librosa_spec)
    
    print("\nGenerating comparison plot...")
    plot_comparison(rust_spec, librosa_spec)
    
    # # Save comparison data
    # np.savez('spectrogram_comparison.npz', 
    #          rust=rust_spec, 
    #          librosa=librosa_spec,
    #          parameters={'n_fft': n_fft, 'hop_length': hop_length, 'win_length': win_length})
    # print("Comparison data saved as: spectrogram_comparison.npz")

if __name__ == "__main__":
    main()
