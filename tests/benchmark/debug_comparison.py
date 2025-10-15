# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "librosa",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
Debug script to understand differences between spectrs and librosa.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python debug_comparison.py <spectrs_json> <librosa_json> [output_image]"
        )
        sys.exit(1)

    spectrs_file = sys.argv[1]
    librosa_file = sys.argv[2]
    output_image = sys.argv[3] if len(sys.argv) > 3 else "comparison.png"

    # Load spectrograms
    with open(spectrs_file, "r") as f:
        spectrs_data = json.load(f)
    with open(librosa_file, "r") as f:
        librosa_data = json.load(f)

    spectrs_spec = np.array(spectrs_data["data"])
    librosa_spec = np.array(librosa_data["data"])

    print(f"Spectrs shape: {spectrs_spec.shape}")
    print(f"Librosa shape: {librosa_spec.shape}")
    print(f"\nSpectrs stats:")
    print(f"  Min: {spectrs_spec.min():.6e}")
    print(f"  Max: {spectrs_spec.max():.6e}")
    print(f"  Mean: {spectrs_spec.mean():.6e}")
    print(f"  Std: {spectrs_spec.std():.6e}")
    print(f"\nLibrosa stats:")
    print(f"  Min: {librosa_spec.min():.6e}")
    print(f"  Max: {librosa_spec.max():.6e}")
    print(f"  Mean: {librosa_spec.mean():.6e}")
    print(f"  Std: {librosa_spec.std():.6e}")

    # Compute correlation
    flat1 = spectrs_spec.flatten()
    flat2 = librosa_spec.flatten()
    min_len = min(len(flat1), len(flat2))
    flat1 = flat1[:min_len]
    flat2 = flat2[:min_len]

    correlation = np.corrcoef(flat1, flat2)[0, 1]
    print(f"\nCorrelation: {correlation:.6f}")

    # Compute scale ratio
    ratio = np.mean(spectrs_spec) / np.mean(librosa_spec)
    print(f"Mean ratio (spectrs/librosa): {ratio:.6f}")

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot spectrograms
    im1 = axes[0, 0].imshow(
        np.log1p(spectrs_spec), aspect="auto", origin="lower", cmap="viridis"
    )
    axes[0, 0].set_title("Spectrs (log scale)")
    axes[0, 0].set_ylabel("Frequency bin")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(
        np.log1p(librosa_spec), aspect="auto", origin="lower", cmap="viridis"
    )
    axes[0, 1].set_title("Librosa (log scale)")
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot difference
    min_shape = (
        min(spectrs_spec.shape[0], librosa_spec.shape[0]),
        min(spectrs_spec.shape[1], librosa_spec.shape[1]),
    )
    diff = (
        spectrs_spec[: min_shape[0], : min_shape[1]]
        - librosa_spec[: min_shape[0], : min_shape[1]]
    )
    im3 = axes[0, 2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r")
    axes[0, 2].set_title("Difference (spectrs - librosa)")
    plt.colorbar(im3, ax=axes[0, 2])

    # Plot histograms
    axes[1, 0].hist(spectrs_spec.flatten(), bins=100, alpha=0.7, label="Spectrs")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Spectrs histogram")
    axes[1, 0].set_yscale("log")

    axes[1, 1].hist(
        librosa_spec.flatten(), bins=100, alpha=0.7, label="Librosa", color="orange"
    )
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Librosa histogram")
    axes[1, 1].set_yscale("log")

    # Scatter plot
    sample_indices = np.random.choice(min_len, min(10000, min_len), replace=False)
    axes[1, 2].scatter(flat2[sample_indices], flat1[sample_indices], alpha=0.1, s=1)
    axes[1, 2].set_xlabel("Librosa")
    axes[1, 2].set_ylabel("Spectrs")
    axes[1, 2].set_title(f"Scatter plot (corr={correlation:.3f})")
    axes[1, 2].plot(
        [flat2.min(), flat2.max()], [flat2.min(), flat2.max()], "r--", label="y=x"
    )
    axes[1, 2].legend()
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to {output_image}")


if __name__ == "__main__":
    main()
