# spectrs

[![Crates.io](https://img.shields.io/crates/v/spectrs.svg)](https://crates.io/crates/spectrs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fast spectrogram creation library.

## Table of Contents

- [What It Does](#what-it-does)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Examples](#examples)
- [Colormaps](#colormaps)
- [Librosa Compatibility](#librosa-compatibility)
- [Performance](#performance)
- [API Reference](#api-reference)

## What It Does

spectrs is a Rust library for creating spectrograms from audio files. It provides:

- **STFT (Short-Time Fourier Transform)** spectrograms with power or magnitude scaling
- **Mel-scaled** spectrograms using HTK or Slaney scales
- **Audio I/O** for reading WAV files (mono/stereo, 8/16/32-bit)
- **Resampling** to arbitrary sample rates
- **Image export** of spectrograms with multiple colormaps (Viridis, Magma, Inferno, Plasma, Gray)
- **Compatibility** with librosa

## Installation

### As a Command-Line Tool

Install the binary with full features (CLI + image support):

```bash
cargo install spectrs
```

### As a Library

When using spectrs as a library dependency, you typically want minimal dependencies. Add it with:

```bash
# Minimal dependencies (core audio and spectrogram functionality only)
cargo add spectrs --no-default-features

# With image export support
cargo add spectrs --no-default-features --features image
```

## Quick Start

### Command-Line Usage

After installing with `cargo install spectrs`, you can process audio files from the command line.
The CLI will automatically create PNG images with the same name as the input file(s).

```bash
# Process a single file with default settings
spectrs audio.wav

# Create a mel spectrogram with 128 mel bands
spectrs audio.wav --n-mels 128

# Customize spectrogram parameters
spectrs audio.wav \
  --n-fft 2048 \
  --hop-length 512 \
  --n-mels 128 \
  --spec-type power \
  --colormap viridis

# Walk a directory and process all WAV files, placing output files alongside WAV files
spectrs audio_folder/

# Walk a directory and process all WAV files, placing output files in another directory
# with the same nested structure (if any) of the input directory
spectrs audio_folder/ --output-dir processed_audio_folder/

# See all available options
spectrs --help
```

## Colormaps

spectrs supports multiple colormaps for spectrogram visualization, including *viridis*, *magma*, *inferno*, *plasma* and simply *gray*. All colormaps implementations are based on the [matplotlib colormaps](https://github.com/BIDS/colormap).

## Librosa Compatibility

spectrs is designed to replicate librosa's behaviour in the Rust ecosystem. Compatibility tests are part of the test suite, ensuring correlation between librosa's and spectrs' above 99% and relative error on relevant bands < 2%.
```
