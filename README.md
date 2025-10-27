# spectrs

[![Crates.io](https://img.shields.io/crates/v/spectrs.svg)](https://crates.io/crates/spectrs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A fast spectrogram creation library for Rust.

![](https://github.com/giacomopiccinini/spectrs/blob/main/assets/spectrogram.png)

## Table of Contents

- [What It Does](#what-it-does)
- [Why spectrs?](#why-spectrs)
- [Installation](#installation)
- [Quick Start](#quick-start)

## What It Does

spectrs is a pure-Rust library for creating spectrograms from WAV audio files. It's designed to be a batteries-included crate that provides both a **library** (for integrating spectrs into any downstream app) and a **CLI**. By "batteries-included," I mean that spectrs comes equipped with modules for:

1. **Audio Input**: Read WAV files (no MP3 support, sorry!) and convert them to mono
2. **Resampling**: Resample mono audio files to your desired sample rate
3. **STFT**: Perform Short-Time Fourier Transform with power or magnitude scaling
4. **Mel-scaling**: Convert spectrograms to mel scale using HTK or Slaney scales
5. **Image Export**: Save spectrograms to disk as images with multiple colormaps (Viridis, Magma, Inferno, Plasma, Gray)

I've made sure to maintain compatibility with Librosa's results and implementation.

## Why spectrs?

spectrs was born out of frustration with the lack of simple, comprehensive tools for either:
- Creating spectrograms within the Rust ecosystem, or 
- Creating and saving spectrograms from the CLI without resorting to Python or FFmpeg

Specifically, the pain points I wanted to address are:

**Python Issues:**
- No single binary for creating spectrograms. You need `uv` pointing to *some* file *somewhere* on your computer
- Poor parallelization
- Massive, unnecessary dependencies when using *Torch Audio* instead of Librosa
- All the baggage that comes with Matplotlib for saving images

**FFmpeg Issues:**
- Obscure and esoteric syntax
- Limited colormap and spectrogram type availability
- FFmpeg is always a pain to install

Don't get me wrong: Librosa, Torch Audio, and FFmpeg are all incredible tools. I just wanted something simple and self-contained.

## Installation

As mentioned, spectrs can function as both a library and a CLI tool, depending on your needs.

The basic implementation (without additional features) can compute spectrograms but can't save them locally as images. For that functionality, you'll need the `image` feature. If you want CLI support, you'll need the `cli` feature.

### As a Library

When using spectrs as a library dependency, you typically want minimal dependencies:

```bash
# Minimal dependencies (core audio and spectrogram functionality only)
cargo add spectrs --no-default-features

# With image export support
cargo add spectrs --no-default-features --features image
```

### As a Command-Line Tool

Install the binary with full features (CLI + image support):

```bash
cargo install spectrs
```

## Quick Start

### Command-Line Usage

After installing with `cargo install spectrs`, you can process audio files from the command line. The CLI automatically creates PNG images with the same name as your input file(s). 

You can pass either a single file or an entire directory. Note the parallelization behavior:
- **Single file**: Parallelization happens at the spectrogram creation level
- **Multiple files**: Parallelization applies to files (spectrogram generation is serial, but multiple files are processed in parallel)

```bash
# See all available options
spectrs --help

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

# Process all WAV files in a directory, placing output files alongside input files
spectrs audio_folder/

# Process all WAV files in a directory, placing output files in another directory
# (preserves the nested structure of the input directory, if any)
spectrs audio_folder/ --output-dir processed_audio_folder/
```

### Colormaps

spectrs supports multiple colormaps for spectrogram visualization: *viridis*, *magma*, *inferno*, *plasma*, and *gray*. All colormap implementations are based on the [matplotlib colormaps](https://github.com/BIDS/colormap).