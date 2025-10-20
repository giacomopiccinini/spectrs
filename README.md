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

Add spectrs to your project with

```
cargo add spectrs
```

or with image export support

```
cargo add spectrs --features image
```

## Examples

### Basic STFT Spectrogram

```rust
use spectrs::io::audio::read_audio_file_mono;
use spectrs::spectrogram::stft::{par_compute_spectrogram, SpectrogramType};

let (audio, _sr) = read_audio_file_mono("speech.wav")?;

// Compute magnitude spectrogram
let spec = par_compute_spectrogram(
    &audio,
    2048,   // n_fft: higher = better frequency resolution
    512,    // hop_length: lower = better time resolution
    2048,   // win_length: window size
    true,   // center: pad the signal
    SpectrogramType::Magnitude
);
```

### Mel Spectrogram

```rust
use spectrs::spectrogram::mel::{convert_to_mel, MelScale};

// First compute STFT, then convert to mel
let mel_spec = convert_to_mel(
    &spec,
    22050,              // sample_rate
    2048,               // n_fft
    128,                // n_mels: number of mel bands
    Some(20.0),         // f_min: lowest frequency
    Some(8000.0),       // f_max: highest frequency
    MelScale::Slaney    // librosa default
);
```

### Save Spectrogram as Image

```rust
use spectrs::io::image::{save_spectrogram_image, Colormap};

// Requires "image" feature
// Use default Viridis colormap (librosa/matplotlib default)
save_spectrogram_image(&mel_spec, "spectrogram.png", Colormap::Viridis)?;

// Or use other colormaps:
// save_spectrogram_image(&mel_spec, "spectrogram.png", Colormap::Magma)?;
// save_spectrogram_image(&mel_spec, "spectrogram.png", Colormap::Inferno)?;
// save_spectrogram_image(&mel_spec, "spectrogram.png", Colormap::Plasma)?;
// save_spectrogram_image(&mel_spec, "spectrogram.png", Colormap::Gray)?;
```

### Single-Threaded Mode

For small files or embedded systems:

```rust
use spectrs::spectrogram::stft::compute_spectrogram;

// Non-parallelized version (lower overhead)
let spec = compute_spectrogram(
    &audio,
    512,
    160,
    400,
    false,
    SpectrogramType::Power
);
```

### Complete Pipeline

```rust
use spectrs::io::audio::{read_audio_file_mono, resample};
use spectrs::spectrogram::stft::{par_compute_spectrogram, SpectrogramType};
use spectrs::spectrogram::mel::{convert_to_mel, MelScale};

// Read and process audio file
let (audio, sr) = read_audio_file_mono("song.wav")?;

// Resample to standard rate
let audio = resample(audio, sr, 22050)?;

// Compute spectrogram
let spec = par_compute_spectrogram(&audio, 2048, 512, 2048, true, SpectrogramType::Power);

// Convert to mel
let mel = convert_to_mel(&spec, 22050, 2048, 128, None, None, MelScale::HTK);

// Save as image (requires "image" feature)
#[cfg(feature = "image")]
{
    use spectrs::io::image::Colormap;
    spectrs::io::image::save_spectrogram_image(&mel, "output.png", Colormap::Viridis)?;
}
```

## Colormaps

spectrs supports multiple colormaps for spectrogram visualization, including *viridis*, *magma*, *inferno*, *plasma* and simply *gray*. All colormaps implementations are based on the [matplotlib colormaps](https://github.com/BIDS/colormap).

## Librosa Compatibility

spectrs is designed to replicate librosa's behaviour in the Rust ecosystem. Compatibility tests are part of the test suite, ensuring correlation between librosa's and spectrs' above 99% and relative error on relevant bands < 2%. 

Run compatibility tests:

```bash
cargo test --test test_librosa_compatibility
```
