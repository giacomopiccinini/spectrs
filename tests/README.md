# Spectrs Tests

This directory contains comprehensive tests for the spectrs library, including unit tests, integration tests, and librosa compatibility benchmarks.

## Test Structure

- **`common/`**: Shared test utilities for creating test audio files and helper functions
- **`test_io.rs`**: Unit tests for I/O functions (`read_audio_file_mono`, `resample`)
- **`test_spectrogram.rs`**: Unit tests for STFT spectrogram computation
- **`test_mel.rs`**: Unit tests for mel spectrogram conversion
- **`test_integration.rs`**: Integration tests for the full pipeline (read → resample → STFT → mel)
- **`test_cli.rs`**: Integration tests for the CLI binary and `--output-dir` functionality
- **`test_librosa_compatibility.rs`**: Benchmark tests comparing spectrs output with librosa (Python)
- **`benchmark/`**: Python scripts for librosa comparison

## Running Tests

### Basic Tests

Run all unit and integration tests:

```bash
cargo test
```

This will run all tests except the librosa compatibility tests (which are ignored by default).

### Test Categories

Run only unit tests:
```bash
cargo test --test test_io
cargo test --test test_spectrogram
cargo test --test test_mel
```

Run only integration tests:
```bash
cargo test --test test_integration
```

Run CLI tests:
```bash
cargo test --test test_cli
```

### Librosa Compatibility Tests

The librosa compatibility tests require Python 3.12+ and `uv` to be installed.

Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run librosa compatibility tests:
```bash
cargo test --test test_librosa_compatibility -- --ignored --show-output
```

These tests:
1. Generate test audio files
2. Compute spectrograms using spectrs (Rust)
3. Compute spectrograms using librosa (Python via uv)
4. Compare the outputs using correlation and relative error metrics
5. Assert that compatibility thresholds are met

### Compatibility Thresholds

The librosa compatibility tests use the following thresholds:
- **Correlation**: ≥ 0.95 (95% correlation expected)
- **Relative Error**: ≤ 0.15 (15% relative error allowed)

## Test Coverage

### Unit Tests

#### I/O Tests (`test_io.rs`)
- ✓ Read mono audio files (8-bit, 16-bit, 32-bit)
- ✓ Read stereo audio files and convert to mono
- ✓ Resample audio (downsample, upsample, same rate, extreme rates)
- ✓ Different sample rates (8000, 16000, 22050, 44100, 48000 Hz)

#### Spectrogram Tests (`test_spectrogram.rs`)
- ✓ Basic STFT computation
- ✓ Power vs magnitude spectrograms
- ✓ Centered vs non-centered windowing
- ✓ Different FFT sizes (256, 512, 1024, 2048)
- ✓ Different hop lengths
- ✓ Complex multi-frequency signals
- ✓ Short and long audio

#### Mel Tests (`test_mel.rs`)
- ✓ Basic mel spectrogram conversion
- ✓ HTK vs Slaney mel scales
- ✓ Different numbers of mel bins (20, 40, 80, 128)
- ✓ Custom frequency ranges
- ✓ Energy conservation
- ✓ Different sample rates
- ✓ Power vs magnitude inputs

### Integration Tests (`test_integration.rs`)

Full pipeline tests with various combinations:
- ✓ Mono/stereo audio (8-bit, 16-bit, 32-bit)
- ✓ Different FFT sizes
- ✓ Different mel bin counts
- ✓ Different sample rates
- ✓ Magnitude vs power spectrograms
- ✓ Centered windowing
- ✓ HTK vs Slaney mel scales
- ✓ Custom frequency ranges
- ✓ Complex multi-frequency signals
- ✓ Short (0.1s) and long (10s) audio

### CLI Tests (`test_cli.rs`)

Command-line interface tests:
- ✓ Single file with default output (same directory)
- ✓ Single file with custom output directory (`--output-dir`)
- ✓ Directory input with default output
- ✓ Directory input with custom output directory (preserves structure)
- ✓ Nested directory structure preservation
- ✓ CLI parameters with output directory (resampling, mel, colormap)
- ✓ Non-WAV files are ignored in directory processing
- ✓ Parent directories are created automatically
- ✓ Error handling for non-existent input files

### Librosa Compatibility Tests (`test_librosa_compatibility.rs`)

- ✓ STFT compatibility (basic)
- ✓ STFT with different FFT sizes
- ✓ Mel spectrogram with HTK scale
- ✓ Mel spectrogram with Slaney scale
- ✓ Mel spectrogram with different mel bin counts
- ✓ Complex multi-frequency signals
- ✓ Different sample rates

## Python Scripts

### `benchmark/generate_librosa_spectrogram.py`

Generates spectrograms using librosa with configurable parameters.

Usage:
```bash
uv run tests/benchmark/generate_librosa_spectrogram.py <audio_file> <output_json> [params_json]
```

### `benchmark/compare_spectrograms.py`

Compares two spectrograms and computes similarity metrics.

Usage:
```bash
uv run tests/benchmark/compare_spectrograms.py <spectrs_json> <librosa_json> <output_json>
```

### `benchmark/compare_with_librosa.py`

Original comparison script (kept for reference).

## Test Data

All tests use programmatically generated WAV files (no pre-recorded audio files).
Test files are created in temporary directories and cleaned up after each test.

Test audio generation includes:
- Simple sine waves (440 Hz)
- Complex multi-frequency signals (220, 440, 880, 1320 Hz)
- Various durations (0.1s to 10s)
- Various sample rates (8000 to 48000 Hz)
- Mono and stereo configurations
- 8-bit, 16-bit, and 32-bit depths

## Continuous Integration

All tests (except librosa compatibility) run automatically on CI.
Librosa compatibility tests are marked with `#[ignore]` and can be run manually.

## Notes

- Tests are designed to be deterministic and reproducible
- No network access required (except for installing dependencies)
- All temporary files are cleaned up after tests
- Tests use parallelization where appropriate (via rayon)

