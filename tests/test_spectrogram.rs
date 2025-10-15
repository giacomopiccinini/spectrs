mod common;

use anyhow::Result;
use common::{cleanup_test_dir, create_complex_test_wav, create_test_wav, setup_test_dir};
use spectrs::io::audio::read_audio_file_mono;
use spectrs::spectrogram::stft::{SpectrogramType, compute_spectrogram, par_compute_spectrogram};

#[test]
fn test_compute_spectrogram_basic() -> Result<()> {
    // Create a simple sine wave
    let sr = 16000;
    let duration = 1.0;
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    // Compute spectrogram
    let n_fft = 512;
    let hop_length = 160;
    let win_length = 400;

    let spec = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    // Check dimensions
    let n_freq_bins = n_fft / 2 + 1;
    let expected_n_frames = (num_samples - win_length) / hop_length + 1;

    assert_eq!(spec.len(), n_freq_bins);
    assert!(spec[0].len() >= expected_n_frames - 1); // Allow small variation

    // Check that values are non-negative (power spectrogram)
    for freq_bin in &spec {
        for &value in freq_bin {
            assert!(value >= 0.0);
        }
    }

    Ok(())
}

#[test]
fn test_compute_spectrogram_magnitude() -> Result<()> {
    let sr = 16000;
    let duration = 0.5;
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let n_fft = 256;
    let hop_length = 128;
    let win_length = 256;

    let spec = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Magnitude,
    );

    // Check that values are non-negative
    for freq_bin in &spec {
        for &value in freq_bin {
            assert!(value >= 0.0);
        }
    }

    Ok(())
}

#[test]
fn test_compute_spectrogram_centered() -> Result<()> {
    let sr = 16000;
    let duration = 0.5;
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let n_fft = 512;
    let hop_length = 160;
    let win_length = 400;

    // Compute with centering
    let spec_centered = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        true,
        SpectrogramType::Power,
    );

    // Compute without centering
    let spec_not_centered = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    // Both should have valid shapes
    assert!(spec_centered.len() > 0);
    assert!(spec_not_centered.len() > 0);
    assert!(spec_centered[0].len() > 0);
    assert!(spec_not_centered[0].len() > 0);

    Ok(())
}

#[test]
fn test_compute_spectrogram_different_fft_sizes() -> Result<()> {
    let sr = 16000;
    let duration = 0.5;
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let fft_sizes = vec![256, 512, 1024, 2048];

    for n_fft in fft_sizes {
        let hop_length = n_fft / 4;
        let win_length = n_fft / 2;

        let spec = par_compute_spectrogram(
            &samples,
            n_fft,
            hop_length,
            win_length,
            false,
            SpectrogramType::Power,
        );

        let expected_freq_bins = n_fft / 2 + 1;
        assert_eq!(spec.len(), expected_freq_bins);
        assert!(spec[0].len() > 0);
    }

    Ok(())
}

#[test]
fn test_compute_spectrogram_different_hop_lengths() -> Result<()> {
    let sr = 16000;
    let duration = 1.0;
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let n_fft = 512;
    let win_length = 400;
    let hop_lengths = vec![80, 160, 320];

    for hop_length in hop_lengths {
        let spec = par_compute_spectrogram(
            &samples,
            n_fft,
            hop_length,
            win_length,
            false,
            SpectrogramType::Power,
        );

        // Smaller hop length should give more frames
        assert!(spec.len() > 0);
        assert!(spec[0].len() > 0);
    }

    Ok(())
}

#[test]
fn test_compute_spectrogram_from_file() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_spec.wav");

    // Create test file
    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    // Read audio
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    assert_eq!(sr, 16000);

    // Compute spectrogram
    let n_fft = 512;
    let hop_length = 160;
    let win_length = 400;

    let spec = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    // Verify dimensions
    let n_freq_bins = n_fft / 2 + 1;
    assert_eq!(spec.len(), n_freq_bins);
    assert!(spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_compute_spectrogram_complex_signal() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_complex.wav");

    // Create complex test file with multiple frequencies
    create_complex_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    // Read audio
    let (samples, _sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    // Compute spectrogram
    let n_fft = 1024;
    let hop_length = 256;
    let win_length = 512;

    let spec = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    // Check that the spectrogram captured multiple frequencies
    // (should have energy in multiple frequency bins)
    let mut bins_with_energy = 0;
    for freq_bin in &spec {
        let max_value = freq_bin.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_value > 1.0 {
            bins_with_energy += 1;
        }
    }

    // Should have energy in multiple bins for a complex signal
    assert!(bins_with_energy > 5);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_compute_spectrogram_power_vs_magnitude() -> Result<()> {
    let sr = 16000;
    let duration = 0.5;
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let n_fft = 512;
    let hop_length = 160;
    let win_length = 400;

    let spec_power = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    let spec_magnitude = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Magnitude,
    );

    // Both should have same shape
    assert_eq!(spec_power.len(), spec_magnitude.len());
    assert_eq!(spec_power[0].len(), spec_magnitude[0].len());

    // Power values should generally be larger than magnitude values (squared)
    // Check a few random positions
    for i in (0..spec_power.len()).step_by(50) {
        for j in (0..spec_power[0].len()).step_by(10) {
            if spec_magnitude[i][j] > 0.1 {
                // Power ≈ Magnitude^2 (allowing for floating point errors)
                let ratio = spec_power[i][j] / (spec_magnitude[i][j] * spec_magnitude[i][j]);
                assert!(
                    (ratio - 1.0).abs() < 0.01,
                    "Power should be magnitude squared"
                );
            }
        }
    }

    Ok(())
}

#[test]
fn test_compute_spectrogram_short_audio() -> Result<()> {
    // Test with very short audio
    let sr = 16000;
    let duration = 0.1; // 100ms
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let n_fft = 512;
    let hop_length = 160;
    let win_length = 400;

    let spec = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    // Should still produce valid output
    assert!(spec.len() > 0);
    assert!(spec[0].len() > 0);

    Ok(())
}

#[test]
fn test_compute_spectrogram_stereo_to_mono() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_stereo.wav");

    // Create stereo test file
    create_test_wav(&audio_path, 1.0, 16000, 2, 16)?;

    // Read as mono (should average channels)
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    assert_eq!(sr, 16000);

    // Compute spectrogram
    let n_fft = 512;
    let hop_length = 160;
    let win_length = 400;

    let spec = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    // Verify it worked
    assert!(spec.len() > 0);
    assert!(spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_compute_vs_par_compute_same_results() -> Result<()> {
    // Ensure single-threaded and parallel versions produce identical results
    let sr = 16000;
    let duration = 1.0;
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let n_fft = 512;
    let hop_length = 160;
    let win_length = 400;

    let spec_single = compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    let spec_parallel = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    // Check they have the same shape
    assert_eq!(spec_single.len(), spec_parallel.len());
    assert_eq!(spec_single[0].len(), spec_parallel[0].len());

    // Check they have the same values (allowing for floating point precision)
    for (i, (row_single, row_parallel)) in spec_single.iter().zip(spec_parallel.iter()).enumerate()
    {
        for (j, (&val_single, &val_parallel)) in
            row_single.iter().zip(row_parallel.iter()).enumerate()
        {
            assert!(
                (val_single - val_parallel).abs() < 1e-5,
                "Mismatch at [{}, {}]: {} vs {}",
                i,
                j,
                val_single,
                val_parallel
            );
        }
    }

    Ok(())
}

#[test]
fn test_compute_spectrogram_single_threaded() -> Result<()> {
    // Test the single-threaded version
    let sr = 16000;
    let duration = 0.5;
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let n_fft = 256;
    let hop_length = 128;
    let win_length = 256;

    let spec = compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    // Check dimensions
    let n_freq_bins = n_fft / 2 + 1;
    assert_eq!(spec.len(), n_freq_bins);
    assert!(spec[0].len() > 0);

    // Check that values are non-negative
    for freq_bin in &spec {
        for &value in freq_bin {
            assert!(value >= 0.0);
        }
    }

    Ok(())
}
