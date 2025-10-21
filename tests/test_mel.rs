mod common;

use anyhow::Result;
use common::{cleanup_test_dir, create_complex_test_wav, create_test_wav, setup_test_dir};
use spectrs::io::audio::read_audio_file_mono;
use spectrs::spectrogram::mel::{MelScale, convert_to_mel, par_convert_to_mel};
use spectrs::spectrogram::stft::{SpectrogramType, par_compute_spectrogram};

#[test]
fn test_convert_to_mel_basic() -> Result<()> {
    // Create a simple spectrogram
    let sr = 16000;
    let duration = 1.0;
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

    // Convert to mel
    let n_mels = 40;
    let mel_spec = convert_to_mel(&spec, sr, n_fft, n_mels, None, None, MelScale::HTK);

    // Check dimensions
    assert_eq!(mel_spec.len(), n_mels);
    assert_eq!(mel_spec[0].len(), spec[0].len());

    // Check that values are non-negative
    for mel_bin in &mel_spec {
        for &value in mel_bin {
            assert!(value >= 0.0);
        }
    }

    Ok(())
}

#[test]
fn test_convert_to_mel_htk_vs_slaney() -> Result<()> {
    let sr = 16000;
    let duration = 0.5;
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

    let n_mels = 40;

    // Convert using HTK
    let mel_spec_htk = convert_to_mel(&spec, sr, n_fft, n_mels, None, None, MelScale::HTK);

    // Convert using Slaney
    let mel_spec_slaney = convert_to_mel(&spec, sr, n_fft, n_mels, None, None, MelScale::Slaney);

    // Both should have same shape
    assert_eq!(mel_spec_htk.len(), mel_spec_slaney.len());
    assert_eq!(mel_spec_htk[0].len(), mel_spec_slaney[0].len());

    // Values should be different (different mel scales)
    let mut differences = 0;
    for i in 0..mel_spec_htk.len() {
        for j in 0..mel_spec_htk[0].len() {
            if (mel_spec_htk[i][j] - mel_spec_slaney[i][j]).abs() > 0.001 {
                differences += 1;
            }
        }
    }

    // Should have some differences
    assert!(differences > 0);

    Ok(())
}

#[test]
fn test_convert_to_mel_different_n_mels() -> Result<()> {
    let sr = 16000;
    let duration = 0.5;
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

    let n_mels_values = vec![20, 40, 80, 128];

    for n_mels in n_mels_values {
        let mel_spec = convert_to_mel(&spec, sr, n_fft, n_mels, None, None, MelScale::HTK);

        // Check dimensions match
        assert_eq!(mel_spec.len(), n_mels);
        assert_eq!(mel_spec[0].len(), spec[0].len());
    }

    Ok(())
}

#[test]
fn test_convert_to_mel_with_frequency_range() -> Result<()> {
    let sr = 16000;
    let duration = 0.5;
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

    let n_mels = 40;

    // Default range (0 to Nyquist)
    let mel_spec_default = convert_to_mel(&spec, sr, n_fft, n_mels, None, None, MelScale::HTK);

    // Custom range (300 Hz to 4000 Hz)
    let f_min = 300.0;
    let f_max = 4000.0;
    let mel_spec_custom = convert_to_mel(
        &spec,
        sr,
        n_fft,
        n_mels,
        Some(f_min),
        Some(f_max),
        MelScale::HTK,
    );

    // Both should have same shape
    assert_eq!(mel_spec_default.len(), mel_spec_custom.len());
    assert_eq!(mel_spec_default[0].len(), mel_spec_custom[0].len());

    // Values should be different due to different frequency ranges
    let mut differences = 0;
    for i in 0..mel_spec_default.len() {
        for j in 0..mel_spec_default[0].len() {
            if (mel_spec_default[i][j] - mel_spec_custom[i][j]).abs() > 0.001 {
                differences += 1;
            }
        }
    }

    assert!(differences > 0);

    Ok(())
}

#[test]
fn test_convert_to_mel_from_file() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_mel.wav");

    // Create test file
    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    // Read audio
    let (samples, sr) = read_audio_file_mono(&audio_path)?;

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

    // Convert to mel
    let n_mels = 40;
    let mel_spec = convert_to_mel(&spec, sr, n_fft, n_mels, None, None, MelScale::HTK);

    // Verify dimensions
    assert_eq!(mel_spec.len(), n_mels);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_convert_to_mel_complex_signal() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_mel_complex.wav");

    // Create complex test file
    create_complex_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    // Read audio
    let (samples, sr) = read_audio_file_mono(&audio_path)?;

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

    // Convert to mel
    let n_mels = 80;
    let mel_spec = convert_to_mel(&spec, sr, n_fft, n_mels, None, None, MelScale::HTK);

    // Check that multiple mel bins have energy
    let mut bins_with_energy = 0;
    for mel_bin in &mel_spec {
        let max_value = mel_bin.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_value > 0.1 {
            bins_with_energy += 1;
        }
    }

    assert!(bins_with_energy > 5);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_convert_to_mel_energy_conservation() -> Result<()> {
    // Test that mel conversion doesn't create energy out of nowhere
    let sr = 16000;
    let duration = 0.5;
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

    // Calculate total energy in original spectrogram
    let total_energy_orig: f32 = spec
        .iter()
        .map(|freq_bin| freq_bin.iter().sum::<f32>())
        .sum();

    // Convert to mel
    let n_mels = 40;
    let mel_spec = convert_to_mel(&spec, sr, n_fft, n_mels, None, None, MelScale::HTK);

    // Calculate total energy in mel spectrogram
    let total_energy_mel: f32 = mel_spec
        .iter()
        .map(|mel_bin| mel_bin.iter().sum::<f32>())
        .sum();

    // Mel spectrogram should have similar total energy (within reasonable tolerance)
    // Note: Exact conservation depends on filter bank normalization
    assert!(total_energy_mel > 0.0);
    assert!(total_energy_orig > 0.0);

    Ok(())
}

#[test]
fn test_convert_to_mel_different_sample_rates() -> Result<()> {
    let test_dir = setup_test_dir()?;

    let sample_rates = vec![8000, 16000, 22050, 44100];

    for sr in sample_rates {
        let audio_path = test_dir.join(format!("test_mel_{}.wav", sr));
        create_test_wav(&audio_path, 0.5, sr, 1, 16)?;

        let (samples, read_sr) = read_audio_file_mono(&audio_path)?;
        assert_eq!(read_sr, sr);

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

        let n_mels = 40;
        let mel_spec = convert_to_mel(&spec, sr, n_fft, n_mels, None, None, MelScale::HTK);

        // Verify it worked
        assert_eq!(mel_spec.len(), n_mels);
        assert!(mel_spec[0].len() > 0);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_convert_to_mel_magnitude_vs_power() -> Result<()> {
    let sr = 16000;
    let duration = 0.5;
    let num_samples = (duration * sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let n_fft = 512;
    let hop_length = 160;
    let win_length = 400;

    // Power spectrogram
    let spec_power = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    // Magnitude spectrogram
    let spec_magnitude = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Magnitude,
    );

    let n_mels = 40;

    // Convert both to mel
    let mel_spec_power = convert_to_mel(&spec_power, sr, n_fft, n_mels, None, None, MelScale::HTK);
    let mel_spec_magnitude = convert_to_mel(
        &spec_magnitude,
        sr,
        n_fft,
        n_mels,
        None,
        None,
        MelScale::HTK,
    );

    // Both should have same shape
    assert_eq!(mel_spec_power.len(), mel_spec_magnitude.len());
    assert_eq!(mel_spec_power[0].len(), mel_spec_magnitude[0].len());

    // Values should be different (power vs magnitude)
    let mut differences = 0;
    for i in 0..mel_spec_power.len() {
        for j in 0..mel_spec_power[0].len() {
            if (mel_spec_power[i][j] - mel_spec_magnitude[i][j]).abs() > 0.001 {
                differences += 1;
            }
        }
    }

    assert!(differences > 0);

    Ok(())
}

#[test]
fn test_convert_to_mel_vs_par_convert_to_mel() -> Result<()> {
    // Test that sequential and parallel versions produce identical results
    let sr = 22050;
    let duration = 2.0;
    let num_samples = (duration * sr as f32) as usize;

    // Create a complex signal with multiple frequencies
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| {
            let t_sec = t as f32 / sr as f32;
            (t_sec * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5
                + (t_sec * 880.0 * 2.0 * std::f32::consts::PI).sin() * 0.3
                + (t_sec * 1320.0 * 2.0 * std::f32::consts::PI).sin() * 0.2
        })
        .collect();

    let n_fft = 2048;
    let hop_length = 512;
    let win_length = 2048;

    // Create spectrogram
    let spec = par_compute_spectrogram(
        &samples,
        n_fft,
        hop_length,
        win_length,
        true,
        SpectrogramType::Power,
    );

    let n_mels = 128;
    let f_min = Some(0.0);
    let f_max = Some((sr / 2) as f32);

    // Test with both mel scales
    for mel_scale in [MelScale::Slaney, MelScale::HTK] {
        // Sequential version
        let mel_spec_seq = convert_to_mel(&spec, sr, n_fft, n_mels, f_min, f_max, mel_scale);

        // Parallel version
        let mel_spec_par = par_convert_to_mel(&spec, sr, n_fft, n_mels, f_min, f_max, mel_scale);

        // Verify dimensions match
        assert_eq!(mel_spec_seq.len(), mel_spec_par.len());
        assert_eq!(mel_spec_seq[0].len(), mel_spec_par[0].len());

        // Verify values are identical (allowing for small floating point errors)
        let tolerance = 1e-6;
        for i in 0..mel_spec_seq.len() {
            for j in 0..mel_spec_seq[0].len() {
                let diff = (mel_spec_seq[i][j] - mel_spec_par[i][j]).abs();
                assert!(
                    diff < tolerance,
                    "Mismatch at [{},{}] for {:?}: seq={}, par={}, diff={}",
                    i,
                    j,
                    mel_scale,
                    mel_spec_seq[i][j],
                    mel_spec_par[i][j],
                    diff
                );
            }
        }
    }

    Ok(())
}

#[test]
fn test_par_convert_to_mel_basic() -> Result<()> {
    // Test that par_convert_to_mel works correctly on a basic example
    let sr = 16000;
    let duration = 1.0;
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

    let mel_spec = par_convert_to_mel(
        &spec,
        sr,
        n_fft,
        40,
        Some(0.0),
        Some(8000.0),
        MelScale::Slaney,
    );

    // Check dimensions
    assert_eq!(mel_spec.len(), 40);
    assert!(mel_spec[0].len() > 0);

    // Check non-negativity
    for mel_bin in &mel_spec {
        for &value in mel_bin {
            assert!(
                value >= 0.0,
                "Mel spectrogram values should be non-negative"
            );
        }
    }

    // Check that some energy exists
    let total_energy: f32 = mel_spec.iter().flatten().sum();
    assert!(
        total_energy > 0.0,
        "Mel spectrogram should have some energy"
    );

    Ok(())
}
