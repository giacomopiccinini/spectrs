mod common;

use anyhow::Result;
use common::{cleanup_test_dir, create_complex_test_wav, create_test_wav, setup_test_dir};
use spectrs::io::audio::{read_audio_file_mono, resample};
use spectrs::spectrogram::mel::{MelScale, convert_to_mel};
use spectrs::spectrogram::stft::{SpectrogramType, par_compute_spectrogram};

/// Integration test: Full pipeline with mono 16-bit audio
#[test]
fn test_full_pipeline_mono_16bit() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_full_mono_16.wav");

    // Create test file
    create_test_wav(&audio_path, 2.0, 44100, 1, 16)?;

    // Read audio
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    assert_eq!(sr, 44100);
    assert_eq!(samples.len(), 88200);

    // Resample to 16000 Hz
    let resampled = resample(samples, sr, 16000)?;
    assert!((resampled.len() as i32 - 32000).abs() < 100);

    // Compute spectrogram
    let n_fft = 512;
    let hop_length = 160;
    let win_length = 400;

    let spec = par_compute_spectrogram(
        &resampled,
        n_fft,
        hop_length,
        win_length,
        false,
        SpectrogramType::Power,
    );

    assert_eq!(spec.len(), 257); // n_fft / 2 + 1
    assert!(spec[0].len() > 0);

    // Convert to mel
    let n_mels = 40;
    let mel_spec = convert_to_mel(&spec, 16000, n_fft, n_mels, None, None, MelScale::HTK);

    assert_eq!(mel_spec.len(), 40);
    assert_eq!(mel_spec[0].len(), spec[0].len());

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Full pipeline with stereo 16-bit audio
#[test]
fn test_full_pipeline_stereo_16bit() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_full_stereo_16.wav");

    // Create stereo test file
    create_test_wav(&audio_path, 2.0, 44100, 2, 16)?;

    // Read audio (should average to mono)
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    assert_eq!(sr, 44100);
    assert_eq!(samples.len(), 88200); // Same as mono

    // Resample
    let resampled = resample(samples, sr, 16000)?;

    // Compute spectrogram
    let spec = par_compute_spectrogram(&resampled, 512, 160, 400, false, SpectrogramType::Power);

    // Convert to mel
    let mel_spec = convert_to_mel(&spec, 16000, 512, 40, None, None, MelScale::HTK);

    assert_eq!(mel_spec.len(), 40);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Full pipeline with mono 8-bit audio
#[test]
fn test_full_pipeline_mono_8bit() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_full_mono_8.wav");

    // Create 8-bit test file
    create_test_wav(&audio_path, 1.0, 22050, 1, 8)?;

    // Read audio
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    assert_eq!(sr, 22050);

    // Resample to 16000 Hz
    let resampled = resample(samples, sr, 16000)?;

    // Compute spectrogram
    let spec = par_compute_spectrogram(&resampled, 512, 160, 400, false, SpectrogramType::Power);

    // Convert to mel
    let mel_spec = convert_to_mel(&spec, 16000, 512, 40, None, None, MelScale::HTK);

    assert_eq!(mel_spec.len(), 40);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Full pipeline with mono 32-bit audio
#[test]
fn test_full_pipeline_mono_32bit() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_full_mono_32.wav");

    // Create 32-bit test file
    create_test_wav(&audio_path, 1.0, 48000, 1, 32)?;

    // Read audio
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    assert_eq!(sr, 48000);

    // Resample to 16000 Hz
    let resampled = resample(samples, sr, 16000)?;

    // Compute spectrogram
    let spec = par_compute_spectrogram(&resampled, 512, 160, 400, false, SpectrogramType::Power);

    // Convert to mel
    let mel_spec = convert_to_mel(&spec, 16000, 512, 40, None, None, MelScale::HTK);

    assert_eq!(mel_spec.len(), 40);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Full pipeline with stereo 32-bit audio
#[test]
fn test_full_pipeline_stereo_32bit() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_full_stereo_32.wav");

    // Create stereo 32-bit test file
    create_test_wav(&audio_path, 1.0, 48000, 2, 32)?;

    // Full pipeline
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    let resampled = resample(samples, sr, 16000)?;
    let spec = par_compute_spectrogram(&resampled, 512, 160, 400, false, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, 16000, 512, 40, None, None, MelScale::HTK);

    assert_eq!(mel_spec.len(), 40);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with different FFT sizes
#[test]
fn test_full_pipeline_different_fft_sizes() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_fft_sizes.wav");

    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

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

        let mel_spec = convert_to_mel(&spec, sr, n_fft, 40, None, None, MelScale::HTK);

        assert_eq!(mel_spec.len(), 40);
        assert!(mel_spec[0].len() > 0);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with different mel bin counts
#[test]
fn test_full_pipeline_different_mel_bins() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_mel_bins.wav");

    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);

    let mel_bin_counts = vec![20, 40, 80, 128];

    for n_mels in mel_bin_counts {
        let mel_spec = convert_to_mel(&spec, sr, 512, n_mels, None, None, MelScale::HTK);

        assert_eq!(mel_spec.len(), n_mels);
        assert_eq!(mel_spec[0].len(), spec[0].len());
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with different sample rates (no resampling)
#[test]
fn test_full_pipeline_different_sample_rates() -> Result<()> {
    let test_dir = setup_test_dir()?;

    let sample_rates = vec![8000, 16000, 22050, 44100, 48000];

    for sr in sample_rates {
        let audio_path = test_dir.join(format!("test_sr_{}.wav", sr));
        create_test_wav(&audio_path, 0.5, sr, 1, 16)?;

        let (samples, read_sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
        assert_eq!(read_sr, sr);

        let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);
        let mel_spec = convert_to_mel(&spec, sr, 512, 40, None, None, MelScale::HTK);

        assert_eq!(mel_spec.len(), 40);
        assert!(mel_spec[0].len() > 0);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with magnitude spectrogram
#[test]
fn test_full_pipeline_magnitude_spectrogram() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_magnitude.wav");

    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Magnitude);
    let mel_spec = convert_to_mel(&spec, sr, 512, 40, None, None, MelScale::HTK);

    assert_eq!(mel_spec.len(), 40);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with centered windowing
#[test]
fn test_full_pipeline_centered() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_centered.wav");

    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    let spec = par_compute_spectrogram(&samples, 512, 160, 400, true, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, sr, 512, 40, None, None, MelScale::HTK);

    assert_eq!(mel_spec.len(), 40);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with Slaney mel scale
#[test]
fn test_full_pipeline_slaney_mel_scale() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_slaney.wav");

    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, sr, 512, 40, None, None, MelScale::Slaney);

    assert_eq!(mel_spec.len(), 40);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with custom frequency range
#[test]
fn test_full_pipeline_custom_freq_range() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_freq_range.wav");

    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, sr, 512, 40, Some(300.0), Some(4000.0), MelScale::HTK);

    assert_eq!(mel_spec.len(), 40);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with complex multi-frequency audio
#[test]
fn test_full_pipeline_complex_audio() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_complex_full.wav");

    create_complex_test_wav(&audio_path, 2.0, 22050, 1, 16)?;

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    let resampled = resample(samples, sr, 16000)?;
    let spec = par_compute_spectrogram(&resampled, 1024, 256, 512, false, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, 16000, 1024, 80, None, None, MelScale::HTK);

    // Check that multiple mel bins captured energy
    let mut bins_with_energy = 0;
    for mel_bin in &mel_spec {
        let max_value = mel_bin.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_value > 0.1 {
            bins_with_energy += 1;
        }
    }

    assert!(bins_with_energy > 10);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with very short audio
#[test]
fn test_full_pipeline_short_audio() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_short.wav");

    create_test_wav(&audio_path, 0.1, 16000, 1, 16)?; // 100ms

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    let spec = par_compute_spectrogram(&samples, 256, 128, 256, false, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, sr, 256, 20, None, None, MelScale::HTK);

    assert_eq!(mel_spec.len(), 20);
    assert!(mel_spec[0].len() > 0);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Integration test: Test with long audio
#[test]
fn test_full_pipeline_long_audio() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_long.wav");

    create_test_wav(&audio_path, 10.0, 16000, 1, 16)?; // 10 seconds

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, sr, 512, 40, None, None, MelScale::HTK);

    assert_eq!(mel_spec.len(), 40);
    // Should have many frames for 10 seconds
    assert!(mel_spec[0].len() > 500);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}
