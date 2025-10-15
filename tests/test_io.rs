mod common;

use anyhow::Result;
use common::{cleanup_test_dir, create_test_wav, setup_test_dir};
use spectrs::io::audio::{read_audio_file_mono, resample};

#[test]
fn test_read_audio_file_mono_mono_16bit() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_mono.wav");

    // Create mono 16-bit WAV file
    create_test_wav(&audio_path, 1.0, 44100, 1, 16)?;

    // Read the file
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    // Verify
    assert_eq!(sr, 44100);
    assert_eq!(samples.len(), 44100); // 1 second at 44100 Hz

    // Check that values are normalized between -1 and 1
    for sample in &samples {
        assert!(sample.abs() <= 1.0);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_read_audio_file_mono_stereo_16bit() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_stereo.wav");

    // Create stereo 16-bit WAV file
    create_test_wav(&audio_path, 1.0, 44100, 2, 16)?;

    // Read the file (should average channels)
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    // Verify
    assert_eq!(sr, 44100);
    assert_eq!(samples.len(), 44100); // 1 second at 44100 Hz

    // Check that values are normalized between -1 and 1
    for sample in &samples {
        assert!(sample.abs() <= 1.0);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_read_audio_file_mono_8bit() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_8bit.wav");

    // Create mono 8-bit WAV file
    create_test_wav(&audio_path, 0.5, 22050, 1, 8)?;

    // Read the file
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    // Verify
    assert_eq!(sr, 22050);
    assert_eq!(samples.len(), 11025); // 0.5 seconds at 22050 Hz

    // Check that values are normalized
    for sample in &samples {
        assert!(sample.abs() <= 1.0);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_read_audio_file_mono_32bit() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_32bit.wav");

    // Create mono 32-bit WAV file
    create_test_wav(&audio_path, 0.5, 48000, 1, 32)?;

    // Read the file
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    // Verify
    assert_eq!(sr, 48000);
    assert_eq!(samples.len(), 24000); // 0.5 seconds at 48000 Hz

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_read_audio_file_mono_different_sample_rates() -> Result<()> {
    let test_dir = setup_test_dir()?;

    let sample_rates = vec![8000, 16000, 22050, 44100, 48000];

    for sr in sample_rates {
        let audio_path = test_dir.join(format!("test_{}.wav", sr));
        create_test_wav(&audio_path, 0.1, sr, 1, 16)?;

        let (samples, read_sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

        assert_eq!(read_sr, sr);
        assert_eq!(samples.len(), (0.1 * sr as f32) as usize);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
fn test_resample_downsample() -> Result<()> {
    // Create test samples at 44100 Hz
    let original_sr = 44100;
    let target_sr = 22050;
    let duration = 1.0;

    let num_samples = (duration * original_sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / original_sr as f32).sin())
        .collect();

    // Resample
    let resampled = resample(samples, original_sr, target_sr)?;

    // Check length (should be approximately half)
    let expected_len = (duration * target_sr as f32) as usize;
    assert!((resampled.len() as i32 - expected_len as i32).abs() < 100); // Allow small difference

    // Check values are still normalized
    for sample in &resampled {
        assert!(sample.abs() <= 1.0);
    }

    Ok(())
}

#[test]
fn test_resample_upsample() -> Result<()> {
    // Create test samples at 22050 Hz
    let original_sr = 22050;
    let target_sr = 44100;
    let duration = 1.0;

    let num_samples = (duration * original_sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / original_sr as f32).sin())
        .collect();

    // Resample
    let resampled = resample(samples, original_sr, target_sr)?;

    // Check length (should be approximately double)
    let expected_len = (duration * target_sr as f32) as usize;
    assert!((resampled.len() as i32 - expected_len as i32).abs() < 100); // Allow small difference

    // Check values are still normalized
    for sample in &resampled {
        assert!(sample.abs() <= 1.0);
    }

    Ok(())
}

#[test]
fn test_resample_same_rate() -> Result<()> {
    // Create test samples at 44100 Hz
    let original_sr = 44100;
    let target_sr = 44100;
    let duration = 0.5;

    let num_samples = (duration * original_sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / original_sr as f32).sin())
        .collect();

    let original_len = samples.len();

    // Resample
    let resampled = resample(samples, original_sr, target_sr)?;

    // Length should be approximately the same
    assert!((resampled.len() as i32 - original_len as i32).abs() < 100);

    Ok(())
}

#[test]
fn test_resample_extreme_rates() -> Result<()> {
    // Test with extreme sample rate change
    let original_sr = 8000;
    let target_sr = 48000;
    let duration = 0.5; // Increased duration to ensure enough samples for resampler

    let num_samples = (duration * original_sr as f32) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|t| (t as f32 * 440.0 * 2.0 * std::f32::consts::PI / original_sr as f32).sin())
        .collect();

    // Resample
    let resampled = resample(samples, original_sr, target_sr)?;

    // Check that it worked and values are valid
    assert!(resampled.len() > 0);
    for sample in &resampled {
        assert!(sample.abs() <= 1.1); // Allow slight overshoot due to interpolation
    }

    Ok(())
}

#[test]
fn test_read_and_resample_integration() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_integration.wav");

    // Create test file at 44100 Hz
    create_test_wav(&audio_path, 1.0, 44100, 2, 16)?;

    // Read the file
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    assert_eq!(sr, 44100);

    // Resample to 16000 Hz
    let resampled = resample(samples, sr, 16000)?;

    // Verify resampled length
    let expected_len = 16000; // 1 second at 16000 Hz
    assert!((resampled.len() as i32 - expected_len as i32).abs() < 100);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}
