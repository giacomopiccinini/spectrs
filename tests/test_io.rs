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
    let (samples, sr) = read_audio_file_mono(&audio_path)?;

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
    let (samples, sr) = read_audio_file_mono(&audio_path)?;

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
    let (samples, sr) = read_audio_file_mono(&audio_path)?;

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
    let (samples, sr) = read_audio_file_mono(&audio_path)?;

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

        let (samples, read_sr) = read_audio_file_mono(&audio_path)?;

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
    let (samples, sr) = read_audio_file_mono(&audio_path)?;
    assert_eq!(sr, 44100);

    // Resample to 16000 Hz
    let resampled = resample(samples, sr, 16000)?;

    // Verify resampled length
    let expected_len = 16000; // 1 second at 16000 Hz
    assert!((resampled.len() as i32 - expected_len as i32).abs() < 100);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[cfg(feature = "image")]
#[test]
fn test_save_spectrogram_image_all_colormaps() -> Result<()> {
    use spectrs::io::image::{Colormap, save_spectrogram_image};
    use spectrs::spectrogram::stft::{SpectrogramType, par_compute_spectrogram};

    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_audio.wav");

    // Create test audio file with complex signal
    common::create_complex_test_wav(&audio_path, 1.0, 22050, 1, 16)?;

    // Read audio
    let (audio, _sr) = read_audio_file_mono(&audio_path)?;

    // Compute spectrogram
    let spec = par_compute_spectrogram(&audio, 512, 128, 512, true, SpectrogramType::Magnitude);

    // Test all colormaps
    let colormaps = vec![
        (Colormap::Viridis, "viridis.png"),
        (Colormap::Magma, "magma.png"),
        (Colormap::Inferno, "inferno.png"),
        (Colormap::Plasma, "plasma.png"),
        (Colormap::Gray, "gray.png"),
    ];

    for (colormap, filename) in colormaps {
        let output_path = test_dir.join(filename);
        save_spectrogram_image(&spec, output_path.clone(), colormap)?;

        // Verify file was created
        assert!(
            output_path.exists(),
            "Image file was not created for {:?}",
            colormap
        );

        // Verify file has non-zero size
        let metadata = std::fs::metadata(&output_path)?;
        assert!(metadata.len() > 0, "Image file is empty for {:?}", colormap);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[cfg(feature = "image")]
#[test]
fn test_save_spectrogram_image_viridis() -> Result<()> {
    use spectrs::io::image::{Colormap, save_spectrogram_image};
    use spectrs::spectrogram::stft::{SpectrogramType, par_compute_spectrogram};

    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_audio.wav");

    // Create test audio
    create_test_wav(&audio_path, 0.5, 22050, 1, 16)?;

    // Read and process
    let (audio, _sr) = read_audio_file_mono(&audio_path)?;
    let spec = par_compute_spectrogram(&audio, 512, 128, 512, true, SpectrogramType::Power);

    // Save with viridis (default colormap)
    let output_path = test_dir.join("spec_viridis.png");
    save_spectrogram_image(&spec, output_path.clone(), Colormap::Viridis)?;

    // Verify
    assert!(output_path.exists());
    let metadata = std::fs::metadata(&output_path)?;
    assert!(metadata.len() > 100); // Should be reasonably sized

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[cfg(feature = "image")]
#[test]
fn test_save_mel_spectrogram_image() -> Result<()> {
    use spectrs::io::image::{Colormap, save_spectrogram_image};
    use spectrs::spectrogram::mel::{MelScale, convert_to_mel};
    use spectrs::spectrogram::stft::{SpectrogramType, par_compute_spectrogram};

    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_audio.wav");

    // Create test audio
    common::create_complex_test_wav(&audio_path, 1.0, 22050, 1, 16)?;

    // Read and process
    let (audio, sr) = read_audio_file_mono(&audio_path)?;
    let spec = par_compute_spectrogram(&audio, 1024, 256, 1024, true, SpectrogramType::Power);

    // Convert to mel
    let mel_spec = convert_to_mel(
        &spec,
        sr,
        1024,
        64,
        Some(20.0),
        Some(8000.0),
        MelScale::Slaney,
    );

    // Save with different colormaps
    let output_viridis = test_dir.join("mel_viridis.png");
    let output_magma = test_dir.join("mel_magma.png");

    save_spectrogram_image(&mel_spec, output_viridis.clone(), Colormap::Viridis)?;
    save_spectrogram_image(&mel_spec, output_magma.clone(), Colormap::Magma)?;

    // Verify both files exist
    assert!(output_viridis.exists());
    assert!(output_magma.exists());

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[cfg(feature = "image")]
#[test]
fn test_save_spectrogram_image_edge_cases() -> Result<()> {
    use spectrs::io::image::{Colormap, save_spectrogram_image};

    let test_dir = setup_test_dir()?;

    // Test with small spectrogram
    let small_spec = vec![vec![1.0, 2.0, 3.0]; 10];
    let output_path = test_dir.join("small_spec.png");
    save_spectrogram_image(&small_spec, output_path.clone(), Colormap::Viridis)?;
    assert!(output_path.exists());

    // Test with all zeros
    let zero_spec = vec![vec![0.0; 100]; 50];
    let output_path = test_dir.join("zero_spec.png");
    save_spectrogram_image(&zero_spec, output_path.clone(), Colormap::Gray)?;
    assert!(output_path.exists());

    // Test with uniform values
    let uniform_spec = vec![vec![5.0; 100]; 50];
    let output_path = test_dir.join("uniform_spec.png");
    save_spectrogram_image(&uniform_spec, output_path.clone(), Colormap::Plasma)?;
    assert!(output_path.exists());

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[cfg(feature = "image")]
#[test]
fn test_save_spectrogram_different_dimensions() -> Result<()> {
    use spectrs::io::image::{Colormap, save_spectrogram_image};

    let test_dir = setup_test_dir()?;

    // Wide spectrogram (many time frames, few frequency bins)
    let wide_spec = vec![vec![1.0; 500]; 20];
    let output_path = test_dir.join("wide_spec.png");
    save_spectrogram_image(&wide_spec, output_path.clone(), Colormap::Inferno)?;
    assert!(output_path.exists());

    // Tall spectrogram (few time frames, many frequency bins)
    let tall_spec = vec![vec![1.0; 20]; 500];
    let output_path = test_dir.join("tall_spec.png");
    save_spectrogram_image(&tall_spec, output_path.clone(), Colormap::Viridis)?;
    assert!(output_path.exists());

    // Square spectrogram
    let square_spec = vec![vec![1.0; 128]; 128];
    let output_path = test_dir.join("square_spec.png");
    save_spectrogram_image(&square_spec, output_path.clone(), Colormap::Magma)?;
    assert!(output_path.exists());

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[cfg(feature = "image")]
#[test]
fn test_complete_pipeline_with_image() -> Result<()> {
    use spectrs::io::image::{Colormap, save_spectrogram_image};
    use spectrs::spectrogram::mel::{MelScale, convert_to_mel};
    use spectrs::spectrogram::stft::{SpectrogramType, par_compute_spectrogram};

    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("pipeline_test.wav");

    // Create test audio
    common::create_complex_test_wav(&audio_path, 1.0, 44100, 1, 16)?;

    // Full pipeline: read -> resample -> spectrogram -> mel -> save image
    let (audio, sr) = read_audio_file_mono(&audio_path)?;
    let audio = resample(audio, sr, 22050)?;

    let spec = par_compute_spectrogram(&audio, 2048, 512, 2048, true, SpectrogramType::Power);
    let mel = convert_to_mel(
        &spec,
        22050,
        2048,
        128,
        Some(20.0),
        Some(8000.0),
        MelScale::Slaney,
    );

    // Save with default librosa-style colormap
    let output_path = test_dir.join("pipeline_output.png");
    save_spectrogram_image(&mel, output_path.clone(), Colormap::Viridis)?;
    assert!(output_path.exists());

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[cfg(not(feature = "image"))]
#[test]
fn test_save_spectrogram_image_feature_disabled() {
    use std::path::PathBuf;

    use spectrs::io::image::{Colormap, save_spectrogram_image};

    let spec = vec![vec![1.0; 10]; 10];
    let result = save_spectrogram_image(&spec, PathBuf::from("test.png"), Colormap::Viridis);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Image feature not enabled")
    );
}
