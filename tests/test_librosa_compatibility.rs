mod common;

use anyhow::{Context, Result};
use common::{cleanup_test_dir, create_complex_test_wav, create_test_wav, setup_test_dir};
use serde_json::Value;
use spectrs::io::audio::read_audio_file_mono;
use spectrs::spectrogram::mel::{MelScale, convert_to_mel};
use spectrs::spectrogram::stft::{SpectrogramType, par_compute_spectrogram};
use std::fs;
use std::process::Command;

/// Compatibility thresholds for librosa comparison
const CORRELATION_THRESHOLD: f32 = 0.95; // High correlation expected
const RELATIVE_ERROR_THRESHOLD: f32 = 0.15; // Allow 15% relative error

/// Helper function to save spectrogram as JSON
fn save_spectrogram_json(spec: &[Vec<f32>], output_path: &str) -> Result<()> {
    let json = serde_json::json!({
        "data": spec,
        "shape": [spec.len(), spec[0].len()],
    });

    fs::write(output_path, serde_json::to_string(&json)?)?;
    Ok(())
}

/// Helper function to run Python script using uv
fn run_python_script(script_path: &str, args: &[&str]) -> Result<()> {
    let output = Command::new("uv")
        .arg("run")
        .arg(script_path)
        .args(args)
        .output()
        .context("Failed to execute Python script with uv")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("Python script stderr: {}", stderr);
        eprintln!("Python script stdout: {}", stdout);
        anyhow::bail!("Python script failed with status: {}", output.status);
    }

    Ok(())
}

/// Helper function to compare spectrograms and check thresholds
fn compare_with_librosa(
    spectrs_json: &str,
    librosa_json: &str,
    comparison_json: &str,
) -> Result<(f32, f32)> {
    // Run comparison script
    run_python_script(
        "tests/benchmark/compare_spectrograms.py",
        &[spectrs_json, librosa_json, comparison_json],
    )?;

    // Load comparison results
    let results_str = fs::read_to_string(comparison_json)?;
    let results: Value = serde_json::from_str(&results_str)?;

    let correlation = results["correlation"].as_f64().unwrap() as f32;
    let relative_error = results["relative_error"].as_f64().unwrap() as f32;

    Ok((correlation, relative_error))
}

#[test]
#[ignore] // Ignore by default as it requires Python/librosa
fn test_stft_compatibility_basic() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_librosa_stft.wav");

    // Create test file
    create_test_wav(&audio_path, 2.0, 16000, 1, 16)?;

    // Compute spectrogram with spectrs
    let (samples, _sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);

    // Save spectrs output
    let spectrs_json = test_dir.join("spectrs_stft.json");
    save_spectrogram_json(&spec, spectrs_json.to_str().unwrap())?;

    // Generate librosa output
    let librosa_json = test_dir.join("librosa_stft.json");
    let params_json = test_dir.join("params.json");
    fs::write(
        &params_json,
        serde_json::json!({
            "type": "stft",
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 400,
            "center": false,
        })
        .to_string(),
    )?;

    run_python_script(
        "tests/benchmark/generate_librosa_spectrogram.py",
        &[
            audio_path.to_str().unwrap(),
            librosa_json.to_str().unwrap(),
            params_json.to_str().unwrap(),
        ],
    )?;

    // Compare
    let comparison_json = test_dir.join("comparison.json");
    let (correlation, relative_error) = compare_with_librosa(
        spectrs_json.to_str().unwrap(),
        librosa_json.to_str().unwrap(),
        comparison_json.to_str().unwrap(),
    )?;

    println!(
        "STFT Basic - Correlation: {:.4}, Relative Error: {:.4}",
        correlation, relative_error
    );

    assert!(
        correlation >= CORRELATION_THRESHOLD,
        "Correlation {:.4} below threshold {:.4}",
        correlation,
        CORRELATION_THRESHOLD
    );
    assert!(
        relative_error <= RELATIVE_ERROR_THRESHOLD,
        "Relative error {:.4} above threshold {:.4}",
        relative_error,
        RELATIVE_ERROR_THRESHOLD
    );

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
#[ignore]
fn test_stft_compatibility_different_fft_sizes() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_librosa_fft.wav");

    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    let (samples, _sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;

    let fft_sizes = vec![256, 512, 1024];

    for n_fft in fft_sizes {
        let hop_length = n_fft / 4;
        let win_length = n_fft / 2;

        // Spectrs
        let spec = par_compute_spectrogram(
            &samples,
            n_fft,
            hop_length,
            win_length,
            false,
            SpectrogramType::Power,
        );
        let spectrs_json = test_dir.join(format!("spectrs_stft_{}.json", n_fft));
        save_spectrogram_json(&spec, spectrs_json.to_str().unwrap())?;

        // Librosa
        let librosa_json = test_dir.join(format!("librosa_stft_{}.json", n_fft));
        let params_json = test_dir.join(format!("params_{}.json", n_fft));
        fs::write(
            &params_json,
            serde_json::json!({
                "type": "stft",
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "center": false,
            })
            .to_string(),
        )?;

        run_python_script(
            "tests/benchmark/generate_librosa_spectrogram.py",
            &[
                audio_path.to_str().unwrap(),
                librosa_json.to_str().unwrap(),
                params_json.to_str().unwrap(),
            ],
        )?;

        // Compare
        let comparison_json = test_dir.join(format!("comparison_{}.json", n_fft));
        let (correlation, relative_error) = compare_with_librosa(
            spectrs_json.to_str().unwrap(),
            librosa_json.to_str().unwrap(),
            comparison_json.to_str().unwrap(),
        )?;

        println!(
            "STFT n_fft={} - Correlation: {:.4}, Relative Error: {:.4}",
            n_fft, correlation, relative_error
        );

        assert!(correlation >= CORRELATION_THRESHOLD);
        assert!(relative_error <= RELATIVE_ERROR_THRESHOLD);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
#[ignore]
fn test_mel_compatibility_htk() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_librosa_mel.wav");

    create_test_wav(&audio_path, 2.0, 16000, 1, 16)?;

    // Spectrs
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, sr, 512, 40, None, None, MelScale::HTK);

    let spectrs_json = test_dir.join("spectrs_mel.json");
    save_spectrogram_json(&mel_spec, spectrs_json.to_str().unwrap())?;

    // Librosa
    let librosa_json = test_dir.join("librosa_mel.json");
    let params_json = test_dir.join("params_mel.json");
    fs::write(
        &params_json,
        serde_json::json!({
            "type": "mel",
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 400,
            "center": false,
            "n_mels": 40,
            "f_min": 0.0,
            "f_max": sr as f32 / 2.0,
            "htk": true,
        })
        .to_string(),
    )?;

    run_python_script(
        "tests/benchmark/generate_librosa_spectrogram.py",
        &[
            audio_path.to_str().unwrap(),
            librosa_json.to_str().unwrap(),
            params_json.to_str().unwrap(),
        ],
    )?;

    // Compare
    let comparison_json = test_dir.join("comparison_mel.json");
    let (correlation, relative_error) = compare_with_librosa(
        spectrs_json.to_str().unwrap(),
        librosa_json.to_str().unwrap(),
        comparison_json.to_str().unwrap(),
    )?;

    println!(
        "Mel HTK - Correlation: {:.4}, Relative Error: {:.4}",
        correlation, relative_error
    );

    assert!(correlation >= CORRELATION_THRESHOLD);
    assert!(relative_error <= RELATIVE_ERROR_THRESHOLD);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
#[ignore]
fn test_mel_compatibility_different_n_mels() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_librosa_nmels.wav");

    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);

    let n_mels_values = vec![20, 40, 80];

    for n_mels in n_mels_values {
        // Spectrs
        let mel_spec = convert_to_mel(&spec, sr, 512, n_mels, None, None, MelScale::HTK);
        let spectrs_json = test_dir.join(format!("spectrs_mel_{}.json", n_mels));
        save_spectrogram_json(&mel_spec, spectrs_json.to_str().unwrap())?;

        // Librosa
        let librosa_json = test_dir.join(format!("librosa_mel_{}.json", n_mels));
        let params_json = test_dir.join(format!("params_mel_{}.json", n_mels));
        fs::write(
            &params_json,
            serde_json::json!({
                "type": "mel",
                "n_fft": 512,
                "hop_length": 160,
                "win_length": 400,
                "center": false,
                "n_mels": n_mels,
                "f_min": 0.0,
                "f_max": sr as f32 / 2.0,
                "htk": true,
            })
            .to_string(),
        )?;

        run_python_script(
            "tests/benchmark/generate_librosa_spectrogram.py",
            &[
                audio_path.to_str().unwrap(),
                librosa_json.to_str().unwrap(),
                params_json.to_str().unwrap(),
            ],
        )?;

        // Compare
        let comparison_json = test_dir.join(format!("comparison_mel_{}.json", n_mels));
        let (correlation, relative_error) = compare_with_librosa(
            spectrs_json.to_str().unwrap(),
            librosa_json.to_str().unwrap(),
            comparison_json.to_str().unwrap(),
        )?;

        println!(
            "Mel n_mels={} - Correlation: {:.4}, Relative Error: {:.4}",
            n_mels, correlation, relative_error
        );

        assert!(correlation >= CORRELATION_THRESHOLD);
        assert!(relative_error <= RELATIVE_ERROR_THRESHOLD);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
#[ignore]
fn test_mel_compatibility_slaney() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_librosa_slaney.wav");

    create_test_wav(&audio_path, 1.0, 16000, 1, 16)?;

    // Spectrs
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, sr, 512, 40, None, None, MelScale::Slaney);

    let spectrs_json = test_dir.join("spectrs_mel_slaney.json");
    save_spectrogram_json(&mel_spec, spectrs_json.to_str().unwrap())?;

    // Librosa (htk=false means Slaney)
    let librosa_json = test_dir.join("librosa_mel_slaney.json");
    let params_json = test_dir.join("params_mel_slaney.json");
    fs::write(
        &params_json,
        serde_json::json!({
            "type": "mel",
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 400,
            "center": false,
            "n_mels": 40,
            "f_min": 0.0,
            "f_max": sr as f32 / 2.0,
            "htk": false,
        })
        .to_string(),
    )?;

    run_python_script(
        "tests/benchmark/generate_librosa_spectrogram.py",
        &[
            audio_path.to_str().unwrap(),
            librosa_json.to_str().unwrap(),
            params_json.to_str().unwrap(),
        ],
    )?;

    // Compare
    let comparison_json = test_dir.join("comparison_mel_slaney.json");
    let (correlation, relative_error) = compare_with_librosa(
        spectrs_json.to_str().unwrap(),
        librosa_json.to_str().unwrap(),
        comparison_json.to_str().unwrap(),
    )?;

    println!(
        "Mel Slaney - Correlation: {:.4}, Relative Error: {:.4}",
        correlation, relative_error
    );

    assert!(correlation >= CORRELATION_THRESHOLD);
    assert!(relative_error <= RELATIVE_ERROR_THRESHOLD);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
#[ignore]
fn test_compatibility_complex_signal() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let audio_path = test_dir.join("test_librosa_complex.wav");

    create_complex_test_wav(&audio_path, 2.0, 16000, 1, 16)?;

    // Spectrs
    let (samples, sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
    let spec = par_compute_spectrogram(&samples, 1024, 256, 512, false, SpectrogramType::Power);
    let mel_spec = convert_to_mel(&spec, sr, 1024, 80, None, None, MelScale::HTK);

    let spectrs_json = test_dir.join("spectrs_complex.json");
    save_spectrogram_json(&mel_spec, spectrs_json.to_str().unwrap())?;

    // Librosa
    let librosa_json = test_dir.join("librosa_complex.json");
    let params_json = test_dir.join("params_complex.json");
    fs::write(
        &params_json,
        serde_json::json!({
            "type": "mel",
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 512,
            "center": false,
            "n_mels": 80,
            "f_min": 0.0,
            "f_max": sr as f32 / 2.0,
            "htk": true,
        })
        .to_string(),
    )?;

    run_python_script(
        "tests/benchmark/generate_librosa_spectrogram.py",
        &[
            audio_path.to_str().unwrap(),
            librosa_json.to_str().unwrap(),
            params_json.to_str().unwrap(),
        ],
    )?;

    // Compare
    let comparison_json = test_dir.join("comparison_complex.json");
    let (correlation, relative_error) = compare_with_librosa(
        spectrs_json.to_str().unwrap(),
        librosa_json.to_str().unwrap(),
        comparison_json.to_str().unwrap(),
    )?;

    println!(
        "Complex Signal - Correlation: {:.4}, Relative Error: {:.4}",
        correlation, relative_error
    );

    assert!(correlation >= CORRELATION_THRESHOLD);
    assert!(relative_error <= RELATIVE_ERROR_THRESHOLD);

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

#[test]
#[ignore]
fn test_compatibility_different_sample_rates() -> Result<()> {
    let test_dir = setup_test_dir()?;

    let sample_rates = vec![8000, 16000, 22050];

    for sr in sample_rates {
        let audio_path = test_dir.join(format!("test_librosa_sr_{}.wav", sr));
        create_test_wav(&audio_path, 1.0, sr, 1, 16)?;

        // Spectrs
        let (samples, read_sr) = read_audio_file_mono(audio_path.to_str().unwrap())?;
        let spec = par_compute_spectrogram(&samples, 512, 160, 400, false, SpectrogramType::Power);
        let mel_spec = convert_to_mel(&spec, read_sr, 512, 40, None, None, MelScale::HTK);

        let spectrs_json = test_dir.join(format!("spectrs_sr_{}.json", sr));
        save_spectrogram_json(&mel_spec, spectrs_json.to_str().unwrap())?;

        // Librosa
        let librosa_json = test_dir.join(format!("librosa_sr_{}.json", sr));
        let params_json = test_dir.join(format!("params_sr_{}.json", sr));
        fs::write(
            &params_json,
            serde_json::json!({
                "type": "mel",
                "n_fft": 512,
                "hop_length": 160,
                "win_length": 400,
                "center": false,
                "n_mels": 40,
                "f_min": 0.0,
                "f_max": sr as f32 / 2.0,
                "htk": true,
            })
            .to_string(),
        )?;

        run_python_script(
            "tests/benchmark/generate_librosa_spectrogram.py",
            &[
                audio_path.to_str().unwrap(),
                librosa_json.to_str().unwrap(),
                params_json.to_str().unwrap(),
            ],
        )?;

        // Compare
        let comparison_json = test_dir.join(format!("comparison_sr_{}.json", sr));
        let (correlation, relative_error) = compare_with_librosa(
            spectrs_json.to_str().unwrap(),
            librosa_json.to_str().unwrap(),
            comparison_json.to_str().unwrap(),
        )?;

        println!(
            "Sample Rate {} - Correlation: {:.4}, Relative Error: {:.4}",
            sr, correlation, relative_error
        );

        assert!(correlation >= CORRELATION_THRESHOLD);
        assert!(relative_error <= RELATIVE_ERROR_THRESHOLD);
    }

    cleanup_test_dir(&test_dir)?;
    Ok(())
}
