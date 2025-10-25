mod common;

use anyhow::Result;
use common::{cleanup_test_dir, create_test_wav, setup_test_dir};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Helper function to get the path to the compiled binary
fn get_binary_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push("debug");
    path.push("spectrs");
    path
}

/// Test CLI with single file and default output (same directory as input)
#[test]
fn test_cli_single_file_default_output() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let input_wav = test_dir.join("test_audio.wav");
    let expected_output = test_dir.join("test_audio.png");

    // Create test audio file
    create_test_wav(&input_wav, 1.0, 16000, 1, 16)?;

    // Run the CLI
    let output = Command::new(get_binary_path())
        .arg(input_wav.to_str().unwrap())
        .output()
        .expect("Failed to execute spectrs");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that output file was created in same directory
    assert!(
        expected_output.exists(),
        "Output file not created at: {}",
        expected_output.display()
    );

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Test CLI with single file and custom output directory
#[test]
fn test_cli_single_file_custom_output_dir() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let input_wav = test_dir.join("test_audio.wav");
    let output_dir = test_dir.join("output");
    let expected_output = output_dir.join("test_audio.png");

    // Create test audio file and output directory
    create_test_wav(&input_wav, 1.0, 16000, 1, 16)?;
    fs::create_dir(&output_dir)?;

    // Run the CLI with --output-dir
    let output = Command::new(get_binary_path())
        .arg(input_wav.to_str().unwrap())
        .arg("--output-dir")
        .arg(output_dir.to_str().unwrap())
        .output()
        .expect("Failed to execute spectrs");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that output file was created in output directory
    assert!(
        expected_output.exists(),
        "Output file not created at: {}",
        expected_output.display()
    );

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Test CLI with input directory and custom output directory (preserves structure)
#[test]
fn test_cli_directory_custom_output_dir() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let input_dir = test_dir.join("input");
    let subdir = input_dir.join("subdir");
    let output_dir = test_dir.join("output");

    // Create directory structure and test audio files
    fs::create_dir(&input_dir)?;
    fs::create_dir(&subdir)?;
    fs::create_dir(&output_dir)?;

    let audio1 = input_dir.join("audio1.wav");
    let audio2 = subdir.join("audio2.wav");
    create_test_wav(&audio1, 1.0, 16000, 1, 16)?;
    create_test_wav(&audio2, 1.0, 16000, 1, 16)?;

    // Expected outputs preserving directory structure
    let expected_output1 = output_dir.join("audio1.png");
    let expected_output2 = output_dir.join("subdir").join("audio2.png");

    // Run the CLI with directory input and --output-dir
    let output = Command::new(get_binary_path())
        .arg(input_dir.to_str().unwrap())
        .arg("--output-dir")
        .arg(output_dir.to_str().unwrap())
        .output()
        .expect("Failed to execute spectrs");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that output files were created with preserved structure
    assert!(
        expected_output1.exists(),
        "Output file not created at: {}",
        expected_output1.display()
    );
    assert!(
        expected_output2.exists(),
        "Output file not created at: {}",
        expected_output2.display()
    );

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Test CLI with directory input and default output (same as input)
#[test]
fn test_cli_directory_default_output() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let input_dir = test_dir.join("audio_files");
    fs::create_dir(&input_dir)?;

    let audio1 = input_dir.join("audio1.wav");
    let audio2 = input_dir.join("audio2.wav");
    create_test_wav(&audio1, 1.0, 16000, 1, 16)?;
    create_test_wav(&audio2, 1.0, 16000, 1, 16)?;

    // Expected outputs in same directory as inputs
    let expected_output1 = input_dir.join("audio1.png");
    let expected_output2 = input_dir.join("audio2.png");

    // Run the CLI with directory input
    let output = Command::new(get_binary_path())
        .arg(input_dir.to_str().unwrap())
        .output()
        .expect("Failed to execute spectrs");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that output files were created in same directory
    assert!(
        expected_output1.exists(),
        "Output file not created at: {}",
        expected_output1.display()
    );
    assert!(
        expected_output2.exists(),
        "Output file not created at: {}",
        expected_output2.display()
    );

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Test CLI with nested directory structure and custom output
#[test]
fn test_cli_nested_directory_structure() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let input_dir = test_dir.join("input");
    let subdir1 = input_dir.join("dir1");
    let subdir2 = input_dir.join("dir2");
    let nested = subdir1.join("nested");
    let output_dir = test_dir.join("output");

    // Create nested directory structure
    fs::create_dir(&input_dir)?;
    fs::create_dir(&subdir1)?;
    fs::create_dir(&subdir2)?;
    fs::create_dir(&nested)?;
    fs::create_dir(&output_dir)?;

    // Create audio files in various locations
    let audio1 = input_dir.join("root.wav");
    let audio2 = subdir1.join("audio1.wav");
    let audio3 = subdir2.join("audio2.wav");
    let audio4 = nested.join("deep.wav");

    create_test_wav(&audio1, 1.0, 16000, 1, 16)?;
    create_test_wav(&audio2, 1.0, 16000, 1, 16)?;
    create_test_wav(&audio3, 1.0, 16000, 1, 16)?;
    create_test_wav(&audio4, 1.0, 16000, 1, 16)?;

    // Expected outputs preserving full structure
    let expected_output1 = output_dir.join("root.png");
    let expected_output2 = output_dir.join("dir1").join("audio1.png");
    let expected_output3 = output_dir.join("dir2").join("audio2.png");
    let expected_output4 = output_dir.join("dir1").join("nested").join("deep.png");

    // Run the CLI
    let output = Command::new(get_binary_path())
        .arg(input_dir.to_str().unwrap())
        .arg("--output-dir")
        .arg(output_dir.to_str().unwrap())
        .output()
        .expect("Failed to execute spectrs");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check all output files
    assert!(expected_output1.exists(), "root.png not created");
    assert!(expected_output2.exists(), "dir1/audio1.png not created");
    assert!(expected_output3.exists(), "dir2/audio2.png not created");
    assert!(
        expected_output4.exists(),
        "dir1/nested/deep.png not created"
    );

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Test CLI with various spectrogram parameters and output directory
#[test]
fn test_cli_with_parameters_and_output_dir() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let input_wav = test_dir.join("test.wav");
    let output_dir = test_dir.join("spectrograms");
    let expected_output = output_dir.join("test.png");

    create_test_wav(&input_wav, 2.0, 44100, 1, 16)?;
    fs::create_dir(&output_dir)?;

    // Run with various parameters (using win_length equal to n_fft to avoid issues)
    let output = Command::new(get_binary_path())
        .arg(input_wav.to_str().unwrap())
        .arg("--output-dir")
        .arg(output_dir.to_str().unwrap())
        .arg("--sr")
        .arg("16000")
        .arg("--n-fft")
        .arg("512")
        .arg("--hop-length")
        .arg("160")
        .arg("--win-length")
        .arg("512")
        .arg("--n-mels")
        .arg("80")
        .arg("--colormap")
        .arg("magma")
        .output()
        .expect("Failed to execute spectrs");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(expected_output.exists(), "Output file not created");

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Test that non-WAV files are ignored in directory processing
#[test]
fn test_cli_ignores_non_wav_files() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let input_dir = test_dir.join("mixed");
    let output_dir = test_dir.join("output");

    fs::create_dir(&input_dir)?;
    fs::create_dir(&output_dir)?;

    // Create WAV and non-WAV files
    let audio_wav = input_dir.join("audio.wav");
    let text_file = input_dir.join("readme.txt");
    let png_file = input_dir.join("image.png");

    create_test_wav(&audio_wav, 1.0, 16000, 1, 16)?;
    fs::write(&text_file, "This is a text file")?;
    fs::write(&png_file, "fake png data")?;

    // Run the CLI
    let output = Command::new(get_binary_path())
        .arg(input_dir.to_str().unwrap())
        .arg("--output-dir")
        .arg(output_dir.to_str().unwrap())
        .output()
        .expect("Failed to execute spectrs");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Only the WAV file should have generated output
    let expected_output = output_dir.join("audio.png");
    let unexpected_txt = output_dir.join("readme.png");
    let unexpected_png = output_dir.join("image.png");

    assert!(expected_output.exists(), "WAV output not created");
    assert!(!unexpected_txt.exists(), "Non-WAV file was processed");
    assert!(
        !unexpected_png.exists() || png_file.metadata()?.len() != expected_output.metadata()?.len(),
        "PNG file should not be reprocessed"
    );

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Test CLI creates parent directories when needed
#[test]
fn test_cli_creates_parent_directories() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let input_dir = test_dir.join("input");
    let subdir = input_dir.join("sub");
    let output_dir = test_dir.join("output");

    fs::create_dir(&input_dir)?;
    fs::create_dir(&subdir)?;
    fs::create_dir(&output_dir)?;

    let audio = subdir.join("audio.wav");
    create_test_wav(&audio, 1.0, 16000, 1, 16)?;

    // Expected output should create the subdirectory
    let expected_output = output_dir.join("sub").join("audio.png");

    // Run the CLI (should create output_dir/sub automatically)
    let output = Command::new(get_binary_path())
        .arg(input_dir.to_str().unwrap())
        .arg("--output-dir")
        .arg(output_dir.to_str().unwrap())
        .output()
        .expect("Failed to execute spectrs");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        expected_output.exists(),
        "Output file with parent dirs not created"
    );

    cleanup_test_dir(&test_dir)?;
    Ok(())
}

/// Test CLI error handling for non-existent input
#[test]
fn test_cli_nonexistent_input() -> Result<()> {
    let test_dir = setup_test_dir()?;
    let nonexistent = test_dir.join("does_not_exist.wav");

    let output = Command::new(get_binary_path())
        .arg(nonexistent.to_str().unwrap())
        .output()
        .expect("Failed to execute spectrs");

    // Should fail with non-zero exit code
    assert!(
        !output.status.success(),
        "CLI should fail for non-existent input"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("does not exist")
            || stderr.contains("not found")
            || stderr.contains("No such"),
        "Error message should mention missing file"
    );

    cleanup_test_dir(&test_dir)?;
    Ok(())
}
