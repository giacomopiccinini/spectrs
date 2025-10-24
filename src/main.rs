use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use spectrs::io::audio::{read_audio_file_mono, resample};
use spectrs::io::image::{Colormap, save_spectrogram_image};
use spectrs::spectrogram::mel::{MelScale, convert_to_mel, par_convert_to_mel};
use spectrs::spectrogram::stft::{SpectrogramType, compute_spectrogram, par_compute_spectrogram};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Input file or directory
    #[arg(required = true)]
    pub input: String,

    /// Output directory path (optional). PNG files are created inside this directory with the same
    /// relative structure as inputs.
    #[arg(long)]
    pub output_dir: Option<String>,

    /// Target sample rate (optional). If specified, resampling is applied before spectrogram creation.
    #[arg(long)]
    pub sr: Option<u32>,

    /// FFT window size
    #[arg(long, default_value = "2048")]
    pub n_fft: usize,

    /// Hop length
    #[arg(long, default_value = "512")]
    pub hop_length: usize,

    /// Window length
    #[arg(long, default_value = "2048")]
    pub win_length: usize,

    /// Enable centering in the FFT window
    #[arg(long, default_value = "true")]
    pub center: bool,

    /// Spectrogram type
    #[arg(long, default_value = "power")]
    pub spec_type: SpectrogramType,

    /// Number of mel bands (optional, for mel spectrograms)
    #[arg(long)]
    pub n_mels: Option<usize>,

    /// Minimum frequency (Hz)
    #[arg(long, default_value = "0.0")]
    pub f_min: Option<f32>,

    /// Maximum frequency (Hz, optional). In unspecified, it's sr/2 by Nyquist theorem
    #[arg(long)]
    pub f_max: Option<f32>,

    /// Mel scale type (only applies to mel spectrograms)
    #[arg(long, default_value = "slaney")]
    pub mel_scale: MelScale,

    /// Colormap for visualization
    #[arg(long, default_value = "viridis")]
    pub colormap: Colormap,
}

/// Create spectrogram for a single file (uses parallel spectrogram computation)
fn par_create_spectrogram(
    input: &Path,
    output: &Path,
    sr: Option<u32>,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    center: bool,
    spec_type: SpectrogramType,
    n_mels: Option<usize>,
    f_min: Option<f32>,
    f_max: Option<f32>,
    mel_scale: MelScale,
    colormap: Colormap,
) -> Result<()> {
    // Read audio file and convert to mono
    let (mut audio, original_sr) =
        read_audio_file_mono(input).with_context(|| "Failed to read audio")?;

    // Resample if necessary
    let target_sr;
    if sr.is_some() && sr.unwrap() != original_sr {
        audio = resample(audio, original_sr, sr.unwrap())
            .with_context(|| "Failed to resample audio")?;
        target_sr = sr.unwrap();
    } else {
        target_sr = original_sr;
    }

    // Create spectrogram (parallelized over frames)
    let mut spec =
        par_compute_spectrogram(&audio, n_fft, hop_length, win_length, center, spec_type);

    // Convert to mel if necessary (parallelized over mel bands)
    if n_mels.is_some() {
        spec = par_convert_to_mel(
            &spec,
            target_sr,
            n_fft,
            n_mels.unwrap(),
            f_min,
            f_max,
            mel_scale,
        );
    }

    save_spectrogram_image(&spec, output.to_path_buf(), colormap)
        .with_context(|| "Failed to save spectogram")?;

    Ok(())
}

/// Create spectrogram for batch processing (uses sequential spectrogram computation)
fn create_spectrogram(
    input: &Path,
    output: &Path,
    sr: Option<u32>,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    center: bool,
    spec_type: SpectrogramType,
    n_mels: Option<usize>,
    f_min: Option<f32>,
    f_max: Option<f32>,
    mel_scale: MelScale,
    colormap: Colormap,
) -> Result<()> {
    // Read audio file and convert to mono
    let (mut audio, original_sr) =
        read_audio_file_mono(input).with_context(|| "Failed to read audio")?;

    // Resample if necessary
    let target_sr;
    if sr.is_some() && sr.unwrap() != original_sr {
        audio = resample(audio, original_sr, sr.unwrap())
            .with_context(|| "Failed to resample audio")?;
        target_sr = sr.unwrap();
    } else {
        target_sr = original_sr;
    }

    // Create spectrogram (sequential - parallelism is at file level)
    let mut spec = compute_spectrogram(&audio, n_fft, hop_length, win_length, center, spec_type);

    // Convert to mel if necessary (sequential - parallelism is at file level)
    if n_mels.is_some() {
        spec = convert_to_mel(
            &spec,
            target_sr,
            n_fft,
            n_mels.unwrap(),
            f_min,
            f_max,
            mel_scale,
        );
    }

    save_spectrogram_image(&spec, output.to_path_buf(), colormap)
        .with_context(|| "Failed to save spectogram")?;

    Ok(())
}

/// Compute the output path for a given input file
fn compute_output_path(
    file_path: &Path,
    base_path: &Path,
    output_dir: Option<&str>,
) -> Result<PathBuf> {
    if let Some(out_dir) = output_dir {
        let relative = if file_path == base_path {
            // Single file case - use just the filename
            // Example: file_path="raw/sound.wav", base_path="raw/sound.wav"
            //   → relative="sound.wav" → output="processed/sound.png"
            file_path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("Invalid file path: {}", file_path.display()))?
                .as_ref()
        } else {
            // Directory case - preserve subdirectory structure
            // Example: file_path="raw/b/sound.wav", base_path="raw/"
            //   → relative="b/sound.wav" → output="processed/b/sound.png"
            file_path.strip_prefix(base_path).with_context(|| {
                format!(
                    "Failed to compute relative path for: {}",
                    file_path.display()
                )
            })?
        };
        Ok(Path::new(out_dir).join(relative).with_extension("png"))
    } else {
        // Default: same directory as input
        Ok(file_path.with_extension("png"))
    }
}

fn main() -> Result<()> {
    // Parse the arguments
    let args = Cli::parse();

    // Parse the arguments
    let input = Path::new(&args.input);

    if !input.exists() {
        anyhow::bail!("Input path does not exist: {}", input.display());
    }

    // Case of single input file - use parallel spectrogram computation
    if input.is_file() && input.extension().and_then(|ext| ext.to_str()) == Some("wav") {
        let output = compute_output_path(&input, &input, args.output_dir.as_deref())?;

        par_create_spectrogram(
            &input,
            &output,
            args.sr,
            args.n_fft,
            args.hop_length,
            args.win_length,
            args.center,
            args.spec_type,
            args.n_mels,
            args.f_min,
            args.f_max,
            args.mel_scale,
            args.colormap,
        )
        .with_context(|| "Failed to create spectrogram")?;
    }
    // Case of input being a directory - parallelize over files, sequential spectrogram
    else {
        let files: Vec<_> = WalkDir::new(input)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("wav"))
            .map(|e| e.path().to_path_buf())
            .collect();

        files
            .par_iter()
            .try_for_each(|file| -> Result<()> {
                let output = compute_output_path(file, input, args.output_dir.as_deref())?;

                create_spectrogram(
                    &file,
                    &output,
                    args.sr,
                    args.n_fft,
                    args.hop_length,
                    args.win_length,
                    args.center,
                    args.spec_type,
                    args.n_mels,
                    args.f_min,
                    args.f_max,
                    args.mel_scale,
                    args.colormap,
                )
            })
            .with_context(|| "Failed to create spectrogram")?;
    };

    Ok(())
}
