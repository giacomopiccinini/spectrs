use anyhow::{Context, Result};
use spectrs::utils::audio::{read_audio_file_mono, resample};

fn main() -> Result<()> {
    let (audio, sr) = read_audio_file_mono("harvard.wav")?;

    let target_sr: u32 = 16_000;

    let resampled = resample(audio, sr, target_sr);

    Ok(())
}
