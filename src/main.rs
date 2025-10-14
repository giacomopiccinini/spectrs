use spectrs::utils::audio::read_audio_file_mono;
use anyhow::{Context, Result};

fn main() -> Result<()>{
    let (audio, sr) = read_audio_file_mono("harvard.wav")?;

    Ok(())
}
