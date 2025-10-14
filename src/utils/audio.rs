use anyhow::{Context, Result};
use hound::WavReader;

/// Read audio file from file path and convert to mono by averaging left and right channel
pub fn read_audio_file_mono(audio_file_path: &str) -> Result<(Vec<f64>, u32)> {
    // Open the WAV file
    let mut reader =
        WavReader::open(audio_file_path).with_context(|| "Failed to open audio file")?;

    // Extract info from file
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;
    let bits_per_sample = spec.bits_per_sample;

    // Exit if if more than 2 channels
    if channels > 2 {
        return Err(anyhow::anyhow!(
            "Unsupported number of channels: {}. Only mono and stereo are supported.",
            channels
        ));
    }

    // Init samples vec
    let mut samples: Vec<f64> = Vec::new();

    // Calculate the maximum value based on bits_per_sample
    let max_value = 2_f64.powi(bits_per_sample as i32 - 1);

    // Define accumulator to compute average in case of stereo
    let mut acc = 0_i32;

    // Read into samples vec
    reader
        .samples::<i32>()
        .map(|s| s.with_context(|| "Couldn't read samples"))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .enumerate()
        .for_each(|(i, &sample)| {
            if channels == 2 {
                acc += sample;
                if i % 2 != 0 {
                    // Average and normalize by dividing by max_value
                    samples.push(acc as f64 / 2.0 / max_value);
                    acc = 0_i32;
                }
            } else if channels == 1 {
                // Normalize by dividing by max_value
                samples.push(sample as f64 / max_value);
            }
        });

    Ok((samples, sr))
}
