#[cfg(feature = "image")]
use anyhow::Context;
use anyhow::Result;

/// Save a spectrogram as an image file
/// This function applies log scaling (log1p) to better visualize the spectrogram dynamics.
/// The image is oriented with frequency on the Y-axis (bottom to top) and time on the X-axis.
#[cfg(feature = "image")]
pub fn save_spectrogram_image(spectrogram: &[Vec<f32>], output_path: &str) -> Result<()> {
    use image::{ImageBuffer, Luma};

    let n_freq_bins = spectrogram.len();
    let n_frames = spectrogram[0].len();

    // Find min and max values after log scaling for normalization
    let log_values: Vec<Vec<f32>> = spectrogram
        .iter()
        .map(|row| row.iter().map(|&v| (v + 1.0).ln()).collect())
        .collect();

    let min_val = log_values
        .iter()
        .flatten()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_val = log_values
        .iter()
        .flatten()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let range = max_val - min_val;

    // Create image buffer (width = time, height = frequency)
    let mut img = ImageBuffer::new(n_frames as u32, n_freq_bins as u32);

    // Fill the image (flip vertically so low frequencies are at bottom)
    for (freq_idx, row) in log_values.iter().enumerate() {
        for (time_idx, &value) in row.iter().enumerate() {
            // Normalize to 0-255
            let normalized = if range > 0.0 {
                ((value - min_val) / range * 255.0) as u8
            } else {
                128u8
            };

            // Flip vertically: y = height - 1 - freq_idx
            let y = (n_freq_bins - 1 - freq_idx) as u32;
            let x = time_idx as u32;

            img.put_pixel(x, y, Luma([normalized]));
        }
    }

    // Save the image
    img.save(output_path)
        .with_context(|| format!("Failed to save image to {}", output_path))?;

    Ok(())
}

#[cfg(not(feature = "image"))]
pub fn save_spectrogram_image(_spectrogram: &[Vec<f32>], _output_path: &str) -> Result<()> {
    anyhow::bail!("Image feature not enabled. Compile with --features image to use this function.")
}
