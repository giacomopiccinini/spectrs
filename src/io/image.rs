#[cfg(feature = "image")]
use anyhow::Context;
use anyhow::Result;

/// Available colormaps for spectrogram visualization
#[derive(Debug, Clone, Copy, Default)]
pub enum Colormap {
    /// Perceptually uniform, great for spectrograms (matplotlib/librosa default)
    #[default]
    Viridis,
    /// Perceptually uniform, good for dark backgrounds
    Magma,
    /// Perceptually uniform, high contrast
    Inferno,
    /// Perceptually uniform, bright
    Plasma,
    /// Grayscale
    Gray,
}

/// Apply a colormap to a normalized value (0.0 to 1.0)
/// Returns RGB values as (r, g, b) in 0-255 range
fn apply_colormap(value: f32, colormap: Colormap) -> [u8; 3] {
    let v = value.clamp(0.0, 1.0);

    match colormap {
        Colormap::Viridis => viridis(v),
        Colormap::Magma => magma(v),
        Colormap::Inferno => inferno(v),
        Colormap::Plasma => plasma(v),
        Colormap::Gray => {
            let gray = (v * 255.0) as u8;
            [gray, gray, gray]
        }
    }
}

/// Viridis colormap (approximation using polynomial fits)
fn viridis(t: f32) -> [u8; 3] {
    let r = 0.267004
        + t * (0.004874
            + t * (2.827049
                + t * (-14.180787 + t * (23.635307 + t * (-18.567452 + t * 5.024689)))));
    let g = 0.004975
        + t * (0.408503
            + t * (2.820819 + t * (-11.188263 + t * (14.949977 + t * (-9.419796 + t * 2.192326)))));
    let b = 0.329415
        + t * (1.586134 + t * (-3.126778 + t * (6.283825 + t * (-7.528898 + t * 3.456374))));

    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// Magma colormap (approximation using polynomial fits)
fn magma(t: f32) -> [u8; 3] {
    let r = 0.001462
        + t * (0.062542
            + t * (4.777823
                + t * (-16.339474 + t * (26.246972 + t * (-20.125255 + t * 5.376646)))));
    let g = 0.001512
        + t * (0.142844
            + t * (2.433026 + t * (-9.125778 + t * (13.268_85 + t * (-10.347974 + t * 2.848073)))));
    let b = 0.013952
        + t * (1.581272 + t * (-4.124382 + t * (8.499725 + t * (-10.004358 + t * 4.043804))));

    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// Inferno colormap (approximation using polynomial fits)
fn inferno(t: f32) -> [u8; 3] {
    let r = 0.001788
        + t * (0.200942
            + t * (3.526908
                + t * (-11.642545 + t * (17.366_99 + t * (-12.351348 + t * 3.914439)))));
    let g = 0.000262
        + t * (0.034534
            + t * (2.445295 + t * (-9.520904 + t * (14.683786 + t * (-11.257943 + t * 3.114629)))));
    let b = 0.014227
        + t * (1.534345 + t * (-5.444_98 + t * (11.756179 + t * (-14.117606 + t * 6.258159))));

    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// Plasma colormap (approximation using polynomial fits)
fn plasma(t: f32) -> [u8; 3] {
    let r = 0.050383
        + t * (1.022623
            + t * (0.676103 + t * (-2.828869 + t * (5.187948 + t * (-3.728022 + t * 0.999914)))));
    let g = 0.029803
        + t * (0.237302
            + t * (2.524446 + t * (-9.892_82 + t * (14.450_55 + t * (-10.611676 + t * 2.963018)))));
    let b = 0.527975
        + t * (1.415069 + t * (-4.203048 + t * (8.238769 + t * (-9.742888 + t * 4.764384))));

    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// Save a spectrogram as an image file with colormap support
/// This function applies log scaling (log1p) to better visualize the spectrogram dynamics.
/// The image is oriented with frequency on the Y-axis (bottom to top) and time on the X-axis.
#[cfg(feature = "image")]
pub fn save_spectrogram_image(
    spectrogram: &[Vec<f32>],
    output_path: &str,
    colormap: Colormap,
) -> Result<()> {
    use image::{ImageBuffer, Rgb};

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
            // Normalize to 0.0-1.0
            let normalized = if range > 0.0 {
                (value - min_val) / range
            } else {
                0.5
            };

            // Apply colormap
            let rgb = apply_colormap(normalized, colormap);

            // Flip vertically: y = height - 1 - freq_idx
            let y = (n_freq_bins - 1 - freq_idx) as u32;
            let x = time_idx as u32;

            img.put_pixel(x, y, Rgb(rgb));
        }
    }

    // Save the image
    img.save(output_path)
        .with_context(|| format!("Failed to save image to {}", output_path))?;

    Ok(())
}

#[cfg(not(feature = "image"))]
pub fn save_spectrogram_image(
    _spectrogram: &[Vec<f32>],
    _output_path: &str,
    _colormap: Colormap,
) -> Result<()> {
    anyhow::bail!("Image feature not enabled. Compile with --features image to use this function.")
}
