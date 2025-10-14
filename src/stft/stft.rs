use anyhow::Result;
use hound::{SampleFormat, WavReader};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Create Hann window, see e.g. https://en.wikipedia.org/wiki/Hann_function
fn create_hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (length - 1) as f32).cos()))
        .collect()
}

/// Compute the Mel spectrogram
/// n_samples: number of samples in each Fast Fourier Transform (FFT) window
/// hop_length: stride between windows, i.e. number of samples between successive FFT frames
/// win_length: number of samples in the window function applied before FFT
/// Pad with zeros if needed. This is because usually win_length < n_samples
/// and the missing are just zeros (in this case complex zeros)
fn par_compute_stft(
    audio: &[f32],
    n_samples: usize,
    hop_length: usize,
    win_length: usize,
    center: bool,
) -> Vec<Vec<f32>> {
    // Set-up FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_samples);

    // Create (Hann) window
    let window = create_hann_window(win_length);

    // Determine the number of frames
    let n_frames = (audio.len().saturating_sub(win_length)) / hop_length + 1;

    // Determine number of frequency bins
    let n_freq_bins = n_samples / 2 + 1;

    // Frame-major stft for safe parallel writes: stft[frame][freq]
    // Eventually to be transposed
    let mut transposed_stft = vec![vec![0.0f32; n_freq_bins]; n_frames];

    // Parallel loop over frames
    transposed_stft
        .par_iter_mut() // Auto-parallelize with rayon
        .enumerate() // Extract frame idx
        .for_each(|(frame_idx, out_row)| {
            // Determine start and end sample for each frame, recalling that hop_length is a stride
            // If the end is after the end of the audio it might still be good (depending on start, see after)
            let start = frame_idx * hop_length;
            let end = (start + win_length).clamp(0, audio.len());

            // Start is beyond the end of the file
            if start > audio.len() {
                return;
            }

            // Init thread-local buffers to be filled with windowed audio
            let mut frame = vec![Complex::<f32>::new(0.0, 0.0); n_samples];

            // Add an offset if the window needs to be centered
            let centering_offset = if center {
                (n_samples - win_length) / 2 as usize
            } else {
                0 as usize
            };

            // Window & copy into complex buffer
            let src = &audio[start..end];
            let win = &window[..src.len()];
            for (dst, (&s, &w)) in frame
                .iter_mut()
                .skip(centering_offset)
                .zip(src.iter().zip(win.iter()))
            {
                dst.re = s * w; // Convolve audio and window
                dst.im = 0.0; // No imaginary part
            }

            // Run FFT
            fft.process(&mut frame);

            // Store positive freqs; use power (norm_sqr) for speed + typical usage
            for (k, c) in frame.iter().take(n_freq_bins).enumerate() {
                out_row[k] = c.norm();
            }
        });

    // If your downstream expects [freq][frame], transpose once (cache-friendly)
    let mut stft = vec![vec![0.0f32; n_frames]; n_freq_bins];
    for (t, row) in transposed_stft.into_iter().enumerate() {
        for (f, v) in row.into_iter().enumerate() {
            stft[f][t] = v;
        }
    }
    stft
}
