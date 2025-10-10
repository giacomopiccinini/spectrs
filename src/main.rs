use anyhow::Result;
use hound::{SampleFormat, WavReader};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

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
fn compute_mel_spectrogram(
    audio: &[f32],
    n_samples: usize,
    hop_length: usize,
    win_length: usize,
) -> Vec<Vec<f32>> {
    // Set-up FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_samples);

    // Create Hann window
    let hann_window = create_hann_window(win_length);

    // Determine the number of frames
    let n_frames = (audio.len().saturating_sub(win_length)) / hop_length + 1;

    // Determine number of frequency bins
    let n_freq_bins = n_samples / 2 + 1;

    // Init spectrogram
    let mut spectrogram = vec![vec![0.0f32; n_frames]; n_freq_bins];

    for frame_idx in 0..n_frames {
        // Determine start and end sample for each frame, recalling that hop_length is a stride
        // If the end is after the end of the audio it might still be good (depending on start, see after)
        let start = frame_idx * hop_length;
        let end = (start + win_length).clamp(0, audio.len());

        // Start is beyond the end of the file
        if start > audio.len() {
            break;
        }

        // Apply Hann window and convert to complex
        let mut frame: Vec<Complex<f32>> = audio[start..end]
            .iter()
            .zip(hann_window.iter())
            .map(|(&sample, &win)| Complex::new(sample * win, 0.0))
            .collect();

        // Pad with zeros if needed. This is because usually win_length < n_samples
        // and the missing are just zeros (in this case complex zeros)
        frame.resize(n_samples, Complex::new(0.0, 0.0));

        // Compute FFT in-place
        fft.process(&mut frame);

        // Store magnitude of first half (positive frequencies only)
        for (freq_idx, complex_val) in frame
            .iter()
            .take(n_freq_bins) // Only take positive frequencies (i.e. all the bins we expect)
            .enumerate()
        {
            spectrogram[freq_idx][frame_idx] = complex_val.norm();
        }
    }

    spectrogram
}

fn compute_mel_spectrogram_par(
    audio: &[f32],
    n_samples: usize,
    hop_length: usize,
    win_length: usize,
) -> Vec<Vec<f32>> {
    // FFT plan can be shared across threads (Arc<dyn Fft<_>> is Send + Sync)
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_samples);

    let hann = create_hann_window(win_length);

    let n_frames = (audio.len().saturating_sub(win_length)) / hop_length + 1;
    let n_freq_bins = n_samples / 2 + 1;

    // Frame-major for safe parallel writes: spectrogram[frame][freq]
    let mut spec_frame_major = vec![vec![0.0f32; n_freq_bins]; n_frames];

    // Parallel loop over frames
    spec_frame_major
        .par_iter_mut()
        .enumerate()
        .for_each(|(frame_idx, out_row)| {
            // Thread-local buffers
            let mut frame = vec![Complex::<f32>::new(0.0, 0.0); n_samples];

            // Slice indices
            let start = frame_idx * hop_length;
            if start >= audio.len() {
                return;
            }
            let end = (start + win_length).min(audio.len());

            // Window & copy into complex buffer (in-place, zero-padded)
            let src = &audio[start..end];
            let win = &hann[..src.len()];
            for (dst, (&s, &w)) in frame.iter_mut().zip(src.iter().zip(win.iter())) {
                dst.re = s * w;
                dst.im = 0.0;
            }

            // Optional: allocate scratch once per thread if you want
            // let mut scratch = vec![Complex::<f32>::new(0.0, 0.0); fft.get_inplace_scratch_len()];
            // fft.process_with_scratch(&mut frame, &mut scratch);
            fft.process(&mut frame);

            // Store positive freqs; use power (norm_sqr) for speed + typical usage
            for (k, c) in frame.iter().take(n_freq_bins).enumerate() {
                out_row[k] = c.norm_sqr();
            }
        });

    // If your downstream expects [freq][frame], transpose once (cache-friendly)
    let mut spectrogram = vec![vec![0.0f32; n_frames]; n_freq_bins];
    for (t, row) in spec_frame_major.into_iter().enumerate() {
        for (f, v) in row.into_iter().enumerate() {
            spectrogram[f][t] = v;
        }
    }
    spectrogram
}

fn mel_spec_from_wav(
    path: &str,
    n_samples: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Vec<Vec<f32>>> {
    // 1) Open WAV
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    // 2) Read samples → f32 in [-1, 1]
    let mut mono_f32: Vec<f32> = if spec.sample_format == SampleFormat::Float {
        // Float WAV
        let mut s: Vec<f32> = reader.samples::<f32>().map(|r| r.unwrap_or(0.0)).collect();

        // Downmix to mono if needed (interleaved channels)
        if spec.channels > 1 {
            s = downmix_interleaved_to_mono(&s, spec.channels as usize);
        }
        s
    } else {
        // Integer WAV
        match spec.bits_per_sample {
            0..=16 => {
                // Read as i16, scale to [-1,1]
                let s_i16: Vec<i16> = reader.samples::<i16>().map(|r| r.unwrap_or(0)).collect();
                let mut s: Vec<f32> = s_i16.iter().map(|&x| x as f32 / 32768.0).collect();
                if spec.channels > 1 {
                    s = downmix_interleaved_to_mono(&s, spec.channels as usize);
                }
                s
            }
            _ => {
                // Read as i32 for 24/32-bit PCM, scale to [-1,1]
                let s_i32: Vec<i32> = reader.samples::<i32>().map(|r| r.unwrap_or(0)).collect();
                let mut s: Vec<f32> = s_i32.iter().map(|&x| x as f32 / 2147483648.0).collect();
                if spec.channels > 1 {
                    s = downmix_interleaved_to_mono(&s, spec.channels as usize);
                }
                s
            }
        }
    };

    // (Optional) DC offset removal / light normalization could go here if you like.
    // keep it minimal as requested.

    // 3) Compute spectrogram (your function)
    let spec = compute_mel_spectrogram(&mono_f32, n_samples, hop_length, win_length);
    Ok(spec)
}

// Helper: average interleaved channels into mono
fn downmix_interleaved_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let frames = samples.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut acc = 0.0f32;
        let base = f * channels;
        for c in 0..channels {
            acc += samples[base + c];
        }
        mono.push(acc / (channels as f32));
    }
    mono
}

pub fn mel_spec_from_wav_par(
    path: &str,
    n_samples: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Vec<Vec<f32>>> {
    // 1) Open WAV
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    // 2) Read samples → f32 in [-1, 1]
    let mut mono_f32: Vec<f32> = if spec.sample_format == SampleFormat::Float {
        // Float WAV
        let mut s: Vec<f32> = reader.samples::<f32>().map(|r| r.unwrap_or(0.0)).collect();

        // Downmix to mono if needed (interleaved channels)
        if spec.channels > 1 {
            s = downmix_interleaved_to_mono(&s, spec.channels as usize);
        }
        s
    } else {
        // Integer WAV
        match spec.bits_per_sample {
            0..=16 => {
                // Read as i16, scale to [-1,1]
                let s_i16: Vec<i16> = reader.samples::<i16>().map(|r| r.unwrap_or(0)).collect();
                let mut s: Vec<f32> = s_i16.iter().map(|&x| x as f32 / 32768.0).collect();
                if spec.channels > 1 {
                    s = downmix_interleaved_to_mono(&s, spec.channels as usize);
                }
                s
            }
            _ => {
                // Read as i32 for 24/32-bit PCM, scale to [-1,1]
                let s_i32: Vec<i32> = reader.samples::<i32>().map(|r| r.unwrap_or(0)).collect();
                let mut s: Vec<f32> = s_i32.iter().map(|&x| x as f32 / 2147483648.0).collect();
                if spec.channels > 1 {
                    s = downmix_interleaved_to_mono(&s, spec.channels as usize);
                }
                s
            }
        }
    };

    // (Optional) DC offset removal / light normalization could go here if you like.
    // keep it minimal as requested.

    // 3) Compute spectrogram (your function)
    let spec = compute_mel_spectrogram_par(&mono_f32, n_samples, hop_length, win_length);
    Ok(spec)
}

fn main() -> Result<()> {
    mel_spec_from_wav_par("harvard.wav", 512, 160, 400);

    Ok(())
}
