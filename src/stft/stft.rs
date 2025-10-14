use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

// Different spectrogram types
#[derive(Debug, Clone, Copy)]
pub enum SpectrogramType {
    Magnitude,
    Power,
}

// Different sconversions to mel scale
#[derive(Debug, Clone, Copy)]
pub enum MelScale {
    HTK,
    Slaney,
}

/// Create Hann window, see e.g. https://en.wikipedia.org/wiki/Hann_function
fn create_hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (length - 1) as f32).cos()))
        .collect()
}

/// Compute the spectrogram
/// n_samples: number of samples in each Fast Fourier Transform (FFT) window
/// hop_length: stride between windows, i.e. number of samples between successive FFT frames
/// win_length: number of samples in the window function applied before FFT
/// Pad with zeros if needed. This is because usually win_length < n_samples
/// and the missing are just zeros (in this case complex zeros)
pub fn par_compute_spectrogram(
    audio: &[f32],
    n_samples: usize,
    hop_length: usize,
    win_length: usize,
    center: bool,
    spectrogram_type: SpectrogramType,
) -> Vec<Vec<f32>> {
    // Set-up FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_samples);

    // Choose the transformation function to create the spectrogram
    let transform_fn: fn(&Complex<f32>) -> f32 = match spectrogram_type {
        SpectrogramType::Magnitude => |c| c.norm(),
        SpectrogramType::Power => |c| c.norm_sqr(),
    };

    // Create (Hann) window
    let window = create_hann_window(win_length);

    // Determine the number of frames
    let n_frames = (audio.len().saturating_sub(win_length)) / hop_length + 1;

    // Determine number of frequency bins
    let n_freq_bins = n_samples / 2 + 1;

    // Frame-major spectrogram for safe parallel writes: spectrogram[frame][freq]
    // Eventually to be transposed
    let mut transposed_spectrogram = vec![vec![0.0f32; n_freq_bins]; n_frames];

    // Parallel loop over frames
    transposed_spectrogram
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

            // Store positive freqs only and apply transformation fn depending on request
            for (k, c) in frame.iter().take(n_freq_bins).enumerate() {
                out_row[k] = transform_fn(c);
            }
        });

    // If your downstream expects [freq][frame], transpose once (cache-friendly)
    let mut spectrogram = vec![vec![0.0f32; n_frames]; n_freq_bins];
    for (t, row) in transposed_spectrogram.into_iter().enumerate() {
        for (f, v) in row.into_iter().enumerate() {
            spectrogram[f][t] = v;
        }
    }
    spectrogram
}

/// Convert frequency in Hz to mel scale
fn hz_to_mel(hz: f32, mel_scale: MelScale) -> f32 {
    match mel_scale {
        MelScale::HTK => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney => {
            if hz < 1000.0 {
                3.0 * hz / 200.0
            } else {
                15.0 + 27.0 * (hz / 1000.0).log(6.4)
            }
        }
    }
}

/// Convert mel scale back to Hz (inverse formula of the above)
fn mel_to_hz(mel: f32, mel_scale: MelScale) -> f32 {
    match mel_scale {
        MelScale::HTK => 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            if mel < 15.0 {
                200.0 * mel / 3.0
            } else {
                6.4f32.powf((mel - 15.0) / 27.0) * 1000.0
            }
        }
    }
}

/// Compute an array of acoustic frequencies tuned to the mel scale
/// Because of psycho-acoustic there are two definitions, see (see e.g. https://en.wikipedia.org/wiki/Mel_scale)
/// for additional information.
fn create_mel_frequencies(f_min: f32, f_max: f32, n_mels: usize, mel_scale: MelScale) -> Vec<f32> {
    // Convert to mel scale
    let mel_min = hz_to_mel(f_min, mel_scale);
    let mel_max = hz_to_mel(f_max, mel_scale);

    // Create n_mels points linearly spaced in mel scale
    // Conforms to Librosa implementation
    // librosa.mel_frequencies(n_mels=40)
    // array([     0.   ,     85.317,    170.635,    255.952,
    //           341.269,    426.586,    511.904,    597.221,
    //           682.538,    767.855,    853.173,    938.49 ,
    //          1024.856,   1119.114,   1222.042,   1334.436,
    //          1457.167,   1591.187,   1737.532,   1897.337,
    //          2071.84 ,   2262.393,   2470.47 ,   2697.686,
    //          2945.799,   3216.731,   3512.582,   3835.643,
    //          4188.417,   4573.636,   4994.285,   5453.621,
    //          5955.205,   6502.92 ,   7101.009,   7754.107,
    //          8467.272,   9246.028,  10096.408,  11025.   ])
    let mel_freqs: Vec<f32> = (0..=n_mels - 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels - 1) as f32)
        .map(|f| mel_to_hz(f, mel_scale))
        .collect();

    mel_freqs
}

fn create_mel_filter_bank(
    sr: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: Option<f32>, // Lower cut-off frequency
    f_max: Option<f32>, // Upper cut-off frequency
    mel_scale: MelScale,
) -> Vec<Vec<f32>> {
    // Use provided values or defaults
    let f_min = f_min.unwrap_or(0.0);
    let f_max = f_max.unwrap_or(sr as f32 / 2.0); // (Nyquist theorem)

    //let weights = todo!();

    // Compute fft frequencies.
    // From librosa official doc
    // librosa.fft_frequencies(sr=22050, n_fft=16)
    // array([     0.   ,   1378.125,   2756.25 ,   4134.375,
    //          5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
    // In Python would be [i * sr/n_fft for i in range(0, n_fft//2 + 1)]
    let fft_freqs: Vec<f32> = (0..=n_fft / 2 as usize)
        .map(|i| i as f32 * sr as f32 / n_fft as f32)
        .collect();

    // Extract mel frequencies
    // Equivalent to Librosa mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    let mel_freqs: Vec<f32> = create_mel_frequencies(f_min, f_max, n_mels + 2, mel_scale);

    // Compute differences between subsequent mel frequencies
    // Equivalent to fdiff = np.diff(mel_f) in Librosa implementation
    let mel_freqs_diffs: Vec<f32> = mel_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    todo!();
}

// if fmax is None:
//         fmax = float(sr) / 2

//     # Initialize the weights
//     n_mels = int(n_mels)
//     weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

//     # Center freqs of each FFT bin
//     fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

//     # 'Center freqs' of mel bands - uniformly spaced between limits
//     mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

//     fdiff = np.diff(mel_f)
//     ramps = np.subtract.outer(mel_f, fftfreqs)

//     for i in range(n_mels):
//         # lower and upper slopes for all bins
//         lower = -ramps[i] / fdiff[i]
//         upper = ramps[i + 2] / fdiff[i + 1]

//         # .. then intersect them with each other and zero
//         weights[i] = np.maximum(0, np.minimum(lower, upper))

//     if isinstance(norm, str):
//         if norm == "slaney":
//             # Slaney-style mel is scaled to be approx constant energy per channel
//             enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
//             weights *= enorm[:, np.newaxis]
//         else:
//             raise ParameterError(f"Unsupported norm={norm}")
//     else:
//         weights = util.normalize(weights, norm=norm, axis=-1)

//     # Only check weights if f_mel[0] is positive
//     if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
//         # This means we have an empty channel somewhere
//         warnings.warn(
//             "Empty filters detected in mel frequency basis. "
//             "Some channels will produce empty responses. "
//             "Try increasing your sampling rate (and fmax) or "
//             "reducing n_mels.",
//             stacklevel=2,
//         )

//     return weights

// /// Create mel filter bank matrix
// /// Returns: Vec<Vec<f32>> where each inner Vec is a filter (one per mel bin)
// /// Each filter has length = n_freq_bins and contains the triangular filter weights
// fn create_mel_filter_bank(
//     n_freq_bins: usize,
//     sr: u32,
//     n_mels: usize,      // Number of mel filters to create
//     f_min: Option<f32>, // Lower cut-off frequency
//     f_max: Option<f32>, // Upper cut-off frequency
// ) -> Vec<Vec<f32>> {
//     // Use provided values or defaults
//     let f_min = f_min.unwrap_or(0.0);
//     let f_max = f_max.unwrap_or(sr as f32 / 2.0); // (Nyquist theorem)

//     // Convert to mel scale
//     let mel_min = hz_to_mel(f_min);
//     let mel_max = hz_to_mel(f_max);

//     // Create n_mels + 2 points linearly spaced in mel scale
//     let mel_points: Vec<f32> = (0..=n_mels + 1)
//         .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
//         .collect();

//     // Convert mel points back to Hz
//     let hz_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();

//     // Convert Hz to FFT bin indices
//     let bin_points: Vec<usize> = hz_points
//         .iter()
//         .map(|&hz| ((hz / sr as f32) * (n_freq_bins - 1) as f32 * 2.0).round() as usize)
//         .map(|bin| bin.min(n_freq_bins - 1)) // Clamp to valid range
//         .collect();

//     // Create triangular filters
//     let mut filter_bank = vec![vec![0.0f32; n_freq_bins]; n_mels];

//     for i in 0..n_mels {
//         let left = bin_points[i];
//         let center = bin_points[i + 1];
//         let right = bin_points[i + 2];

//         // Left slope (rising edge)
//         for bin in left..center {
//             if center > left {
//                 filter_bank[i][bin] = (bin - left) as f32 / (center - left) as f32;
//             }
//         }

//         // Right slope (falling edge)
//         for bin in center..=right.min(n_freq_bins - 1) {
//             if right > center {
//                 filter_bank[i][bin] = (right - bin) as f32 / (right - center) as f32;
//             }
//         }
//     }

//     filter_bank
// }

// /// Apply Mel filters to an already created spectrogram
// pub fn convert_to_mel(
//     spectrogram: &Vec<Vec<f32>>,
//     sr: u32,
//     n_freq_bins: usize,
//     n_mels: usize,
//     f_min: Option<f32>, // Lower cut-off frequency
//     f_max: Option<f32>, // Upper cut-off frequency
// ) -> Vec<Vec<f32>> {
//     // Create mel filter bank matrix
//     let mel_filters = create_mel_filter_bank(n_freq_bins, sr, n_mels, f_min, f_max);

//     // Apply filters: mel_spec[mel_bin][time] = sum(spec[freq][time] * filter[mel_bin][freq])
//     let mut mel_spec = vec![vec![0.0; spectrogram[0].len()]; n_mels];

//     for (mel_idx, filter) in mel_filters.iter().enumerate() {
//         for time_idx in 0..spectrogram[0].len() {
//             mel_spec[mel_idx][time_idx] = spectrogram
//                 .iter()
//                 .zip(filter.iter())
//                 .map(|(freq_bin, &filter_val)| freq_bin[time_idx] * filter_val)
//                 .sum();
//         }
//     }

//     mel_spec
// }
