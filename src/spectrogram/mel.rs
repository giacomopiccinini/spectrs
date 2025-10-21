//use clap::ValueEnum;

// Different sconversions to mel scale
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum MelScale {
    HTK,
    Slaney,
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

    // Create weights
    // Equivalent to weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    let n_freq_bins = 1 + n_fft / 2;
    let mut weights: Vec<Vec<f32>> = vec![vec![0.0f32; n_freq_bins]; n_mels];

    // Compute fft frequencies.
    // From librosa official doc
    // librosa.fft_frequencies(sr=22050, n_fft=16)
    // array([     0.   ,   1378.125,   2756.25 ,   4134.375,
    //          5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
    // In Python would be [i * sr/n_fft for i in range(0, n_fft//2 + 1)]
    let fft_freqs: Vec<f32> = (0..=n_fft / 2_usize)
        .map(|i| i as f32 * sr as f32 / n_fft as f32)
        .collect();

    // Extract mel frequencies
    // Equivalent to Librosa mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    let mel_freqs: Vec<f32> = create_mel_frequencies(f_min, f_max, n_mels + 2, mel_scale);

    // Compute differences between subsequent mel frequencies
    // Equivalent to fdiff = np.diff(mel_f) in Librosa implementation
    let mel_freqs_diffs: Vec<f32> = mel_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    // Create ramps matrix: ramps[i][j] = mel_freqs[i] - fft_freqs[j]
    // Equivalent to ramps = np.subtract.outer(mel_f, fftfreqs) in Librosa
    let ramps: Vec<Vec<f32>> = mel_freqs
        .iter()
        .map(|&mel_freq| {
            fft_freqs
                .iter()
                .map(|&fft_freq| mel_freq - fft_freq)
                .collect()
        })
        .collect();

    // Creates triangular mel filter banks that convert linear frequency spectrograms
    // to perceptually-motivated mel-scale representations.
    for i in 0..n_mels {
        // Lower and upper slopes for all bins
        let lower: Vec<f32> = ramps[i].iter().map(|&r| -r / mel_freqs_diffs[i]).collect();

        let upper: Vec<f32> = ramps[i + 2]
            .iter()
            .map(|&r| r / mel_freqs_diffs[i + 1])
            .collect();

        // .. then intersect them with each other and zero
        weights[i] = lower
            .iter()
            .zip(upper.iter())
            .map(|(&l, &u)| 0.0f32.max(l.min(u)))
            .collect();
    }

    // Apply Slaney normalization (librosa's default, regardless of choice for mel scale)
    // Compute normalization factors: 2.0 / (mel_f[2:n_mels+2] - mel_f[0:n_mels])
    let enorm: Vec<f32> = (0..n_mels)
        .map(|i| 2.0 / (mel_freqs[i + 2] - mel_freqs[i]))
        .collect();

    // Apply normalization to each filter
    for i in 0..n_mels {
        for j in 0..n_freq_bins {
            weights[i][j] *= enorm[i];
        }
    }

    weights
}

/// Apply Mel filters to an already created spectrogram
pub fn convert_to_mel(
    spectrogram: &[Vec<f32>],
    sr: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: Option<f32>, // Lower cut-off frequency
    f_max: Option<f32>, // Upper cut-off frequency
    mel_scale: MelScale,
) -> Vec<Vec<f32>> {
    // Create mel filter bank matrix
    let mel_filters = create_mel_filter_bank(sr, n_fft, n_mels, f_min, f_max, mel_scale);

    // Apply filters: mel_spec[mel_bin][time] = sum(spec[freq][time] * filter[mel_bin][freq])
    let mut mel_spec = vec![vec![0.0; spectrogram[0].len()]; n_mels];

    for (mel_idx, filter) in mel_filters.iter().enumerate() {
        for time_idx in 0..spectrogram[0].len() {
            mel_spec[mel_idx][time_idx] = spectrogram
                .iter()
                .zip(filter.iter())
                .map(|(freq_bin, &filter_val)| freq_bin[time_idx] * filter_val)
                .sum();
        }
    }

    mel_spec
}
