use anyhow::Result;
use spectrs::io::audio::{read_audio_file_mono, resample};
use spectrs::spectrogram::mel::{MelScale, convert_to_mel};
use spectrs::spectrogram::stft::{SpectrogramType, par_compute_spectrogram};

fn main() -> Result<()> {
    let (audio, sr) = read_audio_file_mono("harvard.wav")?;

    let target_sr: u32 = 16_000;

    let resampled_audio = resample(audio, sr, target_sr)?;

    let n_samples = 512;
    let hop_length = 160;
    let win_length = 400;
    let center = true;
    let st = SpectrogramType::Power;
    let ms = MelScale::Slaney;
    let n_freq_bins = n_samples / 2 + 1;
    let n_fft = 2 * (n_freq_bins - 1);
    let n_mels = 80;
    let f_min = Some(0.);
    let f_max = Some(22050.);
    //let n_freq_bins = 1 + n_fft / 2;

    let spectrogram = par_compute_spectrogram(
        &resampled_audio,
        n_samples,
        hop_length,
        win_length,
        center,
        st,
    );

    let _x = convert_to_mel(&spectrogram, target_sr, n_fft, n_mels, f_min, f_max, ms);

    Ok(())
}

// fn main() {
//     let sr = 22050;
//     let n_fft = 16;
//     let n_mels = 40;

//     // Different sconversions to mel scale
//     #[derive(Debug, Clone, Copy)]
//     pub enum MelScale {
//         HTK,
//         Slaney,
//     }

//     /// Convert frequency in Hz to mel scale (see e.g. https://en.wikipedia.org/wiki/Mel_scale)
//     fn hz_to_mel(hz: f32, mel_scale: MelScale) -> f32 {
//         match mel_scale {
//             MelScale::HTK => 2595.0 * (1.0 + hz / 700.0).log10(),
//             MelScale::Slaney => {
//                 if hz < 1000.0 {
//                     3.0 * hz / 200.0
//                 } else {
//                     15.0 + 27.0 * (hz / 1000.0).log(6.4)
//                 }
//             }
//         }
//     }

//     /// Convert mel scale back to Hz (inverse formula of the above)
//     fn mel_to_hz(mel: f32, mel_scale: MelScale) -> f32 {
//         match mel_scale {
//             MelScale::HTK => 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0),
//             MelScale::Slaney => {
//                 if mel < 15.0 {
//                     200.0 * mel / 3.0
//                 } else {
//                     6.4f32.powf((mel - 15.0) / 27.0) * 1000.0
//                 }
//             }
//         }
//     }

//     // Use provided values or defaults
//     let f_min = 0.0;
//     let f_max = sr as f32 / 2.0; // (Nyquist theorem)

//     // Convert to mel scale
//     let mel_min = hz_to_mel(f_min, MelScale::Slaney);
//     let mel_max = hz_to_mel(f_max, MelScale::Slaney);

//     println!("{:?}", mel_max);

//     // Create n_mels + 2 points linearly spaced in mel scale
//     let mel_freqs: Vec<f32> = (0..=n_mels - 1)
//         .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels - 1) as f32)
//         .map(|f| mel_to_hz(f, MelScale::Slaney))
//         .collect();

//     println!("{:?}", mel_freqs.len());

//     println!("{:?}", mel_freqs);
// }

// fn main(){
//     let x = vec![1., 1., 2., 3., 5., 8.];

//     let mel_freqs_diffs: Vec<f32> = x.windows(2).map(|w| w[1] - w[0]).collect();

//     println!("{:?}", mel_freqs_diffs);
// }
