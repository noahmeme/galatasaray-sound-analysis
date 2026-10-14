# Mathematical reasoning and investigation design

## What is observed?

A microphone records a mixture, not a direct label of the source. One useful model is

\[
x(t)=\sum_j (h_j*s_j)(t) + (h_{PA}*p)(t)+\epsilon(t),
\]

where spectator signals are \(s_j\), possible speaker playback is \(p\), propagation and reverberation are impulse responses \(h\), and \(\epsilon\) includes other sound and sensor noise. Broadcast recordings may additionally apply compression, equalization, mixing, and edits. A loud whistle or narrow spectral line can arise from either source. Amplified live whistling need not repeat. Even detection of a reused recording would not establish where it was inserted (stadium PA versus broadcast production).

## Fourier transform and STFT

For samples \(x[n]\) acquired at rate \(f_s\), the N-point discrete Fourier transform is

\[
X[k]=\sum_{n=0}^{N-1}x[n]e^{-i2\pi kn/N},\qquad f_k=kf_s/N.
\]

It measures the coefficient of each complex sinusoidal basis function. Magnitude describes strength; phase describes alignment. An FFT computes this transform efficiently. A transform of a whole interval discards the timing of changing whistles, which motivates short overlapping windows:

\[
X[m,k]=\sum_{n=0}^{N-1}x[n+mH]w[n]e^{-i2\pi kn/N}.
\]

The implementation uses a periodic Hann window \(w[n]=\frac12(1-\cos(2\pi n/N))\), window length equal to FFT length, and hop \(H\). Tapering reduces leakage from abrupt frame boundaries but broadens spectral peaks. Frames contain only real samples (`center=False`); incomplete trailing frames are omitted. Plotted frame centers are \((mH+N/2)/f_s\) plus the segment start. No padded boundary frames are interpreted as evidence.

Bin spacing is \(f_s/N\); window duration is \(N/f_s\), and the step between frames is \(H/f_s\). At 24 kHz, N=2048 and H=512 give 11.72 Hz spacing, an 85.33 ms window, and a 21.33 ms step. Spacing is not the ability to resolve two nearby tones: Hann-window peak width also matters. Longer windows improve frequency discrimination while smearing rapid changes. A smaller hop samples time more densely without shortening the window or making adjacent frames independent.

The returned `stft` retains complex coefficients. The display amplitude is

\[
A[m,k]=c_k\frac{|X[m,k]|}{\sum_n w[n]},
\]

where \(c_k=2\) for interior one-sided bins and 1 for DC and, when N is even, Nyquist. A bin-centered sinusoid of amplitude a has approximately amplitude a in its corresponding bin (away from DC/Nyquist). This is a coherent-tone amplitude normalization; it is not an unbiased broadband power estimator. The raw power spectrogram is \(|X|^2\), which must be scaled appropriately for PSD units.

Color is \(20\log_{10}(A/A_{ref})\), with a numerical floor of \(10^{-10}\). It equals \(10\log_{10}(A^2/A_{ref}^2)\). Comparisons use one common reference and one common displayed range. Independent peak normalization would conceal amplitude differences. A -20 dB change represents a factor of ten in amplitude, or one hundred in squared amplitude. Neither scale represents acoustic dB SPL without calibration.

Whistles often show narrow moving ridges; several people may produce overlapping ridges, changing envelopes and broadband background. Search visually for recurrence of detailed frequency trajectories, envelopes, and surrounding sounds over seconds, rather than only one persistent tone. Similar-looking whistles also arise naturally, and visual recurrence is only a candidate for quantitative testing.

The API and framing choices follow the [librosa STFT documentation](https://librosa.org/doc/0.11.0/generated/librosa.stft.html); the conversion follows [librosa amplitude_to_db](https://librosa.org/doc/0.11.0/generated/librosa.amplitude_to_db.html).

## Stochastic processes and planned methods

Treat spectator contributions as realizations of random, time-varying processes. Independent source summation may reduce fine-scale coherence, but independence is a modeling assumption: chants and responses to play synchronize crowds. Stadium sound is usually nonstationary. For a wide-sense stationary process, mean is constant and covariance depends only on lag; within short intervals, approximate stationarity can make estimates useful. Do not assume it across entire matches.

For a zero-mean discrete stationary process, autocovariance

\[
R_x[\ell]=\mathbb E\{x[n]x[n+\ell]\}
\]

and PSD form a Fourier pair (Wiener–Khinchin):

\[
S_x(f)=\frac1{f_s}\sum_{\ell=-\infty}^{\infty}R_x[\ell]
 e^{-i2\pi f\ell/f_s}.
\]

This is the two-sided density convention, with units of digital-amplitude squared per Hz. Its integral over one Nyquist interval equals variance. Estimates from finite, nonstationary recordings have uncertainty; overlapping observations are correlated.

### Welch PSD (next increment)

Average windowed periodograms over M frames:

\[
\hat S(f_k)=\frac1M\sum_m\frac{|X[m,k]|^2}{f_s\sum_n w[n]^2},
\]

with interior positive-frequency bins doubled for a one-sided real-signal PSD. This normalization differs from the spectrogram amplitude normalization. Averaging reduces variability relative to an individual periodogram, at a cost in frequency resolution and temporal information. Overlap means M is not the count of independent observations. PSD can quantify tonal concentration, broadband content and narrow peaks; it does not locate repetition or identify its source. Compare similar match events and microphone/feed conditions. Implementation will use [SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).

### Autocorrelation (later increment)

Estimate similarity of a mean-centered signal to its delayed version. Normalize by the energies of the overlapping portions, require a substantial overlap, and report lag in seconds. A loop of duration T can produce peaks near T, 2T, etc., despite additive noise. Short-lag peaks are also expected for any tonal whistle: a 2 kHz sinusoid repeats every 0.5 ms. These oscillations are not evidence of a repeated recording. Analyze multi-second structure, envelopes, and spectral trajectories, with locally detrended signals and explicit lag ranges.

### Cross-correlation (later increment)

For candidate segments x and y, compute a lag-dependent normalized inner product of their overlapping, mean-centered samples. This can estimate delay and reveal detailed waveform reuse. Gains are handled by normalization, but compression, reverberation, microphone position, resampling and timing drift can destroy waveform agreement. Broadcasts of the same live event naturally correlate. Compare nonoverlapping moments or distinct matches and exclude shared feeds/edits before interpreting reuse. Define lag sign, minimum overlap, and uncertainty explicitly.

### Repeated spectral patterns (later increment)

Construct time-frequency feature vectors from log magnitudes within a specified band and compare windows of several seconds using normalized similarity. A self-similarity matrix can reveal offset diagonals from recurrent sequences. Local whitening or feature normalization helps reduce stationary tonal backgrounds, but can hide meaningful level differences; retain the original measurements. Search at a common sample rate and hop. Exclude the near-zero-lag diagonal and heavily overlapping windows. Constrain any time-warp alignment: excessive flexibility can match unrelated sound.

## Evidence and validation

1. Retain originals and provenance. Log selected intervals, microphone positions, stream/channel, and all processing; preserve raw channels before mixing. Inspect for clipping, strong automatic gain changes and edits.
2. Choose comparable intervals and controls before searching: other stadiums, independently recorded fan videos, and manually annotated whistle events. Include known replay examples where available.
3. Quantify candidates using several methods. Require agreement across detailed temporal and spectral features, not just narrow peaks. Recheck with different reasonable window sizes and independent recordings.
4. Validate on labeled natural and playback examples with noise, reverberation, gain changes, and codecs. Split by match/source rather than adjacent segments. The included simulations test mechanics only.
5. Searching many lags/segments increases false matches. Calibrate thresholds against controls/surrogate data, account for dependence, and report false-positive rates and the total search scope. Tonality alone is an unsuitable discriminator.
6. Describe conclusions as observations with uncertainty. Distinguish evidence of recurring recorded content from evidence of speaker amplification or stadium attribution. The current project supplies exploratory measurements, not a verdict about the allegations.
