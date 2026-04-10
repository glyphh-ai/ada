"""
Audio feature extraction for Ada Voice.

Extracts openSMILE eGeMAPS features from audio and maps them to the
encoder's role names. Returns a flat dict of floats ready for encoding.

Exports:
    extract_features(audio_path) — WAV/MP3 path → feature dict
    extract_features_from_array(samples, sr) — numpy array → feature dict
    FEATURE_MAP — eGeMAPS column name → encoder role name mapping
"""

import numpy as np

try:
    import opensmile
    _HAS_OPENSMILE = True
except ImportError:
    _HAS_OPENSMILE = False


def _find_peaks_simple(data: np.ndarray, distance: int = 1) -> np.ndarray:
    """Simple peak detection — no scipy dependency."""
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            if not peaks or (i - peaks[-1]) >= distance:
                peaks.append(i)
    return np.array(peaks, dtype=int)

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


# ---------------------------------------------------------------------------
# eGeMAPS feature name → encoder role name
# ---------------------------------------------------------------------------
# We select a 21-feature subset of the 88 eGeMAPS functionals, chosen for
# maximum discrimination across the four voice layers.

FEATURE_MAP = {
    # Identity layer — vocal tract geometry (stable across sessions)
    "mfcc1_sma3_amean": "mfcc_1",
    "mfcc2_sma3_amean": "mfcc_2",
    "mfcc3_sma3_amean": "mfcc_3",
    "mfcc4_sma3_amean": "mfcc_4",
    # MFCCs 5-13: finer vocal tract resonances, harder to consciously fake
    "mfcc5_sma3_amean": "mfcc_5",
    "mfcc6_sma3_amean": "mfcc_6",
    "mfcc7_sma3_amean": "mfcc_7",
    "mfcc8_sma3_amean": "mfcc_8",
    "mfcc9_sma3_amean": "mfcc_9",
    "mfcc10_sma3_amean": "mfcc_10",
    "mfcc11_sma3_amean": "mfcc_11",
    "mfcc12_sma3_amean": "mfcc_12",
    "mfcc13_sma3_amean": "mfcc_13",
    "F1frequency_sma3nz_amean": "formant_f1",
    "F2frequency_sma3nz_amean": "formant_f2",
    "F3frequency_sma3nz_amean": "formant_f3",
    "HNRdBACF_sma3nz_amean": "hnr",

    # Emotional state layer — prosodic features (change per utterance)
    "F0semitoneFrom27.5Hz_sma3nz_amean": "f0_mean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "f0_std",
    "loudness_sma3_amean": "loudness_mean",
    "loudness_sma3_stddevNorm": "loudness_std",
    "slope0-500_sma3_amean": "spectral_slope",
    # Pitch range — liars compress or exaggerate their natural range
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0": "f0_p20",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0": "f0_p80",
    # Spectral flux — rate of spectral change, monotone speech = low flux
    "spectralFlux_sma3_amean": "spectral_flux",

    # Cognitive load layer — voice stability (stress / fatigue / deception)
    "jitterLocal_sma3nz_amean": "jitter",
    "shimmerLocaldB_sma3nz_amean": "shimmer",
    "StddevVoicedSegmentLengthSec": "voiced_std",
    # BUG FIX: was "MeanUnvoicedSegmentLength" which duplicated the cadence
    # entry — pause_std captures irregular pausing under cognitive load
    "StddevUnvoicedSegmentLength": "pause_std",
    # Alpha ratio — energy below/above 1kHz, shifts under laryngeal tension
    "alphaRatioV_sma3nz_amean": "alpha_ratio",
    # Hammarberg index — spectral peak ratio, throat tension marker
    "hammarbergIndexV_sma3nz_amean": "hammarberg_index",
    # F1 bandwidth — widens under stress (harder to consciously control)
    "F1bandwidth_sma3nz_amean": "formant_f1_bw",
    # MFCC1 coefficient of variation — within-utterance instability
    "mfcc1_sma3_stddevNorm": "mfcc_1_cv",

    # Cadence layer — speech rhythm (cognitive style)
    "loudnessPeaksPerSec": "loudness_peaks_per_sec",
    "MeanVoicedSegmentLengthSec": "voiced_segment_mean",
    "MeanUnvoicedSegmentLength": "unvoiced_segment_mean",
    "VoicedSegmentsPerSec": "voiced_segments_per_sec",
}

# Reverse map for lookups
_ROLE_TO_EGEMAP = {v: k for k, v in FEATURE_MAP.items()}


# ---------------------------------------------------------------------------
# Feature extraction via openSMILE
# ---------------------------------------------------------------------------

def _get_smile():
    """Get configured openSMILE extractor (cached)."""
    if not _HAS_OPENSMILE:
        raise ImportError(
            "opensmile not installed. Run: pip install opensmile"
        )
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_features(audio_path: str) -> dict[str, float]:
    """Extract voice features from an audio file.

    Args:
        audio_path: Path to WAV, MP3, or other audio file.

    Returns:
        Dict mapping encoder role names to float values.
    """
    smile = _get_smile()
    df = smile.process_file(audio_path)
    return _map_features(df.iloc[0].to_dict())


def extract_features_from_array(
    samples: np.ndarray, sr: int = 16000
) -> dict[str, float]:
    """Extract voice features from a numpy audio array.

    Args:
        samples: Audio samples as float32 array.
        sr: Sample rate in Hz.

    Returns:
        Dict mapping encoder role names to float values.
    """
    smile = _get_smile()
    df = smile.process_signal(samples, sr)
    return _map_features(df.iloc[0].to_dict())


def _map_features(raw: dict) -> dict[str, float]:
    """Map eGeMAPS feature names to encoder role names."""
    result = {}
    for egemap_name, role_name in FEATURE_MAP.items():
        val = raw.get(egemap_name)
        if val is not None:
            result[role_name] = float(val)
        else:
            result[role_name] = 0.0
    return result


# ---------------------------------------------------------------------------
# Fallback: librosa-based extraction (no openSMILE dependency)
# ---------------------------------------------------------------------------

def _estimate_formants_lpc(y: np.ndarray, sr: int, order: int = 12) -> tuple[float, float, float, float]:
    """Estimate F1, F2, F3 from LPC roots plus F1 bandwidth.

    Uses linear prediction coefficients to find vocal tract resonances.
    Returns (F1, F2, F3, F1_bandwidth) in Hz, or 0.0 for any that can't be found.
    F1 bandwidth widens under stress — harder to consciously control.
    """
    if len(y) < order * 2:
        return 0.0, 0.0, 0.0, 0.0

    # Pre-emphasis
    y_pe = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # LPC via autocorrelation method
    from numpy.linalg import solve
    n = len(y_pe)
    r = np.correlate(y_pe, y_pe, mode="full")[n - 1: n + order]
    # Build Toeplitz matrix
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = r[abs(i - j)]
    try:
        a = solve(R, r[1:order + 1])
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, 0.0

    # Find roots
    roots = np.roots(np.concatenate(([1.0], -a)))
    # Keep roots inside the unit circle with positive imaginary part
    roots = roots[np.imag(roots) > 0]
    roots = roots[np.abs(roots) < 1.0]

    if len(roots) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Convert to frequencies and bandwidths
    freqs = np.abs(np.arctan2(np.imag(roots), np.real(roots))) * (sr / (2 * np.pi))
    # Bandwidth = -sr * ln(|root|) / pi
    bws = -sr * np.log(np.abs(roots) + 1e-12) / np.pi

    # Sort by frequency and filter to speech range
    order_idx = np.argsort(freqs)
    freqs = freqs[order_idx]
    bws = bws[order_idx]
    speech_mask = [(90 < f < 5000) for f in freqs]
    formants = [(freqs[i], bws[i]) for i in range(len(freqs)) if speech_mask[i]]

    f1 = formants[0][0] if len(formants) > 0 else 0.0
    f1_bw = formants[0][1] if len(formants) > 0 else 0.0
    f2 = formants[1][0] if len(formants) > 1 else 0.0
    f3 = formants[2][0] if len(formants) > 2 else 0.0
    return float(f1), float(f2), float(f3), float(np.clip(f1_bw, 0, 500))


def _estimate_hnr(y: np.ndarray, sr: int) -> float:
    """Estimate Harmonics-to-Noise Ratio in dB via frame-based autocorrelation.

    Uses short frames (30ms) and averages across voiced frames for robustness.
    Subtracts DC offset before computing to avoid normalized peaks hitting 1.0.
    """
    if len(y) < sr // 50:
        return 0.0

    # Remove DC offset
    y = y - np.mean(y)

    frame_len = int(sr * 0.030)  # 30ms frames
    hop = int(sr * 0.010)        # 10ms hop
    min_lag = int(sr / 500)      # 500 Hz max
    max_lag = int(sr / 50)       # 50 Hz min

    hnr_values = []
    for start in range(0, len(y) - frame_len, hop):
        frame = y[start:start + frame_len]

        # Skip low-energy frames (silence/noise)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.01:
            continue

        # Autocorrelation of this frame
        n = len(frame)
        ac = np.correlate(frame, frame, mode="full")[n - 1:]
        ac_norm = ac / (ac[0] + 1e-12)

        if max_lag >= len(ac_norm):
            continue

        peak_val = float(np.max(ac_norm[min_lag:max_lag]))
        if peak_val <= 0.01 or peak_val >= 0.999:
            continue

        hnr_frame = 10.0 * np.log10(peak_val / (1.0 - peak_val + 1e-12))
        hnr_values.append(np.clip(hnr_frame, 0.0, 40.0))

    if not hnr_values:
        return 0.0
    return float(np.mean(hnr_values))


def _estimate_jitter_shimmer(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate jitter and shimmer using librosa pyin F0 tracking.

    Uses pyin-detected F0 per frame (robust pitch tracker) instead of raw
    autocorrelation peaks which produce garbage on noisy browser audio.
    Jitter = mean absolute F0 difference / mean F0 (across voiced frames).
    Shimmer = mean absolute amplitude difference / mean amplitude.
    """
    if not _HAS_LIBROSA:
        return 0.0, 0.0

    n_fft = min(2048, len(y))
    hop = 512

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, frame_length=n_fft, hop_length=hop,
        )
    except Exception:
        return 0.0, 0.0

    if voiced_flag is None or not voiced_flag.any():
        return 0.0, 0.0

    # Jitter from consecutive voiced F0 values
    voiced_f0 = f0[voiced_flag]
    voiced_f0 = voiced_f0[np.isfinite(voiced_f0) & (voiced_f0 > 0)]

    if len(voiced_f0) < 3:
        return 0.0, 0.0

    # Convert F0 to periods, compute relative jitter
    periods = 1.0 / voiced_f0
    period_diffs = np.abs(np.diff(periods))
    jitter = float(np.mean(period_diffs) / (np.mean(periods) + 1e-12))

    # Shimmer from frame-level RMS at voiced frames
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    # Align RMS with voiced_flag (may differ in length)
    min_len = min(len(rms), len(voiced_flag))
    voiced_rms = rms[:min_len][voiced_flag[:min_len]]
    voiced_rms = voiced_rms[voiced_rms > 0]

    if len(voiced_rms) < 3:
        return jitter, 0.0

    amp_diffs = np.abs(np.diff(voiced_rms))
    shimmer = float(np.mean(amp_diffs) / (np.mean(voiced_rms) + 1e-12))

    return jitter, shimmer


def extract_features_librosa(audio_path: str) -> dict[str, float]:
    """Fallback feature extraction using librosa.

    Extracts a compatible feature set when openSMILE is not available.
    Now includes formant estimation via LPC roots and HNR via autocorrelation,
    which are critical for speaker identity discrimination.
    """
    if not _HAS_LIBROSA:
        raise ImportError(
            "Neither opensmile nor librosa installed. "
            "Run: pip install opensmile  OR  pip install librosa"
        )
    import soundfile as sf

    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    # Need at least 0.1s of audio for any meaningful features
    if len(y) < int(sr * 0.1):
        return {k: 0.0 for k in FEATURE_MAP.values()}

    # MFCCs 1-13 — use smaller n_fft if audio is short
    n_fft = min(2048, len(y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
    mfcc_means = mfccs.mean(axis=1)

    # MFCC deltas — rate of change captures articulatory dynamics
    mfcc_deltas = librosa.feature.delta(mfccs)
    mfcc_delta_means = mfcc_deltas.mean(axis=1)

    # Formants via LPC (now also returns F1 bandwidth)
    f1, f2, f3, f1_bw = _estimate_formants_lpc(y, sr)

    # HNR
    hnr = _estimate_hnr(y, sr)

    # Jitter & shimmer
    try:
        jitter, shimmer = _estimate_jitter_shimmer(y, sr)
    except Exception:
        jitter, shimmer = 0.0, 0.0

    # F0 via pyin — needs enough frames
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, frame_length=min(n_fft, len(y))
        )
        f0_voiced = f0[voiced_flag] if voiced_flag.any() else np.array([0.0])
    except Exception:
        f0_voiced = np.array([0.0])
        voiced_flag = np.array([False])
    f0_hz_mean = float(np.nanmean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    f0_hz_std = float(np.nanstd(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    # Convert to semitones re 27.5Hz
    f0_semi = 12.0 * np.log2(f0_hz_mean / 27.5) if f0_hz_mean > 0 else 0.0
    f0_std_norm = f0_hz_std / f0_hz_mean if f0_hz_mean > 0 else 0.0

    # Loudness (RMS as proxy)
    rms = librosa.feature.rms(y=y, frame_length=n_fft)[0]
    loudness_mean = float(rms.mean())
    loudness_std = float(rms.std() / rms.mean()) if rms.mean() > 0 else 0.0

    # Spectral slope (linear regression of log-magnitude spectrum)
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    log_mag = np.mean(np.log1p(S), axis=1)
    if len(freqs) > 1 and len(log_mag) > 1:
        slope = float(np.polyfit(freqs[:len(log_mag)], log_mag, 1)[0])
    else:
        slope = 0.0

    # Voiced/unvoiced segmentation
    voiced_lengths = []
    unvoiced_lengths = []
    current_voiced = False
    current_len = 0
    hop = 512
    for v in voiced_flag:
        if v == current_voiced:
            current_len += 1
        else:
            dur = current_len * hop / sr
            if current_voiced:
                voiced_lengths.append(dur)
            else:
                unvoiced_lengths.append(dur)
            current_voiced = v
            current_len = 1
    # Final segment
    dur = current_len * hop / sr
    if current_voiced:
        voiced_lengths.append(dur)
    else:
        unvoiced_lengths.append(dur)

    duration = len(y) / sr

    # Loudness peaks per second
    peaks = _find_peaks_simple(rms, distance=max(1, int(sr / (hop * 4))))
    lps = len(peaks) / duration if duration > 0 else 0.0

    # --- New deception-discriminative features ---

    # F0 percentiles (20th and 80th) in semitones re 27.5Hz
    if len(f0_voiced) > 1:
        f0_p20_hz = float(np.nanpercentile(f0_voiced, 20))
        f0_p80_hz = float(np.nanpercentile(f0_voiced, 80))
    else:
        f0_p20_hz = f0_hz_mean
        f0_p80_hz = f0_hz_mean
    f0_p20_semi = 12.0 * np.log2(f0_p20_hz / 27.5) if f0_p20_hz > 0 else 0.0
    f0_p80_semi = 12.0 * np.log2(f0_p80_hz / 27.5) if f0_p80_hz > 0 else 0.0

    # Spectral flux — frame-to-frame L2 norm of normalized magnitude spectrum
    S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-12)
    if S_norm.shape[1] > 1:
        flux = float(np.mean(np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))))
    else:
        flux = 0.0

    # Alpha ratio — energy below vs above 1kHz (dB)
    # S is magnitude STFT from spectral slope computation above
    mean_power = (S ** 2).mean(axis=1)
    alpha_cutoff = 1000.0
    alpha_lo_mask = freqs < alpha_cutoff
    alpha_hi_mask = freqs >= alpha_cutoff
    lo_energy = mean_power[alpha_lo_mask].sum() if alpha_lo_mask.any() else 1e-12
    hi_energy = mean_power[alpha_hi_mask].sum() if alpha_hi_mask.any() else 1e-12
    alpha_ratio = float(10.0 * np.log10((lo_energy + 1e-12) / (hi_energy + 1e-12)))

    # Hammarberg index — max spectral peak 0-2kHz vs 2-5kHz (dB)
    low_band = (freqs >= 0) & (freqs <= 2000)
    high_band = (freqs > 2000) & (freqs <= 5000)
    max_low = float(np.max(mean_power[low_band])) if low_band.any() else 1e-12
    max_high = float(np.max(mean_power[high_band])) if high_band.any() else 1e-12
    hammarberg = float(10.0 * np.log10((max_low + 1e-12) / (max_high + 1e-12)))

    # MFCC1 coefficient of variation — within-utterance instability
    mfcc1_mean_abs = abs(float(mfcc_means[0])) + 1e-12
    mfcc1_std = float(mfccs[0].std())
    mfcc_1_cv = mfcc1_std / mfcc1_mean_abs

    # Unvoiced segment std — irregular pausing
    unvoiced_std = float(np.std(unvoiced_lengths)) if len(unvoiced_lengths) > 1 else 0.0

    return {
        "mfcc_1": float(mfcc_means[0]),
        "mfcc_2": float(mfcc_means[1]),
        "mfcc_3": float(mfcc_means[2]),
        "mfcc_4": float(mfcc_means[3]),
        "mfcc_5": float(mfcc_means[4]),
        "mfcc_6": float(mfcc_means[5]),
        "mfcc_7": float(mfcc_means[6]),
        "mfcc_8": float(mfcc_means[7]),
        "mfcc_9": float(mfcc_means[8]),
        "mfcc_10": float(mfcc_means[9]),
        "mfcc_11": float(mfcc_means[10]),
        "mfcc_12": float(mfcc_means[11]),
        "mfcc_13": float(mfcc_means[12]),
        "mfcc_delta_1": float(mfcc_delta_means[0]),
        "mfcc_delta_2": float(mfcc_delta_means[1]),
        "mfcc_delta_3": float(mfcc_delta_means[2]),
        "mfcc_delta_4": float(mfcc_delta_means[3]),
        "formant_f1": f1,
        "formant_f2": f2,
        "formant_f3": f3,
        "hnr": hnr,
        "f0_mean": float(f0_semi),
        "f0_std": float(f0_std_norm),
        "loudness_mean": loudness_mean,
        "loudness_std": loudness_std,
        "spectral_slope": slope,
        # New emotional features
        "f0_p20": float(f0_p20_semi),
        "f0_p80": float(f0_p80_semi),
        "spectral_flux": flux,
        # Cognitive load features
        "jitter": jitter,
        "shimmer": shimmer,
        "voiced_std": float(np.std(voiced_lengths)) if voiced_lengths else 0.0,
        "pause_std": unvoiced_std,
        "alpha_ratio": alpha_ratio,
        "hammarberg_index": hammarberg,
        "formant_f1_bw": f1_bw,
        "mfcc_1_cv": mfcc_1_cv,
        # Cadence features
        "loudness_peaks_per_sec": lps,
        "voiced_segment_mean": float(np.mean(voiced_lengths)) if voiced_lengths else 0.0,
        "unvoiced_segment_mean": float(np.mean(unvoiced_lengths)) if unvoiced_lengths else 0.0,
        "voiced_segments_per_sec": len(voiced_lengths) / duration if duration > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Liveness detection — replay/playback attack detection
# ---------------------------------------------------------------------------

def compute_liveness_score(audio_path: str) -> dict[str, float]:
    """Detect replay attacks by analyzing spectral characteristics.

    When audio is played through a speaker and re-recorded by a mic:
    1. High-frequency energy dies — speakers roll off above 4-8kHz
    2. Spectral flatness drops — speaker resonance creates tonal peaks
    3. Sub-band energy variance increases — speaker EQ is uneven

    Returns:
        Dict with:
            hf_energy_ratio: ratio of energy above 4kHz to total (live ~0.05-0.3, replay ~0.001-0.02)
            spectral_flatness: geometric/arithmetic mean of spectrum (live ~0.01-0.1, replay ~0.001-0.01)
            is_live: bool — True if liveness checks pass
            liveness_score: float 0-1, higher = more likely live
    """
    if not _HAS_LIBROSA:
        return {"hf_energy_ratio": 0.0, "spectral_flatness": 0.0, "is_live": True, "liveness_score": 1.0}

    import soundfile as sf

    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    if len(y) < int(sr * 0.3):
        return {"hf_energy_ratio": 0.0, "spectral_flatness": 0.0, "is_live": True, "liveness_score": 1.0}

    n_fft = min(2048, len(y))

    # Power spectrum
    S = np.abs(librosa.stft(y, n_fft=n_fft)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_power = S.mean(axis=1)

    # 1. High-frequency energy ratio
    # Live voice at 16kHz sample rate has energy up to 8kHz.
    # Speakers (especially laptop/phone) roll off hard above 4-6kHz.
    hf_cutoff = 4000.0
    hf_mask = freqs >= hf_cutoff
    lf_mask = freqs < hf_cutoff
    total_energy = mean_power.sum()
    hf_energy = mean_power[hf_mask].sum() if hf_mask.any() else 0.0
    hf_ratio = float(hf_energy / (total_energy + 1e-12))

    # 2. Spectral flatness (Wiener entropy)
    # Geometric mean / arithmetic mean of power spectrum.
    # Live speech: more uniform noise floor → higher flatness.
    # Replay: speaker resonance peaks → lower flatness.
    # Only compute on the mid-range (300Hz-4kHz) where speaker artifacts are most visible.
    mid_mask = (freqs >= 300) & (freqs <= 4000)
    mid_power = mean_power[mid_mask]
    if len(mid_power) > 0 and mid_power.min() > 0:
        log_mean = np.mean(np.log(mid_power + 1e-12))
        geo_mean = np.exp(log_mean)
        arith_mean = np.mean(mid_power)
        sf_val = float(geo_mean / (arith_mean + 1e-12))
    else:
        sf_val = 0.0

    # Scoring:
    # hf_ratio: live speech typically 0.03-0.25, replay 0.001-0.015
    # spectral_flatness: live typically 0.01-0.15, replay 0.001-0.008
    hf_score = min(1.0, hf_ratio / 0.03)  # normalized: 0.03+ → 1.0
    sf_score = min(1.0, sf_val / 0.01)     # normalized: 0.01+ → 1.0
    liveness = 0.7 * hf_score + 0.3 * sf_score

    # Threshold: below 0.4 is almost certainly replay
    is_live = liveness >= 0.4

    return {
        "hf_energy_ratio": round(hf_ratio, 6),
        "spectral_flatness": round(sf_val, 6),
        "is_live": is_live,
        "liveness_score": round(liveness, 4),
    }
