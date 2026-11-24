"""
BioSignal-C v1.0 - deployment-realistic corruption suite for physiological signals.

Every corruption is defined on the RAW signal (never on a rendered image), is applied only
at test time, is deterministic given (family, severity, seed), and is parameterised from a
documented real acquisition condition rather than an abstract perturbation.

Signal convention: X is float32 with shape (N, C, L); fs in Hz.
Severity is 1, 2 or 3.  Severity 2 is calibrated to a condition a clinician would still
accept; severity 3 is the edge of usability.

Usage
-----
    from biosignal_c.corruptions import SUITES, apply
    for fam, sev in SUITES["audio"].cells():
        Xc = apply(X, fs, "audio", fam, sev, seed=0)
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfiltfilt, iirnotch, filtfilt

__all__ = ["SUITES", "apply", "Suite"]


# --------------------------------------------------------------------------- helpers
def _rng(family: str, sev: int, seed: int) -> np.random.Generator:
    """Deterministic generator per (family, severity, seed).

    The seed is derived with BLAKE2b over the UTF-8 key, so the seed itself is identical in
    every process and on every platform. The random numbers and corrupted data drawn from it
    can still depend on the NumPy version and the numerical software environment.

    NOTE: this previously used the builtin ``hash()``. CPython salts string hashing per
    process (PYTHONHASHSEED), so every cell whose corruption draws from the RNG produced
    different corrupted signals in every new process (measured in the paper: 34 of the 75
    measurable cells).
    """
    key = f"{family}|{int(sev)}|{int(seed)}".encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big"))


def _bp(x, fs, lo, hi, order=4):
    ny = fs / 2.0
    lo = max(lo / ny, 1e-4)
    hi = min(hi / ny, 0.999)
    if lo >= hi:
        return x
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def _lp(x, fs, hi, order=4):
    hi = min(hi / (fs / 2.0), 0.999)
    sos = butter(order, hi, btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def _hp(x, fs, lo, order=4):
    lo = max(lo / (fs / 2.0), 1e-4)
    sos = butter(order, lo, btype="high", output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def _rms(x):
    return np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True)) + 1e-12


def _add_at_snr(x, n, snr_db):
    """Scale noise n so that it sits at snr_db relative to x, per trace."""
    n = n * (_rms(x) / (10 ** (snr_db / 20.0)) / _rms(n))
    return (x + n).astype(np.float32)


def _bursts(shape, fs, n_bursts, dur_s, rng, amp):
    """Additive Hanning-shaped transients (electrode pop / handling / motion)."""
    N, C, L = shape
    out = np.zeros(shape, dtype=np.float32)
    w = int(max(8, dur_s * fs))
    win = np.hanning(w).astype(np.float32)
    for i in range(N):
        for _ in range(n_bursts):
            c = rng.integers(0, C)
            s = int(rng.integers(0, max(1, L - w)))
            out[i, c, s:s + w] += float(rng.normal(0, amp)) * win
    return out


# --------------------------------------------------------------------------- audio
# lung sound / heart sound / cough / voice, fs typically 4000-44100 Hz
def audio_ambient_white(x, fs, sev, rng):
    snr = {1: 20, 2: 10, 3: 0}[sev]
    return _add_at_snr(x, rng.standard_normal(x.shape).astype(np.float32), snr)


def audio_ward_rumble(x, fs, sev, rng):
    """Ventilation / traffic / trolley rumble: low-band coloured noise."""
    snr = {1: 15, 2: 6, 3: 0}[sev]
    n = _bp(rng.standard_normal(x.shape).astype(np.float32), fs, 20, 300)
    return _add_at_snr(x, n, snr)


def audio_babble(x, fs, sev, rng):
    """Ward speech babble: amplitude-modulated speech-band noise (3-6 Hz envelope)."""
    snr = {1: 15, 2: 8, 3: 2}[sev]
    n = _bp(rng.standard_normal(x.shape).astype(np.float32), fs, 200, min(3400, fs / 2 - 1))
    t = np.arange(x.shape[-1]) / fs
    env = 1.0 + 0.8 * np.sin(2 * np.pi * rng.uniform(3, 6) * t + rng.uniform(0, 6.28))
    return _add_at_snr(x, (n * env.astype(np.float32)), snr)


def audio_stethoscope(x, fs, sev, rng):
    """Low-cost stethoscope: reduced upper cut-off plus a diaphragm resonance peak."""
    hi, fres = {1: (2000, 300), 2: (1200, 220), 3: (700, 180)}[sev]
    y = _bp(x, fs, 60, min(hi, fs / 2 - 1))
    res = _bp(x, fs, fres * 0.8, fres * 1.25)
    return (y + 0.6 * res).astype(np.float32)


def audio_low_gain(x, fs, sev, rng):
    db = {1: -12, 2: -24, 3: -34}[sev]
    return (x * 10 ** (db / 20.0)).astype(np.float32)


def audio_clipping(x, fs, sev, rng):
    """Pre-amp saturation, expressed as a percentile of each trace's own amplitude."""
    q = {1: 90, 2: 75, 3: 55}[sev]
    t = np.percentile(np.abs(x), q, axis=-1, keepdims=True) + 1e-9
    return (np.clip(x, -t, t) / t).astype(np.float32)


def audio_codec(x, fs, sev, rng):
    """Telehealth path: band restriction + coarse companded quantisation."""
    (lo, hi), bits = {1: ((200, 3400), 10), 2: ((300, 3400), 8), 3: ((300, 1800), 6)}[sev]
    y = _bp(x, fs, lo, min(hi, fs / 2 - 1))
    mu = 255.0
    m = np.max(np.abs(y), axis=-1, keepdims=True) + 1e-9
    c = np.sign(y) * np.log1p(mu * np.abs(y) / m) / np.log1p(mu)
    q = 2 ** (bits - 1)
    c = np.round(c * q) / q
    return (np.sign(c) * m * ((1 + mu) ** np.abs(c) - 1) / mu).astype(np.float32)


def audio_adc_bits(x, fs, sev, rng):
    bits = {1: 10, 2: 8, 3: 6}[sev]
    m = np.max(np.abs(x), axis=-1, keepdims=True) + 1e-9
    q = 2 ** (bits - 1)
    return (np.round(x / m * q) / q * m).astype(np.float32)


def audio_packet_loss(x, fs, sev, rng):
    frac = {1: 0.02, 2: 0.08, 3: 0.20}[sev]
    N, C, L = x.shape
    w = int(0.02 * fs)
    z = x.copy()
    k = max(1, int(frac * L / w))
    for i in range(N):
        for _ in range(k):
            s = int(rng.integers(0, max(1, L - w)))
            z[i, :, s:s + w] = 0.0
    return z


def audio_rate_mismatch(x, fs, sev, rng):
    """Recorded at a lower rate then resampled up to the pipeline's rate."""
    factor = {1: 2, 2: 4, 3: 8}[sev]
    y = resample_poly(x, 1, factor, axis=-1)
    y = resample_poly(y, factor, 1, axis=-1)
    L = x.shape[-1]
    if y.shape[-1] < L:
        y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, L - y.shape[-1])])
    return y[..., :L].astype(np.float32)


# --------------------------------------------------------------------------- ECG
def ecg_baseline_wander(x, fs, sev, rng):
    amp = {1: 0.3, 2: 0.8, 3: 1.8}[sev]
    t = np.arange(x.shape[-1]) / fs
    ph = rng.uniform(0, 2 * np.pi, (x.shape[0], 1, 1))
    w = (np.sin(2 * np.pi * 0.25 * t + ph) + 0.6 * np.sin(2 * np.pi * 0.06 * t + 2 * ph))
    return (x + amp * _rms(x) * w.astype(np.float32)).astype(np.float32)


def _mains(x, fs, sev, rng, f0):
    amp = {1: 0.05, 2: 0.20, 3: 0.60}[sev]
    if f0 >= fs / 2:
        f0 = fs / 2 - 1e-3
    t = np.arange(x.shape[-1]) / fs
    ph = rng.uniform(0, 2 * np.pi, (x.shape[0], 1, 1))
    s = np.sin(2 * np.pi * f0 * t + ph) + 0.3 * np.sin(2 * np.pi * min(2 * f0, fs / 2 - 1e-3) * t)
    return (x + amp * _rms(x) * s.astype(np.float32)).astype(np.float32)


def ecg_mains50(x, fs, sev, rng):
    return _mains(x, fs, sev, rng, 50.0)


def ecg_mains60(x, fs, sev, rng):
    return _mains(x, fs, sev, rng, 60.0)


def ecg_emg(x, fs, sev, rng):
    snr = {1: 15, 2: 8, 3: 2}[sev]
    n = _hp(rng.standard_normal(x.shape).astype(np.float32), fs, min(20, fs / 2 - 1))
    return _add_at_snr(x, n, snr)


def ecg_electrode_motion(x, fs, sev, rng):
    k = {1: 2, 2: 5, 3: 12}[sev]
    return (x + _bursts(x.shape, fs, k, 0.4, rng, 2.0 * float(_rms(x).mean()))).astype(np.float32)


def ecg_electrode_dropout(x, fs, sev, rng):
    frac = {1: 0.03, 2: 0.10, 3: 0.25}[sev]
    N, C, L = x.shape
    w = int(0.5 * fs)
    z = x.copy()
    for i in range(N):
        for _ in range(max(1, int(frac * L / w))):
            c = int(rng.integers(0, C))
            s = int(rng.integers(0, max(1, L - w)))
            z[i, c, s:s + w] = 0.0
    return z


def ecg_gain(x, fs, sev, rng):
    return (x * {1: 0.5, 2: 0.2, 3: 0.06}[sev]).astype(np.float32)


def ecg_adc_bits(x, fs, sev, rng):
    return audio_adc_bits(x, fs, {1: 1, 2: 2, 3: 3}[sev], rng)


def ecg_bandwidth(x, fs, sev, rng):
    hi = {1: 40, 2: 25, 3: 12}[sev]
    return _lp(x, fs, min(hi, fs / 2 - 1))


def ecg_polarity(x, fs, sev, rng):
    """Reversed electrodes / inverted lead - label-preserving for rhythm tasks."""
    if sev == 1:
        return x.astype(np.float32)
    if sev == 2:
        return (-x).astype(np.float32)
    z = -x
    if x.shape[1] > 1:                       # also permute channels at severity 3
        z = z[:, rng.permutation(x.shape[1])]
    return z.astype(np.float32)


def ecg_rate_mismatch(x, fs, sev, rng):
    return audio_rate_mismatch(x, fs, sev, rng)


# --------------------------------------------------------------------------- EEG
def eeg_mains(x, fs, sev, rng):
    return _mains(x, fs, sev, rng, 60.0)


def eeg_emg(x, fs, sev, rng):
    snr = {1: 12, 2: 6, 3: 0}[sev]
    n = _bp(rng.standard_normal(x.shape).astype(np.float32), fs, 25, min(90, fs / 2 - 1))
    return _add_at_snr(x, n, snr)


def eeg_ocular(x, fs, sev, rng):
    k = {1: 2, 2: 5, 3: 10}[sev]
    b = _bursts(x.shape, fs, k, 0.6, rng, 3.0 * float(_rms(x).mean()))
    return (x + _lp(b, fs, 8)).astype(np.float32)


def eeg_electrode_pop(x, fs, sev, rng):
    k = {1: 1, 2: 4, 3: 10}[sev]
    return (x + _bursts(x.shape, fs, k, 0.05, rng, 5.0 * float(_rms(x).mean()))).astype(np.float32)


def eeg_channel_loss(x, fs, sev, rng):
    frac = {1: 0.1, 2: 0.25, 3: 0.5}[sev]
    N, C, L = x.shape
    z = x.copy()
    k = max(1, int(frac * C))
    for i in range(N):
        for c in rng.choice(C, size=k, replace=False):
            z[i, c] = 0.0
    return z


def eeg_amplitude_drift(x, fs, sev, rng):
    d = {1: 0.2, 2: 0.5, 3: 1.0}[sev]
    t = np.linspace(0, 1, x.shape[-1], dtype=np.float32)
    g = 1.0 + d * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28, (x.shape[0], 1, 1)))
    return (x * g.astype(np.float32)).astype(np.float32)


def eeg_bandwidth(x, fs, sev, rng):
    hi = {1: 45, 2: 30, 3: 16}[sev]
    return _lp(x, fs, min(hi, fs / 2 - 1))


def eeg_white(x, fs, sev, rng):
    snr = {1: 15, 2: 8, 3: 2}[sev]
    return _add_at_snr(x, rng.standard_normal(x.shape).astype(np.float32), snr)


def eeg_gain(x, fs, sev, rng):
    return (x * {1: 0.5, 2: 0.2, 3: 0.06}[sev]).astype(np.float32)


def eeg_rate_mismatch(x, fs, sev, rng):
    return audio_rate_mismatch(x, fs, sev, rng)


# --------------------------------------------------------------------------- suites
class Suite:
    def __init__(self, name, families, severities=(1, 2, 3)):
        self.name = name
        self.families = families           # dict family -> callable
        self.severities = tuple(severities)

    def cells(self):
        for f in self.families:
            for s in self.severities:
                yield f, s

    def __len__(self):
        return len(self.families) * len(self.severities)


SUITES = {
    "audio": Suite("audio", {
        "ambient_white": audio_ambient_white,
        "ward_rumble": audio_ward_rumble,
        "babble": audio_babble,
        "cheap_stethoscope": audio_stethoscope,
        "low_gain": audio_low_gain,
        "clipping": audio_clipping,
        "telehealth_codec": audio_codec,
        "adc_bits": audio_adc_bits,
        "packet_loss": audio_packet_loss,
        "rate_mismatch": audio_rate_mismatch,
    }),
    "ecg": Suite("ecg", {
        "baseline_wander": ecg_baseline_wander,
        "mains_50": ecg_mains50,
        "mains_60": ecg_mains60,
        "emg": ecg_emg,
        "electrode_motion": ecg_electrode_motion,
        "electrode_dropout": ecg_electrode_dropout,
        "low_gain": ecg_gain,
        "adc_bits": ecg_adc_bits,
        "bandwidth": ecg_bandwidth,
        "polarity": ecg_polarity,
        "rate_mismatch": ecg_rate_mismatch,
    }),
    "eeg": Suite("eeg", {
        "mains": eeg_mains,
        "emg": eeg_emg,
        "ocular": eeg_ocular,
        "electrode_pop": eeg_electrode_pop,
        "channel_loss": eeg_channel_loss,
        "amplitude_drift": eeg_amplitude_drift,
        "bandwidth": eeg_bandwidth,
        "white": eeg_white,
        "low_gain": eeg_gain,
        "rate_mismatch": eeg_rate_mismatch,
    }),
}

# families withheld from any augmentation policy, used to test generalisation of a fix
HELD_OUT = {
    "audio": ["telehealth_codec", "packet_loss", "cheap_stethoscope", "rate_mismatch"],
    "ecg": ["electrode_dropout", "polarity", "rate_mismatch", "bandwidth"],
    "eeg": ["channel_loss", "electrode_pop", "rate_mismatch", "bandwidth"],
}


def apply(X, fs, suite, family, severity, seed=0):
    """Apply one corruption cell. X: (N, C, L) float32 -> same shape."""
    s = SUITES[suite]
    fn = s.families[family]
    x = np.ascontiguousarray(X, dtype=np.float32)
    out = fn(x, float(fs), int(severity), _rng(family, severity, seed))
    return np.ascontiguousarray(np.nan_to_num(out), dtype=np.float32)
