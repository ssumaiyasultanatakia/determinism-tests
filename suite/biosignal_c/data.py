"""
Dataset loaders for BioSignal-C.

Every loader returns a dict:
    X        float32 (N, C, L)   raw signal, per-trace standardised (zero mean, unit sd)
    y        int64   (N,)
    patient  <U..    (N,)
    fs       float                sampling rate of X
    suite    str                  which corruption suite applies ("audio" | "ecg" | "eeg")
    classes  list[str]
    name     str

Loaders cache to code/_cache/<name>.npz so the grid can be re-run cheaply.
Only public datasets already on disk are used; nothing is downloaded.
"""
from __future__ import annotations

import math
import os
import glob
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "_cache")
os.makedirs(CACHE, exist_ok=True)

MAN = os.path.join(os.environ.get("BIOSIGNAL_DATA", os.path.join(os.path.dirname(HERE), "data")), "manifests")
PASCAL = os.path.join(os.environ.get("BIOSIGNAL_DATA", os.path.join(os.path.dirname(HERE), "data")), "Heartbeat Sounds")
APNEA = os.path.join(os.environ.get("BIOSIGNAL_DATA", os.path.join(os.path.dirname(HERE), "data")), "apnea-ecg-database-1.0.0")
PTBDB = os.path.join(os.environ.get("BIOSIGNAL_DATA", os.path.join(os.path.dirname(HERE), "data")), "ptb-diagnostic-ecg-database-1.0.0")
MITST = os.path.join(os.environ.get("BIOSIGNAL_DATA", os.path.join(os.path.dirname(HERE), "data")), "mit-bih-st-change-database-1.0.0")
CHB = os.path.join(os.environ.get("BIOSIGNAL_DATA", os.path.join(os.path.dirname(HERE), "data")), "chb")

DEFAULT_CAP = 9000        # max segments per dataset (whole patients kept)


# --------------------------------------------------------------------------- utils
def _std(X):
    m = X.mean(axis=-1, keepdims=True)
    s = X.std(axis=-1, keepdims=True) + 1e-6
    return np.clip((X - m) / s, -10, 10).astype(np.float32)


def _cap_by_patient(df, cap, seed=0):
    if len(df) <= cap:
        return df
    rng = np.random.default_rng(seed)
    cnt = df.patient_id.astype(str).value_counts()
    pats = cnt.index.to_numpy().copy()
    rng.shuffle(pats)
    keep, tot = [], 0
    for p in pats:
        if tot >= cap:
            break
        keep.append(p)
        tot += int(cnt[p])
    return df[df.patient_id.astype(str).isin(set(keep))]


def _cached(name):
    def deco(fn):
        def wrapper(cap=DEFAULT_CAP, **kw):
            path = os.path.join(CACHE, f"{name}.npz")
            if os.path.exists(path):
                z = np.load(path, allow_pickle=True)
                return {"X": z["X"], "y": z["y"], "patient": z["patient"],
                        "fs": float(z["fs"]), "suite": str(z["suite"]),
                        "classes": list(z["classes"]),
                        "group": z["group"] if "group" in z.files
                        else np.array([""] * len(z["y"])), "name": name}
            t0 = time.time()
            d = fn(cap=cap, **kw)
            d["name"] = name
            if "group" not in d:
                d["group"] = np.array([""] * len(d["y"]))
            np.savez_compressed(path, X=d["X"], y=d["y"], patient=d["patient"],
                                fs=d["fs"], suite=d["suite"], group=d["group"],
                                classes=np.array(d["classes"]))
            print(f"[data] built {name}: X={d['X'].shape} "
                  f"classes={d['classes']} in {time.time()-t0:.0f}s", flush=True)
            return d
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


# --------------------------------------------------------------------------- audio
def _load_manifest_audio(manifest, classes, fs_out, dur, cap, path_col="audio_path"):
    import soundfile as sf
    from scipy.signal import resample_poly
    df = pd.read_csv(os.path.join(MAN, manifest))
    df = df[df.label_str.isin(classes)]
    df = _cap_by_patient(df, cap).sort_values(path_col).reset_index(drop=True)
    L = int(fs_out * dur)
    cp, cs, cfs = None, None, None
    X, y, pat, rid = [], [], [], []
    for _, r in df.iterrows():
        p = r[path_col]
        if not isinstance(p, str) or not p.lower().endswith(".wav") or not os.path.exists(p):
            continue
        if p != cp:
            try:
                cs, cfs = sf.read(p, dtype="float32", always_2d=False)
            except Exception:
                cp = p
                continue
            if cs.ndim > 1:
                cs = cs.mean(1)
            cp = p
        seg = cs[int(float(r.start_s) * cfs):int(float(r.end_s) * cfs)]
        if len(seg) < 0.2 * cfs:
            continue
        if int(cfs) != fs_out:
            g = math.gcd(int(cfs), fs_out)
            seg = resample_poly(seg, fs_out // g, int(cfs) // g)
        seg = np.asarray(seg, dtype=np.float32)
        seg = np.pad(seg, (0, max(0, L - len(seg))))[:L]
        X.append(seg)
        y.append(classes.index(r.label_str))
        pat.append(str(r.patient_id))
        rid.append(str(r.recording_id))
    X = _std(np.stack(X)[:, None, :])
    return {"X": X, "y": np.array(y, dtype=np.int64), "patient": np.array(pat),
            "fs": float(fs_out), "suite": "audio", "classes": list(classes),
            "recording_id": np.array(rid)}


LUNG4 = ["normal", "crackle", "wheeze", "both"]


@_cached("icbhi")
def icbhi(cap=DEFAULT_CAP):
    d = _load_manifest_audio("icbhi_manifest.csv", LUNG4, 4000, 4.0, cap)
    d["group"] = np.array([r.split("_")[-1] for r in d.pop("recording_id")])
    return d


@_cached("sprsound")
def sprsound(cap=DEFAULT_CAP):
    return _load_manifest_audio("sprsound_manifest.csv", LUNG4, 4000, 4.0, cap)


@_cached("hf_lung")
def hf_lung(cap=DEFAULT_CAP):
    return _load_manifest_audio("hf_lung_v1_manifest.csv", LUNG4, 4000, 4.0, cap)


@_cached("kauh")
def kauh(cap=DEFAULT_CAP):
    return _load_manifest_audio("kauh_manifest.csv",
                                ["normal", "wheeze", "crackle"], 4000, 4.0, cap)


@_cached("circor")
def circor(cap=DEFAULT_CAP):
    """Heart sound. Band-limited to the PCG band and 5 s at 2 kHz (fixes the weak pilot)."""
    from scipy.signal import butter, sosfiltfilt
    d = _load_manifest_audio("circor2022_manifest.csv", ["normal", "abnormal"],
                             2000, 5.0, cap)
    sos = butter(4, [25 / 1000.0, 400 / 1000.0], btype="band", output="sos")
    d["X"] = _std(sosfiltfilt(sos, d["X"], axis=-1).astype(np.float32))
    return d


@_cached("pascal")
def pascal(cap=DEFAULT_CAP):
    """PASCAL heart sound: set_a = phone microphone, set_b = clinical stethoscope.

    Kept as a NATURAL acquisition-shift pair; `patient` encodes the set so a
    leave-one-device-out evaluation is possible.
    """
    import soundfile as sf
    from scipy.signal import resample_poly, butter, sosfiltfilt
    fs_out, dur = 2000, 5.0
    L = int(fs_out * dur)
    # labels are encoded in the PASCAL filenames (normal__, murmur__, extrahls__, ...)
    recs = []
    for st in ("set_a", "set_b"):
        for p in sorted(glob.glob(os.path.join(PASCAL, st, "*.wav"))):
            base = os.path.basename(p).lower()
            if base.startswith("aunlabelled") or base.startswith("bunclassified"):
                continue
            tag = base.split("__")[0].replace("_", "")
            tag = tag[1:] if tag[:1] in "ab" and not tag.startswith("normal") else tag
            if tag.startswith("normal"):
                cls = "normal"
            elif tag.startswith(("murmur", "extrahls", "extrastole", "artifact")):
                cls = "abnormal"
            else:
                continue
            recs.append({"path": p, "cls": cls, "set": st})
    df = pd.DataFrame(recs)
    X, y, pat = [], [], []
    for _, r in df.iterrows():
        p, fn = r["path"], os.path.basename(r["path"])
        try:
            s, fs = sf.read(p, dtype="float32", always_2d=False)
        except Exception:
            continue
        if s.ndim > 1:
            s = s.mean(1)
        g = math.gcd(int(fs), fs_out)
        s = resample_poly(s, fs_out // g, int(fs) // g).astype(np.float32)
        for k in range(max(1, len(s) // L)):
            seg = s[k * L:(k + 1) * L]
            if len(seg) < L * 0.6:
                break
            seg = np.pad(seg, (0, L - len(seg)))
            X.append(seg)
            y.append(0 if r["cls"] == "normal" else 1)
            pat.append(f"{r['set']}__{os.path.basename(fn)}")
    sos = butter(4, [25 / 1000.0, 400 / 1000.0], btype="band", output="sos")
    X = _std(sosfiltfilt(sos, np.stack(X)[:, None, :], axis=-1).astype(np.float32))
    return {"X": X, "y": np.array(y, dtype=np.int64), "patient": np.array(pat),
            "group": np.array([p.split("__")[0] for p in pat]),
            "fs": float(fs_out), "suite": "audio", "classes": ["normal", "abnormal"]}


# --------------------------------------------------------------------------- ECG
@_cached("apnea")
def apnea(cap=DEFAULT_CAP):
    """Apnea-ECG, per-minute apnea vs normal.

    Record naming caveat (verified 2026-07-24): the learning set holds three files per
    subject-night for 8 subjects — the base ECG record (a01), a re-annotated ECG copy with
    extra signals (a01er, 5 channels incl. ECG at ch0), and a signals-only record (a01r,
    channels = Resp C / Resp A / Resp N / SpO2, NO ECG). a01/a01er/a01r are the SAME person.
    We keep ONLY the base records (a01..a20, b01..b05, c01..c10 = 35 single-channel ECG),
    which are one recording per subject, so the record id is a valid patient key and channel 0
    is always ECG. Keeping a01r would (i) read oximetry as ECG and (ii) split one subject
    across folds. A hard assert below refuses any non-ECG channel-0 record.
    """
    import wfdb
    fs, L = 100, 6000
    recs = [l.strip() for l in open(os.path.join(APNEA, "RECORDS")) if l.strip()]
    # base records only: drop the withheld test set (x*) and BOTH derived variants (a01er, a01r)
    recs = [r for r in recs
            if not r.startswith("x") and not r.endswith("er") and not r.endswith("r")]
    X, y, pat = [], [], []
    for rc in recs:
        try:
            rec = wfdb.rdrecord(os.path.join(APNEA, rc))
            ann = wfdb.rdann(os.path.join(APNEA, rc), "apn")
        except Exception:
            continue
        assert rec.sig_name[0].upper().startswith("ECG"), \
            f"apnea record {rc} channel 0 is {rec.sig_name[0]!r}, not ECG"
        sig = np.nan_to_num(rec.p_signal[:, 0].astype(np.float32))
        for i, s in enumerate(ann.symbol):
            if s not in ("A", "N"):
                continue
            a, b = i * L, (i + 1) * L
            if b > len(sig):
                break
            X.append(sig[a:b])
            y.append(1 if s == "A" else 0)
            pat.append(rc)
    X = _std(np.stack(X)[:, None, :])
    return {"X": X, "y": np.array(y, dtype=np.int64), "patient": np.array(pat),
            "fs": float(fs), "suite": "ecg", "classes": ["normal", "apnea"]}


@_cached("ptbdb")
def ptbdb(cap=DEFAULT_CAP):
    """PTB Diagnostic ECG: myocardial infarction vs healthy control, lead II, 10 s @ 250 Hz."""
    import wfdb
    from scipy.signal import resample_poly
    fs_out, dur = 250, 10.0
    L = int(fs_out * dur)
    X, y, pat = [], [], []
    for hea in sorted(glob.glob(os.path.join(PTBDB, "patient*", "*.hea"))):
        base = hea[:-4]
        try:
            rec = wfdb.rdrecord(base)
        except Exception:
            continue
        cmt = " ".join(rec.comments).lower()
        if "myocardial infarction" in cmt:
            lab = 1
        elif "healthy control" in cmt:
            lab = 0
        else:
            continue
        names = [n.lower() for n in rec.sig_name]
        ch = names.index("ii") if "ii" in names else 0
        sig = np.nan_to_num(rec.p_signal[:, ch].astype(np.float32))
        g = math.gcd(int(rec.fs), fs_out)
        sig = resample_poly(sig, fs_out // g, int(rec.fs) // g).astype(np.float32)
        pid = os.path.basename(os.path.dirname(hea))
        for k in range(min(6, len(sig) // L)):
            X.append(sig[k * L:(k + 1) * L])
            y.append(lab)
            pat.append(pid)
    X = _std(np.stack(X)[:, None, :])
    return {"X": X, "y": np.array(y, dtype=np.int64), "patient": np.array(pat),
            "fs": float(fs_out), "suite": "ecg", "classes": ["healthy", "MI"]}


@_cached("mitbihst")
def mitbihst(cap=DEFAULT_CAP):
    """MIT-BIH ST Change: ST-deviation episodes vs normal, driven by the project manifest."""
    import wfdb
    from scipy.signal import resample_poly
    fs_out, dur = 250, 10.0
    L = int(fs_out * dur)
    df = pd.read_csv(os.path.join(MAN, "mitbihst_manifest.csv"))
    df = _cap_by_patient(df, cap).sort_values("recording_id").reset_index(drop=True)
    classes = ["normal", "ST_dev"]
    cache_rec, sig, fs_in = None, None, None
    X, y, pat = [], [], []
    for _, r in df.iterrows():
        rid = str(r.recording_id).split("_")[0]
        if rid != cache_rec:
            try:
                rec = wfdb.rdrecord(os.path.join(MITST, rid))
            except Exception:
                cache_rec = rid
                continue
            sig = np.nan_to_num(rec.p_signal[:, 0].astype(np.float32))
            fs_in = int(rec.fs)
            cache_rec = rid
        if sig is None:
            continue
        a = int(float(r.start_s) * fs_in)
        b = int(float(r.end_s) * fs_in)
        seg = sig[a:b]
        if len(seg) < 0.5 * fs_in:
            continue
        g = math.gcd(fs_in, fs_out)
        seg = resample_poly(seg, fs_out // g, fs_in // g).astype(np.float32)
        seg = np.pad(seg, (0, max(0, L - len(seg))))[:L]
        X.append(seg)
        y.append(classes.index(r.label_str))
        pat.append(str(r.patient_id))
    X = _std(np.stack(X)[:, None, :])
    return {"X": X, "y": np.array(y, dtype=np.int64), "patient": np.array(pat),
            "fs": float(fs_out), "suite": "ecg", "classes": classes}


# --------------------------------------------------------------------------- EEG
@_cached("chbmit")
def chbmit(cap=DEFAULT_CAP):
    """CHB-MIT scalp EEG seizure detection from the existing preprocessed cache.

    Cache holds (n, 18, 1024) windows at 256 Hz. We keep 8 channels to fit the GPU and
    subsample the (very imbalanced) non-seizure class.
    """
    files = sorted(glob.glob(os.path.join(CHB, "*.npz")))
    rng = np.random.default_rng(0)
    X, y, pat = [], [], []
    per_subj = max(50, cap // max(1, len(files)))
    for f in files:
        z = np.load(f)
        xs, ys = z["X"], z["y"].astype(np.int64)
        pid = str(z["pid"])
        pos = np.where(ys == 1)[0]
        neg = np.where(ys == 0)[0]
        if len(pos) == 0:
            continue
        k = min(len(pos), per_subj // 2)
        pos = rng.choice(pos, k, replace=False)
        neg = rng.choice(neg, min(len(neg), k), replace=False)
        idx = np.concatenate([pos, neg])
        X.append(xs[idx][:, :8, :])
        y.append(ys[idx])
        pat += [pid] * len(idx)
    X = _std(np.concatenate(X))
    return {"X": X, "y": np.concatenate(y), "patient": np.array(pat),
            "fs": 256.0, "suite": "eeg", "classes": ["nonseizure", "seizure"]}


REGISTRY = {
    "icbhi": icbhi, "sprsound": sprsound, "hf_lung": hf_lung, "kauh": kauh,
    "circor": circor, "pascal": pascal,
    "apnea": apnea, "ptbdb": ptbdb, "mitbihst": mitbihst,
    "chbmit": chbmit,
}

MODALITY = {"icbhi": "lung", "sprsound": "lung", "hf_lung": "lung", "kauh": "lung",
            "circor": "pcg", "pascal": "pcg",
            "apnea": "ecg", "ptbdb": "ecg", "mitbihst": "ecg",
            "chbmit": "eeg"}


def load(name, cap=DEFAULT_CAP):
    return REGISTRY[name](cap=cap)
