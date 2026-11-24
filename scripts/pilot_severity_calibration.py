"""PILOT - are the three severity levels actually a scale, and is it comparable across families?

A corruption suite that labels its levels 1, 2, 3 is making two implicit promises: that severity
increases within a family, and that a given level means something similar across families. Neither is
guaranteed by construction - each family has its own parameterisation, and the levels were set by
judgement. If either promise fails the suite still works as a set of perturbations but must not be
summarised by severity, and the data descriptor has to say so.

We measure both at signal level, which is model-independent and therefore the right place to check a
data artefact: distortion is quantified as relative L2 against the clean signal, and the within-family
ordering and cross-family spread are reported.

KILL CRITERION. If a substantial share of families are non-monotone in severity, the level labels are
not a scale and the descriptor must report per-family parameters instead of a severity axis.
-> conference/results/pilot_severity_calibration.json
"""
import json, os, sys
import numpy as np

SUITE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "suite")
sys.path.insert(0, SUITE_SRC)
from biosignal_c.corruptions import SUITES, apply

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "pilot_severity_calibration.json")
FS = {"audio": 4000.0, "ecg": 1000.0, "eeg": 256.0}
RNG = np.random.default_rng(5488)


def clean_signal(suite, n_ch, n=4096):
    """A neutral broadband probe: the question is the transform, not the physiology."""
    x = RNG.standard_normal((n_ch, n))
    t = np.arange(n) / FS[suite]
    x += 0.5 * np.sin(2 * np.pi * 1.2 * t)[None, :]          # slow component
    return np.ascontiguousarray(x, dtype=np.float64)


def rel_l2(a, b):
    return float(np.linalg.norm(np.ravel(a) - np.ravel(b)) / (np.linalg.norm(np.ravel(b)) + 1e-12))


res = {"per_family": {}, "fs": FS, "probe": "broadband gaussian + 1.2 Hz component, 4096 samples"}
for sname, suite in SUITES.items():
    sevs = sorted(suite.severities)
    for ch in (2, 1):                                  # some families need multi-channel input
        ok = True
        for fam in suite.families:
            try:
                x = clean_signal(sname, ch)
                apply(x, FS[sname], sname, fam, int(sevs[0]), seed=0)
            except Exception:
                ok = False; break
        if ok: break
    x = clean_signal(sname, ch)
    for fam in suite.families:
        row = {}
        try:
            for s in sevs:
                y = apply(x, FS[sname], sname, fam, int(s), seed=0)
                row[str(s)] = round(rel_l2(y, x), 4)
        except Exception as e:
            res["per_family"][f"{sname}/{fam}"] = {"error": repr(e)[:90]}; continue
        vals = [row[str(s)] for s in sevs]
        res["per_family"][f"{sname}/{fam}"] = {
            "distortion_by_severity": row,
            "monotone_increasing": bool(all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))),
            "range": round(max(vals) - min(vals), 4)}

ok = {k: v for k, v in res["per_family"].items() if "distortion_by_severity" in v}
mono = [k for k, v in ok.items() if v["monotone_increasing"]]
res["n_families_measured"] = len(ok)
res["n_monotone"] = len(mono)
res["non_monotone"] = [k for k in ok if k not in mono]
for s in sevs:
    vals = [v["distortion_by_severity"][str(s)] for v in ok.values()]
    res.setdefault("cross_family_spread", {})[str(s)] = {
        "median": round(float(np.median(vals)), 4),
        "min": round(float(min(vals)), 4), "max": round(float(max(vals)), 4)}
res["PASS"] = bool(len(mono) >= 0.8 * len(ok))
res["verdict"] = ("severity behaves as a scale in %d/%d families" % (len(mono), len(ok)) +
                  ("; report as an axis" if res["PASS"] else "; do NOT summarise by severity"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"measured {len(ok)} families | monotone {len(mono)} | PASS={res['PASS']}")
print("non-monotone:", res["non_monotone"] or "none")
print("cross-family spread:", json.dumps(res.get("cross_family_spread", {})))
