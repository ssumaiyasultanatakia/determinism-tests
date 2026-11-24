"""The defect, isolated and measured: what PYTHONHASHSEED does to a "seeded" benchmark.

BioSignal-C derived its per-cell RNG from ``hash((family, severity, seed))``. That looks
deterministic and is not: CPython salts the hash of str and bytes with a per-process random value
unless PYTHONHASHSEED is set, so the same nominal seed produces a different stream in every new
interpreter. Corruptions that never touch the RNG - clipping, polarity inversion, bandwidth limiting
- were unaffected, which is why the defect survived casual checking: a spot-check that happened to
land on a filter looked reproducible.

This script does three things a reader needs in order to believe the finding:
  1. shows the raw instability of hash() on identical keys across interpreters;
  2. contrasts the old and fixed seed derivations directly, in-process, with no benchmark attached;
  3. reports how far apart two "identically seeded" corruption realisations actually are, so the
     consequence is a measured quantity rather than a worry.
-> conference/results/hash_defect.json
"""
import json, os, subprocess, sys, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "hash_defect.json")
KEYS = [("audio_babble", 2, 0), ("ecg_emg", 3, 0), ("powerline", 1, 7), ("drift", 4, 3)]


def old_seed(key):                      # what the suite used to do
    return abs(hash(key)) % (2 ** 32)


def new_seed(key):                      # what it does now
    k = "|".join(map(str, key)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(k, digest_size=4).digest(), "big")


def child(_):
    return json.dumps({str(k): [old_seed(k), new_seed(k)] for k in KEYS})


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--child":
    print(child(None)); sys.exit(0)

if __name__ == "__main__":
    # 1-2. collect both derivations from several fresh interpreters
    runs = []
    for _ in range(8):
        env = dict(os.environ); env.pop("PYTHONHASHSEED", None)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"],
                           capture_output=True, text=True, env=env)
        runs.append(json.loads(r.stdout))
    res = {"n_interpreters": len(runs), "keys": {}}
    for k in KEYS:
        ks = str(k)
        olds = [r[ks][0] for r in runs]; news = [r[ks][1] for r in runs]
        res["keys"][ks] = {"old_distinct_seeds": len(set(olds)), "new_distinct_seeds": len(set(news)),
                           "old_example": olds[:3], "new_value": news[0]}
    res["old_is_stable"] = all(v["old_distinct_seeds"] == 1 for v in res["keys"].values())
    res["new_is_stable"] = all(v["new_distinct_seeds"] == 1 for v in res["keys"].values())

    # 3. consequence: how different are two realisations that a user believes are the same?
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "suite"))
    from biosignal_c.corruptions import apply
    rng0 = np.random.default_rng(0)
    x = rng0.standard_normal((2, 4000)).astype(np.float64)  # 2 leads x 4 s at 1 kHz
    div = {}
    from biosignal_c.corruptions import SUITES
    suite = "ecg" if "ecg" in SUITES else list(SUITES)[0]
    fams = list(SUITES[suite].families)
    for fam in fams:
        try:
            sev = int(sorted(SUITES[suite].severities)[len(SUITES[suite].severities) // 2])
            a = apply(x, 1000.0, suite, fam, sev, seed=0)
            b = apply(x, 1000.0, suite, fam, sev, seed=12345)   # a stream the old code could have drawn
            d = float(np.linalg.norm(a - b)) / (float(np.linalg.norm(a)) + 1e-12)
            c = float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])
            div[f"{suite}/{fam}"] = {"severity": sev, "relative_L2": round(d, 4), "corr": round(c, 4),
                                     "rng_dependent": bool(d > 1e-9)}
        except Exception as e:
            div[f"{suite}/{fam}"] = {"error": repr(e)[:110]}
    rngdep = [v for v in div.values() if v.get("rng_dependent")]
    res["n_families"] = len(div)
    res["n_rng_dependent"] = len(rngdep)
    res["median_relative_L2_rng_dependent"] = (
        round(float(np.median([v["relative_L2"] for v in rngdep])), 4) if rngdep else None)
    res["realisation_divergence"] = div
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2)[:1400])
