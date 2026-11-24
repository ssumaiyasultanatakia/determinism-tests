"""A third untested invariant: is the suite invariant to how the caller batches its input?

The paper's two findings are instances of one thing -- a property users reasonably assume a
released artefact has, that its authors never wrote a test for. Determinism is assumed because a
seed was fixed. A severity scale is assumed because the levels are numbered. Two instances is an
anecdote. This tests a third, chosen because it is the one a user is most likely to trip over
without ever suspecting it: corrupting a dataset in chunks rather than in a single call.

Nothing in the API suggests chunking is unsafe, and there is every reason to do it -- memory, a
progress bar, parallelism across workers. But a per-cell generator is consumed in draw order, so
if a family draws an array sized to its input, splitting the input changes every draw after the
first chunk. The seed is correct, the code is deterministic, the fix from Section IV is in place,
and the outputs still differ.

Compares apply(X) against concat(apply(X[:h]), apply(X[h:])) for every cell in the ECG suite,
under the CORRECTED seed derivation, so any divergence found is a separate defect and not a
restatement of the first one.
-> conference/results/batch_invariance.json
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "batch_invariance.json")
P6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "suite")
sys.path.insert(0, P6)
from biosignal_c import corruptions as C   # noqa: E402

CACHE = os.path.join(P6, "_cache", "apnea.npz")
N = 256            # segments; small and even, so the split is unambiguous
SEED = 0


def rel_l2(a, b):
    d = np.linalg.norm((a - b).ravel())
    n = np.linalg.norm(a.ravel())
    return float(d / n) if n > 0 else 0.0


def main():
    z = np.load(CACHE, allow_pickle=True)
    X = z["X"][:N].astype(np.float32); fs = float(z["fs"]); suite = str(z["suite"])
    h = N // 2
    rows = {}
    for fam, sev in C.SUITES[suite].cells():
        whole = C.apply(X, fs, suite, fam, sev, seed=SEED)
        a = C.apply(X[:h], fs, suite, fam, sev, seed=SEED)
        b = C.apply(X[h:], fs, suite, fam, sev, seed=SEED)
        chunked = np.concatenate([a, b], axis=0)
        r = rel_l2(whole, chunked)
        rows["%s|%d" % (fam, sev)] = {"relative_L2": round(r, 6),
                                      "identical": bool(np.array_equal(whole, chunked))}
    n = len(rows)
    diff = {k: v for k, v in rows.items() if not v["identical"]}
    fams_hit = sorted({k.split("|")[0] for k in diff})
    vals = [v["relative_L2"] for v in diff.values()]
    out = {"n_cells": n, "n_divergent": len(diff),
           "frac_divergent": round(len(diff) / n, 3),
           "families_affected": fams_hit,
           "median_relative_L2_divergent": round(float(np.median(vals)), 4) if vals else 0.0,
           "max_relative_L2": round(float(np.max(vals)), 4) if vals else 0.0,
           "n_segments": int(N), "split": [int(h), int(N - h)], "seed": SEED,
           "per_cell": rows}
    out["reading"] = (
        "Corrupting %d segments in one call and in two halves, with the corrected seed derivation "
        "and an identical nominal seed, gives different data in %d of %d cells (%.0f%%), across %d "
        "families, at a median relative L2 of %.3f among the affected cells. The API gives a caller "
        "no reason to suspect this and every reason to batch."
        % (N, len(diff), n, 100 * len(diff) / n, len(fams_hit),
           out["median_relative_L2_divergent"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_cell"}, indent=2))


if __name__ == "__main__":
    main()
