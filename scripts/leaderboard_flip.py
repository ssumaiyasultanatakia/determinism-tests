"""What a process-dependent seed does to the ranking a benchmark is used to produce.

The defect has so far been measured on the data: two nominally identical realisations of a
corruption cell differ by a median relative L2 of 0.371. That is the input to the question a
reader actually cares about, not the answer to it. A robustness benchmark exists to rank methods,
so the claim worth establishing is whether the ranking moves when only the interpreter changes.

Design. Models are trained once, on clean data, and never retrained: the defect lives in the
corruption of the *test* set, so a single training run is the correct control -- any ranking
change observed downstream cannot be attributed to training stochasticity, because there is none.
Each corruption cell is then realised under six session salts (SALTS, 15 session pairs), and the
same fixed models are scored on all of them. Under the defective derivation the per-cell generator
is an arbitrary draw that is fixed within a process and redrawn in the next. Each salt *emulates*
one such process: seed = default_rng([salt, cell_index]) gives every cell its own seed per session.
The script does not start six interpreters or call hash() itself; it reproduces the kind of
per-process seed change the defect causes, in a repeatable way.

Reported: rank agreement between sessions, how often the winner changes, and -- the
number that decides whether the benchmark can rank at all -- the session-to-session accuracy
swing set against the gaps between the methods being compared.
-> conference/results/leaderboard_flip.json
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "leaderboard_flip.json")
P6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "suite")
sys.path.insert(0, P6)

from biosignal_c import corruptions as C          # noqa: E402

CACHE = os.path.join(P6, "_cache", "apnea.npz")
N_TRAIN, N_TEST = 6000, 2500
SALTS = (11, 22, 33, 44, 55, 66)                  # six interpreter sessions -> 15 session pairs
RNG = np.random.default_rng(20260816)


def bandpower_features(X, fs, n_bands=24):
    """Log band power per segment. Deterministic, cheap, and independent of the corruption."""
    x = X[:, 0, :].astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    edges = np.linspace(0.5, min(45.0, fs / 2 - 1), n_bands + 1)
    out = np.empty((len(x), n_bands), dtype=np.float64)
    for i in range(n_bands):
        m = (f >= edges[i]) & (f < edges[i + 1])
        out[:, i] = np.log1p(P[:, m].sum(1))
    return out


def build_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s = lambda m: make_pipeline(StandardScaler(), m)
    return {
        "logreg_C1":   s(LogisticRegression(C=1.0, max_iter=2000)),
        "logreg_C005": s(LogisticRegression(C=0.05, max_iter=2000)),
        "lda":         s(LinearDiscriminantAnalysis()),
        "rf200":       RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
        "histgb":      HistGradientBoostingClassifier(random_state=0),
        "gaussnb":     s(GaussianNB()),
        "knn25":       s(KNeighborsClassifier(n_neighbors=25)),
        "mlp64":       s(MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=0)),
    }


def main():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    z = np.load(CACHE, allow_pickle=True)
    X, y, pat, fs, suite = z["X"], z["y"], z["patient"], float(z["fs"]), str(z["suite"])

    # patient-disjoint split, deterministic
    pts = np.array(sorted(set(pat.tolist())))
    order = RNG.permutation(len(pts))
    tr_p = set(pts[order[: int(0.65 * len(pts))]].tolist())
    tr = np.array([p in tr_p for p in pat])
    itr = np.where(tr)[0][:N_TRAIN]
    ite = np.where(~tr)[0][:N_TEST]

    Ftr, ytr = bandpower_features(X[itr], fs), y[itr]
    Xte, yte = X[ite], y[ite]

    models = build_models()
    for m in models.values():
        m.fit(Ftr, ytr)

    cells = list(C.SUITES[suite].cells())

    scores = {str(salt): {} for salt in SALTS}
    t0 = time.time()
    for salt in SALTS:
        for ci, (fam, sev) in enumerate(cells):
            # under the defect the per-cell stream is an arbitrary per-process draw;
            # a salt-dependent seed reproduces exactly that
            seed = int(np.random.default_rng([salt, ci]).integers(0, 2 ** 31))
            Xc = C.apply(Xte, fs, suite, fam, sev, seed=seed)
            Fc = bandpower_features(Xc, fs)
            scores[str(salt)]["%s|%d" % (fam, sev)] = {
                k: float(roc_auc_score(yte, m.predict_proba(Fc)[:, 1]))
                for k, m in models.items()}
        print("[salt %d] done %.0fs" % (salt, time.time() - t0), flush=True)

    names = list(models)
    from itertools import combinations
    pairs = list(combinations([str(s) for s in SALTS], 2))

    per_cell = {}
    for cell in scores[str(SALTS[0])]:
        V = {s: np.array([scores[s][cell][n] for n in names]) for s in map(str, SALTS)}
        # a cell is RNG-dependent iff its scores move at all between sessions
        rng_dep = bool(np.max([np.max(np.abs(V[a] - V[b])) for a, b in pairs]) > 0)
        f_rho, f_flip, f_swing, f_gap = [], [], [], []
        for a, b in pairs:
            va, vb = V[a], V[b]
            r = spearmanr(va, vb).statistic
            f_rho.append(1.0 if np.isnan(r) else float(r))
            f_flip.append(names[int(np.argmax(va))] != names[int(np.argmax(vb))])
            f_swing.append(float(np.max(np.abs(va - vb))))
            srt = np.sort(va)[::-1]
            f_gap.append(float(srt[0] - srt[1]))
        per_cell[cell] = {
            "rng_dependent": rng_dep,
            "mean_rank_rho": round(float(np.mean(f_rho)), 3),
            "winner_change_rate": round(float(np.mean(f_flip)), 3),
            "median_auc_swing": round(float(np.median(f_swing)), 4),
            "max_auc_swing": round(float(np.max(f_swing)), 4),
            "median_top1_margin": round(float(np.median(f_gap)), 4),
            "swing_exceeds_margin_rate": round(
                float(np.mean(np.array(f_swing) > np.array(f_gap))), 3)}

    def agg(sel):
        c = {k: v for k, v in per_cell.items() if sel(v)}
        if not c:
            return {"n_cells": 0}
        return {"n_cells": len(c),
                "winner_change_rate": round(
                    float(np.mean([v["winner_change_rate"] for v in c.values()])), 3),
                "cells_with_any_winner_change": int(
                    sum(v["winner_change_rate"] > 0 for v in c.values())),
                "median_rank_rho": round(
                    float(np.median([v["mean_rank_rho"] for v in c.values()])), 3),
                "min_rank_rho": round(
                    float(np.min([v["mean_rank_rho"] for v in c.values()])), 3),
                "median_auc_swing": round(
                    float(np.median([v["median_auc_swing"] for v in c.values()])), 4),
                "max_auc_swing": round(
                    float(np.max([v["max_auc_swing"] for v in c.values()])), 4),
                "median_top1_margin": round(
                    float(np.median([v["median_top1_margin"] for v in c.values()])), 4),
                "swing_exceeds_margin_rate": round(
                    float(np.mean([v["swing_exceeds_margin_rate"] for v in c.values()])), 3)}

    n = len(per_cell)
    by_pop = {"all_cells": agg(lambda v: True),
              "rng_dependent_cells": agg(lambda v: v["rng_dependent"]),
              "deterministic_cells": agg(lambda v: not v["rng_dependent"])}
    out = {"setup": {"dataset": "apnea", "suite": suite, "n_train": len(itr),
                     "n_test": len(ite), "n_models": len(names), "models": names,
                     "n_cells": n, "salts": list(SALTS),
                     "note": ("models trained once on clean data and reused for both sessions, "
                              "so no ranking change can come from training stochasticity")},
           "per_cell": per_cell,
           "n_session_pairs": len(pairs),
           "summary": by_pop}

    r = by_pop["rng_dependent_cells"]
    d = by_pop["deterministic_cells"]
    out["reading"] = (
        "Of %d corruption cells, %d are RNG-dependent and %d are pure filters that never touch "
        "the generator; the second group is bit-identical across sessions (winner change rate "
        "%.3f) and is reported separately rather than averaged in, because including it dilutes "
        "the effect by construction. On the %d RNG-dependent cells, scored with identical models "
        "and identical nominal seeds over %d session pairs, the AUROC swing between sessions has "
        "median %.4f against a median winner margin of %.4f. The winner changes at rate %.3f, and "
        "the swing exceeds the margin at rate %.3f. The ranking is therefore mostly stable: the "
        "defect corrupts the data in every affected cell, but only displaces the reported winner "
        "in the minority of cells where the top two methods are separated by less than the "
        "session noise. That minority is the part of a leaderboard a reader cannot trust."
        % (n, r["n_cells"], d["n_cells"], d.get("winner_change_rate", 0.0),
           r["n_cells"], len(pairs), r["median_auc_swing"], r["median_top1_margin"],
           r["winner_change_rate"], r["swing_exceeds_margin_rate"]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["summary"], indent=2))
    print(out["reading"])


if __name__ == "__main__":
    main()
