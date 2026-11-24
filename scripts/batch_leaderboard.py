"""Does chunking the input change the leaderboard, the way a process-dependent seed does?

Section VII shows that corrupting a dataset in one call and in two halves gives different data in 18
of 33 ECG cells, under the corrected seed derivation. That is a statement about the data. The paper's
strongest experimental design - freeze the models, change only the thing under test, watch the
ranking - has so far been applied only to the seeding defect. Applying it a second time is what turns
"untested invariant" from an asserted class into a demonstrated one.

Same protocol as the seeding experiment: eight classifiers trained once on clean data and never
retrained, so no ranking change can come from training stochasticity. The only difference between the
two scorings is whether the corruption saw the test set as one block or as two halves.
-> conference/results/batch_leaderboard.json
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "batch_leaderboard.json")
P6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "suite")
sys.path.insert(0, P6)
sys.path.insert(0, HERE)
from biosignal_c import corruptions as C          # noqa: E402
from leaderboard_flip import bandpower_features, build_models, CACHE, N_TRAIN, N_TEST, RNG  # noqa: E402


def main():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    z = np.load(CACHE, allow_pickle=True)
    X, y, pat, fs, suite = z["X"], z["y"], z["patient"], float(z["fs"]), str(z["suite"])
    pts = np.array(sorted(set(pat.tolist())))
    order = RNG.permutation(len(pts))
    tr_p = set(pts[order[: int(0.65 * len(pts))]].tolist())
    tr = np.array([p in tr_p for p in pat])
    itr = np.where(tr)[0][:N_TRAIN]; ite = np.where(~tr)[0][:N_TEST]

    models = build_models()
    Ftr, ytr = bandpower_features(X[itr], fs), y[itr]
    for m in models.values():
        m.fit(Ftr, ytr)
    names = list(models)
    Xte, yte = X[ite], y[ite]
    h = len(Xte) // 2

    per_cell = {}
    for fam, sev in C.SUITES[suite].cells():
        whole = C.apply(Xte, fs, suite, fam, sev, seed=0)
        a = C.apply(Xte[:h], fs, suite, fam, sev, seed=0)
        b = C.apply(Xte[h:], fs, suite, fam, sev, seed=0)
        chunked = np.concatenate([a, b], axis=0)
        identical = bool(np.array_equal(whole, chunked))
        Fw, Fc = bandpower_features(whole, fs), bandpower_features(chunked, fs)
        vw = np.array([roc_auc_score(yte, m.predict_proba(Fw)[:, 1]) for m in models.values()])
        vc = np.array([roc_auc_score(yte, m.predict_proba(Fc)[:, 1]) for m in models.values()])
        r = spearmanr(vw, vc).statistic
        srt = np.sort(vw)[::-1]
        per_cell["%s|%d" % (fam, sev)] = {
            "batch_sensitive": not identical,
            "winner_changed": names[int(np.argmax(vw))] != names[int(np.argmax(vc))],
            "auc_swing": round(float(np.max(np.abs(vw - vc))), 4),
            "top1_margin": round(float(srt[0] - srt[1]), 4),
            "rank_rho": 1.0 if np.isnan(r) else round(float(r), 3)}

    def agg(sel):
        c = {k: v for k, v in per_cell.items() if sel(v)}
        if not c:
            return {"n_cells": 0}
        sw = np.array([v["auc_swing"] for v in c.values()])
        mg = np.array([v["top1_margin"] for v in c.values()])
        return {"n_cells": len(c),
                "winner_changed_cells": int(sum(v["winner_changed"] for v in c.values())),
                "winner_change_rate": round(float(np.mean([v["winner_changed"] for v in c.values()])), 3),
                "median_auc_swing": round(float(np.median(sw)), 4),
                "max_auc_swing": round(float(np.max(sw)), 4),
                "median_top1_margin": round(float(np.median(mg)), 4),
                "swing_exceeds_margin_rate": round(float(np.mean(sw > mg)), 3),
                "median_rank_rho": round(float(np.median([v["rank_rho"] for v in c.values()])), 3)}

    out = {"setup": {"dataset": "apnea", "suite": suite, "n_test": int(len(ite)),
                     "split": [h, len(Xte) - h], "n_models": len(names),
                     "note": "models trained once on clean data; only the batching of the corruption differs"},
           "per_cell": per_cell,
           "summary": {"all_cells": agg(lambda v: True),
                       "batch_sensitive_cells": agg(lambda v: v["batch_sensitive"]),
                       "batch_stable_cells": agg(lambda v: not v["batch_sensitive"])}}
    b = out["summary"]["batch_sensitive_cells"]; st = out["summary"]["batch_stable_cells"]
    out["reading"] = (
        "Across %d cells, %d are batch-sensitive and %d are not. On the batch-sensitive cells the "
        "reported winner changes in %d (rate %.3f) and the swing exceeds the winner's margin at rate "
        "%.3f, median rank correlation %.3f; the batch-stable cells move by exactly zero. Chunking the "
        "input therefore perturbs a leaderboard by the same mechanism a re-seeded interpreter does."
        % (out["summary"]["all_cells"]["n_cells"], b["n_cells"], st["n_cells"],
           b["winner_changed_cells"], b["winner_change_rate"], b["swing_exceeds_margin_rate"],
           b["median_rank_rho"]))
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out["summary"], indent=2)); print(); print(out["reading"])


if __name__ == "__main__":
    main()
