"""95% confidence intervals for the ranking-stability rates (Table III and Section VII).

Cluster bootstrap with the CELL as the resampling unit. Rates are computed per cell first
(over its 15 session pairs for the seeding study, or its single block-vs-halves comparison for
the batching study), so all observations from one cell stay together and the dependence between
session pairs that share a cell, and between pairs that share a session, is never broken.
Cells are resampled with replacement (B = 20,000, fixed seed) and the rate is the mean of the
per-cell rates in each resample; the interval is the 2.5th-97.5th percentile.
Inputs: results/leaderboard_flip.json, results/batch_leaderboard.json -> results/leaderboard_ci.json
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results")
B = 20000


def boot(v, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return {"mean": round(float(v.mean()), 3), "ci95": [round(float(np.percentile(m, 2.5)), 3),
                                                       round(float(np.percentile(m, 97.5)), 3)]}


def boot_family(per_cell, seed=0):
    """Sensitivity check: resample whole corruption families (cells of one family stay together)."""
    fams = {}
    for key, rate in per_cell:
        fams.setdefault(key.split("|")[0], []).append(rate)
    F = list(fams); rng = np.random.default_rng(seed)
    m = [np.mean(sum((fams[F[i]] for i in rng.integers(0, len(F), len(F))), [])) for _ in range(B)]
    return {"n_families": len(F), "ci95": [round(float(np.percentile(m, 2.5)), 3),
                                          round(float(np.percentile(m, 97.5)), 3)]}


lf = json.load(open(os.path.join(RES, "leaderboard_flip.json")))["per_cell"]
rng_cells = [c for c in lf.values() if c["rng_dependent"]]
bl = json.load(open(os.path.join(RES, "batch_leaderboard.json")))["per_cell"]
bs_cells = [c for c in bl.values() if c["batch_sensitive"]]

out = {"seeding": {"n_cells": len(rng_cells), "pairs_per_cell": 15,
                   "winner_change_rate": boot([c["winner_change_rate"] for c in rng_cells]),
                   "swing_exceeds_margin": boot([c["swing_exceeds_margin_rate"] for c in rng_cells])},
       "batching": {"n_cells": len(bs_cells), "comparisons_per_cell": 1,
                    "winner_change_rate": boot([float(c["winner_changed"]) for c in bs_cells]),
                    "swing_exceeds_margin": boot([float(c["auc_swing"] > c["top1_margin"]) for c in bs_cells])},
       "n_boot": B, "method": "cluster (cell-level) percentile bootstrap"}
out["seeding"]["swing_exceeds_margin_family_level"] = boot_family(
    [(k, c["swing_exceeds_margin_rate"]) for k, c in lf.items() if c["rng_dependent"]])
json.dump(out, open(os.path.join(RES, "leaderboard_ci.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
