"""Reproduce the defect on demand, so its magnitude is a measurement rather than a memory.

The suite is fixed, so the released code now passes. A reader cannot check a claim about how bad it
was unless the broken derivation can be restored on request. This monkey-patches the pre-fix seed
function back in, re-runs the same cell sweep, and reports the two numbers that matter together:

  within-process mismatches  - what a conventional determinism test measures
  across-process mismatches  - what reproduction actually requires

The gap between them is the finding. A test that only repeats calls inside one interpreter returns a
clean bill of health on code that cannot be reproduced tomorrow.
-> conference/results/prefix_reproduction.json
"""
import hashlib, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "prefix_reproduction.json")
SUITE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "suite")
sys.path.insert(0, SUITE_SRC)


def digest_cells(mode):
    """Hash every (suite, family, severity) output under the chosen seed derivation."""
    from biosignal_c import corruptions as C
    if mode == "prefix":                      # restore the defective derivation
        C._rng = lambda family, sev, seed: np.random.default_rng(abs(hash((family, sev, seed))) % (2 ** 32))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4000)).astype(np.float64)
    out = {}
    for sname, suite in C.SUITES.items():
        for fam in suite.families:
            for sev in suite.severities:
                key = f"{sname}/{fam}/{sev}"
                try:
                    y = C.apply(x, 1000.0, sname, fam, int(sev), seed=0)
                    out[key] = hashlib.blake2b(np.ascontiguousarray(y, dtype=np.float64).tobytes(),
                                               digest_size=8).hexdigest()
                except Exception:
                    continue
    return out


if len(sys.argv) > 2 and sys.argv[1] == "--child":
    print(json.dumps(digest_cells(sys.argv[2]))); sys.exit(0)

res = {}
for mode in ("prefix", "fixed"):
    within = digest_cells(mode)
    within2 = digest_cells(mode)
    n_within = sum(1 for k in within if within[k] != within2.get(k))
    runs = []
    for _ in range(3):
        env = dict(os.environ); env.pop("PYTHONHASHSEED", None)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", mode],
                           capture_output=True, text=True, env=env)
        try: runs.append(json.loads(r.stdout))
        except Exception: pass
    keys = sorted(set(within) & set.intersection(*[set(r) for r in runs])) if runs else []
    n_across = sum(1 for k in keys if len({r[k] for r in runs}) > 1)
    res[mode] = {"cells": len(keys), "within_process_mismatches": n_within,
                 "across_process_mismatches": n_across,
                 "across_process_rate": round(n_across / max(1, len(keys)), 4)}
    print(f"[{mode:6}] cells={len(keys):3}  within={n_within:3}  across={n_across:3}"
          f"  ({res[mode]['across_process_rate']:.1%})", flush=True)
res["interpretation"] = ("A within-process determinism test returns zero mismatches under the "
                         "defective derivation, so it cannot detect the defect that makes the "
                         "benchmark irreproducible across sessions.")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
