"""Does the three-line co-occurrence rule actually catch the defect it is looking for?

A negative scan result is only as good as the detector that produced it. This script measures the
rule's sensitivity against the known-defective construction (the one this paper reports) and against
deliberately harder variants, so the false-negative envelope is stated rather than assumed.
It also records what the scanned library environment actually contained.
-> conference/results/scan_rule_validation.json
"""
import io, json, os, sys, sysconfig, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_hash_seeding import scan_file

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "scan_rule_validation.json")

CASES = {
 "canonical_defect_one_line": ("""
def _rng(family, sev, seed):
    return np.random.default_rng(abs(hash((family, sev, seed))) % (2 ** 32))
""", True),
 "split_two_lines": ("""
def _rng(family, sev, seed):
    h = abs(hash((family, sev, seed))) % (2 ** 32)
    return np.random.default_rng(h)
""", True),
 "split_three_lines": ("""
def _rng(family, sev, seed):
    h = abs(hash((family, sev, seed)))
    h = h % (2 ** 32)
    return np.random.default_rng(h)
""", True),
 "split_five_lines": ("""
def _rng(family, sev, seed):
    h = abs(hash((family, sev, seed)))
    h = h % (2 ** 32)
    # a comment
    # another comment
    return np.random.default_rng(h)
""", False),
 "across_function_boundary": ("""
def cell_seed(family, sev, seed):
    return abs(hash((family, sev, seed))) % (2 ** 32)

def _rng(family, sev, seed):
    return np.random.default_rng(cell_seed(family, sev, seed))
""", False),
 "torch_manual_seed": ("""
def setup(tag):
    torch.manual_seed(abs(hash(tag)) % (2 ** 31))
""", True),
 "correct_blake2b_no_hit": ("""
def _rng(family, sev, seed):
    key = f"{family}|{sev}|{seed}".encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, 4).digest(), "big"))
""", False),
}


def main():
    res, tp, fn, fp = {}, 0, 0, 0
    for name, (src, should_hit) in CASES.items():
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(src); path = f.name
        hit = len(scan_file(path)) > 0
        os.unlink(path)
        res[name] = {"detected": hit, "rule_should_detect": should_hit,
                     "outcome": ("true positive" if hit and should_hit else
                                 "false negative" if (not hit) and should_hit else
                                 "true negative" if (not hit) and (not should_hit) else
                                 "detected (outside stated window)")}
        tp += hit and should_hit; fn += (not hit) and should_hit; fp += hit and not should_hit

    roots = [p for p in sys.path if p and os.path.isdir(p) and "site-packages" in p] \
            or [sysconfig.get_paths()["purelib"]]
    pkgs = set()
    for r in roots:
        for e in os.listdir(r):
            if e.endswith((".dist-info", ".egg-info")):
                pkgs.add(e.split("-")[0])
    out = {"cases": res,
           "sensitivity_within_window": f"{tp}/{tp+fn}",
           "known_false_negative_modes": [k for k, v in res.items()
                                          if v["outcome"] == "false negative"
                                          or (not v["detected"] and not CASES[k][1] and "boundary" in k)],
           "environment": {"python": sys.version.split()[0],
                           "site_packages_roots": roots,
                           "distributions_installed": len(pkgs)}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for k, v in res.items():
        print(f"  {k:30s} detected={str(v['detected']):5s} {v['outcome']}")
    print("sensitivity within the stated 3-line window:", out["sensitivity_within_window"])
    print("environment:", out["environment"]["python"], out["environment"]["distributions_installed"], "distributions")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
