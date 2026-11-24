"""Is seeding from hash() a local mistake or a shipped pattern?

The defect we found is only worth a paper if it is not peculiar to us. The failure needs three
things to occur together: a value derived from Python's ``hash()``, applied to a str/bytes-containing
key, flowing into a random-number seed. Each is innocuous alone. Together they produce a generator
that is stable within a process and different in the next one, which is precisely the configuration
that survives testing and breaks reproduction.

Two populations are scanned separately, because they answer different questions and only one of them
is the population the paper claims is at risk:

  LIBRARIES  installed third-party packages. A control: if the pattern were endemic to the ecosystem
             it would appear here. Absence here does NOT bound its frequency in research code.
  RESEARCH   public research repositories available locally. This is the at-risk population, and our
             sample of it is small enough that it establishes nothing about prevalence; it is
             reported so that the at-risk population is sampled at all rather than only argued about.

The rule is deliberately conservative: a hit is recorded only when a hash() call and a seeding call
appear within a few lines of each other, and the snippet is recorded so every hit can be checked by
hand rather than trusted. The intent is not to accuse any project - a hit may be benign - but to
establish whether the pattern is one a practitioner could plausibly copy.
-> conference/results/hash_seeding_scan.json
"""
import io, json, os, re, sys, sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "hash_seeding_scan.json")
SEED_CALL = re.compile(r"(default_rng|RandomState|np\.random\.seed|random\.seed|manual_seed|set_seed)\s*\(")
HASH_CALL = re.compile(r"\bhash\s*\(")
WINDOW = 3          # lines between the hash() and the seeding call


def scan_file(path):
    try:
        src = io.open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        return []
    if "hash(" not in src:
        return []
    lines = src.splitlines()
    hits = []
    for i, ln in enumerate(lines):
        if not HASH_CALL.search(ln):
            continue
        lo, hi = max(0, i - WINDOW), min(len(lines), i + WINDOW + 1)
        for j in range(lo, hi):
            if SEED_CALL.search(lines[j]):
                hits.append({"line": i + 1, "snippet": "\n".join(lines[lo:hi])[:300]})
                break
    return hits


# Public research repositories present on this machine. Our own repositories are excluded: scanning
# code we wrote would only rediscover the instance the paper already reports.
RESEARCH_ROOTS = [p for p in os.environ.get("RESEARCH_REPOS", "").split(os.pathsep) if p]


def scan_roots(roots, group_depth=1):
    units, files_scanned, findings = {}, 0, []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", "tests", "test", ".git"}]
            rel = os.path.relpath(dirpath, root)
            unit = (os.path.basename(root) if group_depth == 0 else
                    (rel.split(os.sep)[0] if rel != "." else "_top"))
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                files_scanned += 1
                h = scan_file(os.path.join(dirpath, fn))
                if h:
                    units[unit] = units.get(unit, 0) + len(h)
                    findings.append({"unit": unit, "file": os.path.join(rel, fn), "hits": h[:2]})
    return {"roots": roots, "files_scanned": files_scanned,
            "units_with_pattern": len(units), "hits_total": sum(units.values()),
            "by_unit": dict(sorted(units.items(), key=lambda z: -z[1])[:25]),
            "examples": findings[:12]}


def main():
    lib_roots = [p for p in sys.path if p and os.path.isdir(p) and "site-packages" in p]
    lib_roots = list(dict.fromkeys(lib_roots)) or [sysconfig.get_paths()["purelib"]]

    libraries = scan_roots(lib_roots, group_depth=1)
    research = scan_roots([r for r in RESEARCH_ROOTS if os.path.isdir(r)], group_depth=0)
    research["n_repositories"] = sum(os.path.isdir(r) for r in RESEARCH_ROOTS)

    out = {
        "libraries": libraries,
        "research_repositories": research,
        "confirmed_instances": 1,
        "reading": ("Absence in installed libraries establishes that this is not an ecosystem-wide "
                    "library pattern. It does NOT estimate frequency in research code, which is the "
                    "population the paper argues is at risk; the research sample here is far too "
                    "small for that. One instance is confirmed: our own, reported in this paper."),
        # kept for backwards compatibility with the figure/table code
        "roots": libraries["roots"], "files_scanned": libraries["files_scanned"],
        "packages_with_pattern": libraries["units_with_pattern"],
        "hits_total": libraries["hits_total"], "by_package": libraries["by_unit"],
        "examples": libraries["examples"],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"LIBRARIES : {libraries['files_scanned']} files, "
          f"{libraries['units_with_pattern']} packages with the pattern, "
          f"{libraries['hits_total']} sites")
    print(f"RESEARCH  : {research['n_repositories']} public repos, {research['files_scanned']} files, "
          f"{research['units_with_pattern']} with the pattern, {research['hits_total']} sites")
    for k, v in list(libraries["by_unit"].items())[:12]:
        print(f"   lib {k:26} {v}")
    for k, v in list(research["by_unit"].items())[:12]:
        print(f"   res {k:26} {v}")


if __name__ == "__main__":
    main()
