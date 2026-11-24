# Supplementary material for ICCIT 2026 paper 1873

**Determinism Tests Cannot Detect Per-Process Seed Divergence in Released Benchmarks**

This package contains every script, the configuration each script uses, and every result file
reported in the paper. Configurations (seeds, number of interpreters, cell lists, sample sizes,
bootstrap settings) are written as named constants at the top of each script, so each script is
its own configuration.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | one script per result (table below) |
| `suite/biosignal_c/` | snapshot of the corruption suite studied in the paper (`corruptions.py`, with the corrected BLAKE2b seed derivation) and its data loader (`data.py`) |
| `results/` | the JSON result files behind every number in the paper |

## Script to result map

| Script | Paper | Needs data? | Run time |
|---|---|---|---|
| `reproduce_prefix_defect.py` | Table I (0 within-process, 34/75 across-process mismatches) | no | ~10 s |
| `demonstrate_hash_defect.py` | single-key check (8/8 vs 1 distinct seeds) and Table II; the two electrode families (`electrode_motion`, `electrode_dropout`) need a batch of segments, so they appear as expected errors in `hash_defect.json` and are excluded from Table II | no | ~15 s |
| `leaderboard_flip.py` | Table III (ranking stability, 6 sessions, 15 pairs; each session is emulated by a salt value, `SALTS`, as described in Section IV-B) | Apnea-ECG | minutes |
| `leaderboard_ci.py` | intervals of Table III and Section VII (cluster bootstrap over cells, B = 20,000) and the family-level sensitivity check of Section IV-C ([0.11, 0.33]) | no (reads `results/`) | ~5 s |
| `scan_hash_seeding.py` | Table IV (library scan) | no | minutes |
| `validate_scan_rule.py` | detector validation (7 synthetic cases) | no | <1 s |
| `pilot_severity_calibration.py` | Section VI and Fig. 2 (severity scale) | no | ~1 s |
| `batch_invariance.py` | Section VII (18 of 33 cells change under batching) | Apnea-ECG | ~1 min |
| `batch_leaderboard.py` | Section VII ranking effect | Apnea-ECG | minutes |

## How to run

```
pip install -r requirements.txt
cd scripts
python reproduce_prefix_defect.py      # each script writes its JSON to ../results/
```

Three scripts use the public **Apnea-ECG** database, available at https://physionet.org/content/apnea-ecg/1.0.0/ (T. Penzel, G. B. Moody, R. G. Mark, A. L. Goldberger and J. H. Peter, "The apnea-ECG database," *Computers in Cardiology*, vol. 27, pp. 255-258, 2000). The ranking study uses a patient-disjoint split (65% of patients for training; 6,000 training and 2,500 test segments) and eight classifiers on band-power features. Download the database, then build the cache once:

```
set BIOSIGNAL_DATA=<folder that contains apnea-ecg-database-1.0.0>     (Windows; use export on Linux/macOS)
python -c "import sys; sys.path.insert(0, '../suite'); from biosignal_c import data; data.apnea()"
```

This writes `suite/_cache/apnea.npz`, which `leaderboard_flip.py`, `batch_invariance.py` and
`batch_leaderboard.py` read.

`scan_hash_seeding.py` scans the `site-packages` folders of the Python environment that runs it.
To also scan research repositories, list their folders in the environment variable `RESEARCH_REPOS`
(separated by `;` on Windows, `:` elsewhere).

## What to expect when re-running

* We re-ran the package from a clean copy. `reproduce_prefix_defect.py`, `pilot_severity_calibration.py`,
  `leaderboard_ci.py` and `batch_invariance.py` reproduce the shipped results exactly.
* `demonstrate_hash_defect.py` reports 8 distinct `hash()`-derived seeds out of 8 and 1 BLAKE2b seed, as in
  the paper; the example `hash()` seed values themselves change on every run. That is the defect being demonstrated.
* `validate_scan_rule.py` and `scan_hash_seeding.py` depend on the packages installed in your environment;
  the detection results on the synthetic cases do not.
* All results were produced on one platform (Windows 10, CPython 3.9.13, NumPy 1.26.4). Section III-B of the
  paper explains why other operating systems or library versions may give different data from the same seed.
* Local folder paths in the result files have been replaced by `<local path>`.

## Notes

* `suite/` is a pre-release snapshot of the corruption suite, provided only to reproduce this paper; the
  suite itself will be released with its companion resource paper.
* No patient data are included. Apnea-ECG must be obtained from PhysioNet under its own licence.
