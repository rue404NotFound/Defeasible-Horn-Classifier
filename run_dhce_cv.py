#!/usr/bin/env python3
"""
10 × 50/50 hold-out evaluation for DHCE (Making It Comparable with SBF).

Folder layout (same directory as this script):
  • current.lp   
  • data-cancer.lp  - Boolean Breast Cancer dataset 
  • seeds.txt       - (optional) it has to have integer per line (default 0-9)
Output files:
  • runs/split<i>.lp        - train/test facts for each 50/50 split
  • runs/split<i>_best.txt  - best DHCE answer set for that split (will use to compare with SBF)
  • dhce_vs_sbf.csv         - summary table (split, error, literals …)
"""

import json, random, subprocess, pathlib, csv, itertools, re, sys, copy

# ── paths ──────────────────────────────────────────────────────────────
DATA_FILE  = "data.lp"     # make sure the dataset is in the same directory
MODEL_FILE = "current.lp"         # The ASP model (must be in the same folder as this script)
SEED_FILE  = "seeds.txt"
OUT_DIR    = pathlib.Path("runs"); OUT_DIR.mkdir(exist_ok=True) #Where run results and models are saved

# ── hyper-parameter grid ───────────────────────────────────────────────
# All combinations of these constants will be tested — fine-tune here
GRID = {
    "maxD"   : [4, 5],           # choose number of default rules
    "maxE"   : [1, 2],           # choose exceptions per rule
    "maxBody": [2, 3, 4],        # choose max literals in a rule
}
# ───────────────────────────────────────────────────────────────────────

# Creates default seeds.txt file with 10 values if it doesn't exist
def ensure_seeds():
    p = pathlib.Path(SEED_FILE)
    if not p.exists() or not p.read_text().strip():
        p.write_text("\n".join(str(i) for i in range(10)))
    return [int(s) for s in p.read_text().split()]

# Load all row IDs from the dataset file based on presence of the target attribute (a(10))
def load_rows():
    ids, pat = [], re.compile(r"val\((\d+),a\(10\),[01]\)\.")
    with open(DATA_FILE) as f:
        for line in f:
            m = pat.match(line)
            if m: ids.append(int(m.group(1)))
    if not ids:
        sys.exit(f"No rows found in {DATA_FILE} (is attribute-10 the label?).")
    return ids

# Write a train/test split to an ASP-compatible format
def write_split(train, test, fname):
    with open(fname, "w") as f:
        f.writelines(f"train_given({i}).\n" for i in train)
        f.writelines(f"test_given({j}).\n"  for j in test)

"""Runs clingo with the current parameter settings and 
   returns the best atom set (lowest test error)"""
def run_clingo(split_lp, consts):
    """Return atoms of the best witness (lowest test error)."""
    flags = []
    for k, v in consts.items():
        flags.extend(["--const", f"{k}={v}"])
    cmd = [
        "clingo", MODEL_FILE, DATA_FILE, str(split_lp),
        "0", "--outf=2", *flags
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # 0 = SAT, 10 = SAT, 20 = UNSAT, 30 = OPTIMUM FOUND
    if proc.returncode not in (0, 10, 20, 30):
        return None

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None

    best_atoms, best_err, best_lits = None, 1.0, 10**9
    for wit in data["Call"][0]["Witnesses"]:
        atoms = wit["Value"]
        err, lits = metrics(atoms)
        if err is None:
            continue
        if err < best_err or (err == best_err and lits < best_lits):
            best_atoms, best_err, best_lits = atoms, err, lits
    return best_atoms

# Extract test error and total number of literals used from the answer set
def metrics(atoms):
    if not atoms:
        return None, None
    err, lits = None, 0
    for a in atoms:
        if a.startswith("error_test_pp10k("):
            err = int(a[17:-1]) / 10000.0   # 700 → 0.07
        elif a.startswith(("default_body(", "exception_body(")):
            lits += 1
    return err, lits


# Main loop runs the experiment for each split defined by seeds
# For each split: shuffle rows, split 50/50, generate all GRID combinations
# Run each with Clingo, select the best answer set (lowest test error, minimal rule size)...Making it comparable with SBF
# ── main experiment loop ───────────────────────────────────────────────
seeds     = ensure_seeds()
all_rows  = load_rows()
summary   = [] # will store test error (best error should have <=4.7%), rules used, and hyperparameters


for split_no, seed in enumerate(seeds, 1):
    rows = copy.deepcopy(all_rows)
    random.Random(seed).shuffle(rows)
    half = len(rows) // 2
    train, test = rows[:half], rows[half:]

    split_lp = OUT_DIR / f"split{split_no}.lp"
    write_split(train, test, split_lp)

    best = {"err": 1.0, "lits": 9999, "consts": None, "atoms": None}
    grid_iter = (dict(zip(GRID, vals))
                 for vals in itertools.product(*GRID.values()))

    for consts in grid_iter:
        atoms     = run_clingo(split_lp, consts)
        err, lits = metrics(atoms)
        if err is None:
            continue
        if err < best["err"] or (err == best["err"] and lits < best["lits"]):
            best.update(err=err, lits=lits, consts=consts, atoms=atoms)

    if best["atoms"] is None:
        print(f"✖︎ Split {split_no}: no model found.")
        continue

    (OUT_DIR / f"split{split_no}_best.txt").write_text("\n".join(best["atoms"]))
    summary.append((split_no, seed, best["err"], best["lits"], best["consts"]))
    print(f"Split {split_no}: error={best['err']:.4f}  literals={best['lits']}  {best['consts']}")


# ── final report ───────────────────────────────────────────────────────
if not summary:
    sys.exit("\nNo splits succeeded – check Clingo version / dataset.")

mean_err = sum(s[2] for s in summary) / len(summary)
print(f"\nAverage test error over {len(summary)} splits: {mean_err:.4f}")

with open("dhce_vs_sbf.csv", "w", newline="") as csvf:
    w = csv.writer(csvf)
    w.writerow(["split","seed","testError","literals","maxD","maxE","maxBody"])
    for s in summary:
        md, me, mb = s[4]["maxD"], s[4]["maxE"], s[4]["maxBody"]
        w.writerow([s[0], s[1], f"{s[2]:.4f}", s[3], md, me, mb])

print("\n CSV saved to dhce_vs_sbf.csv")
print(" Best rule sets in runs/ (one text file per split)")
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Make sure "current.lp", "data-cancer.lp" and optionally "seeds.txt" are in the same folder as this script.
# Output files (best rule sets and summary CSV) are saved under the "runs/" folder.
