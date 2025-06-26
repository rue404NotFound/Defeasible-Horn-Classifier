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
import json, random, subprocess, pathlib, csv, itertools, re, sys, copy, time

# ── configuration ─────────────────────────────────────────────────────
DATA_FILE      = "data.lp"      
MODEL_FILE     = "current.lp"
SEED_FILE      = "seeds.txt"
OUT_DIR        = pathlib.Path("runs"); OUT_DIR.mkdir(exist_ok=True)
CLINGO_TIMEOUT = 60             # adjust the seconds used per clingo call

# ── hyper-parameter grid ───────────────────────────────────────────────

# All combinations of these constants will be tested — fine-tune here
GRID = {                        # hyper-parameter grid (compact)
    "maxD":   [3, 4],   # choose number of default rules        
    "maxE":   [2, 3],   # choose exceptions per rule        
    "maxBody":[2],      # choose max literals in a rule        
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
    ids = []
    pat = re.compile(r"val\((\d+),a\(10\),[01]\)\.")
    with open(DATA_FILE) as f:
        for line in f:
            m = pat.match(line)
            if m:
                ids.append(int(m.group(1)))
    if not ids:
        sys.exit(
            f"No rows found in {DATA_FILE} (is attribute-10 the label?)."
        )
    return ids

# Write a train/test split to an ASP-compatible format
def write_split(train, test, fname):
    with open(fname, "w") as f:
        f.writelines(f"train_given({i}).\n" for i in train)
        f.writelines(f"test_given({j}).\n"  for j in test)


# ── run clingo and return the best atom set ─────────────────────────

def run_clingo(split_lp: pathlib.Path, consts: dict):
    flags = sum((["--const", f"{k}={v}"] for k, v in consts.items()), [])
    cmd   = ["clingo", MODEL_FILE, DATA_FILE, str(split_lp),
             "0", "--outf=2", *flags]

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=CLINGO_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, None          # timeout → skip
    runtime = time.perf_counter() - start

    if proc.returncode not in (0, 10, 20, 30):
        return None, None

    try:
        j = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, None

    best_atoms, best_err, best_lits = None, 1.0, 10**9
    for call in j.get("Call", []):
        for wit in call.get("Witnesses", []):
            atoms = wit["Value"]
            err, lits = metrics(atoms)
            if err is None:
                continue
            if err < best_err or (err == best_err and lits < best_lits):
                best_atoms, best_err, best_lits = atoms, err, lits

    return (best_atoms, runtime) if best_atoms else (None, None)

# Extract test error and total number of literals used from the answer set
def metrics(atoms):
    if not atoms:
        return None, None
    err, lits = None, 0
    for a in atoms:
        if a.startswith("error_test_pp10k("):
            err = int(a[17:-1]) / 10000.0
        elif a.startswith(("default_body(", "exception_body(")):
            lits += 1
    return err, lits

# ── main experiment loop ───────────────────────────────────────────────────
seeds    = ensure_seeds()
rows_all = load_rows()
summary  = []  # (split, seed, err, lits, consts, runtime)

for split_no, seed in enumerate(seeds, 1):
    rows = copy.deepcopy(rows_all)
    random.Random(seed).shuffle(rows)
    half = len(rows) // 2
    train, test = rows[:half], rows[half:]

    split_lp = OUT_DIR / f"split{split_no}.lp"
    write_split(train, test, split_lp)

    best = {"err": 1.0, "lits": 9999, "consts": None,
            "atoms": None, "time": None}

    for consts in (dict(zip(GRID, v))
                   for v in itertools.product(*GRID.values())):
        atoms, rt = run_clingo(split_lp, consts)
        err, lits = metrics(atoms)
        if err is None:
            continue
        if err < best["err"] or (err == best["err"] and lits < best["lits"]):
            best.update(err=err, lits=lits, consts=consts,
                        atoms=atoms, time=rt)

    if not best["atoms"]:
        print(f"Split {split_no}: no model found.")
        continue

    (OUT_DIR / f"split{split_no}_best.txt").write_text(
        f"% clingo_time_sec {best['time']:.3f}\n" +
        "\n".join(best["atoms"])
    )

    summary.append((split_no, seed, best["err"],
                    best["lits"], best["consts"], best["time"]))
    print(f"Split {split_no}: error={best['err']:.4f}  "
          f"lits={best['lits']}  t={best['time']:.2f}s  {best['consts']}")

# ───────final report ───────────────────────────────────────────────────────────────
if not summary:
    sys.exit("\nNo splits succeeded – check dataset / clingo.")

best_err   = min(summary, key=lambda x: x[2])
best_lits  = min(summary, key=lambda x: (x[3], x[2]))
mean_err   = sum(s[2] for s in summary) / len(summary)

print(f"\nBest test error: {best_err[2]:.4f} (split {best_err[0]})")
print(f"Average test error: {mean_err:.4f}")
print(f"Most compact model: {best_lits[3]} literals, "
      f"error {best_lits[2]:.4f} (split {best_lits[0]})")

with open("dhce_vs_sbf.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split","seed","testError","literals",
                "maxD","maxE","maxBody","runtimeSec"])
    for s in summary:
        md, me, mb = s[4]["maxD"], s[4]["maxE"], s[4]["maxBody"]
        w.writerow([s[0], s[1], f"{s[2]:.4f}", s[3],
                    md, me, mb, f"{s[5]:.3f}"])

print("\n CSV saved to dhce_vs_sbf.csv")
print(" Rule sets in runs/ (each begins with its clingo runtime)")
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Make sure "current.lp", "data-cancer.lp"/"data.lp" and optionally "seeds.txt" are in the same folder as this script.
# Output files (best rule sets and summary CSV) are saved under the "runs/" folder.