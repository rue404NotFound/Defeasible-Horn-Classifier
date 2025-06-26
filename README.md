# DHCE:
This work implements the DHCE classifier using ASP. It then compares performance against the Short Boolean Formula (SBF) baseline using 10× 50/50 train-test splits.

# Files:
current.lp        - DHCE logic program (the actual classifier)
run_dhce_cv.py    - Python script for 10×50/50 evaluation with grid search
data.lp           - Boolean dataset (attribute 10 is the class label)
seeds.txt         - Optional list of 10 integer seeds (one per line)(0...9)
requirements.txt  - Dependencies
runs/             - Folder for generated split files and best models (to compare with SBF)

# What current.lp Does:
Defines a defeasible Horn-rule classifier using ASP. It:
- Learns `maxD` default rules, each with at most `maxBody` literals
- Allows up to `maxE` exceptions per rule
- Applies rules to training data and evaluates on test data
- Outputs classification errors (`error_pp10k`, `error_test_pp10k`), rule bodies, and coverage stats

# How To run it manually:
'clingo current.lp data.lp --const maxD=4 --const maxE=1 --const maxBody=3'

Output Expectation:
- `default_body/2` and `exception_body/3` atoms for rule interpretation
- Misclassification counts and test error
- Optimization score (overall error)

# How to Run the Python Evaluation:
To evaluate DHCE across 10 splits, like SBF did run the following:
  'python run_dhce_cv.py"

# This script will:
- Generate 10 random 50/50 splits of the dataset (train/test)
- Try all combinations of `maxD`, `maxE`, and `maxBody`
- Select the best-performing model per split
- Save each best model to `runs/split<i>_best.txt`
- Write a summary CSV file `dhce_vs_sbf.csv` with error rates and rule sizes
Each split is evaluated independently using the ASP classifier in `current.lp`.

# Expected Output:
After running the Python script:
- Per-split results will appear in `runs/`
- Summary statistics (split number, test error, rule size, parameters) in `dhce_vs_sbf.csv`
- Console will show best error per split and average test error

# Requirements:
Install dependencies by running:
'pip install -r requirements.txt'



