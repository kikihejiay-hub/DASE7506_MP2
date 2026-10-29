# DASE7506 MP2 - Selecting Reliable LLM Reasoning

This repository package reproduces the submitted selector for the
dase7506-mp2-open-hard-verifier-v5 protocol.

## Reported result

- Official test accuracy: **81.5845824411135%** (381 / 467)
- Validation accuracy: **95.48872180451127%** (762 / 798)
- Hard validation subset: **89.85915492957747%** (319 / 355)
- Checkpoint size: **2,742,942 bytes** (2.62 MiB)
- Checkpoint SHA-256:
  15620F02F0882B872F6FBD83374818DAABAE58BFFA57CC8BD64B7CD3E82FE516

The difference between validation and test is discussed in the report. The
test bank was evaluated once after freezing the method and was not used for
training, tuning, lookup tables, or method selection.

## Repository layout

- REPORT.pdf: final coursework report (at most 10 pages).
- REPORT.md: editable report source.
- code/: prediction, training, feature, and official evaluation code.
- checkpoint/model.joblib: frozen inference asset.
- results/: official predictions and score JSON files.
- metadata/: experiment log, resource record, hashes, and AI disclosure.
- licenses/: retained dataset attribution and license.

The large supplied data files are intentionally not duplicated in this
package. Place this package beside the original MP2 code/data/ directory, or
copy this package's code files into the supplied code/ directory.

## Installation

Python 3.12 was used.

~~~powershell
cd code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_data.py
~~~

On Linux/macOS, activate with source .venv/bin/activate.

## Training

~~~powershell
$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"

python train.py --output ../checkpoint
~~~

The submitted checkpoint is already included; retraining is not required for
evaluation.

## Validation reproduction

~~~powershell
python predict.py --method submission.py --checkpoint ../checkpoint `
  --split validation --output ../results/validation-reproduced-predictions.json

python score_predictions.py --split validation `
  --predictions ../results/validation-reproduced-predictions.json `
  --output ../results/validation-reproduced-score.json
~~~

Expected accuracy: 95.48872180451127.

## Test reproduction

~~~powershell
python predict.py --method submission.py --checkpoint ../checkpoint `
  --split test --output ../results/test-reproduced-predictions.json

python score_predictions.py --split test `
  --predictions ../results/test-reproduced-predictions.json `
  --output ../results/test-reproduced-score.json
~~~

Expected accuracy: 81.5845824411135.

## Before submission

Replace the two placeholders in REPORT.md and regenerate REPORT.pdf:

- [YOUR NAME]
- [YOUR STUDENT ID]

Then push this directory to a public GitHub repository and submit:

1. the official test accuracy value;
2. the GitHub repository URL as the code link;
3. a direct link to checkpoint/model.joblib (or its containing release) as the
   checkpoint link.

