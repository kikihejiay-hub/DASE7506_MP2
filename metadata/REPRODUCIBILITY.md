# Reproducibility record

## Environment

- Python: 3.12.10
- OS: Windows 11 (10.0.26200)
- CPU identifier: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- Logical processors: 16
- Evaluation thread limit: 4

## Frozen method

- Protocol: dase7506-mp2-open-hard-verifier-v5
- Method entry point: code/submission.py
- Training entry point: code/train.py
- Checkpoint: checkpoint/model.joblib
- Checkpoint bytes: 2,742,942
- SHA-256:
  15620F02F0882B872F6FBD83374818DAABAE58BFFA57CC8BD64B7CD3E82FE516

## Exact PowerShell commands

~~~powershell
cd code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_data.py

$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"

# Optional retraining
python train.py --output ../checkpoint

# Validation
python predict.py --method submission.py --checkpoint ../checkpoint `
  --split validation --output ../results/validation-reproduced-predictions.json
python score_predictions.py --split validation `
  --predictions ../results/validation-reproduced-predictions.json `
  --output ../results/validation-reproduced-score.json

# Test reproduction
python predict.py --method submission.py --checkpoint ../checkpoint `
  --split test --output ../results/test-reproduced-predictions.json
python score_predictions.py --split test `
  --predictions ../results/test-reproduced-predictions.json `
  --output ../results/test-reproduced-score.json
~~~

## Expected official results

- Validation: 762 / 798, 95.48872180451127%
- Test: 381 / 467, 81.5845824411135%

The test references were not used for training, tuning, lookup construction, or
method selection. The public test was scored once after the method was frozen.

