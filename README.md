# Adaptive Ultra-Lightweight 12-Lead ECG Reconstruction

Code for the paper **“Adaptive Ultra-Lightweight 12-Lead ECG Reconstruction from Limited Leads Using Vectorcardiogram as Intermediate Representation.”**

The implementation is an LR–MLP cascade:

1. preprocess 12-lead ECG records and derive VCG signals;
2. split records by patient into train/validation/test sets;
3. fit ridge-regression models for all 3-lead combinations;
4. analyse VCG covariance/orthogonality stability;
5. train an MLP to reconstruct the eight independent ECG leads from the predicted VCG;
6. reconstruct the remaining limb leads with Einthoven’s law and evaluate the result.

## Repository contents

| File | Purpose |
| --- | --- |
| `data.py` | Download/read and preprocess PTB Diagnostic ECG records |
| `split_data.py` | Patient-level train/validation/test split |
| `ridge_train.py` | Ridge regression for all 3-lead combinations |
| `ridge_orthogonality.py` | Temporal VCG covariance/orthogonality analysis |
| `mlp_train.py` | MLP/TCN/LSTM training from predicted VCG |
| `mlp_test.py` | Patient-level reconstruction and testing |
| `inverse_normalize_test_data.py` | Inverse normalization and Einthoven reconstruction |
| `metrics_calculator.py` | Evaluation metrics |
| `orthogonalization.py` | Gram–Schmidt utility for VCG signals |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The repository contains code only; no patient data are included. The preprocessing script expects the PTB Diagnostic ECG Database and uses the PhysioNet access terms. Download/access the dataset from [PhysioNet PTBDB](https://physionet.org/content/ptbdb/1.0.0/) and follow its usage requirements.

The scripts expect the following local inputs/outputs:

```text
data_results/
  patients/<record>/ecg_12lead.npy
  patients/<record>/vcg.npy
health_statistics.txt
```

`health_statistics.txt` records the patient-to-record relationship used by `split_data.py`; it is intentionally not included because it may contain dataset metadata. Generated arrays, model checkpoints, metrics, and plots are ignored by Git.

## Suggested execution order

```bash
python data.py
python split_data.py
python ridge_train.py
python ridge_orthogonality.py
python mlp_train.py
python mlp_test.py
python inverse_normalize_test_data.py
```

The scripts use fixed random seed `20251226` to reproduce the original split and training setup. Adjust paths and model settings in the scripts when using a different dataset layout or hardware environment.

## Citation

If you use this code, please cite:

> Adaptive Ultra-Lightweight 12-Lead ECG Reconstruction from Limited Leads Using Vectorcardiogram as Intermediate Representation.

## Disclaimer

This repository is research software and is not medical software. It has not been validated for clinical diagnosis or patient care.
