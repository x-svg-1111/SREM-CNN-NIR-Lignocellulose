# SREM-CNN Reference Code

Reference implementation of SREM-CNN for near-infrared prediction of
lignocellulosic composition in straw.

The real NIR spectra and laboratory measurements are not included. A synthetic
demo dataset with the same `.npz` schema is provided for checking the code path.

## Contents

- SREM three-region construction for 4000-5500, 5500-7500, and
  7500-10000 cm^-1.
- Independent region-wise min-max normalization and cubic-spline alignment.
- Tensor construction in manuscript shape `M x N x P` and PyTorch shape
  `M x P x N`.
- SREM-CNN, plain 1D-CNN, PLSR, SVR, and XGBoost implementations.
- SHAP Expected Gradients and region interaction decomposition.

## Quick Start

```bash
cd /path/to/srem-nir-spectra
python -m pip install -e ".[dev]"

python scripts/make_synthetic_data.py
python scripts/train_baselines_demo.py --target cellulose
python scripts/train_cnn_baseline_demo.py --target cellulose --epochs 1 --patience 1
python scripts/train_srem_demo.py --target cellulose --epochs 1 --patience 1
python scripts/run_shap_demo.py --target cellulose --epochs 1 --patience 1
pytest -q
```

The demo uses synthetic spectra and is intended to verify shapes, preprocessing,
training, and explanation logic.

## Data Schema

Prepare an `.npz` file with:

- `X`: array of shape `(n_samples, n_wavenumbers)`.
- `y`: array of shape `(n_samples, 3)`, ordered as cellulose,
  hemicellulose, and lignin.
- `wavenumbers`: array of shape `(n_wavenumbers,)`, normally descending from
  10000 to 4000 cm^-1.
- Optional `sample_ids` and `crop_labels`.

For real samples, `X` should contain the averaged spectrum used for modeling.
