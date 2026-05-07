# Manuscript Alignment Notes

This repository contains the public reference implementation of the method
described in the SAA manuscript.

The notes below summarize the code-level correspondence between the manuscript
method description and the implementation.

## Confirmed Manuscript Method

- Dataset: 550 straw samples from wheat, maize, rice, rapeseed, and cotton,
  with targets ordered as cellulose, hemicellulose, and lignin.
- Spectrum: diffuse-reflectance FT-NIR, 10000-4000 cm^-1. The public schema
  expects one averaged spectrum per sample after the three scan replicates have
  been arithmetically averaged.
- Split: 6:2:2. Kennard-Stone selects 330 training samples; the remaining 220
  samples are randomly divided into 110 validation and 110 independent test
  samples.
- Preprocessing candidates: SG derivative smoothing, SNV, and MSC. SG uses
  window widths 5-45, derivative orders 1-2, and polynomial order 2.
- Scale handling: baseline spectra are standardized inside the regression
  pipelines; neural-network targets are standardized during training. SREM
  inputs additionally use the manuscript-specific independent region-wise
  min-max normalization.
- SREM regions:
  - Region 1: 4000-5500 cm^-1, combination band.
  - Region 2: 5500-7500 cm^-1, first overtone.
  - Region 3: 7500-10000 cm^-1, second and higher overtone.
- SREM feature construction:
  - independently min-max normalize each region to [0, 1];
  - align region lengths by cubic spline interpolation;
  - stack regions into an `M x N x P` tensor.
- SREM-CNN:
  - three Conv1d blocks per spectral-region branch;
  - SE attention inserted according to the target-specific Table S1 SE mode,
    where `SE1`, `SE2`, and `SE3` mean adding an SE block after Conv1, Conv2,
    and Conv3, respectively;
  - GELU activation;
  - flatten fusion;
  - one fully connected layer;
  - dropout after the fully connected layer;
  - validation-loss early stopping, patience 30, min delta 0.001, max 300
    epochs.
- Baselines: PLSR, SVR, XGBoost, and plain 1D-CNN with Table S1
  hyperparameters.
- Explanation:
  - SHAP Expected Gradients via `shap.GradientExplainer`;
  - validation-set global mean absolute SHAP values;
  - max normalization of SHAP importance;
  - inclusion-exclusion interaction decomposition between spectral regions.
- Runtime reported in the manuscript: Python 3.12.0, PyTorch 2.8.0, CUDA 12.8,
  RTX 4090D, global seed 2025.

## Implementation Scope

- The implementation focuses on the final SREM pipeline: three physical
  regions, independent min-max normalization, cubic-spline alignment, and the
  published model settings.
- The dense 10000-to-4000 cm^-1 synthetic demo axis uses 3001 points, producing
  route lengths 750, 1000, and 1251 before alignment. The code itself also
  supports other wavenumber grids.
- Real spectra, sample identifiers, split files, trained weights, true
  prediction outputs, and raw SHAP/interaction arrays are not included.

## Manuscript Ambiguities Handled In Code

- The prose says "one SE-block attention module", while Table S1 lists SE Mode
  `2,3` for cellulose and hemicellulose SREM-CNN. The code expresses this as
  `SE2: true` and `SE3: true`, meaning attention follows Conv2 and Conv3.
- The SHAP high-contribution threshold is present in equation form in the
  manuscript, but the extracted DOCX text does not preserve the numeric
  threshold. The helper exposes a configurable threshold and defaults to 0.8,
  matching the high-importance ranges discussed in the results text.
- The interaction section defines masked baselines as zero under standardized
  spectra. SREM input channels are min-max normalized, but the public
  interaction helper keeps the paper's zero-baseline masking convention and
  documents it in the function signature.
