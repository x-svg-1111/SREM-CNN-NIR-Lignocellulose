# Detail Audit Against Manuscript

This audit maps each code-relevant manuscript detail to the public
implementation. "Exact" means the repository implements the detail directly.
"Documented" means the detail belongs to acquisition/private data rather than
public code. "Configurable" means the public helper exposes the behavior but
the demo may use smaller runtime settings.

| Manuscript detail | Status | Public implementation |
| --- | --- | --- |
| 550 straw samples | Exact for demo schema | `scripts/make_synthetic_data.py` defaults to 550 synthetic samples; private data remain excluded. |
| Wheat, maize, rice, rapeseed, cotton | Documented | `data_demo` uses synthetic crop labels; public data are not the real crop distribution. |
| Three target order: cellulose, hemicellulose, lignin | Exact | `src/srem_nir/paper_config.py` and scripts use this target order. |
| Spectrum 10000-4000 cm^-1 | Exact | Synthetic and expected private schemas use descending `wavenumbers`. |
| Three scans averaged | Documented | Public schema expects `X` to contain one sample-level mean spectrum. |
| Split 6:2:2 | Exact | `ks_train_random_val_test_split`. |
| KS selects 330 training samples by Euclidean spectral distance | Exact for n=550 | `kennard_stone` selects 60% training spectra. |
| Remaining 220 split equally into validation/test | Exact for n=550 | Remainder is randomly shuffled with seed 2025, then split 110/110. |
| Test set excluded from training/HPO | Exact | Demo scripts fit preprocessing and models without test labels, then evaluate test last. |
| SG, SNV, MSC preprocessing | Exact | `src/srem_nir/spectra/preprocessing.py`. |
| SG window 5-45, derivative 1-2, polyorder 2 | Exact | `SGConfig` and validation in `savitzky_golay_derivative`. |
| Preprocessed spectra standardized | Exact for baselines; SREM-specific for SREM | Baselines use train-fitted `StandardScaler`; SREM uses the paper's region-wise min-max input normalization. |
| Chemical values standardized | Exact | Neural nets use `TargetStandardizer`; scikit baselines use `TransformedTargetRegressor`. |
| SREM Region 1: 4000-5500 cm^-1 | Exact | `DEFAULT_REGIONS`. |
| SREM Region 2: 5500-7500 cm^-1 | Exact | `DEFAULT_REGIONS`. |
| SREM Region 3: 7500-10000 cm^-1 | Exact | `DEFAULT_REGIONS`. |
| Independent region-wise min-max to [0,1] | Exact | `normalize_regions_minmax`; cubic overshoot is clipped back to [0,1]. |
| Cubic spline alignment | Exact | `align_regions_spline`. |
| Tensor shape M x N x P | Exact | `stack_regions_mnp`. |
| PyTorch input shape M x P x N | Exact implementation detail | `to_torch_channel_first`. |
| Differentiated convolutional feature extraction across channels | Exact | `SREMCNN` splits the three channels into region-specific branches before fusion. |
| Three Conv1d layers | Exact | Each SREM branch and the plain 1D-CNN baseline use three Conv1d blocks. |
| SE-block attention | Exact via Table S1 | `SE1`, `SE2`, and `SE3` are explicit switches for adding SE attention after Conv1, Conv2, and Conv3. |
| Flatten layer and one fully connected layer | Exact | `SREMCNN.regressor` and `PlainCNN1D.regressor`. |
| Dropout after FC | Exact | Dropout follows the GELU-activated dense layer. |
| GELU activation | Exact | `nn.GELU`. |
| Early stopping min delta 0.001, patience 30, max 300 | Exact defaults | `TrainConfig`; demo CLIs can be shortened with flags. |
| Augmentation perturbs preprocessed training spectra | Exact in public helper | `perturb_training_spectra` adds conservative noise, drift, and small shifts. |
| PLSR, SVR, XGB, 1D-CNN baselines | Exact | `src/srem_nir/baselines/regression.py`, `PlainCNN1D`, and demo scripts. |
| Table S1 baseline hyperparameters | Exact | `src/srem_nir/paper_config.py`; mirrored in `configs/paper_hyperparameters.yaml`. |
| Table S1 SREM-CNN hyperparameters | Exact | `paper_srem_params` feeds `SREMCNN`. |
| Metrics RMSEtrain/val/test and R2train/val/test | Exact helpers | `regression_metrics`; scripts print compact RMSE summaries. |
| SHAP Expected Gradients by GradientExplainer | Exact | `gradient_explainer_shap_values`. |
| Background distribution from training samples | Configurable | SHAP demo samples from training data; pass larger `--background-size` for fuller runs. |
| Global validation mean absolute SHAP | Exact | `mean_abs_region_importance` and `mean_abs_wavelength_importance`. |
| SHAP max normalization | Exact | `normalize_importance`. |
| High-contribution feature threshold | Configurable | `high_contribution_mask`, default 0.8 because the DOCX extraction omitted the equation's numeric threshold. |
| Interaction masking by retained regions | Exact | `mask_regions`. |
| Pairwise inclusion-exclusion interaction | Exact | `pairwise_region_interactions`. |
| Coupling coefficient as mean absolute interaction over validation samples | Exact | `pairwise_region_interactions` returns `coupling_matrix`. |
| Zero baseline under standardized spectra | Exact convention in helper | `baseline=0.0` default; for SREM min-max tensors this is the lower bound channel value. |
| Python 3.12.0, PyTorch 2.8.0, CUDA 12.8, RTX 4090D | Documented | README/audit record the paper environment; package metadata stays flexible for public inspection. |
| Global random seed 2025 | Exact | Split, synthetic data, augmentation, model training, and XGBoost use seed 2025. |

## Items Not Published

- Real sample spectra, laboratory values, scan replicates, sample identifiers,
  split files, trained weights, raw predictions, and unpublished experiment
  histories.
- Exact numerical reproduction of Table 2, Figure 6, and Figure S1, because
  those require the private data and trained model artifacts.
